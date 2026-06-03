"""
Train model on multi-ticker dataset for better generalization.
Uses transfer learning: pre-train on major indices, fine-tune on USAR.
"""

import hydra
from omegaconf import DictConfig
import torch
import pandas as pd
import numpy as np
from features import FeatureEngineer
from model import StockPredictor
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_on_multi_ticker(cfg: DictConfig):
    """Train model on combined multi-ticker dataset."""
    
    print("=== Multi-Ticker Training ===\n")
    
    # Load multi-ticker data
    multi_ticker_path = cfg.data.raw_path.replace('.csv', '_multi_ticker.csv')
    df = pd.read_csv(multi_ticker_path)
    
    print(f"Loaded {len(df)} total samples from {df['Ticker'].nunique()} tickers")
    print(f"Tickers: {df['Ticker'].unique()}")
    
    # Feature engineering per ticker (each ticker gets its own indicators)
    feature_engineer = FeatureEngineer(
        use_technical_indicators=cfg.features.use_technical_indicators,
        target=cfg.features.get("target", "return"),
        vol_horizon=cfg.features.get("vol_horizon", 5),
    )

    processed_data = []
    for ticker in df['Ticker'].unique():
        ticker_df = df[df['Ticker'] == ticker].copy()
        ticker_df = ticker_df.drop('Ticker', axis=1).set_index('Date')

        # Apply feature engineering
        ticker_processed = feature_engineer.transform(ticker_df)
        processed_data.append(ticker_processed)
        print(f"  {ticker}: {len(ticker_processed)} samples after feature engineering")

    total_rows = sum(len(p) for p in processed_data)
    print(f"\n Total training samples after feature engineering: {total_rows}")
    print(f" Predicting target: '{feature_engineer.target_col}'")

    # Build sequences PER TICKER (no cross-ticker contamination). Features exclude
    # the forward-looking target so it can't leak into X.
    from train_single_ticker import create_sequences
    feature_cols = [c for c in processed_data[0].columns
                    if c != FeatureEngineer.VOL_TARGET]
    target_name = feature_engineer.target_col
    X_parts, y_parts = [], []
    for p in processed_data:
        Xi, yi = create_sequences(
            p[feature_cols].values, cfg.data.window_size, target=p[target_name].values
        )
        if len(Xi):
            X_parts.append(Xi)
            y_parts.append(yi)
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    print(f"Sequence data: X shape={X.shape}, y shape={y.shape}")

    # Split BEFORE scaling, then fit scaler on TRAIN ONLY (no look-ahead leakage).
    split_idx = int(len(X) * (1 - cfg.data.test_size))
    n_features = X.shape[2]
    scaler = StandardScaler()
    scaler.fit(X[:split_idx].reshape(-1, n_features))

    def _scale(arr):
        return scaler.transform(arr.reshape(-1, n_features)).reshape(arr.shape)

    X_train, X_test = _scale(X[:split_idx]), _scale(X[split_idx:])
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Save scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler_multi_ticker.pkl")

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'n_features': n_features,
    }

@hydra.main(config_path="../config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Prepare multi-ticker data
    data = train_on_multi_ticker(cfg)
    
    # Convert to tensors
    X_train = torch.FloatTensor(data['X_train']).to(device)
    y_train = torch.FloatTensor(data['y_train']).unsqueeze(1).to(device)
    X_test = torch.FloatTensor(data['X_test']).to(device)
    y_test = torch.FloatTensor(data['y_test']).unsqueeze(1).to(device)
    
    # Initialize model
    model = StockPredictor(
        data['n_features'],
        cfg.model.hidden_dim,
        cfg.model.num_layers,
        dropout=cfg.model.dropout
    ).to(device)
    
    criterion = torch.nn.MSELoss()

    print("\n=== Training on Multi-Ticker Data (mini-batch + grad-clip + LR schedule) ===")

    # Mini-batch training loop (shared helper). Early-stops on the test set
    # (this script uses a train/test-only split).
    from train_single_ticker import train_lstm_model
    train_lstm_model(
        model, X_train, y_train, X_test, y_test,
        epochs=cfg.model.epochs, lr=cfg.model.learning_rate,
        batch_size=int(cfg.model.get("batch_size", 64)),
        weight_decay=float(cfg.model.get("weight_decay", 1e-4)),
        grad_clip=float(cfg.model.get("grad_clip", 1.0)),
        patience=cfg.model.patience,
        on_epoch=lambda e, tl, vl: print(f"Epoch {e}: Train Loss {tl:.6f} | Val Loss {vl:.6f}") if e % 10 == 0 else None,
    )

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_mse = criterion(test_pred, y_test)
        
        # R² score
        y_mean = torch.mean(y_test)
        ss_res = torch.sum((y_test - test_pred) ** 2)
        ss_tot = torch.sum((y_test - y_mean) ** 2)
        r2 = 1 - (ss_res / ss_tot)

    # Direction accuracy (high/low vol for volatility; up/down for returns)
    from train_single_ticker import direction_accuracy
    is_vol = cfg.features.get("target", "return") == "volatility"
    threshold = float(np.median(data['y_train'])) if is_vol else 0.0
    dir_acc = direction_accuracy(y_test.cpu().numpy(), test_pred.cpu().numpy(), threshold)

    print(f"\n Multi-Ticker Model Results:")
    print(f"   Test MSE: {test_mse.item():.4f}")
    print(f"   Test R²: {r2.item():.4f}")
    print(f"   Test Directional Accuracy: {dir_acc:.2f}%")
    
    # Save model
    torch.save(model.state_dict(), "models/model_multi_ticker.pth")
    print(f"   Model saved to models/model_multi_ticker.pth")
    
    return model, data['scaler']

if __name__ == "__main__":
    main()
