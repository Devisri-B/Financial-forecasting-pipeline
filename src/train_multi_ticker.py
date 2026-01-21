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
    feature_engineer = FeatureEngineer(use_technical_indicators=cfg.features.use_technical_indicators)
    
    processed_data = []
    for ticker in df['Ticker'].unique():
        ticker_df = df[df['Ticker'] == ticker].copy()
        ticker_df = ticker_df.drop('Ticker', axis=1).set_index('Date')
        
        # Apply feature engineering
        ticker_processed = feature_engineer.transform(ticker_df)
        processed_data.append(ticker_processed)
        print(f"  {ticker}: {len(ticker_processed)} samples after feature engineering")
    
    # Combine all processed data
    all_features = pd.concat(processed_data, ignore_index=True)
    print(f"\n Total training samples after feature engineering: {len(all_features)}")
    
    # Prepare data
    feature_cols = [col for col in all_features.columns if col != 'Close']
    data = all_features[feature_cols + ['Close']].values
    
    # Normalize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Save scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler_multi_ticker.pkl")
    
    # Create sequences
    from src.train_single_ticker import create_sequences
    X, y = create_sequences(data_scaled, cfg.data.window_size)
    
    print(f"Sequence data: X shape={X.shape}, y shape={y.shape}")
    
    # Train/test split
    split_idx = int(len(X) * (1 - cfg.data.test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'n_features': X_train.shape[2]
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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
    criterion = torch.nn.MSELoss()
    
    print("\n=== Training on Multi-Ticker Data ===")
    
    # Training loop (simplified - full version in train.py)
    from src.train_single_ticker import EarlyStopping
    early_stopper = EarlyStopping(patience=cfg.model.patience)
    
    for epoch in range(cfg.model.epochs):
        model.train()
        optimizer.zero_grad()
        
        pred = model(X_train)
        loss = criterion(pred, y_train)
        
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_test)
            val_loss = criterion(val_pred, y_test)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss {loss.item():.4f} | Val Loss {val_loss.item():.4f}")
        
        # Early stopping
        early_stopper(val_loss.item(), model)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch}")
            model.load_state_dict(early_stopper.best_model_state)
            break
    
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
    
    print(f"\n Multi-Ticker Model Results:")
    print(f"   Test MSE: {test_mse.item():.4f}")
    print(f"   Test R²: {r2.item():.4f}")
    
    # Save model
    torch.save(model.state_dict(), "models/model_multi_ticker.pth")
    print(f"   Model saved to models/model_multi_ticker.pth")
    
    return model, data['scaler']

if __name__ == "__main__":
    main()
