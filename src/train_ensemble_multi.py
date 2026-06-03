"""
Multi-ticker training with ensemble methods.
Uses an LSTM base model and trains an ensemble on top of it.
"""

import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import mlflow
import os
import time
import matplotlib.pyplot as plt
from features import FeatureEngineer
from model import StockPredictor
from train_ensemble_single import EnsemblePredictor
from sklearn.preprocessing import StandardScaler
import joblib
from train_single_ticker import create_sequences, EarlyStopping, set_seeds, train_lstm_model, direction_accuracy
from hydra.utils import get_original_cwd

@hydra.main(config_path="../config", config_name="main", version_base=None)
def train_ensemble_multi_ticker(cfg: DictConfig):
    set_seeds(cfg.app.random_state)
    
    # MLflow setup
    tracking_uri = f"sqlite:///{get_original_cwd()}/mlruns.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.app.name)
    
    run_name = f"multi_ticker_ensemble_lr={cfg.model.learning_rate}"
    
    with mlflow.start_run(run_name=run_name):
        start_time = time.perf_counter()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("=== Loading Multi-Ticker Dataset ===\n")
        
        # Load multi-ticker data
        multi_ticker_path = os.path.join(get_original_cwd(), cfg.data.raw_path.replace('.csv', '_multi_ticker.csv'))
        df = pd.read_csv(multi_ticker_path)
        
        print(f"Loaded {len(df)} samples from {df['Ticker'].nunique()} tickers")
        mlflow.log_param("num_tickers", df['Ticker'].nunique())
        mlflow.log_param("total_raw_samples", len(df))
        
        # Feature engineering per ticker
        feature_engineer = FeatureEngineer(
            use_technical_indicators=cfg.features.use_technical_indicators,
            target=cfg.features.get("target", "return"),
            vol_horizon=cfg.features.get("vol_horizon", 5),
        )
        mlflow.log_params({
            "target": feature_engineer.target,
            "vol_horizon": feature_engineer.vol_horizon,
        })

        processed_data = []
        for ticker in df['Ticker'].unique():
            ticker_df = df[df['Ticker'] == ticker].copy()
            ticker_df = ticker_df.drop('Ticker', axis=1).set_index('Date')
            ticker_processed = feature_engineer.transform(ticker_df)
            processed_data.append(ticker_processed)
            print(f"  {ticker}: {len(ticker_processed)} samples")
        
        total_rows = sum(len(p) for p in processed_data)
        print(f"\n Total: {total_rows} samples after feature engineering")

        # Build sequences PER TICKER (no cross-ticker contamination). Features are
        # every column EXCEPT the forward-looking target, which is passed
        # separately so it can never leak into X.
        target_name = feature_engineer.target_col
        feature_cols = [c for c in processed_data[0].columns
                        if c != FeatureEngineer.VOL_TARGET]

        X_parts, y_parts = [], []
        for p in processed_data:
            Xi, yi = create_sequences(
                p[feature_cols].values, cfg.data.window_size,
                target=p[target_name].values,
            )
            if len(Xi):
                X_parts.append(Xi)
                y_parts.append(yi)
        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        print(f"Predicting '{target_name}'  |  Sequences: X={X.shape}, y={y.shape}")
        
        # Split: 70% train, 15% val, 15% test (BEFORE scaling)
        n = len(X)
        train_idx = int(n * 0.7)
        val_idx = int(n * 0.85)
        
        X_train_raw, y_train = X[:train_idx], y[:train_idx]
        X_val_raw, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
        X_test_raw, y_test = X[val_idx:], y[val_idx:]
        
        # Fit scaler ONLY on training data to prevent look-ahead bias
        X_train_2d = X_train_raw.reshape(-1, X_train_raw.shape[-1])
        scaler = StandardScaler()
        scaler.fit(X_train_2d)
        
        # Scale all sets using training statistics
        X_train = scaler.transform(X_train_2d).reshape(X_train_raw.shape)
        X_val = scaler.transform(X_val_raw.reshape(-1, X_val_raw.shape[-1])).reshape(X_val_raw.shape)
        X_test = scaler.transform(X_test_raw.reshape(-1, X_test_raw.shape[-1])).reshape(X_test_raw.shape)
        
        os.makedirs(os.path.join(get_original_cwd(), "models"), exist_ok=True)
        joblib.dump(scaler, os.path.join(get_original_cwd(), "models/scaler_ensemble_multi.pkl"))
        
        print(f"Split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
        
        mlflow.log_params({
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
        })
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)
        y_test_t = torch.FloatTensor(y_test).unsqueeze(1).to(device)
        
        # Initialize LSTM
        input_dim = X_train.shape[2]
        model = StockPredictor(input_dim, cfg.model.hidden_dim, cfg.model.num_layers, dropout=cfg.model.dropout).to(device)
        
        criterion = nn.MSELoss()
        batch_size = int(cfg.model.get("batch_size", 64))
        weight_decay = float(cfg.model.get("weight_decay", 1e-4))
        grad_clip = float(cfg.model.get("grad_clip", 1.0))

        mlflow.log_params({
            "learning_rate": cfg.model.learning_rate,
            "hidden_dim": cfg.model.hidden_dim,
            "num_layers": cfg.model.num_layers,
            "dropout": cfg.model.dropout,
            "patience": cfg.model.patience,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
        })

        print("\n=== Training LSTM (mini-batch + grad-clip + LR schedule) ===")

        def _log_epoch(epoch, train_loss, val_loss):
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train={train_loss:.6f} | Val={val_loss:.6f}")
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        train_losses, val_losses = train_lstm_model(
            model, X_train_t, y_train_t, X_val_t, y_val_t,
            epochs=cfg.model.epochs, lr=cfg.model.learning_rate,
            batch_size=batch_size, weight_decay=weight_decay, grad_clip=grad_clip,
            patience=cfg.model.patience, on_epoch=_log_epoch,
        )

        # Test LSTM
        model.eval()
        with torch.no_grad():
            lstm_pred = model(X_test_t)
            lstm_mse = criterion(lstm_pred, y_test_t)
            
            y_mean = torch.mean(y_test_t)
            ss_res = torch.sum((y_test_t - lstm_pred) ** 2)
            ss_tot = torch.sum((y_test_t - y_mean) ** 2)
            lstm_r2 = 1 - (ss_res / ss_tot)
        
        print(f"\nLSTM Results: R²={lstm_r2.item():.4f}, MSE={lstm_mse.item():.4f}")
        
        # Now train ensemble with proper train/val/test split
        print("\n=== Training Ensemble ===")
        ensemble = EnsemblePredictor(lstm_model=model, weights=[0.5, 0.3, 0.2])
        ensemble.fit(X_train, y_train, X_val, y_val)
        
        # Get predictions on all sets to detect overfitting
        with torch.no_grad():
            lstm_train_pred = model(X_train_t)
            lstm_val_pred = model(X_val_t)
        
        lstm_train_pred_np = lstm_train_pred.cpu().numpy().flatten()
        lstm_val_pred_np = lstm_val_pred.cpu().numpy().flatten()
        
        ensemble_train_pred = ensemble.predict(X_train)
        ensemble_val_pred = ensemble.predict(X_val)
        ensemble_test_pred = ensemble.predict(X_test)
        
        # Calculate R² for all sets to detect overfitting
        def calc_r2(y_true, y_pred):
            y_mean = np.mean(y_true)
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_mean) ** 2)
            return 1 - (ss_res / ss_tot)
        
        # LSTM R² on all sets
        lstm_train_r2 = calc_r2(y_train, lstm_train_pred_np)
        lstm_val_r2 = calc_r2(y_val, lstm_val_pred_np)
        lstm_test_r2 = calc_r2(y_test, lstm_pred.cpu().numpy().flatten())
        
        # Ensemble R² on all sets
        ensemble_train_r2 = calc_r2(y_train, ensemble_train_pred)
        ensemble_val_r2 = calc_r2(y_val, ensemble_val_pred)
        ensemble_test_r2 = calc_r2(y_test, ensemble_test_pred)
        
        print(f"\n=== LSTM R² by Set (Overfitting Check) ===")
        print(f"  Train: {lstm_train_r2:.4f}")
        print(f"  Val:   {lstm_val_r2:.4f}")
        print(f"  Test:  {lstm_test_r2:.4f}")
        print(f"  Overfitting indicator: Val/Test gap = {abs(lstm_val_r2 - lstm_test_r2):.4f}")
        if abs(lstm_val_r2 - lstm_test_r2) > 0.05:
            print(f"  ⚠️  HIGH OVERFITTING: Val/Test gap > 0.05")
        
        print(f"\n=== Ensemble R² by Set (Overfitting Check) ===")
        print(f"  Train: {ensemble_train_r2:.4f}")
        print(f"  Val:   {ensemble_val_r2:.4f}")
        print(f"  Test:  {ensemble_test_r2:.4f}")
        print(f"  Overfitting indicator: Val/Test gap = {abs(ensemble_val_r2 - ensemble_test_r2):.4f}")
        if abs(ensemble_val_r2 - ensemble_test_r2) > 0.05:
            print(f"  ⚠️  HIGH OVERFITTING: Val/Test gap > 0.05")
        
        # Ensemble predictions on test set
        ensemble_pred = ensemble_test_pred
        ensemble_mse = np.mean((y_test - ensemble_pred) ** 2)
        
        print(f"Ensemble Results: R²={ensemble_test_r2:.4f}, MSE={ensemble_mse:.4f}")
        
        # Absolute R2 gain of the ensemble over the LSTM (a ratio is unstable
        # when the LSTM baseline R2 is near zero).
        improvement = ensemble_test_r2 - lstm_test_r2
        print(f"R2 gain vs LSTM: {improvement:+.4f}")
        
        # Calculate additional metrics
        from sklearn.metrics import mean_absolute_error
        
        # LSTM metrics (test set)
        lstm_pred_flat = lstm_pred.cpu().numpy().flatten()
        lstm_mae = mean_absolute_error(y_test, lstm_pred_flat)
        lstm_rmse = np.sqrt(lstm_mse.item())
        lstm_mape = np.mean(np.abs((y_test - lstm_pred_flat) / (np.abs(y_test) + 1e-8))) * 100
        
        # Ensemble metrics (test set)
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        ensemble_rmse = np.sqrt(ensemble_mse)
        ensemble_mape = np.mean(np.abs((y_test - ensemble_pred) / (y_test + 1e-8))) * 100
        
        # Direction accuracy: is the prediction on the correct side of a threshold?
        # For volatility -> high vs low vol (threshold = median of TRAIN targets).
        # For returns     -> up vs down (threshold = 0).
        ensemble_pred_flat = ensemble_pred.flatten()
        dir_threshold = float(np.median(y_train)) if feature_engineer.target == "volatility" else 0.0
        lstm_dir_accuracy = direction_accuracy(y_test, lstm_pred_flat, dir_threshold)
        ensemble_dir_accuracy = direction_accuracy(y_test, ensemble_pred_flat, dir_threshold)
        
        # Log metrics
        mlflow.log_metrics({
            "lstm_train_r2": lstm_train_r2,
            "lstm_val_r2": lstm_val_r2,
            "lstm_test_r2": lstm_test_r2,
            "lstm_test_mse": lstm_mse.item(),
            "lstm_test_rmse": lstm_rmse,
            "lstm_test_mae": lstm_mae,
            "lstm_test_mape": lstm_mape,
            "lstm_directional_accuracy": lstm_dir_accuracy,
            "ensemble_train_r2": ensemble_train_r2,
            "ensemble_val_r2": ensemble_val_r2,
            "ensemble_test_r2": ensemble_test_r2,
            "ensemble_test_mse": ensemble_mse,
            "ensemble_test_rmse": ensemble_rmse,
            "ensemble_test_mae": ensemble_mae,
            "ensemble_test_mape": ensemble_mape,
            "ensemble_directional_accuracy": ensemble_dir_accuracy,
            "ensemble_r2_gain_vs_lstm": improvement,
            "overfitting_indicator_lstm": abs(lstm_val_r2 - lstm_test_r2),
            "overfitting_indicator_ensemble": abs(ensemble_val_r2 - ensemble_test_r2),
        })
        
        # Save artifacts
        try:
            artifacts_dir = os.path.join(get_original_cwd(), "outputs/ensemble_multi_ticker")
            os.makedirs(artifacts_dir, exist_ok=True)
            
            # Plot comparison
            plt.figure(figsize=(14, 6))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss', alpha=0.7)
            plt.plot(val_losses, label='Val Loss', alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.legend()
            plt.title('Training History')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            test_samples = min(200, len(y_test))
            plt.plot(y_test[:test_samples], label='Actual', linewidth=2, alpha=0.8)
            plt.plot(lstm_pred.cpu().numpy()[:test_samples], label=f'LSTM (R²={lstm_r2.item():.3f})', linestyle='--', alpha=0.7)
            plt.plot(ensemble_pred[:test_samples], label=f'Ensemble (R²={ensemble_test_r2:.3f})', linestyle=':', alpha=0.7, linewidth=2)
            plt.xlabel('Sample')
            plt.ylabel('Normalized Price')
            plt.legend()
            plt.title('Predictions Comparison (First 200 test samples)')
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(artifacts_dir, "ensemble_results.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Log artifact safely
            try:
                mlflow.log_artifact(plot_path)
            except Exception as e:
                print(f"Warning: Could not log artifact: {e}")
        except Exception as e:
            print(f"Warning: Could not save artifacts: {e}")
            plt.close()
        
        # Save models
        torch.save(model.state_dict(), os.path.join(get_original_cwd(), "models/lstm_multi_ticker.pth"))
        
        runtime = time.perf_counter() - start_time
        mlflow.log_metric("runtime_sec", runtime)
        
        print(f"\n Training complete in {runtime:.1f}s")
        print(f"\n=== LSTM Metrics (by set) ===")
        print(f"  Train R²: {lstm_train_r2:.4f}")
        print(f"  Val R²:   {lstm_val_r2:.4f}")
        print(f"  Test R²:  {lstm_test_r2:.4f}")
        print(f"  Test RMSE: {lstm_rmse:.4f}")
        print(f"  Test MAE: {lstm_mae:.4f}")
        print(f"  Test MAPE: {lstm_mape:.2f}%")
        print(f"  Test Directional Accuracy: {lstm_dir_accuracy:.2f}%")
        
        print(f"\n=== Ensemble Metrics (by set) ===")
        print(f"  Train R²: {ensemble_train_r2:.4f}")
        print(f"  Val R²:   {ensemble_val_r2:.4f}")
        print(f"  Test R²:  {ensemble_test_r2:.4f}")
        print(f"  Test RMSE: {ensemble_rmse:.4f}")
        print(f"  Test MAE: {ensemble_mae:.4f}")
        print(f"  Test MAPE: {ensemble_mape:.2f}%")
        print(f"  Test Directional Accuracy: {ensemble_dir_accuracy:.2f}%")
        print(f"  R2 gain vs LSTM: {improvement:+.4f}")
        
        # Save calibration score
        calibration_file = os.path.join(get_original_cwd(), "models/calibration_score.txt")
        with open(calibration_file, 'w') as f:
            f.write(f"=== Ensemble Test Metrics ===\n")
            f.write(f"RMSE: {ensemble_rmse:.4f}\n")
            f.write(f"MAE: {ensemble_mae:.4f}\n")
            f.write(f"MAPE: {ensemble_mape:.2f}%\n")
            f.write(f"R²: {ensemble_test_r2:.4f}\n")
            f.write(f"Directional Accuracy: {ensemble_dir_accuracy:.2f}%\n")
            f.write(f"\n=== Overfitting Check ===\n")
            f.write(f"Train R²: {ensemble_train_r2:.4f}\n")
            f.write(f"Val R²:   {ensemble_val_r2:.4f}\n")
            f.write(f"Test R²:  {ensemble_test_r2:.4f}\n")
            f.write(f"Val/Test gap: {abs(ensemble_val_r2 - ensemble_test_r2):.6f}\n")

if __name__ == "__main__":
    train_ensemble_multi_ticker()
