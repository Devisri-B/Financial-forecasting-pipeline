"""
Multi-ticker training with ensemble methods.
Now that we have sufficient data (11K samples), ensemble can work properly.
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
from ensemble import EnsemblePredictor
from sklearn.preprocessing import StandardScaler
import joblib
from train import create_sequences, EarlyStopping, set_seeds
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
        feature_engineer = FeatureEngineer(use_technical_indicators=cfg.features.use_technical_indicators)
        
        processed_data = []
        for ticker in df['Ticker'].unique():
            ticker_df = df[df['Ticker'] == ticker].copy()
            ticker_df = ticker_df.drop('Ticker', axis=1).set_index('Date')
            ticker_processed = feature_engineer.transform(ticker_df)
            processed_data.append(ticker_processed)
            print(f"  {ticker}: {len(ticker_processed)} samples")
        
        all_features = pd.concat(processed_data, ignore_index=True)
        print(f"\n Total: {len(all_features)} samples after feature engineering")
        
        # Prepare data
        feature_cols = [col for col in all_features.columns if col != 'Close']
        data = all_features[feature_cols + ['Close']].values
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        os.makedirs(os.path.join(get_original_cwd(), "models"), exist_ok=True)
        joblib.dump(scaler, os.path.join(get_original_cwd(), "models/scaler_ensemble_multi.pkl"))
        
        # Create sequences
        X, y = create_sequences(data_scaled, cfg.data.window_size)
        print(f"Sequences: X={X.shape}, y={y.shape}")
        
        # Split: 70% train, 15% val, 15% test
        n = len(X)
        train_idx = int(n * 0.7)
        val_idx = int(n * 0.85)
        
        X_train, y_train = X[:train_idx], y[:train_idx]
        X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
        X_test, y_test = X[val_idx:], y[val_idx:]
        
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
        
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
        criterion = nn.MSELoss()
        early_stopper = EarlyStopping(patience=cfg.model.patience)
        
        mlflow.log_params({
            "learning_rate": cfg.model.learning_rate,
            "hidden_dim": cfg.model.hidden_dim,
            "num_layers": cfg.model.num_layers,
            "dropout": cfg.model.dropout,
            "patience": cfg.model.patience,
        })
        
        print("\n=== Training LSTM ===")
        train_losses, val_losses = [], []
        
        for epoch in range(cfg.model.epochs):
            model.train()
            optimizer.zero_grad()
            
            pred = model(X_train_t)
            loss = criterion(pred, y_train_t)
            
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred, y_val_t)
            
            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train={loss.item():.4f} | Val={val_loss.item():.4f}")
            
            mlflow.log_metrics({"train_loss": loss.item(), "val_loss": val_loss.item()}, step=epoch)
            
            early_stopper(val_loss.item(), model)
            if early_stopper.early_stop:
                print(f"Early stopping at epoch {epoch}")
                model.load_state_dict(early_stopper.best_model_state)
                break
        
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
        
        # Ensemble predictions on test set
        ensemble_pred = ensemble.predict(X_test)
        ensemble_mse = np.mean((y_test - ensemble_pred) ** 2)
        
        y_mean_np = np.mean(y_test)
        ensemble_ss_res = np.sum((y_test - ensemble_pred) ** 2)
        ensemble_ss_tot = np.sum((y_test - y_mean_np) ** 2)
        ensemble_r2 = 1 - (ensemble_ss_res / ensemble_ss_tot)
        
        print(f"Ensemble Results: R²={ensemble_r2:.4f}, MSE={ensemble_mse:.4f}")
        
        improvement = ((ensemble_r2 - lstm_r2.item()) / abs(lstm_r2.item() + 1e-6)) * 100
        print(f"Improvement: {improvement:+.1f}%")
        
        # Log metrics
        mlflow.log_metrics({
            "lstm_test_r2": lstm_r2.item(),
            "lstm_test_mse": lstm_mse.item(),
            "ensemble_test_r2": ensemble_r2,
            "ensemble_test_mse": ensemble_mse,
            "ensemble_improvement_pct": improvement,
        })
        
        # Save artifacts
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
        plt.plot(ensemble_pred[:test_samples], label=f'Ensemble (R²={ensemble_r2:.3f})', linestyle=':', alpha=0.7, linewidth=2)
        plt.xlabel('Sample')
        plt.ylabel('Normalized Price')
        plt.legend()
        plt.title('Predictions Comparison (First 200 test samples)')
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(artifacts_dir, "ensemble_results.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        mlflow.log_artifact(plot_path)
        
        # Save models
        torch.save(model.state_dict(), os.path.join(get_original_cwd(), "models/lstm_multi_ticker.pth"))
        
        runtime = time.perf_counter() - start_time
        mlflow.log_metric("runtime_sec", runtime)
        
        print(f"\n Training complete in {runtime:.1f}s")
        print(f"   LSTM: R²={lstm_r2.item():.4f}")
        print(f"   Ensemble: R²={ensemble_r2:.4f} ({improvement:+.1f}%)")

if __name__ == "__main__":
    train_ensemble_multi_ticker()
