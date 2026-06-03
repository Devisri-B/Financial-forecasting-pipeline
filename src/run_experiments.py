"""
Experimental Analysis: A, B, C configurations on multi-ticker data.
Demonstrates hyperparameter optimization and MLflow experiment tracking.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import mlflow
import os
import time
import copy
from features import FeatureEngineer
from model import StockPredictor
from sklearn.preprocessing import StandardScaler
import joblib
from train_single_ticker import create_sequences, EarlyStopping, set_seeds, train_lstm_model, direction_accuracy
from hydra.utils import get_original_cwd

class ExperimentRunner:
    """Run multiple hyperparameter configurations."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = self._prepare_data()
    
    def _prepare_data(self):
        """Load and prepare multi-ticker data."""
        print("Loading multi-ticker dataset...")
        
        multi_ticker_path = os.path.join(
            get_original_cwd(), 
            self.cfg.data.raw_path.replace('.csv', '_multi_ticker.csv')
        )
        df = pd.read_csv(multi_ticker_path)
        
        # Feature engineering
        feature_engineer = FeatureEngineer(
            use_technical_indicators=self.cfg.features.use_technical_indicators,
            target=self.cfg.features.get("target", "return"),
            vol_horizon=self.cfg.features.get("vol_horizon", 5),
        )
        self.target_name = feature_engineer.target_col
        self.target_kind = feature_engineer.target
        processed_data = []

        for ticker in df['Ticker'].unique():
            ticker_df = df[df['Ticker'] == ticker].copy()
            ticker_df = ticker_df.drop('Ticker', axis=1).set_index('Date')
            ticker_processed = feature_engineer.transform(ticker_df)
            processed_data.append(ticker_processed)

        # Build sequences PER TICKER (no cross-ticker contamination). Features
        # exclude the forward-looking target so it can't leak into X.
        feature_cols = [c for c in processed_data[0].columns
                        if c != FeatureEngineer.VOL_TARGET]
        X_parts, y_parts = [], []
        for p in processed_data:
            Xi, yi = create_sequences(
                p[feature_cols].values, self.cfg.data.window_size,
                target=p[self.target_name].values,
            )
            if len(Xi):
                X_parts.append(Xi)
                y_parts.append(yi)
        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)

        # Split BEFORE scaling, then fit the scaler on TRAIN ONLY (no leakage).
        n = len(X)
        train_idx = int(n * 0.7)
        val_idx = int(n * 0.85)

        n_features = X.shape[2]
        scaler = StandardScaler()
        scaler.fit(X[:train_idx].reshape(-1, n_features))

        def _scale(arr):
            return scaler.transform(arr.reshape(-1, n_features)).reshape(arr.shape)

        X_train = _scale(X[:train_idx])
        X_val = _scale(X[train_idx:val_idx])
        X_test = _scale(X[val_idx:])

        return {
            'X_train': torch.FloatTensor(X_train).to(self.device),
            'y_train': torch.FloatTensor(y[:train_idx]).unsqueeze(1).to(self.device),
            'X_val': torch.FloatTensor(X_val).to(self.device),
            'y_val': torch.FloatTensor(y[train_idx:val_idx]).unsqueeze(1).to(self.device),
            'X_test': torch.FloatTensor(X_test).to(self.device),
            'y_test': torch.FloatTensor(y[val_idx:]).unsqueeze(1).to(self.device),
            'input_dim': n_features,
            'scaler': scaler,
        }
    
    def run_experiment(self, exp_name, exp_config):
        """Run a single experiment configuration."""
        
        print(f"\n{'='*60}")
        print(f"Running Experiment: {exp_name}")
        print(f"{'='*60}")
        
        set_seeds(self.cfg.app.random_state)
        
        # MLflow setup
        tracking_uri = f"sqlite:///{get_original_cwd()}/mlruns.db"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.cfg.app.name)
        
        run_name = f"exp_{exp_name}_lr={exp_config['learning_rate']}_hd={exp_config['hidden_dim']}"
        
        with mlflow.start_run(run_name=run_name):
            start_time = time.perf_counter()
            
            # Log experiment config
            mlflow.log_params({
                "experiment": exp_name,
                "learning_rate": exp_config['learning_rate'],
                "hidden_dim": exp_config['hidden_dim'],
                "num_layers": exp_config['num_layers'],
                "dropout": exp_config['dropout'],
                "batch_size": exp_config.get('batch_size', 32),
                "patience": exp_config['patience'],
            })
            
            mlflow.log_param("description", exp_config.get('description', ''))
            
            # Initialize model
            model = StockPredictor(
                self.data['input_dim'],
                exp_config['hidden_dim'],
                exp_config['num_layers'],
                dropout=exp_config['dropout']
            ).to(self.device)
            
            criterion = nn.MSELoss()

            print(f"Model: hidden={exp_config['hidden_dim']}, layers={exp_config['num_layers']}, "
                  f"dropout={exp_config['dropout']}, lr={exp_config['learning_rate']}")

            # Proper mini-batch training (shared helper).
            def _log_epoch(epoch, train_loss, val_loss):
                if epoch % 10 == 0:
                    print(f"  Epoch {epoch}: Train={train_loss:.6f} | Val={val_loss:.6f}")
                mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            train_losses, val_losses = train_lstm_model(
                model, self.data['X_train'], self.data['y_train'],
                self.data['X_val'], self.data['y_val'],
                epochs=exp_config.get('epochs', 150),
                lr=exp_config['learning_rate'],
                batch_size=exp_config.get('batch_size', 64),
                weight_decay=exp_config.get('weight_decay', 1e-4),
                grad_clip=exp_config.get('grad_clip', 1.0),
                patience=exp_config['patience'],
                on_epoch=_log_epoch,
            )
            
            # Test evaluation
            model.eval()
            with torch.no_grad():
                test_pred = model(self.data['X_test'])
                test_mse = criterion(test_pred, self.data['y_test'])
                
                # R² calculation
                y_mean = torch.mean(self.data['y_test'])
                ss_res = torch.sum((self.data['y_test'] - test_pred) ** 2)
                ss_tot = torch.sum((self.data['y_test'] - y_mean) ** 2)
                test_r2 = 1 - (ss_res / ss_tot)
            
            # Additional metrics
            mae = torch.mean(torch.abs(self.data['y_test'] - test_pred))
            rmse = torch.sqrt(test_mse)

            # Direction accuracy (high/low vol for volatility; up/down for returns)
            y_train_np = self.data['y_train'].cpu().numpy()
            threshold = float(np.median(y_train_np)) if self.target_kind == "volatility" else 0.0
            dir_acc = direction_accuracy(
                self.data['y_test'].cpu().numpy(), test_pred.cpu().numpy(), threshold
            )

            # Log final metrics
            metrics = {
                "test_mse": test_mse.item(),
                "test_rmse": rmse.item(),
                "test_r2": test_r2.item(),
                "test_mae": mae.item(),
                "test_directional_accuracy": dir_acc,
                "final_train_loss": train_losses[-1],
                "final_val_loss": val_losses[-1],
                "converged_epoch": len(train_losses),
            }
            
            mlflow.log_metrics(metrics)
            
            runtime = time.perf_counter() - start_time
            # Include runtime in both MLflow and returned metrics for summary printing
            metrics["runtime_sec"] = runtime
            mlflow.log_metric("runtime_sec", runtime)
            
            print(f"\n Results for {exp_name}:")
            print(f"   R²: {test_r2.item():.4f}")
            print(f"   Directional Accuracy: {dir_acc:.2f}%")
            print(f"   MSE: {test_mse.item():.6f}")
            print(f"   RMSE: {rmse.item():.4f}")
            print(f"   MAE: {mae.item():.4f}")
            print(f"   Converged at epoch: {len(train_losses)}")
            print(f"   Runtime: {runtime:.1f}s")
            
            return metrics

@hydra.main(config_path="../config", config_name="main", version_base=None)
def run_experiments(cfg: DictConfig):
    """Run A, B, C experiments."""
    
    runner = ExperimentRunner(cfg)
    
    # Define experiment configurations
    # Three configurations for the mini-batch trainer (grad-clipping + LR scheduler).
    experiments = {
        'A': {
            'description': 'Baseline: Conservative hyperparameters',
            'learning_rate': 0.0005,
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'patience': 10,
            'epochs': 100,
        },
        'B': {
            'description': 'Optimized: Best from tuning',
            'learning_rate': 0.001,
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'patience': 12,
            'epochs': 120,
        },
        'C': {
            'description': 'Aggressive: High capacity with strong regularization',
            'learning_rate': 0.002,
            'hidden_dim': 256,
            'num_layers': 3,
            'dropout': 0.3,
            'batch_size': 128,
            'weight_decay': 3e-4,
            'patience': 15,
            'epochs': 150,
        }
    }
    
    results = {}
    for exp_name, exp_config in experiments.items():
        results[exp_name] = runner.run_experiment(exp_name, exp_config)
    
    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    for exp_name, metrics in results.items():
        print(f"\nExperiment {exp_name}:")
        print(f"  R²: {metrics['test_r2']:.4f}")
        print(f"  MSE: {metrics['test_mse']:.6f}")
        print(f"  Runtime: {metrics['runtime_sec']:.1f}s")
    
    best_exp = max(results.items(), key=lambda x: x[1]['test_r2'])
    print(f"\n Best Experiment: {best_exp[0]} with R²={best_exp[1]['test_r2']:.4f}")

if __name__ == "__main__":
    run_experiments()
