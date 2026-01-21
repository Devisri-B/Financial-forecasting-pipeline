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
from train import create_sequences, EarlyStopping, set_seeds
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
        feature_engineer = FeatureEngineer(use_technical_indicators=self.cfg.features.use_technical_indicators)
        processed_data = []
        
        for ticker in df['Ticker'].unique():
            ticker_df = df[df['Ticker'] == ticker].copy()
            ticker_df = ticker_df.drop('Ticker', axis=1).set_index('Date')
            ticker_processed = feature_engineer.transform(ticker_df)
            processed_data.append(ticker_processed)
        
        all_features = pd.concat(processed_data, ignore_index=True)
        
        # Normalize
        feature_cols = [col for col in all_features.columns if col != 'Close']
        data = all_features[feature_cols + ['Close']].values
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Create sequences
        X, y = create_sequences(data_scaled, self.cfg.data.window_size)
        
        # Split
        n = len(X)
        train_idx = int(n * 0.7)
        val_idx = int(n * 0.85)
        
        return {
            'X_train': torch.FloatTensor(X[:train_idx]).to(self.device),
            'y_train': torch.FloatTensor(y[:train_idx]).unsqueeze(1).to(self.device),
            'X_val': torch.FloatTensor(X[train_idx:val_idx]).to(self.device),
            'y_val': torch.FloatTensor(y[train_idx:val_idx]).unsqueeze(1).to(self.device),
            'X_test': torch.FloatTensor(X[val_idx:]).to(self.device),
            'y_test': torch.FloatTensor(y[val_idx:]).unsqueeze(1).to(self.device),
            'input_dim': X.shape[2],
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
            
            optimizer = torch.optim.Adam(model.parameters(), lr=exp_config['learning_rate'])
            criterion = nn.MSELoss()
            early_stopper = EarlyStopping(patience=exp_config['patience'])
            
            print(f"Model: hidden={exp_config['hidden_dim']}, layers={exp_config['num_layers']}, "
                  f"dropout={exp_config['dropout']}, lr={exp_config['learning_rate']}")
            
            # Training
            train_losses, val_losses = [], []
            
            for epoch in range(exp_config.get('epochs', 150)):
                model.train()
                optimizer.zero_grad()
                
                pred = model(self.data['X_train'])
                loss = criterion(pred, self.data['y_train'])
                
                loss.backward()
                optimizer.step()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_pred = model(self.data['X_val'])
                    val_loss = criterion(val_pred, self.data['y_val'])
                
                train_losses.append(loss.item())
                val_losses.append(val_loss.item())
                
                if epoch % 10 == 0:
                    print(f"  Epoch {epoch}: Train={loss.item():.4f} | Val={val_loss.item():.4f}")
                
                mlflow.log_metrics({
                    "train_loss": loss.item(),
                    "val_loss": val_loss.item()
                }, step=epoch)
                
                early_stopper(val_loss.item(), model)
                if early_stopper.early_stop:
                    print(f"  Early stopping at epoch {epoch}")
                    model.load_state_dict(early_stopper.best_model_state)
                    break
            
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
            
            # Log final metrics
            metrics = {
                "test_mse": test_mse.item(),
                "test_rmse": rmse.item(),
                "test_r2": test_r2.item(),
                "test_mae": mae.item(),
                "final_train_loss": train_losses[-1],
                "final_val_loss": val_losses[-1],
                "converged_epoch": len(train_losses),
            }
            
            mlflow.log_metrics(metrics)
            
            runtime = time.perf_counter() - start_time
            # Include runtime in both MLflow and returned metrics for summary printing
            metrics["runtime_sec"] = runtime
            mlflow.log_metric("runtime_sec", runtime)
            
            print(f"\n✅ Results for {exp_name}:")
            print(f"   R²: {test_r2.item():.4f}")
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
    experiments = {
        'A': {
            'description': 'Baseline: Conservative hyperparameters',
            'learning_rate': 0.005,
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'patience': 10,
            'epochs': 100,
        },
        'B': {
            'description': 'Optimized: Best from tuning',
            'learning_rate': 0.01,
            'hidden_dim': 128,
            'num_layers': 3,
            'dropout': 0.05,
            'patience': 15,
            'epochs': 150,
        },
        'C': {
            'description': 'Aggressive: High capacity with strong regularization',
            'learning_rate': 0.02,
            'hidden_dim': 256,
            'num_layers': 3,
            'dropout': 0.15,
            'patience': 20,
            'epochs': 200,
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
    print(f"\n🏆 Best Experiment: {best_exp[0]} with R²={best_exp[1]['test_r2']:.4f}")

if __name__ == "__main__":
    run_experiments()
