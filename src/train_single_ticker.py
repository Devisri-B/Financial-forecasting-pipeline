import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import mlflow
import os
import copy  # To save best model weights
import random
import time
import hashlib
import platform
import subprocess
import json
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature
from features import FeatureEngineer
from model import StockPredictor
from sklearn.preprocessing import StandardScaler
import joblib
from hydra.utils import get_original_cwd

# --- Helper Class for Early Stopping ---
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0


def set_seeds(seed: int):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length][0] 
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

@hydra.main(config_path="../config", config_name="main", version_base=None)
def train_pipeline(cfg: DictConfig):
    set_seeds(cfg.app.random_state)

    # Use a stable tracking URI so all Hydra runs land in the same MLflow DB
    tracking_uri = f"sqlite:///{get_original_cwd()}/mlruns.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.app.name)

    # Give each run a readable name for the MLflow UI
    run_name = f"lr={cfg.model.learning_rate}_hd={cfg.model.hidden_dim}_win={cfg.data.window_size}"

    with mlflow.start_run(run_name=run_name):
        start_time = time.perf_counter()

        # Optional tags to make the UI filterable
        try:
            code_version = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            code_version = "unknown"

        # Data hash for provenance
        raw_bytes = open(cfg.data.raw_path, "rb").read()
        data_hash = hashlib.md5(raw_bytes).hexdigest()
        data_size = os.path.getsize(cfg.data.raw_path)
        data_mtime = os.path.getmtime(cfg.data.raw_path)

        mlflow.set_tags({
            "stage": "exp",
            "owner": "devisri",
            "data_version": data_hash,
            "code_version": code_version,
            "notes": "",
        })

        #  Log Hyperparameters and context
        mlflow.log_params(OmegaConf.to_container(cfg.model, resolve=True))
        mlflow.log_params({
            "window_size": cfg.data.window_size,
            "raw_path": cfg.data.raw_path,
            "raw_hash_md5": data_hash,
            "raw_size_bytes": data_size,
            "raw_mtime": data_mtime,
        })

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlflow.log_params({
            "device": str(device),
            "hardware": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
        })

        # 1. Load & Process
        df = pd.read_csv(cfg.data.raw_path, index_col=0)
        engineer = FeatureEngineer(cfg.features.use_technical_indicators)
        df_processed = engineer.transform(df)
        
        os.makedirs(os.path.dirname(cfg.data.reference_path), exist_ok=True)
        df_processed.to_csv(cfg.data.reference_path)
        os.makedirs(os.path.dirname(cfg.data.processed_path), exist_ok=True)
        df_processed.to_csv(cfg.data.processed_path)
        
        # 2. Scale Data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df_processed)
        
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/scaler.pkl")
        
        # 3. Create Sequences
        X, y = create_sequences(data_scaled, cfg.data.window_size)
        
        # Split Train/Test
        train_size = int(len(X) * (1 - cfg.data.test_size))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        mlflow.log_params({
            "train_size": len(X_train),
            "test_size": len(X_test),
        })
        
        X_train = torch.FloatTensor(X_train).to(device)
        y_train = torch.FloatTensor(y_train).unsqueeze(1).to(device)
        X_test = torch.FloatTensor(X_test).to(device)
        y_test = torch.FloatTensor(y_test).unsqueeze(1).to(device)
        
        # 4. Initialize Model (Learns Uncertainty)
        input_dim = X_train.shape[2]
        model = StockPredictor(input_dim, cfg.model.hidden_dim, cfg.model.num_layers, dropout=cfg.model.dropout).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)

        # Log optimizer details
        mlflow.log_params({
            "optimizer": "adam",
            "optimizer_betas": list(optimizer.defaults.get("betas")),
            "optimizer_weight_decay": optimizer.defaults.get("weight_decay"),
        })
        
        # Use Mean Squared Error Loss for stable training
        # This forces the model to actually learn patterns instead of high uncertainty
        criterion = nn.MSELoss()
        
        # Initialize Early Stopping using the configured patience
        early_stopper = EarlyStopping(patience=cfg.model.patience, min_delta=0.001)

        train_loss_history, val_loss_history = [], []

        # 5. Training Loop
        for epoch in range(cfg.model.epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass returns prediction
            pred = model(X_train)
            loss = criterion(pred, y_train)
            
            loss.backward()
            optimizer.step()
            
            # Validation Step (for Early Stopping)
            model.eval()
            with torch.no_grad():
                val_pred = model(X_test)
                val_loss = criterion(val_pred, y_test)
            
            train_loss_history.append(loss.item())
            val_loss_history.append(val_loss.item())

            print(f"Epoch {epoch}: Train Loss {loss.item():.4f} | Val Loss {val_loss.item():.4f}")
            
            # Log Loss per Epoch 
            mlflow.log_metrics({"train_loss": loss.item(), "val_loss": val_loss.item()}, step=epoch)
            
            # Check Early Stopping
            early_stopper(val_loss.item(), model)
            if early_stopper.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                model.load_state_dict(early_stopper.best_model_state)
                break
        
        # Ensure we load the best weights if we finished loops without stopping
        if not early_stopper.early_stop and early_stopper.best_model_state:
            model.load_state_dict(early_stopper.best_model_state)

        # 6. Evaluation with Comprehensive Metrics
        model.eval()
        with torch.no_grad():
            pred = model(X_test)
            final_loss = criterion(pred, y_test)
            
        mlflow.log_metric("final_test_mse", final_loss.item())

        # Regression metrics for evaluating model quality
        mae = torch.mean(torch.abs(pred - y_test))
        mse = torch.mean((pred - y_test) ** 2)
        rmse = torch.sqrt(mse)
        
        # R² Score: measures goodness of fit (1.0 = perfect, 0 = baseline, <0 = worse than baseline)
        y_mean = torch.mean(y_test)
        ss_res = torch.sum((y_test - pred) ** 2)  # Residual sum of squares
        ss_tot = torch.sum((y_test - y_mean) ** 2)  # Total sum of squares
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calibration: mean residual (should be close to 0 if well-calibrated)
        residuals = y_test - pred
        mean_residual = torch.mean(residuals)
        std_residual = torch.std(residuals)
        
        # MAPE: Mean Absolute Percentage Error (useful for scale-independent error)
        mape = torch.mean(torch.abs((y_test - pred) / (torch.abs(y_test) + 1e-6))) * 100
        
        # Detection: Train-Val divergence indicates overfitting
        # Get final train predictions for comparison
        with torch.no_grad():
            train_pred = model(X_train)
            train_mse = torch.mean((X_train.shape[0] * (train_pred - y_train) ** 2))
        
        overfitting_ratio = rmse.item() / (torch.sqrt(train_mse).item() + 1e-6)

        # Inference latency on a single batch
        start_inf = time.perf_counter()
        with torch.no_grad():
            _ = model(X_test[:1])
        inference_ms = (time.perf_counter() - start_inf) * 1000

        mlflow.log_metrics({
            "mae": mae.item(),
            "mse": mse.item(),
            "rmse": rmse.item(),
            "r_squared": r_squared.item(),
            "mape": mape.item(),
            "mean_residual": mean_residual.item(),
            "std_residual": std_residual.item(),
            "overfitting_ratio": overfitting_ratio,
            "inference_ms": inference_ms,
        })
        
        # Save RMSE for reference
        os.makedirs(os.path.dirname(cfg.deployment.calibration_path), exist_ok=True)
        with open(cfg.deployment.calibration_path, "w") as f:
            f.write(f"RMSE: {rmse.item():.4f}\nMAE: {mae.item():.4f}")
            
        # 7. Export to ONNX (Two outputs!)
        artifacts_dir = "artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)

        # Feature list artifact
        feature_list_path = os.path.join("models", "features.txt")
        with open(feature_list_path, "w") as f:
            f.write("\n".join(df_processed.columns))

        # Save config used for the run
        config_path = os.path.join("models", "run_config.yaml")
        OmegaConf.save(cfg, config_path)

        # Feature stats for drift monitoring
        train_flat = X_train.detach().cpu().reshape(-1, input_dim).numpy()
        test_flat = X_test.detach().cpu().reshape(-1, input_dim).numpy()
        stats = {
            "train_mean": train_flat.mean(axis=0).tolist(),
            "train_std": train_flat.std(axis=0).tolist(),
            "test_mean": test_flat.mean(axis=0).tolist(),
            "test_std": test_flat.std(axis=0).tolist(),
        }
        stats_path = os.path.join(artifacts_dir, "feature_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        # Plots
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_history, label="train", linewidth=2)
        plt.plot(val_loss_history, label="val", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.title("Training and Validation Loss Curves")
        plt.grid(True, alpha=0.3)
        loss_plot_path = os.path.join(artifacts_dir, "loss_curves.png")
        plt.savefig(loss_plot_path, dpi=100, bbox_inches='tight')
        plt.close()

        # Inverse transform predictions back to original scale for interpretable plots
        # Note: We're predicting the first feature (Close price) which is at index 0
        y_test_np = y_test.detach().cpu().numpy().flatten()
        pred_np = pred.detach().cpu().numpy().flatten()
        
        # Create dummy arrays with all features for inverse transform
        # (scaler expects all features, we only care about first one)
        y_test_full = np.zeros((len(y_test_np), data_scaled.shape[1]))
        pred_full = np.zeros((len(pred_np), data_scaled.shape[1]))
        y_test_full[:, 0] = y_test_np
        pred_full[:, 0] = pred_np
        
        # Inverse transform to get original price scale
        y_test_original = scaler.inverse_transform(y_test_full)[:, 0]
        pred_original = scaler.inverse_transform(pred_full)[:, 0]
        
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_original, label="Actual", linewidth=2, alpha=0.8)
        plt.plot(pred_original, label="Predicted", linewidth=2, alpha=0.8)
        plt.xlabel("Time Step")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.title("Prediction vs Actual (Original Scale)")
        plt.grid(True, alpha=0.3)
        pred_plot_path = os.path.join(artifacts_dir, "pred_vs_actual.png")
        plt.savefig(pred_plot_path, dpi=100, bbox_inches='tight')
        plt.close()

        # Residual Analysis Plot (for detecting systematic errors and overfitting)
        residuals_np = residuals.detach().cpu().numpy().flatten()
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Residuals over time (detect patterns/trends)
        plt.subplot(1, 2, 1)
        plt.plot(residuals_np, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.xlabel("Time Step")
        plt.ylabel("Residual (Actual - Predicted)")
        plt.title("Residuals Over Time (should be random around 0)")
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Residual distribution (check for normality, check calibration)
        plt.subplot(1, 2, 2)
        plt.hist(residuals_np, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label=f'Mean={mean_residual.item():.4f}')
        plt.xlabel("Residual Value")
        plt.ylabel("Frequency")
        plt.title("Residual Distribution (Calibration Check)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        residual_plot_path = os.path.join(artifacts_dir, "residuals_analysis.png")
        plt.savefig(residual_plot_path, dpi=100, bbox_inches='tight')
        plt.close()

        # Summary artifact for quick comparison
        summary = {
            "learning_rate": cfg.model.learning_rate,
            "hidden_dim": cfg.model.hidden_dim,
            "dropout": cfg.model.dropout,
            "window_size": cfg.data.window_size,
            "final_test_mse": final_loss.item(),
            "final_test_rmse": rmse.item(),
            "mae": mae.item(),
            "mse": mse.item(),
            "r_squared": r_squared.item(),
            "mape": mape.item(),
            "mean_residual": mean_residual.item(),
            "std_residual": std_residual.item(),
            "overfitting_ratio": overfitting_ratio,
            "runtime_sec": None,  # filled below
            "data_hash": data_hash,
        }

        # TorchScript export
        dummy_input = torch.randn(1, cfg.data.window_size, input_dim)
        script_model = torch.jit.trace(model.cpu(), dummy_input)
        torchscript_path = os.path.join("models", "model_scripted.pt")
        script_model.save(torchscript_path)

        # ONNX export
        new_onnx_path = os.path.join(os.path.dirname(cfg.deployment.onnx_path), "model_fixed.onnx")
        torch.onnx.export(
            model.cpu(),
            dummy_input,
            new_onnx_path,
            input_names=["input"],
            output_names=["prediction"],  # Single output: Point prediction
            dynamic_axes={"input": {0: "batch_size"}, "prediction": {0: "batch_size"}},
            opset_version=18,
            dynamo=False,
        )
        print(f"Training complete. Model exported to {new_onnx_path}")

        # Model signature and MLflow model log
        input_example = X_test[:1].detach().cpu().numpy()
        output_example = pred[:1].detach().cpu().numpy()
        signature = infer_signature(input_example, output_example)
        mlflow.pytorch.log_model(
            model.cpu(),
            artifact_path="model_torch",
            signature=signature,
            input_example=input_example,
            pip_requirements="requirements.txt",
        )

        # Runtime metric
        runtime_sec = time.perf_counter() - start_time
        mlflow.log_metric("runtime_sec", runtime_sec)
        summary["runtime_sec"] = runtime_sec

        # Run summary artifact
        summary_path = os.path.join(artifacts_dir, "run_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Log artifacts
        mlflow.log_artifact("models/scaler.pkl")
        mlflow.log_artifact(feature_list_path)
        mlflow.log_artifact(config_path)
        mlflow.log_artifact(loss_plot_path)
        mlflow.log_artifact(pred_plot_path)
        mlflow.log_artifact(stats_path)
        mlflow.log_artifact(summary_path)
        mlflow.log_artifact(torchscript_path)
        mlflow.log_artifact(new_onnx_path)

if __name__ == "__main__":
    train_pipeline()