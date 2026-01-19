import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import mlflow
import os
from features import FeatureEngineer
from model import StockPredictor
from sklearn.preprocessing import StandardScaler
import joblib

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
    mlflow.set_experiment(cfg.app.name)
    
    with mlflow.start_run():
        # 1. Load & Process
        df = pd.read_csv(cfg.data.raw_path, index_col=0)
        engineer = FeatureEngineer(cfg.features.use_technical_indicators)
        df_processed = engineer.transform(df)
        
        # Save reference data
        os.makedirs(os.path.dirname(cfg.data.reference_path), exist_ok=True)
        df_processed.to_csv(cfg.data.reference_path)

        # Save processed data
        os.makedirs(os.path.dirname(cfg.data.processed_path), exist_ok=True)
        df_processed.to_csv(cfg.data.processed_path)
        
        # 2. Scale Data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df_processed)
        
        # Create the 'models' directory explicitly before saving
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/scaler.pkl")
        
        # 3. Create Sequences
        X, y = create_sequences(data_scaled, cfg.data.window_size)
        
        # Split Train/Test
        train_size = int(len(X) * (1 - cfg.data.test_size))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # To Tensor
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train).unsqueeze(1)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test).unsqueeze(1)
        
        # 4. Initialize Model
        input_dim = X_train.shape[2]
        model = StockPredictor(input_dim, cfg.model.hidden_dim, cfg.model.num_layers, dropout=cfg.model.dropout)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
        criterion = nn.MSELoss()
        
        # 5. Training Loop
        for epoch in range(cfg.model.epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}: Loss {loss.item()}")
            
        # 6. Evaluation & Calibration
        model.eval()
        with torch.no_grad():
            preds = model(X_test)
            test_loss = criterion(preds, y_test)
            residuals = torch.abs(y_test - preds).numpy()
            
        mlflow.log_metric("test_mse", test_loss.item())
        
        # Save Conformal Prediction Score
        q_95 = np.quantile(residuals, 0.95)
        # Directory already created above, but safe to keep this
        os.makedirs(os.path.dirname(cfg.deployment.calibration_path), exist_ok=True)
        with open(cfg.deployment.calibration_path, "w") as f:
            f.write(str(q_95))
            
        # 7. Export to ONNX
        dummy_input = torch.randn(1, cfg.data.window_size, input_dim)
        torch.onnx.export(
            model, dummy_input, cfg.deployment.onnx_path,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print("Training complete. Model exported.")

if __name__ == "__main__":
    train_pipeline()