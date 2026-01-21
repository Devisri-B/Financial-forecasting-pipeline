# MLflow Experiment Tracking Showcase

## What is MLflow?
MLflow is an open-source platform for managing ML workflow, including:
- **Experiment Tracking**: Log parameters, metrics, and artifacts
- **Model Registry**: Version control for trained models
- **Model Serving**: Deploy models as REST APIs
- **Runs Management**: Compare multiple experiments

## Our Implementation

### 1. Experiment Architecture
```
┌─────────────────────────────────────────┐
│      MLflow Experiment Tracker          │
│  Financial-forecasting-pipeline         │
└──────────────────┬──────────────────────┘
        │
        ├─ Experiment A: Conservative
        │   └─ Run: exp_A_lr=0.005_hd=64
        │       └─ Metrics: R²=0.8352, MSE=0.046
        │
        ├─ Experiment B: Optimized (BEST)
        │   └─ Run: exp_B_lr=0.01_hd=128
        │       └─ Metrics: R²=0.9826, MSE=0.0049
        │
        └─ Experiment C: Aggressive
            └─ Run: exp_C_lr=0.02_hd=256
                └─ Metrics: R²=-2.66 (overfitted)
```

### 2. Tracked Parameters

#### Experiment A - Conservative
```python
{
    "learning_rate": 0.005,
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.1,
    "batch_size": 32,
    "patience": 10,
    "epochs": 100,
    "description": "Conservative hyperparameters"
}
```

#### Experiment B - Optimized
```python
{
    "learning_rate": 0.01,
    "hidden_dim": 128,
    "num_layers": 3,
    "dropout": 0.05,
    "batch_size": 32,
    "patience": 15,
    "epochs": 150,
    "description": "Best from tuning on multi-ticker data"
}
```

#### Experiment C - Aggressive
```python
{
    "learning_rate": 0.02,
    "hidden_dim": 256,
    "num_layers": 3,
    "dropout": 0.15,
    "batch_size": 32,
    "patience": 20,
    "epochs": 200,
    "description": "High capacity with strong regularization"
}
```

### 3. Tracked Metrics

All experiments track:
- **train_loss**: Per-epoch training loss (monitored for convergence)
- **val_loss**: Per-epoch validation loss (monitored for early stopping)
- **test_mse**: Final mean squared error on test set
- **test_rmse**: Final root mean squared error
- **test_r2**: R² score (primary performance metric)
- **test_mae**: Mean absolute error
- **final_train_loss**: Training loss at convergence
- **final_val_loss**: Validation loss at convergence
- **converged_epoch**: Epoch where early stopping triggered
- **runtime_sec**: Total training time

### 4. Results Comparison Table

| Metric | Exp A | Exp B | Exp C |
|--------|-------|-------|-------|
| **R² Score** | 0.8352 | **0.9826** | -2.6574  |
| **MSE** | 0.046028 | 0.004850 | 1.021713 |
| **RMSE** | 0.2145 | 0.0696 | 1.0108 |
| **MAE** | 0.1721 | 0.0507 | 0.9219 |
| **Converged** | Epoch 14 | Epoch 72 | Epoch 30 |
| **Training Time** | 17.7s | 440.6s | 663.5s |
| **Recommendation** |  |  **USE THIS** |  |

### 5. Key Insights from Experiments

**Experiment A (Conservative)**
- Small model (64 hidden units) underfits despite simple config
- Converges quickly (14 epochs)
- R²=0.8352 leaves significant unexplained variance
- Too conservative, insufficient capacity

**Experiment B (Optimized) **
- **BEST PERFORMER**: R²=0.9826
- Balanced: Enough capacity (128 hidden) without overfitting
- Proper regularization (dropout=0.05)
- Learning rate (0.01) allows convergence without instability
- Patience (15) balances early stopping with full training
- Takes longer (440s) but achieves near-perfect predictions

**Experiment C (Aggressive) **
- **FAILED**: R²=-2.6574 (worse than predicting mean!)
- Large model (256 hidden units) overfits despite high dropout (0.15)
- High learning rate (0.02) causes instability
- Early stopping at epoch 30 suggests oscillating loss
- More capacity ≠ better with finite data (7,666 samples)

### 6. MLflow UI Workflow

#### Starting MLflow UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db --host 127.0.0.1 --port 5000
```
Then open: `http://127.0.0.1:5000`

#### In MLflow UI:
1. **Experiment List**
   - Shows all experiments in `Financial-forecasting-pipeline`
   - Three runs visible: exp_A, exp_B, exp_C
   - Run names show hyperparameters (lr, hd) for quick identification

2. **Metrics Tab**
   - Line plots of train_loss and val_loss over epochs
   - See how each config converges differently
   - Compare final metrics across runs

3. **Parameters Tab**
   - Side-by-side comparison of all hyperparameters
   - Easy to spot differences (learning_rate: 0.005 vs 0.01 vs 0.02)
   - Descriptions show intent of each experiment

4. **Runs Comparison**
   - Select multiple runs to compare metrics
   - Generate comparison charts
   - Export results as CSV

### 7. Integration Code

```python
import mlflow

# Configure tracking
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("Financial-forecasting-pipeline")

# Start run
with mlflow.start_run(run_name="exp_B_optimized"):
    
    # Log parameters
    mlflow.log_params({
        "learning_rate": 0.01,
        "hidden_dim": 128,
        "num_layers": 3,
        "dropout": 0.05,
    })
    
    # Log metrics during training
    for epoch in range(epochs):
        train_loss = train_step()
        val_loss = validate()
        
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss
        }, step=epoch)
    
    # Log final metrics
    mlflow.log_metrics({
        "test_r2": 0.9826,
        "test_mse": 0.0049,
    })
    
    # Log model artifact
    mlflow.pytorch.log_model(model, artifact_path="model")
```

### 8. Project Benefits

This implementation demonstrates:
1. **Experiment Management**: Systematic tracking of A/B/C tests
2. **Reproducibility**: All parameters and metrics logged
3. **Hyperparameter Tuning**: Shows understanding of model optimization
4. **ML Best Practices**: Early stopping, validation set monitoring
5. **Production Readiness**: Proper logging for debugging and auditing
6. **Communication**: Clear metric reporting for stakeholders

### 9. Accessing Results

#### View all experiments:
```bash
mlflow experiments list
```

#### Get specific run details:
```bash
mlflow runs show --experiment-name "Financial-forecasting-pipeline" \
  --run-id <run_id>
```

#### Export metrics to CSV:
```bash
mlflow experiments export --experiment-id 1 --output-dir ./export
```

### 10. Production Integration

MLflow enables:
- **Model Registry**: Transition runs to production/staging/archived
- **Versioning**: Track model lineage and provenance
- **Deployment**: Serve best model via MLflow Model Serving
- **Monitoring**: Compare production model against candidates

---

## Summary

**Experiment Results**:
- Experiment A: Conservative approach → R²=0.8352 (underfitting)
- **Experiment B: Optimal configuration → R²=0.9826 (RECOMMENDED)**
- Experiment C: Aggressive approach → R²=-2.6574 (overfitting)

**MLflow Value**:
- Tracked 3 complete experiments with 100+ metrics each
- Easy comparison via web UI
- Full reproducibility of best performing model
- Foundation for production model monitoring

**Next**: Deploy Exp B model to AWS Lambda for real-time predictions
