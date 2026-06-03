# MLflow Experiment Tracking

MLflow logs every training run's parameters, metrics, and artifacts to a local SQLite
backend, so experiments can be compared and reproduced. `run_experiments.py` runs three
hyper-parameter configurations (A/B/C) on the multi-ticker volatility task and logs each as
a separate MLflow run.

## Tracked parameters

Each run logs its full config, e.g.:

```python
{
    "learning_rate": 0.001,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "batch_size": 64,
    "weight_decay": 0.0001,
    "patience": 12,
    "epochs": 120,
    "target": "volatility",
    "vol_horizon": 5,
}
```

## Tracked metrics

- `train_loss`, `val_loss` — per-epoch (convergence + early stopping)
- `test_mse`, `test_rmse`, `test_mae`
- `test_r2` — primary regression metric
- `test_directional_accuracy` — high-vs-low volatility hit rate
- `converged_epoch`, `runtime_sec`

## A/B/C configurations (`run_experiments.py`)

| Config | learning_rate | hidden_dim | num_layers | dropout | batch_size |
|--------|---------------|------------|------------|---------|------------|
| A (conservative) | 0.0005 | 64 | 2 | 0.10 | 64 |
| B (optimized) | 0.0010 | 128 | 2 | 0.20 | 64 |
| C (high-capacity) | 0.0020 | 256 | 3 | 0.30 | 128 |

## Example results (volatility target)

Out-of-sample on the chronological test split. Numbers vary modestly with seed/split.

| Metric | Exp A | Exp B | Exp C |
|--------|-------|-------|-------|
| **Test R²** | -0.15 | 0.34 | **0.36** |
| **Directional Accuracy** | 59.8% | 71.3% | **76.3%** |

Larger models (B, C) clearly outperform the small, low-LR config (A). The best config is a
good starting point for the production ensemble in `train_ensemble_multi.py`.

## MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db --host 127.0.0.1 --port 5000
# then open http://127.0.0.1:5000
```

In the UI you can plot `train_loss`/`val_loss` per epoch, compare hyper-parameters
side-by-side, and select multiple runs to compare final metrics.

## Integration pattern

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("Financial-forecasting-pipeline")

with mlflow.start_run(run_name="exp_B"):
    mlflow.log_params({"learning_rate": 0.001, "hidden_dim": 128, "target": "volatility"})

    for epoch in range(epochs):
        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

    mlflow.log_metrics({"test_r2": test_r2, "test_directional_accuracy": dir_acc})
    mlflow.pytorch.log_model(model, artifact_path="model")
```
