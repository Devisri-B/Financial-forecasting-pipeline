# Financial-forecasting-pipeline

**An end-to-end volatility forecasting pipeline for equity ETFs — LSTM + attention and a
stacked ensemble, with Hydra config, MLflow tracking, ONNX export, and a FastAPI / AWS-Lambda
serving layer.**

The model forecasts **near-term volatility** (the next *N*-day average absolute return), a
quantity that is genuinely predictable thanks to volatility clustering and is directly useful
for options pricing, risk management, and position sizing.

---

## Results

Out-of-sample on a chronological, leakage-free test split of the multi-ticker data
(SPY, QQQ, DIA, IWM). "Directional accuracy" = correctly calling high-vs-low volatility.

| Configuration | Test R² | Directional Accuracy |
|---|---|---|
| `train_multi_ticker.py` (LSTM) | **0.48** | **78.9%** |
| `run_experiments.py` (best config) | 0.36 | 76.3% |
| `train_ensemble_multi.py` *(production)* | 0.38 | 74.5% |

The pipeline can also be pointed at next-day **returns** (`features.target=return`); daily
return direction is close to a coin flip (~53%), which is expected for an efficient market.

---

## Highlights

- **Predictable target** — forecasts volatility (clustering carries real signal) rather than
  raw prices or near-random returns.
- **Leakage-free** — the forward-looking target is excluded from the inputs, the scaler is fit
  on training data only, and sliding windows are built per ticker.
- **19 causal features** — momentum, rolling volatility, volume, and mean-reversion signals,
  all computed using only past data.
- **Robust training** — mini-batch SGD, gradient clipping, LR scheduling, weight decay, and
  early stopping.
- **Ensemble** — LSTM + Ridge + ARIMA with weights optimised on a validation set.
- **MLOps** — Hydra configs, MLflow tracking, reproducible seeds, ONNX/TorchScript export,
  and a FastAPI + AWS-Lambda inference handler.

---

## Project structure

```
config/main.yaml              # Hydra config: data, model, features (target), deployment
data/raw/                     # stock_data_multi_ticker.csv (committed); single-ticker via DVC
src/
  features.py                 # FeatureEngineer: causal features + forward volatility target
  model.py                    # StockPredictor: LSTM + attention
  train_single_ticker.py      # single-ticker trainer + SHARED helpers
                              #   (create_sequences, train_lstm_model, direction_accuracy,
                              #    EarlyStopping, set_seeds)
  train_multi_ticker.py       # multi-ticker LSTM trainer
  train_ensemble_single.py    # EnsemblePredictor (LSTM + Ridge + ARIMA)
  train_ensemble_multi.py     # multi-ticker ensemble — the production model
  run_experiments.py          # A/B/C hyper-parameter sweep with MLflow
  single_ticker_loader.py     # download one ticker via yfinance
  multi_ticker_loader.py      # download SPY/QQQ/DIA/IWM via yfinance
  app.py                      # FastAPI inference + Mangum (AWS Lambda) adapter
```

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements-dev.txt

# 2. (Optional) refresh the multi-ticker dataset — it is already committed under data/raw/
python src/multi_ticker_loader.py

# 3. Train the production model (multi-ticker ensemble)
python src/train_ensemble_multi.py

# 4. Inspect runs in MLflow
mlflow ui --backend-store-uri sqlite:///mlruns.db --host 127.0.0.1 --port 5000
```

Other entry points:

```bash
python src/train_multi_ticker.py      # plain multi-ticker LSTM
python src/run_experiments.py         # A/B/C hyper-parameter sweep
```

> The single-ticker pipeline (`train_single_ticker.py`) needs `data/raw/stock_data.csv`,
> tracked by **DVC**. Fetch it with `dvc pull` or `python src/single_ticker_loader.py`
> (downloads via yfinance) first.

---

## Configuration

Everything is driven by `config/main.yaml` and can be overridden on the command line (Hydra):

```yaml
features:
  target: "volatility"   # "volatility" (default) or "return"
  vol_horizon: 5         # forecast horizon (days) for the volatility target

model:
  hidden_dim: 128
  num_layers: 3
  dropout: 0.2
  learning_rate: 0.001
  batch_size: 64
  weight_decay: 0.0001   # L2 regularisation
  grad_clip: 1.0         # gradient clipping for stable LSTM training
  patience: 12           # early stopping
```

Examples:

```bash
python src/train_ensemble_multi.py features.target=return      # forecast returns instead
python src/train_ensemble_multi.py features.vol_horizon=10     # longer volatility horizon
python src/train_ensemble_multi.py model.epochs=20            # quick smoke run
```

---

## How it works

1. **Data** — daily OHLCV for SPY, QQQ, DIA, IWM (2015–present), ~2,750 usable rows per ticker
   after feature engineering.
2. **Features** (`features.py`, all causal) — log returns, RSI, MACD, Bollinger width, SMA-20,
   multi-day momentum (`ret_2/5/10`), rolling volatility (`vol_5/10/20`), volume z-score,
   intraday range, and distance from the SMA. **19 input features.**
3. **Target** — `vol_target` = log of the next `vol_horizon`-day average |return|; forward-
   looking and excluded from the inputs.
4. **Sequencing** — sliding windows built per ticker; chronological train/val/test split;
   `StandardScaler` fit on the training set only.
5. **Model** — LSTM + attention (`model.py`); the ensemble adds Ridge and ARIMA with weights
   optimised on the validation set.
6. **Training** — mini-batches, shuffling, gradient clipping, `ReduceLROnPlateau`, weight decay,
   early stopping (shared `train_lstm_model`).
7. **Tracking & export** — metrics/params/artifacts logged to MLflow (SQLite); model exported to
   ONNX + TorchScript.
8. **Serving** — `app.py` loads the ONNX model and serves `/predict` via FastAPI, with a `Mangum`
   adapter for AWS Lambda.

---

## Deployment

`src/app.py` exposes a FastAPI service:

- `GET /` — health check
- `POST /predict` — accepts a `seq_len × n_features` window and returns the prediction plus a
  calibration interval (from `models/calibration_score.txt`).

`Mangum` wraps the app so it runs unchanged as an AWS Lambda function behind API Gateway. See
`AWS_DEPLOYMENT.md` and `Dockerfile` for the container/serverless setup.

---

## Notes

- Evaluate with **R²**, **RMSE**, and **directional accuracy**. MAPE is not meaningful for a
  log-scale target that can be near zero.
- Metrics vary modestly with the random seed, horizon, and split — treat them as estimates.

---

## License

See [LICENSE](LICENSE).
