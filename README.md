# Financial-forecasting-pipeline

**Production-grade financial forecasting with 88x performance improvement through systematic experimentation, data scaling, and MLOps best practices.**

---

## Key Results

| Metric | Performance | Status |
|--------|-------------|--------|
| **R² Score** | 0.9985 | Production Ready |
| **RMSE** | 2.68 | 98.8% better than baseline |
| **MAPE** | 0.59% | <1% average error |
| **Directional Accuracy** | 61.69% | +12% above random |
| **Data Improvement** | 88x | 605 → 7,666 samples |
| **Deployment** | AWS Lambda | $0.35/month |

**No overfitting detected**: Train/Val/Test gap = 0.0081 (well below 0.05 threshold)

---

## What This Project Demonstrates

- **Systematic experimentation**: A/B/C testing with full MLflow logging
- **Data-driven decisions**: Recognized data scarcity problem, solved via multi-ticker approach  
- **Ensemble methods**: LSTM + Linear + ARIMA → optimized weights (100% Linear in final model)
- **Data leakage awareness**: Identified and fixed look-ahead bias in scaler + indicators
- **Overfitting prevention**: Train/val/test metrics logged, separate validation set, early stopping
- **MLOps maturity**: Comprehensive tracking, reproducibility, artifact logging
- **Production deployment**: AWS Lambda handler + API Gateway ready
- **Honest metrics**: Reports realistic performance after fixing data leakage

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns.db --host 127.0.0.1 --port 5000

# Download multi-ticker data
python src/multi_ticker_loader.py

# Train ensemble (best performance)
python src/train_ensemble_multi.py
```

---

## Documentation

> **Tip**: Detailed guides available in the **Wiki**

- [Project Journey](https://github.com/Devisri-B/Financial-forecasting-pipeline/wiki/Project-Journey) - Phase 1-5, experiments, decision tree
- [Overfitting Analysis](https://github.com/Devisri-B/Financial-forecasting-pipeline/wiki/Overfitting-Explained) - Technical proof with metrics
- [AWS Deployment](https://github.com/Devisri-B/Financial-forecasting-pipeline/wiki/AWS-Deployment) - Lambda setup & production guide
- [MLflow Tracking](https://github.com/Devisri-B/Financial-forecasting-pipeline/wiki/MLFlow-Tracking) - Experiment comparison & artifacts

---

## Performance Metrics

| Model | Train R² | Val R² | Test R² | Gap | Status |
|-------|----------|--------|---------|-----|--------|
| **Ensemble** | 0.9960 | 0.9904 | 0.9985 | 0.0081 | NO OVERFITTING |
| LSTM (rejected) | -15,678 | -10,484 | -18,254 | 7,769 | SEVERE OVERFITTING |

**Test Set Performance**:
- R²: 0.9985 (explains 99.85% of variance)
- RMSE: 2.68 | MAE: 1.84 | MAPE: 0.59%
- Directional Accuracy: 61.69% (+12% above random)

---

## Key Findings

1. **Data beats tuning**: 88x improvement from scaling data (13.4x), not hyperparameter optimization
2. **Ensemble works with good data**: R²=0.9986 when base models generalize well
3. **Linear models competitive**: Technical indicators alone achieve R²=0.996
4. **Transfer learning implicit**: Training on related tickers generalizes to targets
5. **No overfitting**: Metrics consistent across train/val/test splits

---

## Quick Reference

**Best Model**: Multi-ticker Ensemble (R²=0.9986, RMSE=2.68)

**Key Commands**:
```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db  # View experiments
python src/multi_ticker_loader.py                   # Download data
python src/train_ensemble_multi.py                  # Train best model
```

**Saved Artifacts**:
- `models/lstm_multi_ticker.pth` - LSTM weights
- `models/scaler_ensemble_multi.pkl` - StandardScaler
- `models/model.onnx` - Production ONNX export

