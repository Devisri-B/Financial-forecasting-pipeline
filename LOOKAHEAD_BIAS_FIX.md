# Fixing Look-Ahead Bias: From R²=0.9986 to Realistic Performance

## Problem Summary

The current R² of 0.9986 is suspiciously high—a red flag in the ML community indicating **data leakage**. This document explains the issue and provides the fix.

## Root Causes Identified

### 1. **Scaler Fitted on Entire Dataset** (Critical)
```python
# WRONG (current code in train_ensemble_multi.py, line 69)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)  # ← Fits on all data including test set!

# Then split happens AFTER
X, y = create_sequences(data_scaled, cfg.data.window_size)
train_idx = int(n * 0.7)
val_idx = int(n * 0.85)
X_train, y_train = X[:train_idx], y[:train_idx]  # ← Uses statistics from future data
```

**Why it's wrong:** The scaler's min/max values are computed from test data, so test samples are normalized using information they shouldn't have access to.

### 2. **Technical Indicators Computed Globally** (High Impact)
```python
# WRONG (current code in features.py)
df['rsi'] = ta.momentum.rsi(df['Close'], window=14)  # Computed on all data
df['macd'] = ta.trend.MACD(df['Close']).macd()       # Future context leaked
df['bb_width'] = ta.volatility.BollingerBands(df['Close']).bollinger_wband()
df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
```

**Why it's wrong:** These indicators use past windows, but when computed on the full dataset, they incorporate future price movements that shouldn't influence historical indicator values.

### 3. **No Walk-Forward Validation** (Conceptual)
```python
# WRONG (current split)
train_idx = int(n * 0.7)      # Random split doesn't respect time ordering
val_idx = int(n * 0.85)
X_train, y_train = X[:train_idx], y[:train_idx]
X_test, y_test = X[val_idx:], y[val_idx:]
```

**Why it's wrong:** Financial data is time-series. A model trained on 2025 data shouldn't be tested on 2024 data.

## The Fix

### Step 1: Fit Scaler on Training Data Only

```python
# In train_ensemble_multi.py, around line 69
# CORRECT approach:
X, y = create_sequences(data_scaled, cfg.data.window_size)  # Don't scale yet!

# Split FIRST
n = len(X)
train_idx = int(n * 0.7)
val_idx = int(n * 0.85)

X_train_unsealed = X[:train_idx]
X_val_unsealed = X[train_idx:val_idx]
X_test_unsealed = X[val_idx:]

# THEN fit scaler only on training data
scaler = StandardScaler()
scaler.fit(X_train_unsealed.reshape(-1, X_train_unsealed.shape[-1]))  # Fit on train only

# Transform all sets with TRAINING statistics
X_train = scaler.transform(X_train_unsealed.reshape(-1, X_train_unsealed.shape[-1])).reshape(X_train_unsealed.shape)
X_val = scaler.transform(X_val_unsealed.reshape(-1, X_val_unsealed.shape[-1])).reshape(X_val_unsealed.shape)
X_test = scaler.transform(X_test_unsealed.reshape(-1, X_test_unsealed.shape[-1])).reshape(X_test_unsealed.shape)

joblib.dump(scaler, os.path.join(get_original_cwd(), "models/scaler_ensemble_multi.pkl"))
```

### Step 2: Compute Technical Indicators Per Split

Create a separate feature engineering function that respects temporal boundaries:

```python
# Add to features.py
def add_technical_indicators_inplace(df: pd.DataFrame, max_lookback: int = 20) -> pd.DataFrame:
    """
    Add technical indicators in a forward-looking manner.
    Only use past data, never future data.
    """
    df = df.copy()
    
    # RSI (needs past 14 bars)
    df['rsi'] = np.nan
    for i in range(14, len(df)):
        df.iloc[i, df.columns.get_loc('rsi')] = ta.momentum.rsi(df['Close'].iloc[:i+1], window=14).iloc[-1]
    
    # Similar for MACD, Bollinger Bands, SMA
    # ... (compute only using data up to current point)
    
    return df
```

### Step 3: Implement Proper Walk-Forward Validation

```python
# For truly production-ready validation:
# Split by DATE, not by index
train_cutoff = df[df['Date'] == '2024-01-01']  # All data before this date
val_cutoff = df[df['Date'] == '2025-01-01']     # Data between train and val cutoff
# Test: All data after val_cutoff
```

## Expected Results After Fix

| Metric | Current (Leaked) | Expected After Fix | Notes |
|--------|------------------|-------------------|-------|
| R² Score | 0.9986 | ~0.5-0.7 | Still decent but realistic |
| RMSE | 0.0197 | ~0.05-0.08 | 2-4x worse (expected) |
| MAPE | 1.64% | ~3-6% | More realistic for stock prediction |
| Directional Accuracy | 61.8% | ~52-55% | Random is 50%, so this is only +2-5% signal |

## Why This Matters for Your Portfolio

This analysis demonstrates:
- ✅ **Understanding of ML pitfalls** - You recognized the red flag
- ✅ **Knowledge of data leakage** - Can explain technical + statistical leakage
- ✅ **Time-series awareness** - Know that stock data isn't i.i.d.
- ✅ **Honest metrics reporting** - Willing to report realistic performance
- ✅ **Production mindset** - Think about what would happen with truly unseen data

## References

- [Temporal Data Leakage](https://machinelearningmastery.com/data-leakage-machine-learning/)
- [Walk-Forward Analysis](https://en.wikipedia.org/wiki/Walk_forward_optimization)
- [Financial ML Best Practices](https://www.mlfinlab.com/)
