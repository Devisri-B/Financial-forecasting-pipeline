import pandas as pd
import numpy as np
import ta
import structlog

logger = structlog.get_logger()

class FeatureEngineer:
    def __init__(self, use_technical_indicators: bool = True):
        self.use_technical_indicators = use_technical_indicators

    def _compute_forward_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicators without look-ahead bias.
        Each indicator only uses data up to current point, never future data.
        """
        df = df.copy()
        
        # RSI - computed only on data available up to current point
        df['rsi'] = np.nan
        for i in range(14, len(df)):
            try:
                df.iloc[i, df.columns.get_loc('rsi')] = ta.momentum.rsi(
                    df['Close'].iloc[:i+1], window=14
                ).iloc[-1]
            except:
                df.iloc[i, df.columns.get_loc('rsi')] = np.nan
        
        # MACD - computed only on data available up to current point
        df['macd'] = np.nan
        for i in range(26, len(df)):
            try:
                macd_obj = ta.trend.MACD(df['Close'].iloc[:i+1])
                df.iloc[i, df.columns.get_loc('macd')] = macd_obj.macd().iloc[-1]
            except:
                df.iloc[i, df.columns.get_loc('macd')] = np.nan
        
        # Bollinger Bands - computed only on data available up to current point
        df['bb_width'] = np.nan
        for i in range(20, len(df)):
            try:
                bb = ta.volatility.BollingerBands(df['Close'].iloc[:i+1])
                df.iloc[i, df.columns.get_loc('bb_width')] = bb.bollinger_wband().iloc[-1]
            except:
                df.iloc[i, df.columns.get_loc('bb_width')] = np.nan
        
        # SMA - computed only on data available up to current point
        df['sma_20'] = np.nan
        for i in range(19, len(df)):
            try:
                df.iloc[i, df.columns.get_loc('sma_20')] = ta.trend.sma_indicator(
                    df['Close'].iloc[:i+1], window=20
                ).iloc[-1]
            except:
                df.iloc[i, df.columns.get_loc('sma_20')] = np.nan
        
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting feature engineering", shape=df.shape)
        
        df = df.copy()
        
        # Ensure 'Close' is present
        if 'Close' not in df.columns:
            # Handle yfinance multi-index case if necessary
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
        cols_to_fix = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 1. Log Returns (Target Stationarity)
        # We predict the next day's return, so we shift back for the target
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        
        if self.use_technical_indicators:
            # Compute indicators without look-ahead bias
            df = self._compute_forward_indicators(df)

        # Drop NaNs created by rolling windows and shifting
        df.dropna(inplace=True)
        
        logger.info("Feature engineering complete", final_shape=df.shape)
        return df