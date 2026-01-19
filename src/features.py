import pandas as pd
import numpy as np
import ta
import structlog

logger = structlog.get_logger()

class FeatureEngineer:
    def __init__(self, use_technical_indicators: bool = True):
        self.use_technical_indicators = use_technical_indicators

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting feature engineering", shape=df.shape)
        
        df = df.copy()
        
        # Ensure 'Close' is present
        if 'Close' not in df.columns:
            # Handle yfinance multi-index case if necessary
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
        # 1. Log Returns (Target Stationarity)
        # We predict the next day's return, so we shift back for the target
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        
        if self.use_technical_indicators:
            # RSI
            df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['macd'] = macd.macd()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['bb_width'] = bollinger.bollinger_wband()
            
            # Simple Moving Average
            df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)

        # Drop NaNs created by rolling windows and shifting
        df.dropna(inplace=True)
        
        logger.info("Feature engineering complete", final_shape=df.shape)
        return df