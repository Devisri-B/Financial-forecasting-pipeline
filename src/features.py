import pandas as pd
import ta
from typing import Tuple
import structlog

logger = structlog.get_logger()

class FeatureEngineer:
    def __init__(self, use_technical_indicators: bool = True):
        self.use_technical_indicators = use_technical_indicators

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates technical indicators: RSI, MACD, Bollinger Bands.
        """
        logger.info("Starting feature engineering", shape=df.shape)
        
        df = df.copy()
        
        # 1. Log Returns (Better stationarity than raw prices)
        df['log_ret'] = df['Close'].pct_change().apply(lambda x: np.log(1 + x))
        
        if self.use_technical_indicators:
            # RSI (Momentum)
            df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
            
            # MACD (Trend)
            macd = ta.trend.MACD(df['Close'])
            df['macd'] = macd.macd()
            df['macd_diff'] = macd.macd_diff()
            
            # Bollinger Bands (Volatility)
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['bb_high'] = bollinger.bollinger_hband()
            df['bb_low'] = bollinger.bollinger_lband()
            
            # Volume Moving Average
            df['vol_ma'] = df['Volume'].rolling(window=20).mean()

        # Drop NaNs created by rolling windows
        df.dropna(inplace=True)
        
        logger.info("Feature engineering complete", final_shape=df.shape)
        return df