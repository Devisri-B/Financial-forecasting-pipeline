"""
Multi-ticker data loader to increase training dataset size.
Downloads and combines data from multiple correlated tickers.
"""

import yfinance as yf
import pandas as pd
import os
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../config", config_name="main", version_base=None)
def load_multi_ticker_data(cfg: DictConfig):
    """
    Download data for multiple tickers to increase training samples.
    Strategy: Major indices have longer history and more data.
    """
    # Major tickers with long history (1980s+)
    tickers = [
        "SPY",   # S&P 500 ETF (1993-present, ~8200 trading days)
        "QQQ",   # Nasdaq 100 ETF (1999-present, ~6800 days)
        "DIA",   # Dow Jones ETF (1998-present, ~7000 days)
        "IWM",   # Russell 2000 ETF (2000-present, ~6500 days)
    ]
    
    print(f"Downloading data for {len(tickers)} tickers...")
    all_data = []
    
    for ticker in tickers:
        print(f"  Fetching {ticker}...")
        df = yf.download(ticker, start="2015-01-01", progress=False)
        
        if df.empty:
            print(f"    ⚠️  No data for {ticker}, skipping...")
            continue
        
        # Flatten multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Add ticker identifier
        df['Ticker'] = ticker
        df['Date'] = df.index
        df.reset_index(drop=True, inplace=True)
        
        print(f"    ✅ {ticker}: {len(df)} samples ({df['Date'].min()} to {df['Date'].max()})")
        all_data.append(df)
    
    # Combine all tickers
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    
    # Create directories
    os.makedirs(os.path.dirname(cfg.data.raw_path), exist_ok=True)
    
    # Save combined data
    multi_ticker_path = cfg.data.raw_path.replace('.csv', '_multi_ticker.csv')
    combined_df.to_csv(multi_ticker_path, index=False)
    
    print(f"\n✅ Total samples: {len(combined_df)}")
    print(f"   Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
    print(f"   Saved to: {multi_ticker_path}")
    
    # Also create USAR-only version for comparison
    print(f"\n📊 Also downloading USAR for target prediction...")
    df_usar = yf.download(cfg.data.ticker, start=cfg.data.start_date, progress=False)
    if isinstance(df_usar.columns, pd.MultiIndex):
        df_usar.columns = df_usar.columns.get_level_values(0)
    df_usar.to_csv(cfg.data.raw_path)
    print(f"   USAR saved to: {cfg.data.raw_path}")
    
    return multi_ticker_path

if __name__ == "__main__":
    load_multi_ticker_data()
