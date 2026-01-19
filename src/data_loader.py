import yfinance as yf
import pandas as pd
import os
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../config", config_name="main", version_base=None)
def load_data(cfg: DictConfig):
    print(f"Downloading data for {cfg.data.ticker}...")
    
    # Download data
    df = yf.download(cfg.data.ticker, start=cfg.data.start_date, progress=False)
    
    if df.empty:
        raise ValueError(f"No data found for ticker {cfg.data.ticker}.")

    # yfinance returns columns like ('Close', 'USAR'). Saving this creates 2 headers.
    # We flatten it to just 'Close' so pandas reads numbers, not strings.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Create directories
    os.makedirs(os.path.dirname(cfg.data.raw_path), exist_ok=True)
    
    # Save raw data
    df.to_csv(cfg.data.raw_path)
    print(f"Data saved to {cfg.data.raw_path}")

if __name__ == "__main__":
    load_data()