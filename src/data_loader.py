import yfinance as yf
import pandas as pd
import os
import hydra
from omegaconf import DictConfig
import os


@hydra.main(config_path="../config", config_name="main", version_base=None)
def load_data(cfg: DictConfig):
    print(f"Downloading data for {cfg.data.ticker}...")
    
    # Download data
    df = yf.download(cfg.data.ticker, start=cfg.data.start_date)
    
    if df.empty:
        raise ValueError(f"No data found for ticker {cfg.data.ticker}. Check symbol or internet connection.")

    # Create directories
    os.makedirs(os.path.dirname(cfg.data.raw_path), exist_ok=True)
    
    # Save raw data
    df.to_csv(cfg.data.raw_path)
    print(f"Data saved to {cfg.data.raw_path}")

if __name__ == "__main__":
    load_data()