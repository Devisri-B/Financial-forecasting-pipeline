import yfinance as yf
import pandas as pd
import os

def load_data(ticker="USAR", start_date="2020-01-01", output_path="data/raw/stock_data.csv"):
    """
    Downloads stock data and saves it. 
    configure DVC here to track this file.
    """
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start=start_date)
    
    if df.empty:
        raise ValueError("No data downloaded. Check ticker symbol.")

    # Create dir if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    load_data()