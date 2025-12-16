# data_utils.py
import pandas as pd
import yfinance as yf
from typing import List, Tuple

def download_price_data(
    tickers: List[str],
    start: str = "2018-01-01",
    end: str = "2024-12-31"
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers.
    """
    data = yf.download(tickers, start=start, end=end, auto_adjust=False)["Adj Close"]
    # If single ticker, make it a DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna()
    return data

def train_test_split_prices(
    prices: pd.DataFrame,
    train_ratio: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split price data into train and test segments by time.
    """
    n = len(prices)
    split_idx = int(n * train_ratio)
    train = prices.iloc[:split_idx].copy()
    test = prices.iloc[split_idx:].copy()
    return train, test
