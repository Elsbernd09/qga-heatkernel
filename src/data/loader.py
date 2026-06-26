from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd
import yfinance as yf


def download_adjusted_close(
    tickers: Iterable[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download clean adjusted close prices for a list of symbols.

    Args:
        tickers: Iterable of ticker symbols.
        start: Optional start date in YYYY-MM-DD format.
        end: Optional end date in YYYY-MM-DD format.
        interval: Data interval, default daily.

    Returns:
        A pandas DataFrame indexed by date containing adjusted close prices.
    """
    symbols = list(tickers)
    if not symbols:
        return pd.DataFrame()

    raw = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
    )

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" not in raw.columns.get_level_values(1):
            raise ValueError("Downloaded data does not contain adjusted close prices.")
        prices = raw.xs("Adj Close", axis=1, level=1)
    else:
        prices = raw["Adj Close"].to_frame(name=symbols[0])

    prices.columns = [str(column) for column in prices.columns]
    prices = prices.sort_index()
    prices = prices.loc[:, prices.notna().any(axis=0)]
    prices = prices.ffill().bfill().astype("float64")
    prices.index = pd.to_datetime(prices.index)

    return prices
