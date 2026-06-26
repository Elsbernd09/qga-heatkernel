from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.loc[:, cleaned.notna().any(axis=0)]
    cleaned.index = pd.to_datetime(cleaned.index)
    return cleaned


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns from adjusted close prices.

    Args:
        prices: DataFrame of adjusted close price series.

    Returns:
        Log return DataFrame.
    """
    returns = np.log(prices / prices.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan)
    return _clean_dataframe(returns)


def rolling_volatility(prices: pd.DataFrame, window: int = 20, annualize: bool = False, trading_days: int = 252) -> pd.DataFrame:
    """
    Compute rolling volatility from log returns.

    Args:
        prices: DataFrame of adjusted close price series.
        window: Rolling window size.
        annualize: If True, annualize the volatility.
        trading_days: Trading days used for annualization.

    Returns:
        Rolling volatility DataFrame.
    """
    returns = log_returns(prices)
    volatility = returns.rolling(window=window, min_periods=1).std()
    if annualize:
        volatility = volatility * np.sqrt(trading_days)
    return _clean_dataframe(volatility)


def momentum(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute simple momentum as percentage change over a rolling window.

    Args:
        prices: DataFrame of adjusted close price series.
        window: Number of periods used for momentum.

    Returns:
        Momentum DataFrame.
    """
    momentum_series = prices / prices.shift(window) - 1
    return _clean_dataframe(momentum_series)


def drawdown(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute drawdown from adjusted close prices.

    Args:
        prices: DataFrame of adjusted close price series.

    Returns:
        Drawdown DataFrame where values are negative or zero.
    """
    rolling_max = prices.cummax()
    drawdown_series = prices / rolling_max - 1
    return _clean_dataframe(drawdown_series)


def rolling_correlation_to_spy(prices: pd.DataFrame, window: int = 20, spy_ticker: str = "SPY") -> pd.DataFrame:
    """
    Compute rolling correlation between each asset and SPY.

    Args:
        prices: DataFrame of adjusted close price series.
        window: Rolling window size for correlation.
        spy_ticker: The ticker symbol used as the benchmark series.

    Returns:
        Rolling correlation DataFrame with SPY.
    """
    returns = log_returns(prices)
    if spy_ticker not in returns.columns:
        raise KeyError(
            f"Benchmark ticker '{spy_ticker}' not found in price DataFrame."
        )

    spy_returns = returns[spy_ticker]
    correlation = returns.rolling(window=window, min_periods=window).corr(spy_returns)
    return _clean_dataframe(correlation)


def z_score_features(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute rolling z-score of log returns.

    Args:
        prices: DataFrame of adjusted close price series.
        window: Rolling window size for mean and standard deviation.

    Returns:
        Z-score DataFrame.
    """
    returns = log_returns(prices)
    rolling_mean = returns.rolling(window=window, min_periods=window).mean()
    rolling_std = returns.rolling(window=window, min_periods=window).std()
    z_score = (returns - rolling_mean) / rolling_std
    return _clean_dataframe(z_score)
