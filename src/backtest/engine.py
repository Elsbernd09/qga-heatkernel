from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class BacktestEngine:
    """Historical simulation engine for geometric signal strategies.

    Backtesting applies a portfolio rule to past prices and evaluates the
    resulting performance. This is a historical research exercise and does not
    guarantee future profitability. Backtests can be affected by overfitting,
    lookahead bias, and survivorship bias.
    """

    @staticmethod
    def compute_forward_returns(prices: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """Compute forward returns aligned to each current price date.

        The returned DataFrame associates each observation date with the return
        over the next `horizon` periods. The last `horizon` rows are NaN.
        """
        if horizon < 1:
            raise ValueError("horizon must be at least 1")

        prices = prices.copy()
        forward_returns = prices.pct_change(periods=horizon).shift(-horizon)
        return forward_returns

    @staticmethod
    def generate_weights_from_scores(
        scores: pd.Series,
        long_top_n: int = 3,
        short_bottom_n: int = 0,
        long_short: bool = False,
    ) -> pd.Series:
        """Create portfolio weights from asset scores.

        Long-only mode selects the highest scoring assets and weights them equally.
        Long-short mode can also take the lowest scoring assets as shorts.
        """
        if long_top_n < 0 or short_bottom_n < 0:
            raise ValueError("long_top_n and short_bottom_n must be non-negative")

        scores = scores.dropna().sort_values(ascending=False)
        weights = pd.Series(0.0, index=scores.index)

        long_assets = list(scores.head(long_top_n).index) if long_top_n > 0 else []
        short_assets = list(scores.tail(short_bottom_n).index) if short_bottom_n > 0 else []

        if long_short and long_assets and short_assets:
            long_weight = 0.5 / len(long_assets)
            short_weight = -0.5 / len(short_assets)
        elif long_assets:
            long_weight = 1.0 / len(long_assets)
            short_weight = 0.0
        else:
            long_weight = 0.0
            short_weight = 0.0

        weights.loc[long_assets] = long_weight
        weights.loc[short_assets] = short_weight
        return weights

    @staticmethod
    def rebalance_dates(prices: pd.DataFrame, frequency: str = "W") -> pd.DatetimeIndex:
        """Return the dates on which the strategy should rebalance.

        Weekly, monthly, and daily rebalancing are supported.
        """
        if not isinstance(prices.index, pd.DatetimeIndex):
            try:
                prices = prices.copy()
                prices.index = pd.to_datetime(prices.index)
            except Exception as exc:
                raise ValueError("prices index must be datetime-like") from exc

        freq = frequency.upper()
        if freq == "D":
            return prices.index
        if freq == "W":
            return prices.resample("W-FRI").last().dropna(how="all").index
        if freq in {"M", "ME"}:
            return prices.resample("ME").last().dropna(how="all").index
        if freq == "MS":
            return prices.resample("MS").last().dropna(how="all").index

        return prices.resample(freq).last().dropna(how="all").index

    @staticmethod
    def calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
        """Compute drawdown from a strategy equity curve."""
        high_water_mark = equity_curve.cummax()
        return equity_curve / high_water_mark - 1.0

    @staticmethod
    def turnover(weights: pd.DataFrame) -> pd.Series:
        """Calculate turnover from period-to-period weight changes."""
        if weights.empty:
            return pd.Series(dtype=float)

        turnover_series = weights.diff().abs().sum(axis=1) / 2.0
        return turnover_series.fillna(turnover_series.iloc[0] if len(weights) else 0.0)

    @staticmethod
    def rolling_sharpe(
        returns: pd.Series,
        window: int = 60,
        periods_per_year: int = 252,
    ) -> pd.Series:
        """Compute a rolling Sharpe ratio for a return series."""
        rolling_mean = returns.rolling(window=window, min_periods=1).mean()
        rolling_std = returns.rolling(window=window, min_periods=1).std()
        factor = np.sqrt(periods_per_year)
        return (rolling_mean / rolling_std).fillna(0.0) * factor

    @staticmethod
    def equal_weight_benchmark(prices: pd.DataFrame) -> pd.Series:
        """Compute an equal-weight benchmark equity curve excluding SPY if present."""
        assets = [col for col in prices.columns if col != "SPY"]
        if not assets:
            raise ValueError("No assets available for equal weight benchmark")

        asset_returns = prices[assets].pct_change().fillna(0.0)
        weights = np.repeat(1.0 / len(assets), len(assets))
        portfolio_returns = asset_returns.dot(weights)
        return (1.0 + portfolio_returns).cumprod()

    @staticmethod
    def spy_benchmark(prices: pd.DataFrame, spy_column: str = "SPY") -> pd.Series:
        """Return a buy-and-hold equity curve for SPY."""
        if spy_column not in prices.columns:
            raise ValueError(f"Benchmark column '{spy_column}' not found in price data")

        spy_prices = prices[spy_column].astype(float)
        spy_returns = spy_prices.pct_change().fillna(0.0)
        return (1.0 + spy_returns).cumprod()

    @staticmethod
    def calculate_performance_metrics(
        equity_curve: pd.Series,
        benchmark_curve: Optional[pd.Series] = None,
        periods_per_year: int = 252,
    ) -> Dict[str, float]:
        """Calculate standard performance metrics from an equity curve."""
        returns = equity_curve.pct_change().fillna(0.0)
        total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)
        periods = len(returns)
        if periods > 0:
            annualized_return = float((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (periods_per_year / periods) - 1.0)
        else:
            annualized_return = 0.0

        volatility = float(returns.std() * np.sqrt(periods_per_year)) if returns.std() != 0 else 0.0
        sharpe = float((returns.mean() / returns.std() * np.sqrt(periods_per_year)) if returns.std() != 0 else 0.0)
        drawdown = BacktestEngine.calculate_drawdown(equity_curve)
        max_drawdown = float(drawdown.min())
        hit_rate = float((returns > 0).sum() / len(returns)) if len(returns) > 0 else 0.0
        best_day = float(returns.max()) if len(returns) > 0 else 0.0
        worst_day = float(returns.min()) if len(returns) > 0 else 0.0

        metrics: Dict[str, float] = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "hit_rate": hit_rate,
            "best_day": best_day,
            "worst_day": worst_day,
            "beta_to_benchmark": 0.0,
            "correlation_to_benchmark": 0.0,
        }

        if benchmark_curve is not None and not benchmark_curve.empty:
            benchmark_returns = benchmark_curve.pct_change().reindex(returns.index).fillna(0.0)
            cov = np.cov(returns, benchmark_returns)[0, 1] if len(returns) > 1 else 0.0
            bench_var = np.var(benchmark_returns) if len(benchmark_returns) > 1 else 0.0
            metrics["beta_to_benchmark"] = float(cov / bench_var) if bench_var > 0 else 0.0
            metrics["correlation_to_benchmark"] = float(returns.corr(benchmark_returns)) if len(returns) > 1 else 0.0

        return metrics

    def run_backtest(
        self,
        prices: pd.DataFrame,
        signal_scores: pd.DataFrame,
        long_top_n: int = 3,
        short_bottom_n: int = 0,
        long_short: bool = False,
        transaction_cost: float = 0.001,
        rebalance_frequency: str = "W",
    ) -> Dict[str, pd.DataFrame]:
        """Execute a periodic rebalancing backtest using signal scores."""
        prices = prices.copy()
        signal_scores = signal_scores.copy()

        if not isinstance(prices.index, pd.DatetimeIndex):
            prices.index = pd.to_datetime(prices.index)
        if not isinstance(signal_scores.index, pd.DatetimeIndex):
            signal_scores.index = pd.to_datetime(signal_scores.index)

        common_assets = prices.columns.intersection(signal_scores.columns)
        if common_assets.empty:
            raise ValueError("No common assets between prices and signal_scores")

        prices = prices[common_assets].astype(float)
        daily_returns = prices.pct_change().fillna(0.0)

        rebalance_dates = self.rebalance_dates(prices, rebalance_frequency)
        rebalance_dates = [date for date in rebalance_dates if date in prices.index]

        weight_records: Dict[pd.Timestamp, pd.Series] = {}
        turnover_records: Dict[pd.Timestamp, float] = {}
        trades: list[Dict[str, object]] = []
        previous_weights = pd.Series(0.0, index=common_assets)

        for date in rebalance_dates:
            available_signals = signal_scores.loc[:date, common_assets]
            if available_signals.empty:
                continue
            latest_signals = available_signals.iloc[-1]
            weights = self.generate_weights_from_scores(
                latest_signals,
                long_top_n=long_top_n,
                short_bottom_n=short_bottom_n,
                long_short=long_short,
            ).reindex(common_assets).fillna(0.0)
            weight_records[date] = weights
            turnover_value = float(np.sum(np.abs(weights - previous_weights)) / 2.0)
            turnover_records[date] = turnover_value
            trades.append({"date": date, "turnover": turnover_value, "weights": weights.to_dict()})
            previous_weights = weights

        weights = pd.DataFrame(index=prices.index, columns=common_assets, dtype=float).fillna(0.0)
        sorted_rebalance_dates = sorted(weight_records.keys())
        for date in sorted_rebalance_dates:
            weights.loc[date:] = weight_records[date].values
        weights = weights.ffill().fillna(0.0)

        strategy_returns = (weights.shift(1).fillna(0.0) * daily_returns).sum(axis=1)
        cost_series = pd.Series(0.0, index=prices.index)
        first_date = prices.index[0] if len(prices.index) else None
        for date, cost in turnover_records.items():
            if date in cost_series.index and date != first_date:
                cost_series.loc[date] = cost * transaction_cost

        adjusted_returns = strategy_returns - cost_series
        equity_curve = (1.0 + adjusted_returns).cumprod()
        turnover_series = pd.Series(turnover_records).reindex(prices.index).fillna(0.0)
        trades_df = pd.DataFrame(trades)

        return {
            "equity_curve": equity_curve,
            "daily_returns": strategy_returns,
            "weights": weights,
            "turnover": turnover_series,
            "trades": trades_df,
        }

    def backtest_report(
        self,
        prices: pd.DataFrame,
        signal_scores: pd.DataFrame,
        benchmark: str = "SPY",
    ) -> Dict[str, object]:
        """Run a backtest and compare results against benchmarks."""
        results = self.run_backtest(prices, signal_scores)
        strategy_curve = results["equity_curve"]

        benchmark_curve = pd.Series(dtype=float)
        spy_metrics = {}
        try:
            benchmark_curve = self.spy_benchmark(prices, spy_column=benchmark)
            spy_metrics = self.calculate_performance_metrics(strategy_curve, benchmark_curve)
        except ValueError:
            spy_metrics = {}

        equal_curve = self.equal_weight_benchmark(prices)
        strategy_metrics = self.calculate_performance_metrics(strategy_curve, benchmark_curve if not benchmark_curve.empty else None)
        equal_metrics = self.calculate_performance_metrics(equal_curve, benchmark_curve if not benchmark_curve.empty else None)
        drawdown = self.calculate_drawdown(strategy_curve)
        rolling_sharpe = self.rolling_sharpe(results["daily_returns"])

        return {
            "strategy_results": results,
            "strategy_metrics": strategy_metrics,
            "spy_metrics": spy_metrics,
            "equal_weight_metrics": equal_metrics,
            "drawdown": drawdown,
            "rolling_sharpe": rolling_sharpe,
        }
