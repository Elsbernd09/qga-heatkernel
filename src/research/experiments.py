from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


class QGAExperimentRunner:
    """Research experiment runner for Quantum Geometric Alpha signals."""

    periods_per_year = 252

    @staticmethod
    def _validate_dataframe(data: pd.DataFrame, name: str) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame")
        if data.empty:
            raise ValueError(f"{name} must not be empty")
        return data.copy()

    @staticmethod
    def _assign_quantile_buckets(
        values: pd.Series,
        n_buckets: int,
    ) -> pd.Series:
        if values.empty:
            return pd.Series(dtype=int)

        ranked = values.rank(method="first", pct=True)
        buckets = (ranked * n_buckets).astype(int)
        buckets = buckets.clip(upper=n_buckets - 1)
        return buckets

    def future_returns(
        self,
        prices: pd.DataFrame,
        horizons: Sequence[int] = (5, 20),
    ) -> Dict[int, pd.DataFrame]:
        prices = self._validate_dataframe(prices, "prices").astype(float)
        futures: Dict[int, pd.DataFrame] = {}

        for horizon in horizons:
            if horizon < 1:
                raise ValueError("horizons must contain positive integers")
            future_returns = prices.pct_change(periods=horizon).shift(-horizon)
            futures[int(horizon)] = future_returns

        return futures

    def future_realized_volatility(
        self,
        returns: pd.DataFrame,
        horizons: Sequence[int] = (5, 20),
    ) -> Dict[int, pd.DataFrame]:
        returns = self._validate_dataframe(returns, "returns").astype(float)
        realized: Dict[int, pd.DataFrame] = {}

        for horizon in horizons:
            if horizon < 1:
                raise ValueError("horizons must contain positive integers")
            volatility = (
                returns.rolling(window=horizon, min_periods=horizon)
                .std(ddof=0)
                .shift(-horizon)
                * np.sqrt(self.periods_per_year)
            )
            realized[int(horizon)] = volatility

        return realized

    def curvature_predictive_test(
        self,
        curvature_df: pd.DataFrame,
        returns: pd.DataFrame,
        horizons: Sequence[int] = (5, 20),
    ) -> Dict[str, pd.DataFrame]:
        curvature_df = self._validate_dataframe(curvature_df, "curvature_df").astype(float)
        returns = self._validate_dataframe(returns, "returns").astype(float)

        future_volatility = self.future_realized_volatility(returns, horizons=horizons)
        correlation_records: List[Dict[str, Any]] = []
        bucket_records: List[Dict[str, Any]] = []

        for horizon in horizons:
            future_vol = future_volatility[horizon]
            left = curvature_df.stack().rename("score").reset_index()
            left.columns = ["date", "asset", "score"]
            right = future_vol.stack().rename("future_vol").reset_index()
            right.columns = ["date", "asset", "future_vol"]
            merged = pd.merge(left, right, how="inner", on=["date", "asset"]).dropna()
            correlation = float(merged["score"].corr(merged["future_vol"])) if not merged.empty else np.nan
            correlation_records.append(
                {
                    "horizon": int(horizon),
                    "curvature_future_vol_correlation": correlation,
                }
            )

            aligned_scores = curvature_df.reindex_like(future_vol)
            for date in aligned_scores.index:
                row_scores = aligned_scores.loc[date]
                row_vol = future_vol.loc[date]
                row_data = pd.DataFrame(
                    {
                        "score": row_scores,
                        "future_vol": row_vol,
                    }
                ).dropna()
                if row_data.empty:
                    continue
                row_data["bucket"] = self._assign_quantile_buckets(row_data["score"], n_buckets=5)
                bucket_summary = (
                    row_data.groupby("bucket").agg(
                        average_score=("score", "mean"),
                        average_future_vol=("future_vol", "mean"),
                        count=("future_vol", "count"),
                    )
                    .reset_index()
                )
                bucket_summary["horizon"] = int(horizon)
                bucket_summary["date"] = date
                bucket_records.extend(bucket_summary.to_dict(orient="records"))

        correlation_df = pd.DataFrame(correlation_records).set_index("horizon")
        bucketed_df = (
            pd.DataFrame(bucket_records).rename(columns={"bucket": "curvature_quintile"})
            if bucket_records
            else pd.DataFrame(
                columns=[
                    "horizon",
                    "date",
                    "curvature_quintile",
                    "average_score",
                    "average_future_vol",
                    "count",
                ]
            )
        )
        if not bucketed_df.empty:
            bucketed_df["curvature_quintile"] = bucketed_df["curvature_quintile"].astype(int) + 1
            bucketed_df = bucketed_df["horizon date curvature_quintile average_score average_future_vol count".split()]

        return {
            "correlation": correlation_df,
            "bucketed": bucketed_df,
        }

    def heat_predictive_test(
        self,
        heat_df: pd.DataFrame,
        returns: pd.DataFrame,
        horizons: Sequence[int] = (5, 20),
    ) -> Dict[str, pd.DataFrame]:
        heat_df = self._validate_dataframe(heat_df, "heat_df").astype(float)
        returns = self._validate_dataframe(returns, "returns").astype(float)

        future_volatility = self.future_realized_volatility(returns, horizons=horizons)
        future_returns = {
            horizon: (
                (1.0 + returns)
                .rolling(window=horizon, min_periods=horizon)
                .apply(np.prod, raw=True)
                .shift(-horizon)
                - 1.0
            )
            for horizon in horizons
        }

        correlation_records: List[Dict[str, Any]] = []
        bucket_records: List[Dict[str, Any]] = []

        for horizon in horizons:
            future_vol = future_volatility[horizon]
            future_ret = future_returns[horizon]
            left_vol = heat_df.stack().rename("score").reset_index()
            left_vol.columns = ["date", "asset", "score"]
            right_vol = future_vol.stack().rename("future_vol").reset_index()
            right_vol.columns = ["date", "asset", "future_vol"]
            merged_vol = pd.merge(left_vol, right_vol, how="inner", on=["date", "asset"]).dropna()
            left_ret = heat_df.stack().rename("score").reset_index()
            left_ret.columns = ["date", "asset", "score"]
            right_ret = future_ret.stack().rename("future_return").reset_index()
            right_ret.columns = ["date", "asset", "future_return"]
            merged_ret = pd.merge(left_ret, right_ret, how="inner", on=["date", "asset"]).dropna()
            vol_corr = float(merged_vol["score"].corr(merged_vol["future_vol"])) if not merged_vol.empty else np.nan
            ret_corr = float(merged_ret["score"].corr(merged_ret["future_return"])) if not merged_ret.empty else np.nan
            correlation_records.append(
                {
                    "horizon": int(horizon),
                    "heat_future_vol_correlation": vol_corr,
                    "heat_future_return_correlation": ret_corr,
                }
            )

            aligned_scores = heat_df.reindex_like(future_vol)
            for date in aligned_scores.index:
                row_scores = aligned_scores.loc[date]
                row_vol = future_vol.loc[date]
                row_ret = future_ret.loc[date]
                row_data = pd.DataFrame(
                    {
                        "score": row_scores,
                        "future_vol": row_vol,
                        "future_return": row_ret,
                    }
                ).dropna()
                if row_data.empty:
                    continue
                row_data["bucket"] = self._assign_quantile_buckets(row_data["score"], n_buckets=5)
                bucket_summary = (
                    row_data.groupby("bucket").agg(
                        average_score=("score", "mean"),
                        average_future_vol=("future_vol", "mean"),
                        average_future_return=("future_return", "mean"),
                        hit_rate=("future_return", lambda x: float((x > 0).sum()) / len(x) if len(x) else 0.0),
                        count=("future_return", "count"),
                    )
                    .reset_index()
                )
                bucket_summary["horizon"] = int(horizon)
                bucket_summary["date"] = date
                bucket_records.extend(bucket_summary.to_dict(orient="records"))

        correlation_df = pd.DataFrame(correlation_records).set_index("horizon")
        bucketed_df = (
            pd.DataFrame(bucket_records).rename(columns={"bucket": "heat_quintile"})
            if bucket_records
            else pd.DataFrame(
                columns=[
                    "horizon",
                    "date",
                    "heat_quintile",
                    "average_score",
                    "average_future_vol",
                    "average_future_return",
                    "hit_rate",
                    "count",
                ]
            )
        )
        if not bucketed_df.empty:
            bucketed_df["heat_quintile"] = bucketed_df["heat_quintile"].astype(int) + 1
            bucketed_df = bucketed_df[
                "horizon date heat_quintile average_score average_future_vol average_future_return hit_rate count".split()
            ]

        return {
            "correlation": correlation_df,
            "bucketed": bucketed_df,
        }

    def signal_bucket_test(
        self,
        signal_scores: pd.DataFrame,
        prices: pd.DataFrame,
        horizon: int = 20,
        n_buckets: int = 5,
    ) -> pd.DataFrame:
        signal_scores = self._validate_dataframe(signal_scores, "signal_scores").astype(float)
        prices = self._validate_dataframe(prices, "prices").astype(float)

        horizon = int(horizon)
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if n_buckets < 1:
            raise ValueError("n_buckets must be at least 1")

        forward_returns = self.future_returns(prices, horizons=[horizon])[horizon]
        daily_returns = prices.pct_change().fillna(0.0)
        forward_volatility = self.future_realized_volatility(daily_returns, horizons=[horizon])[horizon]

        records: List[Dict[str, Any]] = []
        common_assets = prices.columns.intersection(signal_scores.columns)
        if common_assets.empty:
            raise ValueError("No common assets between prices and signal_scores")

        signal_scores = signal_scores[common_assets].reindex(prices.index).dropna(how="all")

        for date in signal_scores.index.intersection(prices.index):
            row_scores = signal_scores.loc[date].dropna()
            if row_scores.empty:
                continue
            row_returns = forward_returns.loc[date].reindex(row_scores.index)
            row_vol = forward_volatility.loc[date].reindex(row_scores.index)
            row_data = pd.DataFrame(
                {
                    "score": row_scores,
                    "future_return": row_returns,
                    "future_volatility": row_vol,
                }
            ).dropna()
            if row_data.empty:
                continue
            row_data["bucket"] = self._assign_quantile_buckets(row_data["score"], n_buckets=n_buckets)
            row_data["date"] = date
            records.extend(row_data.reset_index().rename(columns={"index": "asset"}).to_dict(orient="records"))

        aggregated = pd.DataFrame(records)
        if aggregated.empty:
            return pd.DataFrame(
                columns=[
                    "bucket",
                    "average_signal",
                    "average_future_return",
                    "average_future_volatility",
                    "hit_rate",
                    "count",
                ]
            )

        bucket_metrics = (
            aggregated.groupby("bucket").agg(
                average_signal=("score", "mean"),
                average_future_return=("future_return", "mean"),
                average_future_volatility=("future_volatility", "mean"),
                hit_rate=("future_return", lambda x: float((x > 0).sum()) / len(x) if len(x) else 0.0),
                count=("future_return", "count"),
            )
            .reset_index()
        )
        bucket_metrics["bucket"] = bucket_metrics["bucket"].astype(int) + 1
        bucket_metrics = bucket_metrics.sort_values("bucket").reset_index(drop=True)
        return bucket_metrics

    def _calculate_performance_metrics(
        self,
        equity_curve: pd.Series,
    ) -> Dict[str, float]:
        if not isinstance(equity_curve, pd.Series):
            raise TypeError("equity_curve must be a pandas Series")
        if equity_curve.empty:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "annualized_volatility": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "hit_rate": 0.0,
            }

        returns = equity_curve.pct_change().fillna(0.0)
        total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)
        periods = len(returns)
        annualized_return = float(
            (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (self.periods_per_year / periods) - 1.0
        ) if periods > 0 else 0.0
        volatility = float(returns.std(ddof=0) * np.sqrt(self.periods_per_year)) if returns.std() != 0 else 0.0
        sharpe = float((returns.mean() / returns.std(ddof=0) * np.sqrt(self.periods_per_year))) if returns.std() != 0 else 0.0
        drawdown = equity_curve / equity_curve.cummax() - 1.0
        max_drawdown = float(drawdown.min())
        hit_rate = float((returns > 0).sum() / len(returns)) if len(returns) > 0 else 0.0

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "hit_rate": hit_rate,
        }

    def compare_strategy_to_baselines(
        self,
        prices: pd.DataFrame,
        signal_scores: pd.DataFrame,
        backtester: Any,
    ) -> pd.DataFrame:
        prices = self._validate_dataframe(prices, "prices").astype(float)
        signal_scores = self._validate_dataframe(signal_scores, "signal_scores").astype(float)

        if not hasattr(backtester, "run_backtest"):
            raise ValueError("backtester must implement run_backtest")

        results: Dict[str, Dict[str, float]] = {}
        common_assets = prices.columns.intersection(signal_scores.columns)
        if common_assets.empty:
            raise ValueError("No common assets between prices and signal_scores")

        qga_result = backtester.run_backtest(prices[common_assets], signal_scores[common_assets])
        qga_metrics = (
            backtester.calculate_performance_metrics(qga_result["equity_curve"])
            if hasattr(backtester, "calculate_performance_metrics")
            else self._calculate_performance_metrics(qga_result["equity_curve"])
        )
        results["QGA Strategy"] = qga_metrics

        if hasattr(backtester, "equal_weight_benchmark"):
            equal_curve = backtester.equal_weight_benchmark(prices[common_assets])
        else:
            asset_returns = prices[common_assets].pct_change().fillna(0.0)
            equal_curve = (1.0 + asset_returns.dot(np.repeat(1.0 / len(common_assets), len(common_assets)))).cumprod()
        results["Equal Weight"] = (
            backtester.calculate_performance_metrics(equal_curve)
            if hasattr(backtester, "calculate_performance_metrics")
            else self._calculate_performance_metrics(equal_curve)
        )

        if "SPY" in prices.columns:
            if hasattr(backtester, "spy_benchmark"):
                spy_curve = backtester.spy_benchmark(prices)
            else:
                spy_returns = prices["SPY"].pct_change().fillna(0.0)
                spy_curve = (1.0 + spy_returns).cumprod()
            results["SPY Benchmark"] = (
                backtester.calculate_performance_metrics(spy_curve)
                if hasattr(backtester, "calculate_performance_metrics")
                else self._calculate_performance_metrics(spy_curve)
            )

        momentum_scores = prices.pct_change(periods=20).shift(1).reindex(signal_scores.index).fillna(0.0)
        momentum_result = backtester.run_backtest(prices[common_assets], momentum_scores[common_assets])
        results["Momentum Ranking"] = (
            backtester.calculate_performance_metrics(momentum_result["equity_curve"])
            if hasattr(backtester, "calculate_performance_metrics")
            else self._calculate_performance_metrics(momentum_result["equity_curve"])
        )

        volatility_scores = prices.pct_change().shift(1)
        volatility_scores = 1.0 / (volatility_scores.rolling(window=20, min_periods=1).std(ddof=0).replace(0.0, np.nan)).fillna(0.0)
        volatility_scores = volatility_scores.reindex(signal_scores.index).fillna(0.0)
        inverse_vol_result = backtester.run_backtest(prices[common_assets], volatility_scores[common_assets])
        results["Inverse Volatility Ranking"] = (
            backtester.calculate_performance_metrics(inverse_vol_result["equity_curve"])
            if hasattr(backtester, "calculate_performance_metrics")
            else self._calculate_performance_metrics(inverse_vol_result["equity_curve"])
        )

        rng = np.random.default_rng(42)
        random_scores = pd.DataFrame(
            rng.standard_normal(size=(len(prices.index), len(common_assets))),
            index=prices.index,
            columns=common_assets,
        )
        random_result = backtester.run_backtest(prices[common_assets], random_scores)
        results["Random Ranking"] = (
            backtester.calculate_performance_metrics(random_result["equity_curve"])
            if hasattr(backtester, "calculate_performance_metrics")
            else self._calculate_performance_metrics(random_result["equity_curve"])
        )

        comparison = pd.DataFrame(results).T[
            [
                "total_return",
                "annualized_return",
                "annualized_volatility",
                "sharpe_ratio",
                "max_drawdown",
                "hit_rate",
            ]
        ]
        return comparison

    def summarize_experiment_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(results, dict):
            raise TypeError("results must be a dictionary")

        strongest = "No strong pattern identified from the available results."
        weakest = "No weak pattern identified from the available results."
        support_risk_regime = False
        limitations: List[str] = [
            "Historical findings do not guarantee future performance.",
            "Results may be affected by sample selection and regime changes.",
            "Signal relationships may weaken outside the tested period.",
        ]
        next_steps: List[str] = [
            "Validate signals on an out-of-sample dataset.",
            "Test alternative signal construction and rebalancing horizons.",
            "Quantify transaction costs and turnover impact.",
        ]

        if "curvature" in results and isinstance(results["curvature"], dict):
            corr_df = results["curvature"].get("correlation")
            if isinstance(corr_df, pd.DataFrame) and not corr_df.empty:
                best = corr_df["curvature_future_vol_correlation"].abs().idxmax()
                worst = corr_df["curvature_future_vol_correlation"].abs().idxmin()
                strongest = (
                    f"Curvature scores showed the strongest relationship with future volatility at {best}-day horizon."
                )
                weakest = (
                    f"Curvature evidence was weakest at {worst}-day horizon for future volatility."
                )
                support_risk_regime = bool(corr_df["curvature_future_vol_correlation"].abs().max() >= 0.10)

        if "heat" in results and isinstance(results["heat"], dict):
            corr_df = results["heat"].get("correlation")
            if isinstance(corr_df, pd.DataFrame) and not corr_df.empty:
                best = corr_df["heat_future_vol_correlation"].abs().idxmax()
                weakest = (
                    f"Heat scores were least predictive of future returns at {best}-day horizon."
                )
                if corr_df["heat_future_vol_correlation"].abs().max() >= 0.10:
                    strongest = (
                        f"Heat concentration was predictive of future volatility at {best}-day horizon."
                    )
                    support_risk_regime = True

        if "strategy_comparison" in results and isinstance(results["strategy_comparison"], pd.DataFrame):
            strategy_df = results["strategy_comparison"]
            if not strategy_df.empty:
                best_strategy = strategy_df["sharpe_ratio"].idxmax()
                worst_strategy = strategy_df["sharpe_ratio"].idxmin()
                strongest = (
                    f"The strategy comparison indicated {best_strategy} delivered the strongest risk-adjusted performance."
                )
                weakest = (
                    f"The strategy comparison indicated {worst_strategy} was the weakest performer on Sharpe ratio."
                )
                support_risk_regime = support_risk_regime or best_strategy == "QGA Strategy"

        return {
            "strongest_empirical_result": strongest,
            "weakest_empirical_result": weakest,
            "support_risk_regime_framework": support_risk_regime,
            "limitations": limitations,
            "next_steps": next_steps,
        }
