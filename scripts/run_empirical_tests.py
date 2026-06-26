#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtest.engine import BacktestEngine
from data.features import log_returns
from data.loader import download_adjusted_close
from geometry.heat_kernel import HeatKernelDiffusion
from geometry.ricci_curvature import RicciCurvatureEngine
from signals.geometric_signal import GeometricSignalEngine
from research.experiments import QGAExperimentRunner

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "JPM",
    "GS",
    "XOM",
    "GLD",
    "SPY",
    "QQQ",
    "TLT",
]
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def format_table(data: Any, index: bool = True) -> str:
    if isinstance(data, pd.DataFrame):
        if data.empty:
            return "Empty DataFrame"
        return data.to_string(index=index)
    if isinstance(data, pd.Series):
        if data.empty:
            return "Empty Series"
        return data.to_string()
    return str(data)


def build_signal_timeseries(
    prices: pd.DataFrame,
    curvature_history: pd.DataFrame,
    heat_history: pd.DataFrame,
) -> pd.DataFrame:
    assets = list(prices.columns)
    dates = sorted(curvature_history.index.intersection(heat_history.index))
    signal_engine = GeometricSignalEngine()
    signal_scores = pd.DataFrame(index=dates, columns=assets, dtype=float)

    for date in dates:
        curvature_series = curvature_history.loc[date]
        heat_series = heat_history.loc[date]
        signal_table = signal_engine.generate_signal_table(
            assets=assets,
            curvature_series=curvature_series,
            heat_series=heat_series,
            topology_label="Neutral",
        )
        signal_scores.loc[date, signal_table["asset"]]
        signal_scores.loc[date, signal_table["asset"]] = signal_table["final_score"].values

    return signal_scores


def main() -> None:
    print("QGA Empirical Research Script")
    print("This script is for research demonstration only and does not provide investment advice.")

    prices = pd.DataFrame()
    returns = pd.DataFrame()
    curvature_history = pd.DataFrame()
    heat_history = pd.DataFrame()
    signal_scores = pd.DataFrame()

    try:
        print_section("DATA")
        prices = download_adjusted_close(TICKERS, start=START_DATE, end=END_DATE)
        if prices.empty:
            raise RuntimeError("Price download returned no data.")
        print(f"Downloaded adjusted close prices for {len(prices.columns)} assets.")
        print(prices.tail(5).round(2).to_string())

        returns = log_returns(prices)
        print(f"Computed log returns with {returns.shape[0]} observations.")
    except Exception as exc:
        print(f"Data preparation failed: {exc}")
        return

    try:
        print_section("TABLE 1: Curvature Predictive Test")
        curvature_history = RicciCurvatureEngine.curvature_time_series(
            returns.dropna(), window=60, threshold=0.2
        )
        if curvature_history.empty:
            raise RuntimeError("Curvature history is empty.")

        curvature_summary = curvature_history.tail(1).T.rename(columns={curvature_history.index[-1]: "latest_curvature"})
        print("Latest curvature values:")
        print(format_table(curvature_summary.sort_values("latest_curvature").round(4), index=True))
    except Exception as exc:
        curvature_history = pd.DataFrame()
        print(f"Curvature analysis failed: {exc}")

    try:
        print_section("TABLE 2: Heat Diffusion Predictive Test")
        heat_history = HeatKernelDiffusion.rolling_heat_diffusion(
            returns.dropna(), window=60, diffusion_time=1.0
        )
        if heat_history.empty:
            raise RuntimeError("Heat diffusion history is empty.")

        heat_summary = heat_history.tail(1).T.rename(columns={heat_history.index[-1]: "latest_heat"})
        print("Latest heat diffusion scores:")
        print(format_table(heat_summary.sort_values("latest_heat", ascending=False).round(4), index=True))
    except Exception as exc:
        heat_history = pd.DataFrame()
        print(f"Heat diffusion analysis failed: {exc}")

    try:
        print_section("TABLE 3: Signal Bucket Test")
        if curvature_history.empty or heat_history.empty:
            raise RuntimeError("Signal construction requires both curvature and heat history.")

        signal_scores = build_signal_timeseries(prices, curvature_history, heat_history)
        if signal_scores.empty:
            raise RuntimeError("Generated signal score time series is empty.")

        latest_signal = signal_scores.tail(1).T.rename(columns={signal_scores.index[-1]: "qga_score"})
        print("Latest QGA composite scores using curvature and heat:")
        print(format_table(latest_signal.sort_values("qga_score", ascending=False).round(2), index=True))

        experiment_runner = QGAExperimentRunner()
        signal_bucket = experiment_runner.signal_bucket_test(
            signal_scores, prices, horizon=20, n_buckets=5
        )
        print("Signal bucket summary:")
        print(format_table(signal_bucket.round(4), index=False))
    except Exception as exc:
        signal_scores = pd.DataFrame()
        print(f"Signal bucket testing failed: {exc}")

    strategy_comparison = pd.DataFrame()
    summary_interpretation = {}
    try:
        print_section("TABLE 4: Strategy vs Baselines")
        if signal_scores.empty:
            raise RuntimeError("Backtest requires signal scores.")

        backtest_engine = BacktestEngine()
        strategy_comparison = experiment_runner.compare_strategy_to_baselines(
            prices, signal_scores, backtest_engine
        )
        print(format_table(strategy_comparison.round(4), index=True))
    except Exception as exc:
        print(f"Strategy comparison failed: {exc}")

    try:
        print_section("TABLE 5: Summary Interpretation")
        if signal_scores.empty or curvature_history.empty or heat_history.empty:
            raise RuntimeError("Experiment summary requires prior results.")

        future_test = experiment_runner.curvature_predictive_test(
            curvature_history, returns, horizons=[5, 20]
        )
        heat_test = experiment_runner.heat_predictive_test(
            heat_history, returns, horizons=[5, 20]
        )
        summary_interpretation = experiment_runner.summarize_experiment_results(
            {
                "curvature": future_test,
                "heat": heat_test,
                "strategy_comparison": strategy_comparison,
            }
        )

        summary_df = pd.DataFrame(
            {
                "interpretation": [
                    summary_interpretation.get("strongest_empirical_result", ""),
                    summary_interpretation.get("weakest_empirical_result", ""),
                    str(summary_interpretation.get("support_risk_regime_framework", False)),
                ],
                "notes": [
                    "Strongest empirical result",
                    "Weakest empirical result",
                    "Support risk/regime framework",
                ],
            }
        ).set_index("notes")
        print(format_table(summary_df, index=True))

        print("Limitations:")
        for line in summary_interpretation.get("limitations", []):
            print(f"- {line}")
        print("Next steps:")
        for line in summary_interpretation.get("next_steps", []):
            print(f"- {line}")
    except Exception as exc:
        print(f"Experiment summary failed: {exc}")

    print("\nEmpirical research script completed. This output is intended for research discussion only.")


if __name__ == "__main__":
    main()
