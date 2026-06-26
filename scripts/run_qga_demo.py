#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.engine import BacktestEngine
from src.data.features import log_returns
from src.data.loader import download_adjusted_close
from src.geometry.heat_kernel import HeatKernelDiffusion
from src.geometry.padic_tree import build_default_market_tree
from src.geometry.ricci_curvature import RicciCurvatureEngine
from src.quantum.path_integral import PathIntegralSimulator
from src.signals.geometric_signal import GeometricSignalEngine
from src.topology.persistent_homology import PersistentHomologyRegimeDetector


TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "JPM", "GS", "XOM", "GLD", "SPY", "QQQ", "TLT"]
START_DATE = "2022-01-01"
END_DATE = "2024-01-01"
SELECTED_ASSET = "AAPL"


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    print("QGA Research Pipeline Demo")
    print("This script is for demonstration only and does not provide investment advice.")

    print_section("DATA")
    prices = download_adjusted_close(TICKERS, start=START_DATE, end=END_DATE)
    if prices.empty:
        raise RuntimeError("Failed to download price data for the selected tickers.")

    print(f"Downloaded adjusted close prices for {len(prices.columns)} assets from {prices.index.min().date()} to {prices.index.max().date()}.")
    print(prices.iloc[-5:].round(2).to_string())

    returns = log_returns(prices)
    print(f"Computed log returns with {returns.shape[0]} observations.")

    print_section("P-ADIC MARKET TREE")
    tree = build_default_market_tree()
    depth_report = [node.name for node in tree.path_to_root(SELECTED_ASSET)]
    print(f"Path to root for {SELECTED_ASSET}: {depth_report}")
    distance_matrix = tree.distance_matrix(TICKERS, p=2)
    print("Sample ultrametric distances:")
    print(distance_matrix.loc[[SELECTED_ASSET, "MSFT", "SPY"], [SELECTED_ASSET, "MSFT", "SPY"]].round(3).to_string())

    print_section("HEAT KERNEL DIFFUSION")
    try:
        heat_graph = HeatKernelDiffusion.build_correlation_graph(returns.dropna(), threshold=0.2)
        heat_kernel = HeatKernelDiffusion.heat_kernel_matrix(heat_graph, diffusion_time=1.0)
        heat_scores = HeatKernelDiffusion.diffuse_heat(heat_graph, returns.abs().iloc[-1])
        print(f"Built heat diffusion graph with {heat_graph.number_of_nodes()} nodes and {heat_graph.number_of_edges()} edges.")
        print(heat_scores.sort_values(ascending=False).head(5).round(4).to_string())
    except Exception as exc:
        heat_graph = None
        heat_kernel = None
        heat_scores = pd.Series(dtype=float)
        print(f"Heat kernel diffusion skipped due to data shape or graph construction issue: {exc}")

    print_section("RICCI CURVATURE")
    try:
        ricci_graph = RicciCurvatureEngine.build_correlation_graph(returns.dropna(), threshold=0.2)
        edge_curvature = RicciCurvatureEngine.edge_curvature(ricci_graph)
        node_curvature = RicciCurvatureEngine.node_curvature(ricci_graph)
        curvature_history = RicciCurvatureEngine.curvature_time_series(returns.dropna(), window=60, threshold=0.2)
        collapse_flags = RicciCurvatureEngine.detect_curvature_collapse(curvature_history, z_threshold=-1.5, window=20)

        print(f"Computed curvature for {len(node_curvature)} assets.")
        print(node_curvature.sort_values().head(5).round(4).to_string())
        print("Curvature collapse flags (most recent):")
        if not collapse_flags.empty:
            print(collapse_flags.iloc[-1].astype(int).to_string())
    except Exception as exc:
        edge_curvature = None
        node_curvature = pd.Series(dtype=float)
        curvature_history = pd.DataFrame()
        collapse_flags = pd.DataFrame()
        print(f"Ricci curvature analysis skipped due to data shape or processing issue: {exc}")

    print_section("PERSISTENT HOMOLOGY")
    try:
        window = min(60, len(prices))
        if window < 10:
            raise ValueError("Not enough price history to build persistent homology features.")

        feature_matrix = PersistentHomologyRegimeDetector.build_asset_feature_matrix(prices.iloc[-window:], window=window)
        diagram = PersistentHomologyRegimeDetector.compute_diagram(feature_matrix, maxdim=1)
        summary = PersistentHomologyRegimeDetector.summarize_diagram(diagram)
        regime_label = PersistentHomologyRegimeDetector.classify_regime_from_topology(summary)
        rolling = PersistentHomologyRegimeDetector.rolling_diagrams(prices, window=window, step=10, maxdim=1)

        print("Persistent homology summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        print(f"Regime label: {regime_label}")
        print(f"Rolling diagrams produced: {len(rolling)}")
    except Exception as exc:
        feature_matrix = pd.DataFrame()
        diagram = {}
        summary = {}
        regime_label = "Unknown"
        rolling = pd.DataFrame()
        print(f"Persistent homology skipped due to data shape or module issue: {exc}")

    print_section("PATH-INTEGRAL SIMULATION")
    path_summary = {}
    try:
        last_price = float(prices[SELECTED_ASSET].iloc[-1])
        asset_returns = returns[SELECTED_ASSET].dropna()
        if len(asset_returns) < 10:
            raise ValueError("Not enough asset return history for path simulation.")

        drift = float(asset_returns.mean())
        volatility = float(asset_returns.std())
        paths = PathIntegralSimulator.simulate_paths(
            current_price=last_price,
            drift=drift,
            volatility=volatility,
            horizon=20,
            n_paths=1000,
            random_state=42,
        )
        action = PathIntegralSimulator.economic_action(paths, regime="neutral")
        probabilities = PathIntegralSimulator.path_probabilities(action, temperature=1.0)
        path_summary = PathIntegralSimulator.summarize_distribution(paths, probabilities, current_price=last_price)

        print(f"Path-integral summary for {SELECTED_ASSET}:")
        for key, value in path_summary.items():
            print(f"  {key}: {value:.4f}")
    except Exception as exc:
        path_summary = {}
        print(f"Path-integral simulation skipped due to data shape or processing issue: {exc}")

    print_section("GEOMETRIC SIGNAL TABLE")
    signal_frame = pd.DataFrame()
    try:
        signal_engine = GeometricSignalEngine()
        stressed_assets = list(heat_scores.sort_values(ascending=False).head(3).index) if not heat_scores.empty else []
        topology_map = {asset: regime_label for asset in TICKERS}
        path_reports = {SELECTED_ASSET: path_summary} if path_summary else {}

        signal_frame = signal_engine.generate_signal_table(
            assets=TICKERS,
            curvature_series=node_curvature if not node_curvature.empty else None,
            heat_series=heat_scores if not heat_scores.empty else None,
            topology_label=topology_map,
            topology_summary={asset: summary for asset in TICKERS} if summary else None,
            path_summaries=path_reports,
            distance_matrix=distance_matrix,
            stressed_assets=stressed_assets,
        )
        signal_frame = signal_frame["asset final_score signal confidence".split()]
        print(signal_frame.sort_values(by="final_score", ascending=False).head(10).to_string(index=False))
    except Exception as exc:
        print(f"Geometric signal generation skipped due to processing issue: {exc}")

    print_section("BACKTEST RESULTS")
    try:
        if signal_frame.empty:
            raise ValueError("No signal table available for backtest.")

        latest_scores = signal_frame.set_index("asset")["final_score"]
        signal_scores = pd.DataFrame(
            [latest_scores] * len(prices),
            index=prices.index,
            columns=latest_scores.index,
        )
        backtest_engine = BacktestEngine()
        result = backtest_engine.run_backtest(
            prices[latest_scores.index],
            signal_scores,
            long_top_n=3,
            long_short=False,
            rebalance_frequency="W",
        )
        metrics = backtest_engine.calculate_performance_metrics(result["equity_curve"], benchmark_curve=None)

        print("Backtest equity curve summary:")
        print(result["equity_curve"].iloc[-5:].round(4).to_string())
        print("Backtest metrics:")
        for key in ["total_return", "annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown"]:
            print(f"  {key}: {metrics.get(key, 0.0):.4f}")
    except Exception as exc:
        print(f"Backtest skipped due to data or signal shape issue: {exc}")

    print("\nDemo complete. This example is for research demonstration only.")


if __name__ == "__main__":
    main()
