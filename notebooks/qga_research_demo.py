# %% [markdown]
# # QGA Research Demo
#
# This notebook-style demo walks through the Quantum Geometric Alpha pipeline.
# It uses existing project modules to download market data, compute returns,
# analyze geometric and topological structure, generate signals, and run a backtest.
#
# This example is for research demonstration only and is not investment advice.

# %% [markdown]
# ## 0. Setup and Imports

# %%
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

# %% [markdown]
# ## 1. Project Overview
#
# QGA combines hierarchical market structure, graph diffusion, discrete curvature,
# persistent homology, and path-integral-inspired simulation into a unified
# research pipeline.

# %%
print("QGA Research Demo")
print("This walkthrough uses the project modules to illustrate the main pipeline stages.")

# %% [markdown]
# ## 2. Downloading Market Data
#
# We use a fixed list of tickers and a historical date range for the demo.

# %%
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "JPM", "GS", "XOM", "GLD", "SPY", "QQQ", "TLT"]
START_DATE = "2022-01-01"
END_DATE = "2024-01-01"

prices = download_adjusted_close(TICKERS, start=START_DATE, end=END_DATE)
print(f"Downloaded prices for {len(prices.columns)} tickers with {len(prices)} observations.")
print(prices.iloc[-5:].round(2).to_string())

# %% [markdown]
# ## 3. Computing Returns
#
# We compute log returns from the downloaded adjusted close prices.

# %%
returns = log_returns(prices)
print(f"Computed log returns with {returns.shape[0]} rows and {returns.shape[1]} assets.")
print(returns.iloc[-5:].round(4).to_string())

# %% [markdown]
# ## 4. Building the p-adic Market Tree
#
# The p-adic-inspired market tree captures hierarchical relationships among assets.

# %%
tree = build_default_market_tree()
example_path = tree.path_to_root("AAPL")
print("Example hierarchy path for AAPL:")
print(" -> ".join(node.name for node in example_path))

distance_matrix = tree.distance_matrix(TICKERS, p=2)
print("Ultrametric distance sample:")
print(distance_matrix.loc[["AAPL", "MSFT", "SPY"], ["AAPL", "MSFT", "SPY"]].round(3).to_string())

# %% [markdown]
# ## 5. Running Heat Kernel Diffusion
#
# Heat diffusion is computed on a correlation graph to capture stress propagation.

# %%
try:
    heat_graph = HeatKernelDiffusion.build_correlation_graph(returns.dropna(), threshold=0.2)
    heat_kernel = HeatKernelDiffusion.heat_kernel_matrix(heat_graph, diffusion_time=1.0)
    recent_heat = returns.abs().iloc[-1]
    heat_scores = HeatKernelDiffusion.diffuse_heat(heat_graph, recent_heat)

    print(f"Built heat diffusion graph with {heat_graph.number_of_nodes()} nodes and {heat_graph.number_of_edges()} edges.")
    print(heat_scores.sort_values(ascending=False).head(5).round(4).to_string())
except Exception as exc:
    heat_scores = pd.Series(dtype=float)
    print(f"Heat kernel diffusion skipped: {exc}")

# %% [markdown]
# ## 6. Running Ricci Curvature
#
# Discrete curvature analysis provides a structural liquidity stress proxy.

# %%
try:
    ricci_graph = RicciCurvatureEngine.build_correlation_graph(returns.dropna(), threshold=0.2)
    node_curvature = RicciCurvatureEngine.node_curvature(ricci_graph)
    curvature_history = RicciCurvatureEngine.curvature_time_series(returns.dropna(), window=60, threshold=0.2)
    collapse_flags = RicciCurvatureEngine.detect_curvature_collapse(curvature_history, z_threshold=-1.5, window=20)

    print(f"Computed node curvature for {len(node_curvature)} assets.")
    print(node_curvature.sort_values().head(5).round(4).to_string())
    if not collapse_flags.empty:
        print("Recent curvature collapse flags:")
        print(collapse_flags.iloc[-1].astype(int).to_string())
except Exception as exc:
    node_curvature = pd.Series(dtype=float)
    curvature_history = pd.DataFrame()
    collapse_flags = pd.DataFrame()
    print(f"Ricci curvature analysis skipped: {exc}")

# %% [markdown]
# ## 7. Running Persistent Homology
#
# Persistent homology summarizes regime structure using topological features.

# %%
try:
    window = min(60, len(prices))
    feature_matrix = PersistentHomologyRegimeDetector.build_asset_feature_matrix(prices.iloc[-window:], window=window)
    diagram = PersistentHomologyRegimeDetector.compute_diagram(feature_matrix, maxdim=1)
    summary = PersistentHomologyRegimeDetector.summarize_diagram(diagram)
    regime_label = PersistentHomologyRegimeDetector.classify_regime_from_topology(summary)
    rolling_diagrams = PersistentHomologyRegimeDetector.rolling_diagrams(prices, window=window, step=10, maxdim=1)

    print("Persistent homology summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"Regime label: {regime_label}")
    print(f"Rolling diagram windows: {len(rolling_diagrams)}")
except Exception as exc:
    summary = {}
    regime_label = "Unknown"
    rolling_diagrams = pd.DataFrame()
    print(f"Persistent homology skipped: {exc}")

# %% [markdown]
# ## 8. Running Path-Integral Simulation
#
# The path-integral simulator produces a distribution of future asset paths.

# %%
try:
    selected_asset = "AAPL"
    current_price = float(prices[selected_asset].iloc[-1])
    asset_returns = returns[selected_asset].dropna()
    drift = float(asset_returns.mean())
    volatility = float(asset_returns.std())

    paths = PathIntegralSimulator.simulate_paths(
        current_price=current_price,
        drift=drift,
        volatility=volatility,
        horizon=20,
        n_paths=1000,
        random_state=42,
    )
    action_scores = PathIntegralSimulator.economic_action(paths, regime="neutral")
    probabilities = PathIntegralSimulator.path_probabilities(action_scores, temperature=1.0)
    path_summary = PathIntegralSimulator.summarize_distribution(paths, probabilities, current_price=current_price)

    print(f"Path-integral summary for {selected_asset}:")
    for key, value in path_summary.items():
        print(f"  {key}: {value:.4f}")
except Exception as exc:
    path_summary = {}
    print(f"Path-integral simulation skipped: {exc}")

# %% [markdown]
# ## 9. Generating Geometric Signals
#
# Signals combine curvature, heat, topology, ultrametric structure, and path outcomes.

# %%
signal_table = pd.DataFrame()
try:
    signal_engine = GeometricSignalEngine()
    stressed_assets = list(heat_scores.sort_values(ascending=False).head(3).index) if not heat_scores.empty else []
    topology_map = {asset: regime_label for asset in TICKERS}
    path_summaries = {selected_asset: path_summary} if path_summary else {}

    signal_table = signal_engine.generate_signal_table(
        assets=TICKERS,
        curvature_series=node_curvature if not node_curvature.empty else None,
        heat_series=heat_scores if not heat_scores.empty else None,
        topology_label=topology_map,
        topology_summary={asset: summary for asset in TICKERS} if summary else None,
        path_summaries=path_summaries,
        distance_matrix=distance_matrix,
        stressed_assets=stressed_assets,
    )

    print(signal_table[["asset", "final_score", "signal", "confidence"]].sort_values(by="final_score", ascending=False).head(10).to_string(index=False))
except Exception as exc:
    print(f"Geometric signal generation skipped: {exc}")

# %% [markdown]
# ## 10. Running a Backtest
#
# We use the generated signals to drive a simple backtest.

# %%
try:
    if signal_table.empty:
        raise ValueError("No signal information available for backtest.")

    latest_scores = signal_table.set_index("asset")["final_score"]
    signal_scores = pd.DataFrame(
        [latest_scores] * len(prices),
        index=prices.index,
        columns=latest_scores.index,
    )
    backtest_engine = BacktestEngine()
    backtest_results = backtest_engine.run_backtest(
        prices[latest_scores.index],
        signal_scores,
        long_top_n=3,
        long_short=False,
        rebalance_frequency="W",
    )
    print("Backtest equity curve summary:")
    print(backtest_results["equity_curve"].iloc[-5:].round(4).to_string())
except Exception as exc:
    backtest_results = {}
    print(f"Backtest skipped: {exc}")

# %% [markdown]
# ## 11. Printing Performance Metrics
#
# We summarize the final strategy performance metrics.

# %%
try:
    if backtest_results:
        metrics = backtest_engine.calculate_performance_metrics(backtest_results["equity_curve"], benchmark_curve=None)
        print("Performance metrics:")
        for key in ["total_return", "annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown"]:
            print(f"  {key}: {metrics.get(key, 0.0):.4f}")
except Exception as exc:
    print(f"Performance metrics unavailable: {exc}")

# %% [markdown]
# ## 12. Limitations and Research Notes
#
# - This walkthrough is for research demonstration only and is not financial advice.
# - Historical simulations do not guarantee future outcomes.
# - Discrete geometric and topological metrics are approximations.
# - Signal aggregation is exploratory, not a trading recommendation.
# - Data quality, market impact, and transaction costs are not fully modeled here.

# %%
print("\nDemo complete. This notebook-style walkthrough illustrates the QGA pipeline.")
