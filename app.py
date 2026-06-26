#!/usr/bin/env python3
from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.backtest.engine import BacktestEngine
from src.data.features import log_returns
from src.data.loader import download_adjusted_close
from src.geometry.heat_kernel import HeatKernelDiffusion
from src.geometry.padic_tree import build_default_market_tree
from src.geometry.ricci_curvature import RicciCurvatureEngine
from src.quantum.path_integral import PathIntegralSimulator
from src.signals.geometric_signal import GeometricSignalEngine
from src.topology.persistent_homology import PersistentHomologyRegimeDetector

DEFAULT_TICKERS = [
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
DEFAULT_START_DATE = date(2022, 1, 1)
DEFAULT_END_DATE = date(2024, 1, 1)
REBALANCE_OPTIONS = {"Daily": "D", "Weekly": "W", "Monthly": "M"}


@st.cache_data(show_spinner=False)
def load_prices(tickers: List[str], start_date: date, end_date: date) -> pd.DataFrame:
    return download_adjusted_close(
        tickers,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
    )


def normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return prices
    return prices.div(prices.iloc[0]).fillna(method="ffill")


def build_market_tree(tickers: List[str]) -> pd.DataFrame:
    tree = build_default_market_tree()
    return tree.distance_matrix(tickers, p=2)


@st.cache_data(show_spinner=False)
def compute_heat_scores(returns: pd.DataFrame) -> pd.Series:
    graph = HeatKernelDiffusion.build_correlation_graph(returns, threshold=0.2)
    return HeatKernelDiffusion.diffuse_heat(graph, returns.abs().iloc[-1], diffusion_time=1.0)


@st.cache_data(show_spinner=False)
def compute_curvature_scores(returns: pd.DataFrame) -> pd.Series:
    graph = RicciCurvatureEngine.build_correlation_graph(returns, threshold=0.2)
    return RicciCurvatureEngine.node_curvature(graph)


def compute_topology(prices: pd.DataFrame, returns: pd.DataFrame) -> Dict[str, object]:
    window = min(60, len(prices))
    if window < 10:
        raise ValueError("Not enough history for persistent homology analysis.")

    feature_matrix = PersistentHomologyRegimeDetector.build_asset_feature_matrix(
        prices, returns=returns, window=window
    )
    diagram = PersistentHomologyRegimeDetector.compute_diagram(feature_matrix, maxdim=1)
    summary = PersistentHomologyRegimeDetector.summarize_diagram(diagram)
    regime_label = PersistentHomologyRegimeDetector.classify_regime_from_topology(summary)
    return {
        "feature_matrix": feature_matrix,
        "diagram_summary": summary,
        "regime_label": regime_label,
    }


def compute_path_simulation(
    asset: str,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
) -> Dict[str, object]:
    if asset not in prices.columns or asset not in returns.columns:
        raise ValueError(f"Selected asset '{asset}' is not available in the data.")

    current_price = float(prices[asset].iloc[-1])
    asset_returns = returns[asset].dropna()
    if len(asset_returns) < 10:
        raise ValueError("Not enough history for path-integral simulation.")

    drift = float(asset_returns.mean())
    volatility = float(asset_returns.std())
    simulator = PathIntegralSimulator()
    return simulator.run_simulation(
        current_price=current_price,
        drift=drift,
        volatility=volatility,
        horizon=20,
        n_paths=2000,
        random_state=42,
    )


def generate_signal_table(
    assets: List[str],
    curvature: Optional[pd.Series],
    heat: Optional[pd.Series],
    regime_label: Optional[str],
    path_asset: Optional[str],
    path_summary: Optional[Dict[str, object]],
    distance_matrix: Optional[pd.DataFrame],
    stressed_assets: Optional[List[str]],
) -> pd.DataFrame:
    signal_engine = GeometricSignalEngine()
    topology_map = {asset: regime_label for asset in assets} if regime_label else None
    path_summaries = {path_asset: path_summary} if path_asset and path_summary else None

    return signal_engine.generate_signal_table(
        assets=assets,
        curvature_series=curvature,
        heat_series=heat,
        topology_label=topology_map,
        topology_summary={asset: {"regime_label": regime_label} for asset in assets} if regime_label else None,
        path_summaries=path_summaries,
        distance_matrix=distance_matrix,
        stressed_assets=stressed_assets,
    )


def build_backtest_signals(
    prices: pd.DataFrame,
    signal_table: pd.DataFrame,
) -> pd.DataFrame:
    if signal_table.empty:
        raise ValueError("Signal table is empty and cannot be used for backtesting.")

    latest_scores = signal_table.set_index("asset")["final_score"].reindex(prices.columns).fillna(0.0)
    repeated = pd.DataFrame(
        [latest_scores.values] * len(prices),
        index=prices.index,
        columns=prices.columns,
        dtype=float,
    )
    repeated.index.name = prices.index.name
    return repeated


def plot_probability_wave(wave: pd.DataFrame) -> None:
    if wave.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 3))
    widths = np.diff(wave["terminal_price_bin"].to_numpy())
    widths = widths.mean() if len(widths) > 0 else 1.0
    ax.bar(
        wave["terminal_price_bin"],
        wave["probability_mass"],
        width=widths,
        alpha=0.75,
    )
    ax.set_title("Path-Integral Terminal Price Probability Mass")
    ax.set_xlabel("Terminal price")
    ax.set_ylabel("Probability mass")
    fig.tight_layout()
    st.pyplot(fig)


def main() -> None:
    st.set_page_config(page_title="QGA: Quantum Geometric Alpha", layout="wide")
    st.title("QGA: Quantum Geometric Alpha")
    st.markdown(
        """
        QGA is a quantitative research dashboard that models financial markets as evolving geometric and topological systems.
        This research interface is for exploration only and is not investment advice.
        """
    )

    with st.sidebar:
        st.header("Inputs")
        tickers = st.multiselect("Tickers", DEFAULT_TICKERS, default=DEFAULT_TICKERS)
        start_date = st.date_input("Start date", DEFAULT_START_DATE)
        end_date = st.date_input("End date", DEFAULT_END_DATE)
        selected_asset = st.selectbox(
            "Selected asset for path-integral simulation",
            tickers if tickers else DEFAULT_TICKERS,
            index=0,
        )
        rebalance_label = st.selectbox("Rebalance frequency", list(REBALANCE_OPTIONS.keys()), index=1)
        run_analysis = st.button("Run analysis")

        st.markdown("---")
        st.info("Research only: this dashboard is not financial advice.")
        st.write(
            "Historical results are illustrative and do not guarantee future performance."
        )

    if not tickers:
        st.warning("Please select at least one ticker to run the analysis.")
        return

    if start_date >= end_date:
        st.warning("Please select an end date after the start date.")
        return

    if not run_analysis and "analysis_ran" not in st.session_state:
        st.info("Select inputs and click Run analysis to begin.")
        return

    st.session_state.analysis_ran = True
    error_messages: List[str] = []

    try:
        prices = load_prices(tickers, start_date, end_date)
        if prices.empty:
            raise ValueError("No price data returned for the selected tickers and date range.")
        returns = log_returns(prices)
    except Exception as exc:
        st.error(f"Market data failed: {exc}")
        return

    st.header("Project Overview")
    st.markdown(
        "QGA combines hierarchical market structure, heat diffusion, liquidity curvature, persistent topology, and path-integral return simulation into a research dashboard."
    )

    st.header("Market Data")
    st.write("### Price data preview")
    st.dataframe(prices.tail(5))
    st.write("### Normalized price paths")
    st.line_chart(normalize_prices(prices))

    st.header("p-adic Market Tree")
    try:
        distance_matrix = build_market_tree(list(prices.columns))
        tree = build_default_market_tree()
        tree_path = [node.name for node in tree.path_to_root(selected_asset)]
        st.write(f"**Path to root for {selected_asset}:** {tree_path}")

        if "MSFT" in distance_matrix.index and "SPY" in distance_matrix.index:
            subset = distance_matrix.loc[[selected_asset, "MSFT", "SPY"], [selected_asset, "MSFT", "SPY"]].round(3)
        else:
            subset = distance_matrix.loc[[selected_asset], [selected_asset]].round(3)
        st.dataframe(subset)
    except Exception as exc:
        st.warning(f"p-adic market tree analysis failed: {exc}")
        distance_matrix = pd.DataFrame()
        error_messages.append(str(exc))

    st.header("Heat Kernel Diffusion")
    try:
        heat_scores = compute_heat_scores(returns).sort_values(ascending=False)
        heat_df = heat_scores.rename("diffused_heat").to_frame()
        st.dataframe(heat_df)
        st.bar_chart(heat_df)
    except Exception as exc:
        st.warning(f"Heat kernel diffusion failed: {exc}")
        heat_scores = pd.Series(dtype=float)
        error_messages.append(str(exc))

    st.header("Ricci Liquidity Curvature")
    try:
        curvature = compute_curvature_scores(returns).sort_values(ascending=True)
        curvature_df = curvature.rename("curvature").to_frame()
        st.dataframe(curvature_df)
        st.bar_chart(curvature_df)
    except Exception as exc:
        st.warning(f"Ricci curvature analysis failed: {exc}")
        curvature = pd.Series(dtype=float)
        error_messages.append(str(exc))

    st.header("Persistent Homology Regime Detection")
    topology: Dict[str, object] = {}
    try:
        topology = compute_topology(prices, returns)
        summary_df = pd.DataFrame(topology["diagram_summary"], index=[0]).T.rename(columns={0: "value"})
        st.write(f"**Regime label:** {topology['regime_label']}")
        st.dataframe(summary_df)
    except Exception as exc:
        st.warning(f"Persistent homology analysis failed: {exc}")
        topology = {}
        error_messages.append(str(exc))

    st.header("Path-Integral Simulation")
    path_simulation: Dict[str, object] = {}
    try:
        path_simulation = compute_path_simulation(selected_asset, prices, returns)
        path_summary = path_simulation["summary"]
        summary_df = pd.DataFrame(path_summary, index=[0]).T.rename(columns={0: "value"})
        st.dataframe(summary_df)
        st.write("### Probability wave")
        plot_probability_wave(path_simulation["probability_wave"])
    except Exception as exc:
        st.warning(f"Path-integral simulation failed: {exc}")
        path_simulation = {}
        error_messages.append(str(exc))

    st.header("Unified Geometric Signal Table")
    signal_table = pd.DataFrame()
    try:
        stressed_assets = list(heat_scores.head(3).index) if not heat_scores.empty else []
        signal_table = generate_signal_table(
            assets=list(prices.columns),
            curvature=curvature if not curvature.empty else None,
            heat=heat_scores if not heat_scores.empty else None,
            regime_label=topology.get("regime_label") if topology else None,
            path_asset=selected_asset,
            path_summary=path_simulation if path_simulation else None,
            distance_matrix=distance_matrix if not distance_matrix.empty else None,
            stressed_assets=stressed_assets,
        )
        if not signal_table.empty:
            display_columns = ["asset", "final_score", "signal", "confidence", "explanation"]
            st.dataframe(signal_table[display_columns].sort_values(by="final_score", ascending=False))
        else:
            st.warning("Signal table was not produced.")
    except Exception as exc:
        st.warning(f"Signal generation failed: {exc}")
        error_messages.append(str(exc))

    st.header("Backtest Results")
    try:
        if signal_table.empty:
            raise ValueError("Cannot run backtest without a signal table.")
        signal_scores = build_backtest_signals(prices, signal_table)
        backtest_engine = BacktestEngine()
        rebalance_frequency = REBALANCE_OPTIONS.get(rebalance_label, "W")
        results = backtest_engine.run_backtest(
            prices,
            signal_scores,
            long_top_n=3,
            long_short=False,
            rebalance_frequency=rebalance_frequency,
        )

        strategy_curve = results["equity_curve"]
        st.write("### Equity curve")
        st.line_chart(strategy_curve)

        strategy_metrics = backtest_engine.calculate_performance_metrics(
            strategy_curve,
            benchmark_curve=backtest_engine.spy_benchmark(prices, spy_column="SPY") if "SPY" in prices.columns else None,
        )
        metrics = {"Strategy": strategy_metrics}

        if "SPY" in prices.columns:
            spy_curve = backtest_engine.spy_benchmark(prices, spy_column="SPY")
            metrics["SPY"] = backtest_engine.calculate_performance_metrics(spy_curve)

        metrics["Equal Weight"] = backtest_engine.calculate_performance_metrics(
            backtest_engine.equal_weight_benchmark(prices),
        )

        metrics_df = pd.DataFrame(metrics).T
        st.write("### Performance metrics")
        st.dataframe(metrics_df)
    except Exception as exc:
        st.warning(f"Backtest failed: {exc}")
        error_messages.append(str(exc))

    st.header("Limitations")
    st.markdown(
        """
        - This dashboard is a research tool and is not financial advice.
        - Results are derived from experimental models and historical analysis.
        - Backtests are illustrative and do not guarantee future returns.
        - Some advanced modules may fail on sparse or insufficient data.
        """
    )

    if error_messages:
        st.sidebar.error("Some modules produced warnings. Review the main page output for details.")


if __name__ == "__main__":
    main()
