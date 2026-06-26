from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class QGADashboard:
    """Visualization tools for Quantum Geometric Alpha research outputs.

    This dashboard module converts quantitative outputs into interpretable
    charts for market stress, regime structure, signal scores, and portfolio
    performance. Visualizations are intended for research support and do not
    represent investment advice or guaranteed predictions.
    """

    @staticmethod
    def plot_price_history(
        prices: pd.DataFrame,
        assets: Optional[Sequence[str]] = None,
        title: str = "Asset Price History",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot normalized asset price histories for easy relative comparison."""
        if assets is not None:
            prices = prices[assets].copy()
        else:
            prices = prices.copy()

        normalized = prices.div(prices.iloc[0]).mul(100.0)
        fig, ax = plt.subplots(figsize=(10, 6))
        normalized.plot(ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Price (Base = 100)")
        ax.legend(title="Assets", loc="best")
        fig.tight_layout()
        return fig, ax

    @staticmethod
    def plot_returns_heatmap(
        returns: pd.DataFrame,
        title: str = "Return Correlation Heatmap",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Show return correlations to highlight market structure and co-movement."""
        corr = returns.corr()
        fig, ax = plt.subplots(figsize=(8, 8))
        cax = ax.imshow(corr, aspect="equal", cmap="RdYlBu", vmin=-1, vmax=1)
        ax.set_title(title)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.index)
        fig.colorbar(cax, ax=ax, label="Correlation")
        fig.tight_layout()
        return fig, ax

    @staticmethod
    def plot_heat_diffusion(
        diffused_heat: Union[pd.Series, Dict[str, float]],
        title: str = "Heat Kernel Diffusion",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Display the diffusion of heat or stress across market assets."""
        heat_series = pd.Series(diffused_heat).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        heat_series.plot.bar(ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Asset")
        ax.set_ylabel("Heat Intensity")
        fig.tight_layout()
        return fig, ax

    @staticmethod
    def plot_ricci_curvature(
        curvature_series: Union[pd.Series, Dict[str, float]],
        title: str = "Ricci Curvature by Asset",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot curvature values by asset to show liquidity and structural stress."""
        series = pd.Series(curvature_series)
        fig, ax = plt.subplots(figsize=(10, 6))
        series.plot.bar(ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Asset")
        ax.set_ylabel("Ricci Curvature")
        fig.tight_layout()
        return fig, ax

    @staticmethod
    def plot_curvature_time_series(
        curvature_df: pd.DataFrame,
        assets: Optional[Sequence[str]] = None,
        title: str = "Curvature Time Series",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot curvature changes over time for selected assets."""
        if assets is not None:
            curvature_df = curvature_df[assets].copy()
        fig, ax = plt.subplots(figsize=(10, 6))
        curvature_df.plot(ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Ricci Curvature")
        ax.legend(title="Assets", loc="best")
        fig.tight_layout()
        return fig, ax

    @staticmethod
    def plot_topology_complexity(
        rolling_topology: Union[pd.DataFrame, Sequence[Dict[str, Any]]],
        title: str = "Persistent Homology Complexity",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Visualize topology-based complexity scores and regime labels over time."""
        if isinstance(rolling_topology, list):
            rolling_topology = pd.DataFrame(rolling_topology)

        if not isinstance(rolling_topology, pd.DataFrame):
            raise ValueError("rolling_topology must be a DataFrame or list of dictionaries")

        fig, ax = plt.subplots(figsize=(10, 6))
        if "complexity" in rolling_topology.columns:
            rolling_topology["complexity"].plot(ax=ax)
            ax.set_ylabel("Topology Complexity")
        else:
            ax.text(0.5, 0.5, "No complexity series available", ha="center", va="center")

        ax.set_title(title)
        ax.set_xlabel("Date")
        fig.tight_layout()
        return fig, ax

    @staticmethod
    def plot_signal_scores(
        signal_table: pd.DataFrame,
        title: str = "Geometric Signal Scores",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Show composite signal scores by asset for easy ranking interpretation."""
        sorted_table = signal_table.sort_values(by="final_score", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(sorted_table["asset"], sorted_table["final_score"])
        ax.set_title(title)
        ax.set_xlabel("Asset")
        ax.set_ylabel("Final Score")
        labels = None
        if "signal" in sorted_table.columns:
            labels = [f"{asset}\n({signal})" for asset, signal in zip(sorted_table["asset"], sorted_table["signal"]) ]
        else:
            labels = list(sorted_table["asset"])

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        fig.tight_layout()
        return fig, ax

    @staticmethod
    def plot_equity_curve(
        equity_curve: pd.Series,
        benchmark_curve: Optional[pd.Series] = None,
        title: str = "Backtest Equity Curve",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot a strategy equity curve with optional benchmark comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        equity_curve.plot(ax=ax, label="Strategy")
        if benchmark_curve is not None:
            benchmark_curve.plot(ax=ax, label="Benchmark")
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity Value")
        ax.legend(loc="best")
        fig.tight_layout()
        return fig, ax

    @staticmethod
    def plot_drawdown(
        drawdown: pd.Series,
        title: str = "Portfolio Drawdown",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Show the portfolio drawdown path to highlight peak-to-trough risk."""
        fig, ax = plt.subplots(figsize=(10, 6))
        drawdown.plot(ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        fig.tight_layout()
        return fig, ax

    @staticmethod
    def plot_rolling_sharpe(
        rolling_sharpe: pd.Series,
        title: str = "Rolling Sharpe Ratio",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot a rolling Sharpe ratio to visualize risk-adjusted performance trends."""
        fig, ax = plt.subplots(figsize=(10, 6))
        rolling_sharpe.plot(ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Sharpe Ratio")
        fig.tight_layout()
        return fig, ax

    @staticmethod
    def create_summary_report(
        signal_table: pd.DataFrame,
        strategy_metrics: Dict[str, float],
        benchmark_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Create a compact report summarizing signal and backtest performance."""
        top_long = signal_table.sort_values(by="final_score", ascending=False).head(1)
        worst_risk = signal_table.sort_values(by="final_score").head(1)

        report = {
            "top_long_signal": top_long["asset"].iloc[0] if not top_long.empty else None,
            "worst_risk_signal": worst_risk["asset"].iloc[0] if not worst_risk.empty else None,
            "average_confidence": float(signal_table["confidence"].mean()) if "confidence" in signal_table.columns else 0.0,
            "strategy_total_return": strategy_metrics.get("total_return", 0.0),
            "strategy_sharpe": strategy_metrics.get("sharpe_ratio", 0.0),
            "strategy_max_drawdown": strategy_metrics.get("max_drawdown", 0.0),
        }

        if benchmark_metrics is not None:
            report.update(
                {
                    "benchmark_total_return": benchmark_metrics.get("total_return", 0.0),
                    "benchmark_sharpe": benchmark_metrics.get("sharpe_ratio", 0.0),
                    "benchmark_max_drawdown": benchmark_metrics.get("max_drawdown", 0.0),
                }
            )

        return report
