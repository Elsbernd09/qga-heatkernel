import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from visualization.dashboard import QGADashboard


def test_plot_price_history_returns_figure_axis():
    dashboard = QGADashboard()
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"A": [100, 101, 102, 103, 104], "B": [50, 51, 52, 53, 54]}, index=dates)
    fig, ax = dashboard.plot_price_history(prices)
    assert fig is not None
    assert ax is not None


def test_plot_returns_heatmap_handles_returns():
    dashboard = QGADashboard()
    returns = pd.DataFrame({"A": [0.01, -0.01, 0.02], "B": [0.02, -0.02, 0.01]})
    fig, ax = dashboard.plot_returns_heatmap(returns)
    assert fig is not None
    assert ax is not None


def test_plot_heat_diffusion_sorting_and_output():
    dashboard = QGADashboard()
    fig, ax = dashboard.plot_heat_diffusion({"A": 0.5, "B": 0.8, "C": 0.2})
    assert fig is not None
    assert ax is not None


def test_plot_signal_scores_outputs_chart():
    dashboard = QGADashboard()
    signal_table = pd.DataFrame(
        {
            "asset": ["A", "B"],
            "final_score": [50.0, -20.0],
            "signal": ["Long", "Risk-Off"],
            "confidence": [0.8, 0.3],
        }
    )
    fig, ax = dashboard.plot_signal_scores(signal_table)
    assert fig is not None
    assert ax is not None


def test_create_summary_report_includes_expected_fields():
    dashboard = QGADashboard()
    signal_table = pd.DataFrame(
        {"asset": ["A", "B"], "final_score": [50.0, -20.0], "confidence": [0.8, 0.3]}
    )
    strategy_metrics = {"total_return": 0.1, "sharpe_ratio": 1.2, "max_drawdown": -0.05}
    benchmark_metrics = {"total_return": 0.05, "sharpe_ratio": 0.8, "max_drawdown": -0.08}
    report = dashboard.create_summary_report(signal_table, strategy_metrics, benchmark_metrics)
    assert report["top_long_signal"] == "A"
    assert report["worst_risk_signal"] == "B"
    assert report["benchmark_total_return"] == 0.05
