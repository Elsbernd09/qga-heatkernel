import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from backtest.engine import BacktestEngine


def test_compute_forward_returns_aligns_dates():
    engine = BacktestEngine()
    dates = pd.date_range(start="2024-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"A": [100.0, 102.0, 101.0]}, index=dates)
    forward = engine.compute_forward_returns(prices, horizon=1)

    assert forward.iloc[0, 0] == pytest.approx(0.02)
    assert np.isnan(forward.iloc[-1, 0])


def test_generate_weights_from_scores_long_only():
    engine = BacktestEngine()
    scores = pd.Series({"A": 0.7, "B": 0.2, "C": -0.1})
    weights = engine.generate_weights_from_scores(scores, long_top_n=2)

    assert weights["A"] == pytest.approx(0.5)
    assert weights["B"] == pytest.approx(0.5)
    assert weights.sum() == pytest.approx(1.0)


def test_generate_weights_from_scores_long_short():
    engine = BacktestEngine()
    scores = pd.Series({"A": 0.7, "B": 0.2, "C": -0.5, "D": -0.9})
    weights = engine.generate_weights_from_scores(scores, long_top_n=1, short_bottom_n=1, long_short=True)

    assert weights["A"] == pytest.approx(0.5)
    assert weights["D"] == pytest.approx(-0.5)
    assert weights.sum() == pytest.approx(0.0)


def test_rebalance_dates_supports_frequencies():
    engine = BacktestEngine()
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    prices = pd.DataFrame({"A": np.arange(10.0)}, index=dates)

    assert len(engine.rebalance_dates(prices, "D")) == 10
    assert len(engine.rebalance_dates(prices, "W")) >= 1
    assert len(engine.rebalance_dates(prices, "M")) == 1


def test_run_backtest_returns_equity_curve_and_weights():
    engine = BacktestEngine()
    dates = pd.date_range(start="2024-01-01", periods=6, freq="D")
    prices = pd.DataFrame({"A": [100, 101, 102, 101, 103, 104], "B": [50, 49, 48, 49, 50, 51], "SPY": [400, 402, 401, 403, 404, 406]}, index=dates)
    signals = pd.DataFrame(
        {
            "A": [0.9, 0.8, 0.7, 0.6, 0.9, 0.8],
            "B": [0.1, 0.2, 0.3, 0.4, 0.1, 0.2],
        },
        index=dates,
    )

    result = engine.run_backtest(prices, signals, long_top_n=1, long_short=False, rebalance_frequency="D")

    assert set(result.keys()) == {"equity_curve", "daily_returns", "weights", "turnover", "trades"}
    assert result["weights"].shape[0] == len(dates)
    assert result["equity_curve"].iloc[0] == pytest.approx(1.0)


def test_calculate_drawdown_returns_series():
    engine = BacktestEngine()
    equity_curve = pd.Series([1.0, 1.2, 1.1, 1.3], index=pd.date_range("2024-01-01", periods=4, freq="D"))
    drawdown = engine.calculate_drawdown(equity_curve)

    assert isinstance(drawdown, pd.Series)
    assert drawdown.shape == equity_curve.shape
    assert drawdown.iloc[0] == pytest.approx(0.0)
    assert drawdown.iloc[2] < 0.0


def test_equal_weight_and_spy_benchmarks():
    engine = BacktestEngine()
    dates = pd.date_range(start="2024-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"A": [100, 101, 102], "B": [50, 51, 52], "SPY": [400, 402, 404]}, index=dates)

    equal_curve = engine.equal_weight_benchmark(prices)
    spy_curve = engine.spy_benchmark(prices)

    assert equal_curve.iloc[0] == pytest.approx(1.0)
    assert spy_curve.iloc[-1] == pytest.approx(1.01)


def test_calculate_performance_metrics_returns_expected_keys():
    engine = BacktestEngine()
    series = pd.Series([1.0, 1.01, 1.02], index=pd.date_range("2024-01-01", periods=3, freq="D"))
    metrics = engine.calculate_performance_metrics(series)

    assert metrics["total_return"] == pytest.approx(0.02)
    assert "sharpe_ratio" in metrics


def test_turnover_computes_weight_changes():
    engine = BacktestEngine()
    weights = pd.DataFrame(
        {
            "A": [0.5, 0.5, 0.0],
            "B": [0.5, 0.0, 0.5],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    turnover = engine.turnover(weights)

    assert turnover.iloc[1] == pytest.approx(0.25)
    assert turnover.iloc[2] == pytest.approx(0.5)


def test_backtest_report_computes_metrics():
    engine = BacktestEngine()
    dates = pd.date_range(start="2024-01-01", periods=4, freq="D")
    prices = pd.DataFrame({"A": [100, 101, 102, 103], "B": [50, 51, 50, 52], "SPY": [400, 401, 402, 403]}, index=dates)
    signals = pd.DataFrame({"A": [0.8, 0.8, 0.8, 0.8], "B": [0.2, 0.2, 0.2, 0.2]}, index=dates)

    report = engine.backtest_report(prices, signals)
    assert "strategy_metrics" in report
    assert "equal_weight_metrics" in report
    assert "drawdown" in report
    assert "rolling_sharpe" in report
