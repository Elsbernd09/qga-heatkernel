import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from research.experiments import QGAExperimentRunner


def test_future_returns_computes_forward_returns():
    prices = pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0, 104.0, 105.0],
            "B": [50.0, 49.0, 51.0, 52.0, 53.0],
        },
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
    )

    runner = QGAExperimentRunner()
    result = runner.future_returns(prices, horizons=[1, 2])

    assert set(result.keys()) == {1, 2}
    assert result[1].iloc[0, 0] == pytest.approx(0.01)
    assert result[2].iloc[0, 0] == pytest.approx(0.02)
    assert result[2].index[-1] == prices.index[-1]
    assert result[2].iloc[-1].isna().all()


def test_future_realized_volatility_computes_annualized_vol():
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.01, 0.02, -0.01, 0.0],
            "B": [0.02, -0.02, 0.01, -0.01, 0.01],
        },
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
    )

    runner = QGAExperimentRunner()
    vol = runner.future_realized_volatility(returns, horizons=[2])

    assert 2 in vol
    assert vol[2].shape == returns.shape
    expected = returns.iloc[1:3].std(ddof=0) * np.sqrt(252)
    assert vol[2].iloc[0, 0] == pytest.approx(expected["A"])


def test_signal_bucket_test_groups_by_quintile():
    prices = pd.DataFrame(
        {
            "A": [100.0, 102.0, 104.0, 103.0, 105.0],
            "B": [50.0, 51.0, 52.0, 50.0, 51.0],
            "C": [200.0, 198.0, 199.0, 201.0, 202.0],
        },
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
    )
    signals = pd.DataFrame(
        {
            "A": [10.0, 5.0, -5.0, 0.0, 2.0],
            "B": [0.0, 1.0, 3.0, -2.0, -1.0],
            "C": [-10.0, -5.0, 0.0, 5.0, 10.0],
        },
        index=prices.index,
    )

    runner = QGAExperimentRunner()
    bucketed = runner.signal_bucket_test(signals, prices, horizon=1, n_buckets=3)

    assert "bucket" in bucketed.columns
    assert bucketed["count"].sum() > 0
    assert bucketed["average_signal"].is_monotonic_increasing


def test_summarize_experiment_results_returns_dictionary():
    runner = QGAExperimentRunner()
    results = {
        "curvature": {"correlation": pd.DataFrame({"curvature_future_vol_correlation": [0.2, -0.05]}, index=[5, 20])},
        "heat": {"correlation": pd.DataFrame({"heat_future_vol_correlation": [0.12, 0.03], "heat_future_return_correlation": [0.01, -0.02]}, index=[5, 20])},
        "strategy_comparison": pd.DataFrame(
            {
                "total_return": [0.1, 0.05],
                "annualized_return": [0.08, 0.04],
                "annualized_volatility": [0.12, 0.10],
                "sharpe_ratio": [0.7, 0.4],
                "max_drawdown": [-0.1, -0.15],
                "hit_rate": [0.55, 0.5],
            },
            index=["QGA Strategy", "Equal Weight"],
        ),
    }

    summary = runner.summarize_experiment_results(results)
    assert summary["support_risk_regime_framework"] is True
    assert "strongest_empirical_result" in summary
    assert "weakest_empirical_result" in summary
