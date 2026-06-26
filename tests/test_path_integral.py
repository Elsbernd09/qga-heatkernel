import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from quantum.path_integral import PathIntegralSimulator


def test_simulate_paths_returns_dataframe_with_initial_price():
    simulator = PathIntegralSimulator()
    paths = simulator.simulate_paths(
        current_price=100.0,
        drift=0.001,
        volatility=0.02,
        horizon=5,
        n_paths=10,
        random_state=42,
    )

    assert isinstance(paths, pd.DataFrame)
    assert paths.shape == (6, 10)
    assert (paths.iloc[0] == 100.0).all()
    assert (paths > 0).all().all()


def test_run_simulation_outputs_consistent_summary_and_probabilities():
    simulator = PathIntegralSimulator()
    result = simulator.run_simulation(
        current_price=100.0,
        drift=0.001,
        volatility=0.02,
        horizon=5,
        n_paths=20,
        random_state=123,
    )

    assert "paths" in result
    assert "action_scores" in result
    assert "probabilities" in result
    assert "summary" in result
    assert "probability_wave" in result

    assert result["paths"].shape == (6, 20)
    assert isinstance(result["action_scores"], pd.Series)
    assert result["probabilities"].sum() == pytest.approx(1.0)
    assert result["probability_wave"].shape[0] == 50
    assert result["summary"]["expected_terminal_price"] > 0
    assert 0.0 <= result["summary"]["upside_probability"] <= 1.0
    assert 0.0 <= result["summary"]["downside_probability"] <= 1.0


def test_path_probabilities_prefers_lower_action():
    simulator = PathIntegralSimulator()
    actions = pd.Series({"path_0": 1.0, "path_1": 3.0, "path_2": 2.0})
    probabilities = simulator.path_probabilities(actions, temperature=0.5)

    assert probabilities["path_0"] > probabilities["path_2"]
    assert probabilities["path_2"] > probabilities["path_1"]
    assert probabilities.sum() == pytest.approx(1.0)
