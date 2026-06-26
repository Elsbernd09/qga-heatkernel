import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from geometry.heat_kernel import HeatKernelDiffusion


def test_build_correlation_graph_creates_edges_for_high_corr():
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03, 0.02],
            "B": [0.01, 0.02, 0.0301, 0.0199],
            "C": [0.03, -0.01, 0.01, 0.02],
        }
    )

    graph = HeatKernelDiffusion.build_correlation_graph(returns, threshold=0.9)

    assert set(graph.nodes) == {"A", "B", "C"}
    assert graph.has_edge("A", "B")
    assert not graph.has_edge("A", "C")


def test_graph_laplacian_is_symmetric_and_weighted():
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03, 0.02],
            "B": [0.01, 0.02, 0.0301, 0.0199],
        }
    )
    graph = HeatKernelDiffusion.build_correlation_graph(returns, threshold=0.8)
    lap = HeatKernelDiffusion.graph_laplacian(graph)

    assert lap.shape == (2, 2)
    assert np.allclose(lap.values, lap.values.T)
    assert lap.loc["A", "A"] >= 0


def test_heat_kernel_matrix_is_identity_for_zero_diffusion():
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03, 0.02],
            "B": [0.01, 0.02, 0.0301, 0.0199],
        }
    )
    graph = HeatKernelDiffusion.build_correlation_graph(returns, threshold=0.8)
    kernel = HeatKernelDiffusion.heat_kernel_matrix(graph, diffusion_time=0.0)

    assert np.allclose(kernel.values, np.eye(2))


def test_diffuse_heat_preserves_asset_labels():
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03, 0.02],
            "B": [0.01, 0.02, 0.0301, 0.0199],
        }
    )
    graph = HeatKernelDiffusion.build_correlation_graph(returns, threshold=0.8)
    initial_heat = {"A": 1.0, "B": 0.0}
    diffused = HeatKernelDiffusion.diffuse_heat(graph, initial_heat, diffusion_time=0.5)

    assert set(diffused.index) == {"A", "B"}
    assert diffused.sum() > 0


def test_systemic_heat_score_computes_summary_metrics():
    heat = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    score = HeatKernelDiffusion.systemic_heat_score(heat)

    assert score["total_heat"] == pytest.approx(1.0)
    assert score["max_heat"] == pytest.approx(0.5)
    assert score["concentration_ratio"] == pytest.approx(0.5)


def test_identify_heat_concentration_returns_top_assets():
    heat = pd.Series({"A": 0.9, "B": 0.2, "C": 0.1})
    concentrated = HeatKernelDiffusion.identify_heat_concentration(heat, quantile=0.8)

    assert concentrated == ["A"]
