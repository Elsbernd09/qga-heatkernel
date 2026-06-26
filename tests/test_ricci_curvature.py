import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from geometry.ricci_curvature import RicciCurvatureEngine


def test_build_correlation_graph_creates_weighted_edges():
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03, 0.02],
            "B": [0.01, 0.02, 0.0301, 0.0199],
            "C": [0.04, -0.03, 0.01, 0.02],
        }
    )

    graph = RicciCurvatureEngine.build_correlation_graph(returns, threshold=0.5)

    assert set(graph.nodes) == {"A", "B", "C"}
    assert graph.has_edge("A", "B")
    assert graph["A"]["B"]["distance"] >= 0
    assert graph["A"]["B"]["strength"] == pytest.approx(abs(returns["A"].corr(returns["B"])))


def test_edge_curvature_returns_dataframe():
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03, 0.02],
            "B": [0.01, 0.02, 0.0301, 0.0199],
            "C": [0.02, 0.01, 0.02, 0.01],
        }
    )
    graph = RicciCurvatureEngine.build_correlation_graph(returns, threshold=0.8)
    edge_df = RicciCurvatureEngine.edge_curvature(graph)

    assert "curvature" in edge_df.columns
    assert "distance" in edge_df.columns
    assert "strength" in edge_df.columns
    assert edge_df.shape[0] == graph.number_of_edges()


def test_node_curvature_averages_edge_curvature():
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03, 0.02],
            "B": [0.01, 0.02, 0.0301, 0.0199],
            "C": [0.03, 0.02, 0.01, 0.02],
        }
    )
    graph = RicciCurvatureEngine.build_correlation_graph(returns, threshold=0.8)
    node_curv = RicciCurvatureEngine.node_curvature(graph)

    assert set(node_curv.index) == {"A", "B", "C"}
    assert node_curv.dtype == float


def test_curvature_time_series_shapes():
    returns = pd.DataFrame(
        {
            "A": np.random.normal(size=80),
            "B": np.random.normal(size=80),
            "C": np.random.normal(size=80),
        },
        index=pd.date_range("2025-01-01", periods=80, freq="D"),
    )
    curvature_df = RicciCurvatureEngine.curvature_time_series(returns, window=20, threshold=0.1)

    assert curvature_df.shape[0] == 61
    assert set(curvature_df.columns) == {"A", "B", "C"}


def test_detect_curvature_collapse_flags_negative_zscores():
    index = pd.date_range("2025-01-01", periods=10, freq="D")
    curvature_df = pd.DataFrame(
        {
            "A": np.linspace(1.0, -3.0, 10),
            "B": np.linspace(0.5, 0.2, 10),
        },
        index=index,
    )
    collapse = RicciCurvatureEngine.detect_curvature_collapse(curvature_df, z_threshold=-1.0, window=5)

    assert collapse.shape == curvature_df.shape
    assert bool(collapse.iloc[-1, 0]) is True


def test_liquidity_blackhole_score_outputs_dataframe():
    returns = pd.DataFrame(
        {
            "A": np.random.normal(size=50),
            "B": np.random.normal(size=50),
            "C": np.random.normal(size=50),
        },
        index=pd.date_range("2025-01-01", periods=50, freq="D"),
    )
    curvature_df = RicciCurvatureEngine.curvature_time_series(returns, window=20, threshold=0.1)
    score_df = RicciCurvatureEngine.liquidity_blackhole_score(returns, curvature_df, window=20)

    assert score_df.shape == curvature_df.shape
    assert (score_df >= 0).all().all()


def test_detect_liquidity_blackholes_returns_boolean():
    returns = pd.DataFrame(
        {
            "A": np.linspace(0.1, 1.0, 30),
            "B": np.linspace(-0.1, -0.5, 30),
            "C": np.random.normal(scale=0.01, size=30),
        },
        index=pd.date_range("2025-01-01", periods=30, freq="D"),
    )
    curvature_df = RicciCurvatureEngine.curvature_time_series(returns, window=10, threshold=0.1)
    blackholes = RicciCurvatureEngine.detect_liquidity_blackholes(
        returns, curvature_df, z_threshold=-1.0, window=10
    )

    assert blackholes.shape == curvature_df.shape
    assert blackholes.dtypes.unique().tolist() == [bool]
