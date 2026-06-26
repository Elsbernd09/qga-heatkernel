import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from topology.persistent_homology import PersistentHomologyRegimeDetector


def test_build_asset_feature_matrix_returns_standardized_features():
    dates = pd.date_range("2025-01-01", periods=65, freq="D")
    prices = pd.DataFrame(
        {
            "A": np.linspace(100, 110, 65),
            "B": np.linspace(50, 55, 65),
            "SPY": np.linspace(400, 405, 65),
        },
        index=dates,
    )

    features = PersistentHomologyRegimeDetector.build_asset_feature_matrix(prices, window=60)

    assert list(features.index) == ["A", "B", "SPY"]
    assert "beta_to_SPY" in features.columns
    assert np.allclose(features.mean(axis=0), 0, atol=1e-6)


def test_compute_diagram_returns_ripser_output():
    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    prices = pd.DataFrame(
        {
            "A": np.sin(np.linspace(0, 3.14, 60)) + 10,
            "B": np.cos(np.linspace(0, 3.14, 60)) + 20,
            "SPY": np.linspace(100, 105, 60),
        },
        index=dates,
    )
    features = PersistentHomologyRegimeDetector.build_asset_feature_matrix(prices, window=60)
    diagram = PersistentHomologyRegimeDetector.compute_diagram(features, maxdim=1)

    assert "dgms" in diagram
    assert len(diagram["dgms"]) >= 2


def test_rolling_diagrams_returns_dataframe():
    dates = pd.date_range("2025-01-01", periods=70, freq="D")
    prices = pd.DataFrame(
        {
            "A": np.linspace(10, 20, 70),
            "B": np.linspace(5, 7, 70),
            "SPY": np.linspace(100, 110, 70),
        },
        index=dates,
    )

    result = PersistentHomologyRegimeDetector.rolling_diagrams(prices, window=60, step=5, maxdim=1)

    assert "feature_matrix" in result.columns
    assert "diagram" in result.columns
    assert "summary" in result.columns
    assert result.shape[0] == 3


def test_diagram_distance_and_regime_distance_matrix():
    feature_a = pd.DataFrame(np.random.normal(size=(3, 6)))
    feature_b = pd.DataFrame(np.random.normal(size=(3, 6)))
    diag_a = PersistentHomologyRegimeDetector.compute_diagram(feature_a, maxdim=1)
    diag_b = PersistentHomologyRegimeDetector.compute_diagram(feature_b, maxdim=1)

    dist = PersistentHomologyRegimeDetector.diagram_distance(diag_a, diag_b, homology_dimension=1)
    assert dist >= 0

    matrix = PersistentHomologyRegimeDetector.regime_distance_matrix([diag_a, diag_b], homology_dimension=1)
    assert matrix.shape == (2, 2)
    assert matrix.iloc[0, 1] == matrix.iloc[1, 0]


def test_find_nearest_historical_regimes_returns_sorted_distances():
    feature_current = pd.DataFrame(np.random.normal(size=(3, 6)))
    diag_current = PersistentHomologyRegimeDetector.compute_diagram(feature_current, maxdim=1)
    historical = []
    for i in range(3):
        feature = pd.DataFrame(np.random.normal(size=(3, 6)))
        diagram = PersistentHomologyRegimeDetector.compute_diagram(feature, maxdim=1)
        historical.append({"date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i), "diagram": diagram})

    nearest = PersistentHomologyRegimeDetector.find_nearest_historical_regimes(diag_current, historical, top_k=2)
    assert nearest.shape == (2, 2)
    assert "date" in nearest.columns
    assert "distance" in nearest.columns


def test_summarize_and_classify_diagram():
    feature = pd.DataFrame(np.random.normal(size=(4, 6)))
    diagram = PersistentHomologyRegimeDetector.compute_diagram(feature, maxdim=1)
    summary = PersistentHomologyRegimeDetector.summarize_diagram(diagram)
    label = PersistentHomologyRegimeDetector.classify_regime_from_topology(summary)

    assert "num_h0" in summary
    assert "num_h1" in summary
    assert isinstance(label, str)
