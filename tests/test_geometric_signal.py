import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from signals.geometric_signal import GeometricSignalEngine


def test_normalize_series_maps_values_to_range():
    engine = GeometricSignalEngine()
    normalized = engine.normalize_series({"A": 1.0, "B": 2.0, "C": 3.0})

    assert normalized["A"] == pytest.approx(-100.0)
    assert normalized["C"] == pytest.approx(100.0)
    assert normalized.mean() == pytest.approx(0.0)


def test_score_curvature_positive_and_negative():
    engine = GeometricSignalEngine()
    scores = engine.score_curvature({"A": 0.4, "B": -0.6})

    assert scores["A"] > 0
    assert scores["B"] < 0
    assert scores["A"] <= 100
    assert scores["B"] >= -100


def test_score_heat_penalizes_high_heat():
    engine = GeometricSignalEngine()
    scores = engine.score_heat({"A": 0.05, "B": 0.8})

    assert scores["A"] > scores["B"]
    assert scores["B"] < 0


def test_score_topology_label_mapping():
    engine = GeometricSignalEngine()
    assert engine.score_topology("calm") > 0
    assert engine.score_topology("collapsed") < 0
    assert engine.score_topology("unknown") == 0.0


def test_score_path_integral_reflects_upside():
    engine = GeometricSignalEngine()
    score_good = engine.score_path_integral(
        {
            "upside_probability": 0.7,
            "downside_probability": 0.3,
            "expected_return": 0.05,
            "expected_shortfall_5": -0.02,
        }
    )
    score_bad = engine.score_path_integral(
        {
            "upside_probability": 0.3,
            "downside_probability": 0.7,
            "expected_return": -0.03,
            "expected_shortfall_5": -0.08,
        }
    )

    assert score_good > score_bad


def test_score_ultrametric_isolation_prefers_far_assets():
    engine = GeometricSignalEngine()
    distances = pd.DataFrame(
        [[1.0, 0.1, 0.8], [0.1, 1.0, 0.7], [0.8, 0.7, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    far_score = engine.score_ultrametric_isolation("A", distances, stressed_assets=["C"])
    close_score = engine.score_ultrametric_isolation("A", distances, stressed_assets=["B"])

    assert far_score > close_score


def test_combine_scores_ignores_missing_components():
    engine = GeometricSignalEngine()
    combined = engine.combine_scores({"curvature": 50.0, "heat": -20.0})

    assert -100.0 <= combined <= 100.0


def test_classify_signal_thresholds():
    engine = GeometricSignalEngine()
    assert engine.classify_signal(80) == "Strong Long"
    assert engine.classify_signal(50) == "Long"
    assert engine.classify_signal(0) == "Neutral"
    assert engine.classify_signal(-50) == "Risk-Off"
    assert engine.classify_signal(-80) == "Strong Short"


def test_confidence_from_components_agrees_with_direction():
    engine = GeometricSignalEngine()
    confidence = engine.confidence_from_components({"curvature": 40.0, "heat": 20.0})
    assert confidence == pytest.approx(1.0)
    mixed = engine.confidence_from_components({"curvature": 40.0, "heat": -20.0})
    assert mixed < 1.0


def test_generate_asset_signal_and_table():
    engine = GeometricSignalEngine()
    distances = pd.DataFrame(
        [[1.0, 0.4], [0.4, 1.0]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    signal = engine.generate_asset_signal(
        asset="A",
        curvature=0.1,
        heat=0.15,
        topology="calm",
        path_summary={
            "upside_probability": 0.6,
            "downside_probability": 0.4,
            "expected_return": 0.02,
            "expected_shortfall_5": -0.02,
        },
        distance_matrix=distances,
        stressed_assets=["B"],
    )

    assert signal["asset"] == "A"
    assert signal["signal"] in {"Long", "Neutral", "Strong Long"}
    assert 0.0 <= signal["confidence"] <= 1.0

    table = engine.generate_signal_table(
        assets=["A", "B"],
        curvature_series={"A": 0.1, "B": -0.2},
        heat_series={"A": 0.15, "B": 0.6},
        topology_label={"A": "calm", "B": "collapsed"},
        path_summaries={
            "A": {"upside_probability": 0.6, "downside_probability": 0.4, "expected_return": 0.02, "expected_shortfall_5": -0.02},
            "B": {"upside_probability": 0.3, "downside_probability": 0.7, "expected_return": -0.03, "expected_shortfall_5": -0.08},
        },
        distance_matrix=distances,
        stressed_assets=["B"],
    )

    assert list(table.columns) == [
        "asset",
        "final_score",
        "signal",
        "confidence",
        "curvature_score",
        "heat_score",
        "topology_score",
        "path_integral_score",
        "ultrametric_score",
        "explanation",
    ]
    assert table.shape[0] == 2
