import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from geometry.padic_tree import PAdicMarketTree, build_default_market_tree


def test_build_default_market_tree_contains_expected_assets():
    tree = build_default_market_tree()

    assert tree.get_node("AAPL").name == "AAPL"
    assert tree.get_node("SPY").parent.name == "Broad Market"
    assert tree.get_node("BTC-USD").parent.parent.name == "Crypto"


def test_path_to_root_returns_correct_hierarchy():
    tree = build_default_market_tree()
    path_names = [node.name for node in tree.path_to_root("MSFT")]

    assert path_names == ["Global Market", "Equities", "Technology", "MSFT"]


def test_lowest_common_ancestor_for_same_sector():
    tree = build_default_market_tree()
    lca = tree.lowest_common_ancestor("AAPL", "MSFT")

    assert lca.name == "Technology"
    assert lca.depth == 2


def test_ultrametric_distance_respects_hierarchy():
    tree = build_default_market_tree()

    same_sector = tree.ultrametric_distance("AAPL", "MSFT", p=2)
    different_sector = tree.ultrametric_distance("AAPL", "JPM", p=2)

    assert same_sector < different_sector
    assert same_sector == 2 ** -2
    assert different_sector == 2 ** -1


def test_distance_matrix_is_symmetric():
    tree = build_default_market_tree()
    assets = ["AAPL", "MSFT", "SPY"]
    matrix = tree.distance_matrix(assets, p=2)

    assert list(matrix.index) == assets
    assert list(matrix.columns) == assets
    assert matrix.loc["AAPL", "MSFT"] == matrix.loc["MSFT", "AAPL"]
    assert matrix.loc["SPY", "SPY"] == 1.0


def test_shock_propagation_score_increases_with_shared_depth():
    tree = build_default_market_tree()

    score_same_sector = tree.shock_propagation_score("AAPL", "MSFT", p=2)
    score_diff_sector = tree.shock_propagation_score("AAPL", "JPM", p=2)

    assert score_same_sector > score_diff_sector
    assert score_same_sector == 2 ** 2
    assert score_diff_sector == 2 ** 1
