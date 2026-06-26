from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import pandas as pd


@dataclass
class TreeNode:
    """A node in a market hierarchy tree.

    Attributes:
        name: The unique name of the tree node.
        parent: Optional parent node reference.
        children: Mapping from child name to child node.
    """

    name: str
    parent: Optional["TreeNode"] = None
    children: Dict[str, "TreeNode"] = field(default_factory=dict)

    @property
    def depth(self) -> int:
        """Return the distance from the root node."""
        return 0 if self.parent is None else self.parent.depth + 1

    def add_child(self, child_name: str) -> "TreeNode":
        """Add a child node if it does not exist and return it."""
        if child_name not in self.children:
            self.children[child_name] = TreeNode(name=child_name, parent=self)
        return self.children[child_name]


class PAdicMarketTree:
    """A hierarchical market tree using ultrametric distances.

    This structure is p-adic-inspired because it models market similarity by the depth
    of a shared hierarchy rather than performing literal p-adic arithmetic.
    Assets that share a deeper common ancestor are considered closer in the
    ultrametric space.
    """

    def __init__(self, root_name: str = "Global Market") -> None:
        self.root = TreeNode(name=root_name)
        self._nodes: Dict[str, TreeNode] = {root_name: self.root}

    def add_path(self, path: List[str]) -> None:
        """Add a hierarchical path from root to a leaf asset.

        Args:
            path: Sequence of node names from root to asset.
        """
        if not path:
            raise ValueError("Path must contain at least one node name.")

        current = self.root
        if path[0] != self.root.name:
            path = [self.root.name] + path

        for name in path[1:]:
            current = current.add_child(name)
            self._nodes[name] = current

    def get_node(self, name: str) -> TreeNode:
        """Return the node with the given name."""
        try:
            return self._nodes[name]
        except KeyError as exc:
            raise KeyError(f"Node '{name}' not found in market tree.") from exc

    def path_to_root(self, name: str) -> List[TreeNode]:
        """Return the nodes from the named node up to the root."""
        node = self.get_node(name)
        path: List[TreeNode] = []
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def lowest_common_ancestor(self, asset_a: str, asset_b: str) -> TreeNode:
        """Return the lowest common ancestor node for two assets."""
        path_a = self.path_to_root(asset_a)
        path_b = self.path_to_root(asset_b)

        lca = self.root
        for node_a, node_b in zip(path_a, path_b):
            if node_a is node_b:
                lca = node_a
            else:
                break
        return lca

    def ultrametric_distance(self, asset_a: str, asset_b: str, p: int = 2) -> float:
        """Compute ultrametric distance between two assets.

        In this hierarchical market model, distance is defined by the depth of the
        lowest common ancestor: deeper shared hierarchy implies a smaller distance.
        """
        if p <= 1:
            raise ValueError("Parameter p must be greater than 1 for ultrametric distance.")

        lca = self.lowest_common_ancestor(asset_a, asset_b)
        return float(p) ** (-lca.depth)

    def distance_matrix(self, assets: List[str], p: int = 2) -> pd.DataFrame:
        """Return a symmetric ultrametric distance matrix for a list of assets."""
        data = {
            asset: [self.ultrametric_distance(asset, other, p=p) for other in assets]
            for asset in assets
        }
        return pd.DataFrame(data, index=assets)

    def shock_propagation_score(self, source_asset: str, target_asset: str, p: int = 2) -> float:
        """Score how strongly a shock at source_asset propagates to target_asset.

        The propagation score increases with shared hierarchical depth.
        It is the inverse of ultrametric distance in the p-adic-inspired model.
        """
        if p <= 1:
            raise ValueError("Parameter p must be greater than 1 for shock propagation score.")

        lca = self.lowest_common_ancestor(source_asset, target_asset)
        return float(p) ** lca.depth


def build_default_market_tree() -> PAdicMarketTree:
    """Build the default market hierarchy for QGA.

    The hierarchy captures market structure from Global Market down through asset classes,
    sectors, and individual assets. Assets in the same sector share a deeper ancestor
    and are therefore closer in the ultrametric sense.
    """
    tree = PAdicMarketTree(root_name="Global Market")

    tree.add_path(["Global Market", "Equities", "Technology", "AAPL"])
    tree.add_path(["Global Market", "Equities", "Technology", "MSFT"])
    tree.add_path(["Global Market", "Equities", "Technology", "NVDA"])
    tree.add_path(["Global Market", "Equities", "Technology", "AMZN"])
    tree.add_path(["Global Market", "Equities", "Technology", "META"])
    tree.add_path(["Global Market", "Equities", "Financials", "JPM"])
    tree.add_path(["Global Market", "Equities", "Financials", "GS"])
    tree.add_path(["Global Market", "Equities", "Energy", "XOM"])
    tree.add_path(["Global Market", "Bonds", "Treasury", "TLT"])
    tree.add_path(["Global Market", "Commodities", "Precious Metals", "GLD"])
    tree.add_path(["Global Market", "Crypto", "Major Crypto", "BTC-USD"])
    tree.add_path(["Global Market", "Crypto", "Major Crypto", "ETH-USD"])
    tree.add_path(["Global Market", "Indexes", "Broad Market", "SPY"])
    tree.add_path(["Global Market", "Indexes", "Broad Market", "QQQ"])

    return tree
