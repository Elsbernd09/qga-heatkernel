from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd


class RicciCurvatureEngine:
    """Discrete curvature approximation for financial asset graphs.

    This engine provides a practical proxy for Ollivier-Ricci curvature on an
    asset correlation graph. Negative curvature is interpreted as a fragile
    market region or a potential liquidity bottleneck, while positive curvature
    reflects stable local connectivity.

    It is important to note that this is a discrete graph-based proxy and not
    an exact smooth Ricci flow on a manifold.
    """

    @staticmethod
    def build_correlation_graph(
        returns: pd.DataFrame,
        threshold: float = 0.3,
    ) -> nx.Graph:
        """Build a weighted asset graph from return correlations.

        Args:
            returns: DataFrame of asset returns with assets as columns.
            threshold: Minimum absolute correlation required to add an edge.

        Returns:
            A networkx Graph with edge attributes 'corr', 'distance', and 'strength'.
        """
        if returns.empty:
            raise ValueError("Returns DataFrame must not be empty.")

        correlation = returns.corr()
        graph = nx.Graph()
        assets = list(correlation.columns)
        graph.add_nodes_from(assets)

        for i, asset_a in enumerate(assets):
            for asset_b in assets[i + 1 :]:
                corr_value = correlation.at[asset_a, asset_b]
                if pd.isna(corr_value):
                    continue
                strength = float(abs(corr_value))
                if strength <= threshold:
                    continue

                distance = float(np.sqrt(2.0 * (1.0 - corr_value)))
                graph.add_edge(
                    asset_a,
                    asset_b,
                    corr=float(corr_value),
                    distance=distance,
                    strength=strength,
                )

        return graph

    @staticmethod
    def _average_neighbor_distance(
        graph: nx.Graph,
        neighbors_a: Iterable[str],
        neighbors_b: Iterable[str],
    ) -> Optional[float]:
        distances: List[float] = []

        for source in neighbors_a:
            for target in neighbors_b:
                if source == target:
                    distances.append(0.0)
                    continue
                try:
                    distance = nx.shortest_path_length(
                        graph,
                        source=source,
                        target=target,
                        weight="distance",
                    )
                    distances.append(float(distance))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

        if not distances:
            return None
        return float(np.mean(distances))

    @classmethod
    def edge_curvature(cls, graph: nx.Graph) -> pd.DataFrame:
        """Estimate edge curvature for each edge in the graph.

        Curvature is approximated by comparing local neighbor transport distance
        to the direct edge distance. A lower transport distance relative to the
        original edge indicates positive curvature, while a larger transport
        distance indicates negative curvature.
        """
        records: List[Dict[str, float]] = []

        for u, v, data in graph.edges(data=True):
            original_distance = float(data.get("distance", 1.0))
            neighbors_u = set(graph.neighbors(u)) - {v}
            neighbors_v = set(graph.neighbors(v)) - {u}
            avg_transport_distance = cls._average_neighbor_distance(
                graph, neighbors_u, neighbors_v
            )
            if avg_transport_distance is None:
                curvature = 0.0
            else:
                curvature = 1.0 - avg_transport_distance / max(original_distance, 1e-12)

            records.append(
                {
                    "source": u,
                    "target": v,
                    "curvature": float(curvature),
                    "distance": original_distance,
                    "strength": float(data.get("strength", 0.0)),
                }
            )

        return pd.DataFrame(records)

    @staticmethod
    def node_curvature(graph: nx.Graph) -> pd.Series:
        """Compute node curvature as the average curvature of incident edges."""
        edge_df = RicciCurvatureEngine.edge_curvature(graph)
        if edge_df.empty:
            return pd.Series(dtype=float)

        values: Dict[str, List[float]] = {node: [] for node in graph.nodes}
        for _, row in edge_df.iterrows():
            values[row["source"]].append(row["curvature"])
            values[row["target"]].append(row["curvature"])

        result = {
            node: float(np.mean(curvatures)) if curvatures else 0.0
            for node, curvatures in values.items()
        }
        return pd.Series(result).sort_index()

    @staticmethod
    def curvature_time_series(
        returns: pd.DataFrame,
        window: int = 60,
        threshold: float = 0.3,
    ) -> pd.DataFrame:
        """Compute node curvature across a rolling return window."""
        if returns.empty:
            raise ValueError("Returns DataFrame must not be empty.")
        if window < 2:
            raise ValueError("window must be at least 2.")

        assets = list(returns.columns)
        results: List[pd.Series] = []
        dates: List[pd.Timestamp] = []

        for end_idx in range(window - 1, len(returns)):
            window_returns = returns.iloc[end_idx - window + 1 : end_idx + 1]
            graph = RicciCurvatureEngine.build_correlation_graph(
                window_returns, threshold=threshold
            )
            node_curv = RicciCurvatureEngine.node_curvature(graph)
            results.append(node_curv.reindex(assets).fillna(0.0))
            dates.append(returns.index[end_idx])

        return pd.DataFrame(results, index=dates)

    @staticmethod
    def detect_curvature_collapse(
        curvature_df: pd.DataFrame,
        z_threshold: float = -2.0,
        window: int = 20,
    ) -> pd.DataFrame:
        """Flag assets whose curvature z-score falls below the threshold."""
        if curvature_df.empty:
            return pd.DataFrame(index=curvature_df.index, columns=curvature_df.columns, dtype=bool)

        rolling_mean = curvature_df.rolling(window=window, min_periods=2).mean()
        rolling_std = curvature_df.rolling(window=window, min_periods=2).std()
        z_score = (curvature_df - rolling_mean) / rolling_std.replace({0: np.nan})
        return z_score.lt(z_threshold).fillna(False)

    @staticmethod
    def _correlation_concentration(
        returns: pd.DataFrame,
        dates: Iterable[pd.Timestamp],
        window: int = 20,
    ) -> pd.Series:
        concentrations: Dict[pd.Timestamp, float] = {}
        for date in dates:
            window_returns = returns.loc[:date].tail(window)
            if window_returns.shape[0] < 2:
                concentrations[date] = 0.0
                continue
            corr = window_returns.corr().abs()
            n = corr.shape[0]
            if n <= 1:
                concentrations[date] = 0.0
                continue
            total = corr.values.sum() - n
            concentrations[date] = float(total / (n * (n - 1)))
        return pd.Series(concentrations)

    @staticmethod
    def liquidity_blackhole_score(
        returns: pd.DataFrame,
        curvature_df: pd.DataFrame,
        window: int = 20,
    ) -> pd.DataFrame:
        """Compute a liquidity stress score from curvature and market dynamics."""
        if returns.empty or curvature_df.empty:
            raise ValueError("Returns and curvature DataFrames must not be empty.")

        assets = list(curvature_df.columns)
        volatility = returns[assets].rolling(window=window).std().reindex(curvature_df.index).fillna(0.0)
        abs_return = returns[assets].abs().reindex(curvature_df.index).fillna(0.0)
        corr_concentration = RicciCurvatureEngine._correlation_concentration(
            returns[assets], curvature_df.index, window=window
        )
        corr_concentration_normalized = corr_concentration / max(corr_concentration.max(), 1e-12)

        negative_curvature = (-curvature_df).clip(lower=0.0)
        curvature_component = negative_curvature.divide(
            negative_curvature.replace({0.0: np.nan}).max().replace({np.nan: 1.0})
        ).fillna(0.0)

        vol_component = volatility.divide(volatility.mean().replace({0.0: np.nan})).fillna(0.0)
        shock_component = abs_return.divide(abs_return.mean().replace({0.0: np.nan})).fillna(0.0)
        correlation_component = pd.DataFrame(
            {asset: corr_concentration_normalized for asset in assets},
            index=curvature_df.index,
        )

        score = (
            curvature_component + vol_component + shock_component + correlation_component
        ) / 4.0
        return score.fillna(0.0)

    @staticmethod
    def detect_liquidity_blackholes(
        returns: pd.DataFrame,
        curvature_df: pd.DataFrame,
        z_threshold: float = -2.0,
        window: int = 20,
    ) -> pd.DataFrame:
        """Identify assets exhibiting liquidity black hole signals."""
        collapse = RicciCurvatureEngine.detect_curvature_collapse(
            curvature_df, z_threshold=z_threshold, window=window
        )

        volatility = returns[curvature_df.columns].rolling(window=window).std().reindex(curvature_df.index)
        volatility_z = (volatility - volatility.mean()) / volatility.std().replace({0: np.nan})

        shock = returns[curvature_df.columns].abs().reindex(curvature_df.index)
        shock_z = (shock - shock.mean()) / shock.std().replace({0: np.nan})

        elevated_vol = volatility_z.gt(1.0).fillna(False)
        elevated_shock = shock_z.gt(1.0).fillna(False)

        return collapse & elevated_vol & elevated_shock
