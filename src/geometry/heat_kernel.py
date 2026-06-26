from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Union

import networkx as nx
import numpy as np
import pandas as pd
from scipy.linalg import expm


class HeatKernelDiffusion:
    """Heat diffusion engine for financial asset graphs.

    The heat kernel models how volatility, stress, or liquidity pressure diffuses
    through a market network. A graph Laplacian summarizes node connectivity,
    and the heat kernel is the matrix exponential of the negative Laplacian.
    """

    @staticmethod
    def build_correlation_graph(
        returns: pd.DataFrame,
        threshold: float = 0.3,
    ) -> nx.Graph:
        """Build a weighted asset graph from return correlations.

        Args:
            returns: Asset return series with assets as columns.
            threshold: Minimum absolute correlation to place an edge.

        Returns:
            A networkx Graph containing asset nodes and weighted edges.
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
                strength = abs(corr_value)
                if strength <= threshold:
                    continue
                distance = float(np.sqrt(2.0 * (1.0 - corr_value)))
                graph.add_edge(
                    asset_a,
                    asset_b,
                    distance=distance,
                    strength=strength,
                    correlation=float(corr_value),
                )

        return graph

    @staticmethod
    def graph_laplacian(graph: nx.Graph) -> pd.DataFrame:
        """Return the weighted graph Laplacian matrix.

        The Laplacian L = D - W uses edge strength as weights. In finance, the
        graph Laplacian captures the connectivity of asset stress transmission.
        """
        nodes = list(graph.nodes)
        size = len(nodes)
        laplacian = np.zeros((size, size), dtype=float)

        node_index = {node: idx for idx, node in enumerate(nodes)}

        for u, v, data in graph.edges(data=True):
            weight = float(data.get("strength", 0.0))
            i, j = node_index[u], node_index[v]
            laplacian[i, j] = -weight
            laplacian[j, i] = -weight
            laplacian[i, i] += weight
            laplacian[j, j] += weight

        return pd.DataFrame(laplacian, index=nodes, columns=nodes)

    @staticmethod
    def heat_kernel_matrix(graph: nx.Graph, diffusion_time: float = 1.0) -> pd.DataFrame:
        """Compute the heat kernel matrix exp(-tL) for the graph.

        The heat kernel describes how a localized shock spreads through the network
        over diffusion time t.
        """
        if diffusion_time < 0:
            raise ValueError("diffusion_time must be non-negative.")

        laplacian = HeatKernelDiffusion.graph_laplacian(graph)
        heat_kernel = expm(-diffusion_time * laplacian.values)
        return pd.DataFrame(heat_kernel, index=laplacian.index, columns=laplacian.columns)

    @staticmethod
    def diffuse_heat(
        graph: nx.Graph,
        initial_heat: Union[Dict[str, float], pd.Series, np.ndarray],
        diffusion_time: float = 1.0,
    ) -> pd.Series:
        """Diffuse initial heat through the graph and return asset heat levels."""
        kernel = HeatKernelDiffusion.heat_kernel_matrix(graph, diffusion_time=diffusion_time)
        assets = list(kernel.index)

        if isinstance(initial_heat, np.ndarray):
            if initial_heat.shape[0] != len(assets):
                raise ValueError("Initial heat array length must match number of graph nodes.")
            heat_vector = pd.Series(initial_heat, index=assets, dtype=float)
        elif isinstance(initial_heat, pd.Series):
            heat_vector = initial_heat.reindex(assets).fillna(0.0).astype(float)
        elif isinstance(initial_heat, dict):
            heat_vector = pd.Series(initial_heat).reindex(assets).fillna(0.0).astype(float)
        else:
            raise TypeError("initial_heat must be a dict, pandas Series, or numpy ndarray.")

        diffused = kernel.dot(heat_vector)
        return pd.Series(diffused, index=assets, dtype=float)

    @staticmethod
    def rolling_heat_diffusion(
        returns: pd.DataFrame,
        stress_series: Optional[Union[pd.Series, Dict[str, float]]] = None,
        window: int = 60,
        diffusion_time: float = 1.0,
    ) -> pd.DataFrame:
        """Compute rolling diffusion of stress across a dynamic correlation graph.

        Args:
            returns: Asset return series with assets as columns.
            stress_series: Optional per-date stress values or a Series containing assets.
            window: Rolling window size for correlation estimation.
            diffusion_time: Diffusion time for the heat kernel.

        Returns:
            DataFrame of diffused heat values indexed by rolling window end date.
        """
        if returns.empty:
            raise ValueError("Returns DataFrame must not be empty.")
        if window < 2:
            raise ValueError("window must be at least 2.")

        assets = list(returns.columns)
        results: List[pd.Series] = []
        dates: List[pd.Timestamp] = []

        for end_idx in range(window - 1, len(returns)):
            window_returns = returns.iloc[end_idx - window + 1 : end_idx + 1]
            graph = HeatKernelDiffusion.build_correlation_graph(window_returns)

            current_date = returns.index[end_idx]
            if stress_series is None:
                initial_heat = window_returns.iloc[-1].abs()
            elif isinstance(stress_series, dict):
                initial_heat = stress_series
            else:
                if current_date in stress_series.index:
                    initial_heat = stress_series.loc[current_date]
                else:
                    initial_heat = window_returns.iloc[-1].abs()

            diffused = HeatKernelDiffusion.diffuse_heat(
                graph,
                initial_heat=initial_heat,
                diffusion_time=diffusion_time,
            )
            results.append(diffused)
            dates.append(current_date)

        return pd.DataFrame(results, index=dates)

    @staticmethod
    def systemic_heat_score(
        diffused_heat: Union[pd.Series, Dict[str, float], np.ndarray],
    ) -> Dict[str, float]:
        """Compute systemic heat summary metrics for a diffused heat vector."""
        if isinstance(diffused_heat, np.ndarray):
            values = np.asarray(diffused_heat, dtype=float)
        elif isinstance(diffused_heat, pd.Series):
            values = diffused_heat.to_numpy(dtype=float)
        elif isinstance(diffused_heat, dict):
            values = np.asarray(list(diffused_heat.values()), dtype=float)
        else:
            raise TypeError("diffused_heat must be a pandas Series, dict, or numpy ndarray.")

        total_heat = float(np.sum(values))
        max_heat = float(np.max(values)) if values.size > 0 else 0.0
        abs_values = np.abs(values)
        if abs_values.sum() > 0:
            probabilities = abs_values / abs_values.sum()
            heat_entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-12)))
        else:
            heat_entropy = 0.0

        concentration_ratio = float(max_heat / total_heat) if total_heat > 0 else 0.0
        return {
            "total_heat": total_heat,
            "max_heat": max_heat,
            "heat_entropy": heat_entropy,
            "concentration_ratio": concentration_ratio,
        }

    @staticmethod
    def identify_heat_concentration(
        diffused_heat: Union[pd.Series, Dict[str, float], np.ndarray],
        quantile: float = 0.9,
    ) -> List[str]:
        """Return asset names with heat above the specified quantile threshold."""
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must be between 0 and 1.")

        if isinstance(diffused_heat, np.ndarray):
            raise TypeError("identify_heat_concentration requires asset labels, so pass a pandas Series or dict.")
        if isinstance(diffused_heat, dict):
            series = pd.Series(diffused_heat)
        else:
            series = diffused_heat

        threshold = series.quantile(quantile)
        return list(series[series > threshold].index)
