from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from persim import wasserstein
from ripser import ripser
from sklearn.preprocessing import StandardScaler


class PersistentHomologyRegimeDetector:
    """Persistent homology regime detector for financial market states.

    This class converts rolling market windows into point clouds of asset
    features and uses persistent homology to summarize their topological shape.
    H0 features capture connected components, while H1 features represent loops
    or cyclic structure. Persistent topological structures may indicate regime
    stability, sector rotation, or other market patterns.

    The method is a data-driven topology proxy and should not be interpreted as
    a perfect regime classifier or guaranteed market predictor.
    """

    @staticmethod
    def build_asset_feature_matrix(
        prices: pd.DataFrame,
        returns: Optional[pd.DataFrame] = None,
        window: int = 60,
    ) -> pd.DataFrame:
        """Build standardized asset feature matrix from recent price history.

        Args:
            prices: Price DataFrame with assets as columns and dates as index.
            returns: Optional return DataFrame aligned to prices. If absent,
                returns are computed as simple percentage changes.
            window: Rolling window length in observations.

        Returns:
            DataFrame indexed by asset containing standardized features.
        """
        if prices.empty:
            raise ValueError("Prices DataFrame must not be empty.")
        if window < 2:
            raise ValueError("window must be at least 2.")

        prices = prices.dropna(how="any").copy()
        if len(prices) < window:
            raise ValueError("Not enough price history to build feature matrix.")

        window_prices = prices.iloc[-window:]
        if returns is None:
            returns = window_prices.pct_change().dropna()
        else:
            returns = returns.reindex(window_prices.index).copy()
            returns = returns.dropna(how="any")
            if len(returns) < 2:
                raise ValueError("Not enough return history to build feature matrix.")

        features: Dict[str, Dict[str, float]] = {}
        spy_returns = returns["SPY"] if "SPY" in returns.columns else None

        for asset in window_prices.columns:
            prices_series = window_prices[asset]
            recent_returns = returns[asset]

            cumulative_return = float(prices_series.iloc[-1] / prices_series.iloc[0] - 1.0)
            volatility = float(recent_returns.std() * np.sqrt(252.0))
            momentum = float(prices_series.iloc[-1] / prices_series.iloc[max(0, len(prices_series) - 6)] - 1.0)
            high_water_mark = prices_series.cummax()
            max_drawdown = float((prices_series / high_water_mark - 1.0).min())
            abs_return_shock = float(abs(recent_returns.iloc[-1]))

            if spy_returns is not None and asset != "SPY":
                beta = float(recent_returns.cov(spy_returns) / max(spy_returns.var(), 1e-12))
            elif asset == "SPY":
                beta = 1.0
            else:
                beta = 0.0

            features[asset] = {
                "cumulative_return": cumulative_return,
                "annualized_volatility": volatility,
                "momentum": momentum,
                "max_drawdown": max_drawdown,
                "beta_to_SPY": beta,
                "abs_return_shock": abs_return_shock,
            }

        feature_df = pd.DataFrame.from_dict(features, orient="index")
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(feature_df.values)
        scaled_df = pd.DataFrame(
            scaled_values,
            index=feature_df.index,
            columns=feature_df.columns,
        )
        return scaled_df

    @staticmethod
    def compute_diagram(feature_matrix: pd.DataFrame, maxdim: int = 1) -> Dict[str, Any]:
        """Compute a persistence diagram from an asset feature matrix.

        Args:
            feature_matrix: Asset feature matrix with assets as rows.
            maxdim: Max homology dimension to compute.

        Returns:
            ripser output containing persistence diagrams.
        """
        if feature_matrix.empty:
            raise ValueError("Feature matrix must not be empty.")

        diagram = ripser(feature_matrix.values, maxdim=maxdim)
        return diagram

    @staticmethod
    def rolling_diagrams(
        prices: pd.DataFrame,
        window: int = 60,
        step: int = 5,
        maxdim: int = 1,
    ) -> pd.DataFrame:
        """Compute rolling persistence diagrams over market history."""
        if prices.empty:
            raise ValueError("Prices DataFrame must not be empty.")
        if window < 2:
            raise ValueError("window must be at least 2.")
        if step < 1:
            raise ValueError("step must be at least 1.")

        records: List[Dict[str, Any]] = []
        for start in range(0, len(prices) - window + 1, step):
            window_prices = prices.iloc[start : start + window]
            feature_matrix = PersistentHomologyRegimeDetector.build_asset_feature_matrix(
                window_prices, window=window
            )
            diagram = PersistentHomologyRegimeDetector.compute_diagram(
                feature_matrix, maxdim=maxdim
            )
            summary = PersistentHomologyRegimeDetector.summarize_diagram(diagram)
            records.append(
                {
                    "date": window_prices.index[-1],
                    "feature_matrix": feature_matrix,
                    "diagram": diagram,
                    "summary": summary,
                }
            )

        if not records:
            return pd.DataFrame(columns=["date", "feature_matrix", "diagram", "summary"])

        df = pd.DataFrame(records)
        df = df.set_index("date")
        return df

    @staticmethod
    def _extract_dimension_diagram(
        diagram: Union[Dict[str, Any], Sequence[np.ndarray]],
        homology_dimension: int = 1,
    ) -> np.ndarray:
        if isinstance(diagram, dict):
            diagrams = diagram.get("dgms")
            if diagrams is None or homology_dimension >= len(diagrams):
                return np.empty((0, 2))
            return np.asarray(diagrams[homology_dimension])

        if isinstance(diagram, Sequence):
            if homology_dimension >= len(diagram):
                return np.empty((0, 2))
            return np.asarray(diagram[homology_dimension])

        raise TypeError("Diagram must be a ripser output dict or a sequence of persistence diagrams.")

    @staticmethod
    def diagram_distance(
        diagram_a: Union[Dict[str, Any], Sequence[np.ndarray]],
        diagram_b: Union[Dict[str, Any], Sequence[np.ndarray]],
        homology_dimension: int = 1,
    ) -> float:
        """Compute distance between two persistence diagrams using Wasserstein."""
        dgm_a = PersistentHomologyRegimeDetector._extract_dimension_diagram(
            diagram_a, homology_dimension
        )
        dgm_b = PersistentHomologyRegimeDetector._extract_dimension_diagram(
            diagram_b, homology_dimension
        )
        if dgm_a.size == 0 and dgm_b.size == 0:
            return 0.0
        return float(wasserstein(dgm_a, dgm_b))

    @staticmethod
    def regime_distance_matrix(
        diagrams: Union[pd.DataFrame, List[Union[Dict[str, Any], Sequence[np.ndarray]]]],
        homology_dimension: int = 1,
    ) -> pd.DataFrame:
        """Compute pairwise diagram distances between a collection of regimes."""
        if isinstance(diagrams, pd.DataFrame) and "diagram" in diagrams.columns:
            items = list(diagrams["diagram"])
            labels = list(diagrams.index)
        else:
            items = list(diagrams)
            labels = list(range(len(items)))

        size = len(items)
        matrix = np.zeros((size, size), dtype=float)
        for i in range(size):
            for j in range(i + 1, size):
                dist = PersistentHomologyRegimeDetector.diagram_distance(
                    items[i], items[j], homology_dimension=homology_dimension
                )
                matrix[i, j] = dist
                matrix[j, i] = dist

        return pd.DataFrame(matrix, index=labels, columns=labels)

    @staticmethod
    def find_nearest_historical_regimes(
        current_diagram: Union[Dict[str, Any], Sequence[np.ndarray]],
        historical_diagrams: Union[pd.DataFrame, List[Dict[str, Any]]],
        top_k: int = 5,
        homology_dimension: int = 1,
    ) -> pd.DataFrame:
        """Find the closest historical regimes to the current topology."""
        histories: List[Tuple[pd.Timestamp, Any]] = []
        if isinstance(historical_diagrams, pd.DataFrame) and "diagram" in historical_diagrams.columns:
            histories = [(date, row["diagram"]) for date, row in historical_diagrams.iterrows()]
        elif isinstance(historical_diagrams, list):
            for item in historical_diagrams:
                if isinstance(item, dict) and "date" in item and "diagram" in item:
                    histories.append((item["date"], item["diagram"]))
                else:
                    raise ValueError(
                        "Historical diagrams list must contain dicts with 'date' and 'diagram' keys."
                    )
        else:
            raise TypeError("historical_diagrams must be a DataFrame or a list of dicts.")

        distances = []
        for date, diagram in histories:
            distance = PersistentHomologyRegimeDetector.diagram_distance(
                current_diagram, diagram, homology_dimension=homology_dimension
            )
            distances.append({"date": date, "distance": distance})

        result = pd.DataFrame(distances).sort_values("distance").head(top_k)
        return result.reset_index(drop=True)

    @staticmethod
    def summarize_diagram(diagram: Union[Dict[str, Any], Sequence[np.ndarray]]) -> Dict[str, float]:
        """Summarize topological features from a persistence diagram."""
        h0 = PersistentHomologyRegimeDetector._extract_dimension_diagram(diagram, homology_dimension=0)
        h1 = PersistentHomologyRegimeDetector._extract_dimension_diagram(diagram, homology_dimension=1)

        num_h0 = int(len(h0))
        num_h1 = int(len(h1))
        persistence = h1[:, 1] - h1[:, 0] if num_h1 > 0 else np.array([], dtype=float)
        finite_persistence = persistence[np.isfinite(persistence)]
        max_persistence = float(np.max(finite_persistence)) if finite_persistence.size > 0 else 0.0
        avg_persistence = float(np.mean(finite_persistence)) if finite_persistence.size > 0 else 0.0
        total_persistence = float(np.sum(finite_persistence))
        complexity_score = float(total_persistence * (1.0 + num_h1 / 10.0))

        return {
            "num_h0": num_h0,
            "num_h1": num_h1,
            "max_h1_persistence": max_persistence,
            "avg_h1_persistence": avg_persistence,
            "total_persistence": total_persistence,
            "topological_complexity_score": complexity_score,
        }

    @staticmethod
    def classify_regime_from_topology(diagram_summary: Dict[str, float]) -> str:
        """Assign a regime label based on topological complexity and loop structure."""
        num_h1 = diagram_summary.get("num_h1", 0)
        total_persistence = diagram_summary.get("total_persistence", 0.0)
        avg_persistence = diagram_summary.get("avg_h1_persistence", 0.0)
        complexity_score = diagram_summary.get("topological_complexity_score", 0.0)

        if num_h1 == 0 or total_persistence < 0.05:
            return "Low Complexity / Calm"
        if num_h1 <= 1 and avg_persistence < 0.05:
            return "Collapsed / Correlation Compression"
        if complexity_score > 1.0:
            return "High Complexity / Stress"
        if num_h1 >= 3 and avg_persistence >= 0.07:
            return "Loop-Dominant / Sector Rotation"
        if num_h1 >= 1 and avg_persistence >= 0.03:
            return "Fragmented / Rotational"
        return "Low Complexity / Calm"
