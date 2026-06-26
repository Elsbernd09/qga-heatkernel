from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd


class GeometricSignalEngine:
    """Composite market signal engine combining geometric market evidence.

    This engine aggregates several mathematical views of market structure:
    ultrametric asset isolation, heat diffusion concentration, curvature
    liquidity stress, persistent topology regime labels, and path-integral
    distribution summaries. The output is an interpretable signal score
    between -100 and +100.

    The signal is a research-oriented summary of evidence, not a guarantee of
    arbitrage or investment performance. It is intended to make multiple
    geometric perspectives easier to compare, while preserving the uncertainty
    inherent in probabilistic models.
    """

    @staticmethod
    def normalize_series(
        series: Union[pd.Series, Sequence[float], Dict[str, float]],
        higher_is_better: bool = True,
    ) -> pd.Series:
        """Normalize a numeric series to [-100, 100].

        Higher values become better when `higher_is_better` is True. When the
        input is constant, the result is zero to avoid false confidence.
        """
        if series is None:
            raise ValueError("series must not be None")

        values = pd.Series(series).astype(float)
        if values.empty:
            raise ValueError("series must not be empty")

        if not higher_is_better:
            values = -values

        minimum = float(values.min())
        maximum = float(values.max())
        if np.isclose(minimum, maximum):
            return pd.Series(0.0, index=values.index)

        normalized = (values - minimum) / (maximum - minimum)
        scaled = normalized * 200.0 - 100.0
        return pd.Series(scaled, index=values.index)

    @staticmethod
    def score_curvature(curvature_series: Union[pd.Series, Sequence[float], float]) -> pd.Series:
        """Translate discrete curvature estimates into an interpretable score.

        Positive curvature is treated as supportive, while strong negative
        curvature is treated as a stress signal.
        """
        values = pd.Series([curvature_series]) if np.isscalar(curvature_series) else pd.Series(curvature_series)
        score = 100.0 * np.tanh(values.astype(float) / 0.25)
        score.index = values.index
        return score

    @staticmethod
    def score_heat(diffused_heat: Union[pd.Series, Sequence[float], float]) -> pd.Series:
        """Score heat concentration: low heat is benign, high heat is risky.

        Heat diffusion measures how concentrated volatility or stress is across
        assets. High heat concentration suggests market stress, which should
        reduce the composite signal.
        """
        values = pd.Series([diffused_heat]) if np.isscalar(diffused_heat) else pd.Series(diffused_heat)
        clipped = np.clip(values.astype(float), 0.0, 1.0)
        score = 100.0 * np.tanh((0.2 - clipped) * 4.0)
        score.index = values.index
        return score

    @staticmethod
    def score_topology(
        regime_label: str,
        diagram_summary: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Convert topology regime labels into a signal score.

        Topology encodes market regime structure from persistent homology.
        Lower complexity and calm regimes are positive, while collapsed or
        compressed regimes are negative.
        """
        if regime_label is None:
            return 0.0

        label = str(regime_label).strip().lower()
        mapping = {
            "low complexity": 70.0,
            "calm": 70.0,
            "fragmented": 10.0,
            "rotational": 5.0,
            "loop-dominant": 15.0,
            "loop_dominant": 15.0,
            "sector rotation": 15.0,
            "sector_rotation": 15.0,
            "high complexity": -40.0,
            "stress": -40.0,
            "collapsed": -80.0,
            "correlation compression": -80.0,
            "correlation_compression": -80.0,
        }
        score = mapping.get(label, 0.0)

        if diagram_summary is not None:
            complexity = diagram_summary.get("complexity")
            if isinstance(complexity, (int, float)):
                score -= float(np.clip((complexity - 0.5) * 80.0, -20.0, 20.0))

        return float(np.clip(score, -100.0, 100.0))

    @staticmethod
    def score_path_integral(path_summary: Dict[str, Any]) -> float:
        """Score the path-integral simulator output as a single component.

        Positive expected return and upside probability increase the score,
        while downside probability and expected shortfall reduce it.
        """
        if not path_summary:
            return 0.0

        upside = float(path_summary.get("upside_probability", 0.5))
        downside = float(path_summary.get("downside_probability", 0.5))
        expected_return = float(path_summary.get("expected_return", 0.0))
        expected_shortfall_5 = float(path_summary.get("expected_shortfall_5", 0.0))

        score = 50.0 * (upside - downside)
        score += 40.0 * np.tanh(expected_return * 8.0)
        score -= 30.0 * np.clip(-expected_shortfall_5, 0.0, 1.0)
        score -= 10.0 * downside
        return float(np.clip(score, -100.0, 100.0))

    @staticmethod
    def score_ultrametric_isolation(
        asset: str,
        distance_matrix: pd.DataFrame,
        stressed_assets: Optional[Iterable[str]] = None,
    ) -> float:
        """Score an asset by its ultrametric proximity to stressed assets.

        Assets isolated from stress are slightly positive, while assets close to
        stressed names receive a negative score.
        """
        if distance_matrix is None or stressed_assets is None:
            return 0.0

        if asset not in distance_matrix.index and asset not in distance_matrix.columns:
            return 0.0

        stressed = [a for a in stressed_assets if a in distance_matrix.index and a != asset]
        if not stressed:
            return 0.0

        distances = distance_matrix.loc[asset, stressed].astype(float)
        if distances.empty:
            return 0.0

        min_distance = float(distances.min())
        score = 40.0 * np.tanh((min_distance - 0.4) * 3.0)
        return float(np.clip(score, -100.0, 100.0))

    @staticmethod
    def combine_scores(
        component_scores: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Combine component scores into a single final signal score.

        Missing components are ignored and the remaining weights are rescaled.
        """
        if not component_scores:
            return 0.0

        default_weights = {
            "curvature": 0.25,
            "heat": 0.20,
            "topology": 0.20,
            "path_integral": 0.25,
            "ultrametric": 0.10,
        }
        weights = weights or default_weights
        present_scores = {
            name: float(value)
            for name, value in component_scores.items()
            if value is not None and name in weights
        }
        if not present_scores:
            return 0.0

        present_weights = {name: float(weights.get(name, 0.0)) for name in present_scores}
        total_weight = sum(present_weights.values())
        if total_weight <= 0:
            normalized_weights = {name: 1.0 for name in present_scores}
            total_weight = float(len(present_scores))
        else:
            normalized_weights = {
                name: w / total_weight for name, w in present_weights.items()
            }

        combined = sum(present_scores[name] * normalized_weights[name] for name in present_scores)
        return float(np.clip(combined, -100.0, 100.0))

    @staticmethod
    def classify_signal(score: float) -> str:
        """Translate the final score into a discrete signal label."""
        if score >= 70.0:
            return "Strong Long"
        if score >= 30.0:
            return "Long"
        if score > -30.0:
            return "Neutral"
        if score > -70.0:
            return "Risk-Off"
        return "Strong Short"

    @staticmethod
    def confidence_from_components(component_scores: Dict[str, float]) -> float:
        """Estimate confidence based on whether component directions agree."""
        values = [float(v) for v in component_scores.values() if v is not None]
        if not values:
            return 0.5

        signed = [np.sign(v) for v in values if abs(v) >= 5.0]
        if not signed:
            return 0.5

        agreement = float(abs(sum(signed)) / len(signed))
        return float(np.clip(agreement, 0.0, 1.0))

    @staticmethod
    def explain_signal(
        asset: str,
        component_scores: Dict[str, float],
        final_score: float,
        signal_label: str,
    ) -> str:
        """Build a human-readable explanation for a composite asset signal."""
        descriptions = []
        for name, value in component_scores.items():
            if value is None:
                continue
            if value > 20:
                descriptor = "positive"
            elif value < -20:
                descriptor = "negative"
            else:
                descriptor = "moderate"
            descriptions.append(f"{name} is {descriptor}")

        explanation = (
            f"{asset} receives a {signal_label} signal with score {final_score:.1f}"
        )
        if descriptions:
            explanation += " because " + ", ".join(descriptions) + "."
        else:
            explanation += "."
        return explanation

    def generate_asset_signal(
        self,
        asset: str,
        curvature: Optional[float] = None,
        heat: Optional[float] = None,
        topology: Optional[str] = None,
        path_summary: Optional[Dict[str, Any]] = None,
        distance_matrix: Optional[pd.DataFrame] = None,
        stressed_assets: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """Generate a composite signal for one asset."""
        component_scores: Dict[str, float] = {}
        if curvature is not None:
            component_scores["curvature"] = float(self.score_curvature({asset: curvature}).iloc[0])
        if heat is not None:
            component_scores["heat"] = float(self.score_heat({asset: heat}).iloc[0])
        if topology is not None:
            component_scores["topology"] = float(self.score_topology(topology))
        if path_summary is not None:
            component_scores["path_integral"] = float(self.score_path_integral(path_summary))
        if distance_matrix is not None and stressed_assets is not None:
            component_scores["ultrametric"] = float(
                self.score_ultrametric_isolation(asset, distance_matrix, stressed_assets)
            )

        final_score = self.combine_scores(component_scores)
        signal_label = self.classify_signal(final_score)
        confidence = self.confidence_from_components(component_scores)
        explanation = self.explain_signal(asset, component_scores, final_score, signal_label)

        return {
            "asset": asset,
            "final_score": final_score,
            "signal": signal_label,
            "confidence": confidence,
            "curvature_score": component_scores.get("curvature", 0.0),
            "heat_score": component_scores.get("heat", 0.0),
            "topology_score": component_scores.get("topology", 0.0),
            "path_integral_score": component_scores.get("path_integral", 0.0),
            "ultrametric_score": component_scores.get("ultrametric", 0.0),
            "explanation": explanation,
        }

    def generate_signal_table(
        self,
        assets: Sequence[str],
        curvature_series: Optional[Union[pd.Series, Dict[str, float]]] = None,
        heat_series: Optional[Union[pd.Series, Dict[str, float]]] = None,
        topology_label: Optional[Union[str, pd.Series, Dict[str, str]]] = None,
        topology_summary: Optional[Union[Dict[str, Dict[str, Any]], Dict[str, Any]]] = None,
        path_summaries: Optional[Dict[str, Dict[str, Any]]] = None,
        distance_matrix: Optional[pd.DataFrame] = None,
        stressed_assets: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Build a table of composite signals for multiple assets."""
        curvature_series = pd.Series(curvature_series) if curvature_series is not None else pd.Series(dtype=float)
        heat_series = pd.Series(heat_series) if heat_series is not None else pd.Series(dtype=float)

        if isinstance(topology_label, (dict, pd.Series)):
            topology_map = pd.Series(topology_label)
        else:
            topology_map = None

        topology_summary_map = None
        if isinstance(topology_summary, dict) and any(isinstance(v, dict) for v in topology_summary.values()):
            topology_summary_map = topology_summary

        rows = []
        for asset in assets:
            topology = (
                topology_map.get(asset)
                if topology_map is not None
                else topology_label
            )
            topology_details = (
                topology_summary_map.get(asset) if topology_summary_map is not None else topology_summary
            )
            path_summary = None
            if path_summaries is not None:
                path_summary = path_summaries.get(asset)

            row = self.generate_asset_signal(
                asset=asset,
                curvature=curvature_series.get(asset) if not curvature_series.empty else None,
                heat=heat_series.get(asset) if not heat_series.empty else None,
                topology=topology,
                path_summary=path_summary,
                distance_matrix=distance_matrix,
                stressed_assets=stressed_assets,
            )
            rows.append(row)

        return pd.DataFrame(rows)
