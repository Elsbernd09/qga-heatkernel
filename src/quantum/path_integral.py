from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


class PathIntegralSimulator:
    """Monte Carlo path ensemble simulator inspired by path integrals.

    This class generates many possible future asset price paths using
    geometric Brownian motion and evaluates each path with an economic
    action functional. Lower-action paths receive higher probability
    weights through a Boltzmann-like softmax over negative action.

    The design is intentionally probabilistic and metaphorical: this is a
    path-ensemble model inspired by the structure of imaginary-time path
    integrals, not a literal quantum-mechanical price predictor.
    """

    @staticmethod
    def simulate_paths(
        current_price: float,
        drift: float,
        volatility: float,
        horizon: int = 20,
        n_paths: int = 5000,
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """Simulate price paths using geometric Brownian motion.

        Args:
            current_price: The price at time zero.
            drift: Expected per-step drift (same time scale as horizon).
            volatility: Expected per-step volatility.
            horizon: Number of daily time steps to simulate.
            n_paths: Number of Monte Carlo paths.
            random_state: Seed or numpy Generator for reproducibility.

        Returns:
            DataFrame of simulated prices with rows as time steps and
            columns as individual paths.
        """
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        if n_paths < 1:
            raise ValueError("n_paths must be at least 1")
        if current_price <= 0:
            raise ValueError("current_price must be positive")

        rng = np.random.default_rng(random_state)
        time_steps = horizon
        dt = 1.0
        drift_term = (drift - 0.5 * volatility ** 2) * dt
        diffusion_scale = volatility * np.sqrt(dt)

        shocks = rng.standard_normal(size=(time_steps, n_paths))
        log_increments = drift_term + diffusion_scale * shocks
        log_paths = np.cumsum(log_increments, axis=0)
        simulated_prices = current_price * np.exp(log_paths)

        index = list(range(time_steps + 1))
        columns = [f"path_{i}" for i in range(n_paths)]
        result = pd.DataFrame(
            np.vstack([np.full((1, n_paths), current_price), simulated_prices]),
            index=index,
            columns=columns,
            dtype=float,
        )
        result.index.name = "time_step"
        return result

    @staticmethod
    def economic_action(
        paths: pd.DataFrame,
        curvature_stress: float = 0.0,
        liquidity_stress: float = 0.0,
        regime: str = "neutral",
    ) -> pd.Series:
        """Score each path with an economic action functional.

        The action penalizes realized volatility, drawdown, negative terminal
        return, and external stress measures. Regime preferences alter the
        score shape so paths that match the investment regime are favored.

        Args:
            paths: Simulated price paths from simulate_paths.
            curvature_stress: Additional curvature-related penalty applied to
                every path.
            liquidity_stress: Additional liquidity-related penalty.
            regime: Market regime with different path preferences.
                Supported values: 'neutral', 'momentum', 'mean_reversion',
                'risk_off'.

        Returns:
            Series of action scores indexed by path column name.
        """
        if paths.empty:
            raise ValueError("paths must not be empty")
        if not isinstance(paths, pd.DataFrame):
            raise TypeError("paths must be a pandas DataFrame")

        start_prices = paths.iloc[0]
        terminal_prices = paths.iloc[-1]
        log_returns = np.log(paths / paths.shift(1)).fillna(0.0)
        realized_volatility = log_returns.std(axis=0)

        running_max = paths.cummax()
        drawdown = 1.0 - paths / running_max
        max_drawdown = drawdown.max(axis=0)

        terminal_return = terminal_prices / start_prices - 1.0
        negative_return_penalty = np.clip(-terminal_return, 0.0, None)

        action = (
            0.6 * realized_volatility
            + 3.0 * max_drawdown
            + 3.5 * negative_return_penalty
            + float(curvature_stress)
            + float(liquidity_stress)
        )

        regime = regime.lower().strip()
        if regime == "momentum":
            action += 2.0 * np.clip(-terminal_return, 0.0, None)
            action -= 2.0 * np.clip(terminal_return, 0.0, None)
        elif regime == "mean_reversion":
            deviation_range = (paths.max(axis=0) - paths.min(axis=0)) / start_prices
            closeness = np.exp(-np.abs(terminal_return) * 8.0)
            mean_reversion_bonus = 4.0 * closeness * deviation_range
            action -= mean_reversion_bonus
        elif regime == "risk_off":
            action += 8.0 * max_drawdown
            action += 4.0 * np.clip(-terminal_return, 0.0, None)
        elif regime != "neutral":
            raise ValueError(
                f"Unknown regime '{regime}'. Supported regimes: neutral, momentum, mean_reversion, risk_off."
            )

        return pd.Series(action.astype(float), index=paths.columns)

    @staticmethod
    def path_probabilities(
        action_scores: pd.Series,
        temperature: float = 1.0,
    ) -> pd.Series:
        """Convert action scores into probabilities using a softmax on -action.

        Lower action corresponds to higher probability. The softmax is applied to
        the negative action values to emphasize low-action (more plausible)
        path scenarios.
        """
        if action_scores.empty:
            raise ValueError("action_scores must not be empty")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        values = action_scores.astype(float).to_numpy(dtype=float)
        shifted = values - np.min(values)
        exp_values = np.exp(-shifted / temperature)
        probabilities = exp_values / np.sum(exp_values)
        return pd.Series(probabilities, index=action_scores.index)

    @staticmethod
    def _weighted_quantile(
        values: np.ndarray,
        weights: np.ndarray,
        quantile: float,
    ) -> float:
        order = np.argsort(values)
        values_sorted = values[order]
        weights_sorted = weights[order]
        cumulative = np.cumsum(weights_sorted)
        total = cumulative[-1]
        if total <= 0:
            return float(values_sorted[0])
        target = quantile * total
        return float(np.interp(target, cumulative, values_sorted))

    @classmethod
    def terminal_distribution(
        cls,
        paths: pd.DataFrame,
        probabilities: pd.Series,
    ) -> Dict[str, float]:
        """Summarize the weighted terminal price distribution.

        Returns expected values and tail probabilities under the path ensemble.
        """
        if paths.empty:
            raise ValueError("paths must not be empty")
        if probabilities.empty:
            raise ValueError("probabilities must not be empty")

        terminal_prices = paths.iloc[-1]
        start_prices = paths.iloc[0]
        terminal_returns = terminal_prices / start_prices - 1.0

        weights = probabilities.reindex(terminal_prices.index).astype(float).to_numpy(dtype=float)
        weights = weights / np.sum(weights)

        expected_price = float(np.dot(weights, terminal_prices.to_numpy(dtype=float)))
        expected_return = float(np.dot(weights, terminal_returns.to_numpy(dtype=float)))
        upside_probability = float(np.sum(weights[terminal_prices > start_prices]))
        downside_probability = float(np.sum(weights[terminal_prices < start_prices]))
        weighted_5th = cls._weighted_quantile(terminal_returns.to_numpy(dtype=float), weights, 0.05)
        weighted_95th = cls._weighted_quantile(terminal_returns.to_numpy(dtype=float), weights, 0.95)

        return {
            "expected_terminal_price": expected_price,
            "expected_return": expected_return,
            "upside_probability": upside_probability,
            "downside_probability": downside_probability,
            "weighted_5th_percentile": weighted_5th,
            "weighted_95th_percentile": weighted_95th,
        }

    @classmethod
    def summarize_distribution(
        cls,
        paths: pd.DataFrame,
        probabilities: pd.Series,
        current_price: float,
    ) -> Dict[str, float]:
        """Produce a compact summary of the terminal return distribution."""
        dist = cls.terminal_distribution(paths, probabilities)
        terminal_returns = paths.iloc[-1] / paths.iloc[0] - 1.0
        weights = probabilities.reindex(terminal_returns.index).astype(float).to_numpy(dtype=float)
        weights = weights / np.sum(weights)

        sorted_idx = np.argsort(terminal_returns.to_numpy(dtype=float))
        sorted_returns = terminal_returns.to_numpy(dtype=float)[sorted_idx]
        sorted_weights = weights[sorted_idx]
        cumulative = np.cumsum(sorted_weights)

        var_5 = float(cls._weighted_quantile(sorted_returns, sorted_weights, 0.05))
        var_1 = float(cls._weighted_quantile(sorted_returns, sorted_weights, 0.01))
        es_5_mask = cumulative <= 0.05
        if np.any(es_5_mask):
            es_5 = float(np.sum(sorted_returns[es_5_mask] * sorted_weights[es_5_mask]) / np.sum(sorted_weights[es_5_mask]))
        else:
            es_5 = var_5

        concentration_fraction = float(np.sum(np.sort(weights)[-max(1, int(len(weights) * 0.1)):]))

        return {
            "expected_terminal_price": dist["expected_terminal_price"],
            "expected_return": dist["expected_return"],
            "upside_probability": dist["upside_probability"],
            "downside_probability": dist["downside_probability"],
            "confidence_concentration": concentration_fraction,
            "value_at_risk_5": var_5,
            "value_at_risk_1": var_1,
            "expected_shortfall_5": es_5,
            "best_case_95": float(dist["weighted_95th_percentile"]),
            "worst_case_5": float(dist["weighted_5th_percentile"]),
        }

    @staticmethod
    def probability_wave(
        paths: pd.DataFrame,
        probabilities: pd.Series,
        bins: int = 50,
    ) -> pd.DataFrame:
        """Build a weighted density estimate over terminal prices."""
        if paths.empty:
            raise ValueError("paths must not be empty")
        if probabilities.empty:
            raise ValueError("probabilities must not be empty")
        if bins < 1:
            raise ValueError("bins must be at least 1")

        terminal_prices = paths.iloc[-1].to_numpy(dtype=float)
        weights = probabilities.reindex(paths.columns).astype(float).to_numpy(dtype=float)
        weights = weights / np.sum(weights)

        hist, bin_edges = np.histogram(terminal_prices, bins=bins, weights=weights, density=False)
        bin_widths = np.diff(bin_edges)
        density = hist / np.sum(hist) / bin_widths
        bin_centers = bin_edges[:-1] + bin_widths / 2.0

        return pd.DataFrame(
            {
                "terminal_price_bin": bin_centers,
                "probability_mass": hist.astype(float),
                "probability_density": density,
            }
        )

    def run_simulation(
        self,
        current_price: float,
        drift: float,
        volatility: float,
        curvature_stress: float = 0.0,
        liquidity_stress: float = 0.0,
        regime: str = "neutral",
        horizon: int = 20,
        n_paths: int = 5000,
        random_state: Optional[int] = None,
    ) -> Dict[str, object]:
        """Run the full path simulation pipeline.

        This method ties together path generation, economic action scoring,
        probability weighting, distribution summarization, and the final
        probability wave. It is a Monte Carlo finance workflow inspired by the
        structure of path integrals, but it remains a probabilistic ensemble
        model with no guarantee of prediction.
        """
        paths = self.simulate_paths(
            current_price=current_price,
            drift=drift,
            volatility=volatility,
            horizon=horizon,
            n_paths=n_paths,
            random_state=random_state,
        )
        action_scores = self.economic_action(
            paths,
            curvature_stress=curvature_stress,
            liquidity_stress=liquidity_stress,
            regime=regime,
        )
        probabilities = self.path_probabilities(action_scores)
        summary = self.summarize_distribution(paths, probabilities, current_price)
        wave = self.probability_wave(paths, probabilities)

        return {
            "paths": paths,
            "action_scores": action_scores,
            "probabilities": probabilities,
            "summary": summary,
            "probability_wave": wave,
        }
