# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Portfolio evaluation metrics.

These are applied post-solve to compare solver quality in Paper ① experiments.
All metrics operate on the original (non-normalised) returns/covariance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PortfolioStats:
    """Evaluated portfolio performance statistics."""

    selected_assets: list[int]
    n_selected: int
    expected_return: float      # Annualised
    volatility: float           # Annualised standard deviation
    sharpe_ratio: float         # (return − rf) / volatility
    portfolio_variance: float   # σ²
    weights: np.ndarray         # Equal weights by default
    feasible: bool              # True if n_selected == target K


class PortfolioMetrics:
    """Compute portfolio statistics from a solver result."""

    def __init__(self, risk_free_rate: float = 0.001) -> None:
        self.risk_free_rate = risk_free_rate

    def evaluate(
        self,
        selected_assets: list[int],
        returns: np.ndarray,
        covariance: np.ndarray,
        target_k: int,
        weights: np.ndarray | None = None,
    ) -> PortfolioStats:
        """
        Compute portfolio performance statistics.

        Args:
            selected_assets: Indices of selected assets.
            returns: Full-universe expected annual returns, shape (N,).
            covariance: Full-universe annual covariance matrix, shape (N, N).
            target_k: Expected number of selected assets (for feasibility check).
            weights: Portfolio weights. Defaults to equal weighting.

        Returns:
            PortfolioStats with annualised metrics.
        """
        k = len(selected_assets)
        if k == 0:
            return PortfolioStats(
                selected_assets=[],
                n_selected=0,
                expected_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                portfolio_variance=0.0,
                weights=np.array([]),
                feasible=False,
            )

        if weights is None:
            weights = np.full(k, 1.0 / k)
        else:
            weights = np.asarray(weights)
            weights = weights / weights.sum()

        idx = np.array(selected_assets)
        mu_p = float(weights @ returns[idx])
        cov_p = covariance[np.ix_(idx, idx)]
        var_p = float(weights @ cov_p @ weights)
        vol_p = float(np.sqrt(max(var_p, 0.0)))
        rf = self.risk_free_rate
        sharpe = (mu_p - rf) / vol_p if vol_p > 1e-10 else 0.0

        return PortfolioStats(
            selected_assets=list(selected_assets),
            n_selected=k,
            expected_return=mu_p,
            volatility=vol_p,
            sharpe_ratio=sharpe,
            portfolio_variance=var_p,
            weights=weights,
            feasible=(k == target_k),
        )

    def compare(
        self,
        stats_list: list[tuple[str, PortfolioStats]],
    ) -> dict[str, dict]:
        """
        Build a comparison table for multiple solver results.

        Args:
            stats_list: List of (solver_name, PortfolioStats) pairs.

        Returns:
            Dict keyed by solver name with metric sub-dicts.
        """
        return {
            name: {
                "expected_return": s.expected_return,
                "volatility": s.volatility,
                "sharpe_ratio": s.sharpe_ratio,
                "portfolio_variance": s.portfolio_variance,
                "n_selected": s.n_selected,
                "feasible": s.feasible,
            }
            for name, s in stats_list
        }
