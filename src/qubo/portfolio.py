# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Layer 1: Device-independent QUBO formulation for portfolio optimization.

This formulation is permanent — it does not depend on quantum hardware.
The Markowitz mean-variance problem is cast as:

    min  λ · xᵀΣx − (1−λ) · μᵀx
    s.t. Σᵢ xᵢ = K

via the QUBO penalty Lagrangian:

    H(x) = λ · xᵀΣ̂x − (1−λ) · μ̂ᵀx + A · (Σᵢ xᵢ − K)²

Normalization (divide by max-abs) keeps coefficients in [-1, 1] so that
the penalty strength A is meaningful across different datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class QUBOProblem:
    """Container for a QUBO problem instance."""

    Q: np.ndarray          # Upper-triangular QUBO matrix, shape (N, N)
    offset: float          # Constant energy term (for total energy accounting)
    n_variables: int       # N
    asset_names: list[str]
    metadata: dict = field(default_factory=dict)

    def evaluate(self, x: np.ndarray) -> float:
        """Compute QUBO energy  xᵀQx  for binary vector x."""
        x = np.asarray(x, dtype=float)
        return float(x @ self.Q @ x) + self.offset

    def evaluate_no_offset(self, x: np.ndarray) -> float:
        """Compute xᵀQx without the constant offset (for solver comparison)."""
        x = np.asarray(x, dtype=float)
        return float(x @ self.Q @ x)


class PortfolioQUBO:
    """
    Layer 1: Persistent QUBO formulation for equal-weight portfolio selection.

    Given N candidate assets, selects K assets that minimize risk
    while maximizing expected return.
    """

    def formulate(
        self,
        returns: np.ndarray,
        covariance: np.ndarray,
        num_select: int,
        risk_aversion: float = 0.5,
        penalty_strength: Optional[float] = None,
        asset_names: Optional[list[str]] = None,
        scales: Optional[tuple[float, float]] = None,
    ) -> QUBOProblem:
        """
        Build the upper-triangular QUBO matrix Q.

        Args:
            returns: Expected annual returns μ, shape (N,)
            covariance: Annual covariance matrix Σ, shape (N, N)
            num_select: Number of assets to select K
            risk_aversion: λ ∈ [0,1].  0 → max return only,  1 → min risk only
            penalty_strength: Constraint penalty A. Auto-set if None.
            asset_names: Optional labels for assets.
            scales: ``(mu_scale, cov_scale)`` to normalize by. Default None
                takes each from this instance's own max-abs value, which is
                what every result in paper ① uses. Passing them lets a caller
                hold the scale fixed across instances, so that the near-optimal
                structure can be measured independently of a per-instance
                rescaling -- see ``run_normalization_sweep.py``.

        Returns:
            QUBOProblem with upper-triangular Q.
        """
        n = len(returns)
        if covariance.shape != (n, n):
            raise ValueError(f"covariance must be ({n},{n}), got {covariance.shape}")
        if not (0 < num_select <= n):
            raise ValueError(f"num_select must be in (0, {n}]")
        if not (0.0 <= risk_aversion <= 1.0):
            raise ValueError("risk_aversion must be in [0, 1]")

        lam = risk_aversion
        K = num_select

        # Normalize to [-1, 1] for numerical stability across datasets
        if scales is None:
            mu_scale = np.max(np.abs(returns)) + 1e-12
            cov_scale = np.max(np.abs(covariance)) + 1e-12
        else:
            mu_scale, cov_scale = scales
        mu_hat = returns / mu_scale
        cov_hat = covariance / cov_scale

        # Auto-compute penalty: must dominate the objective to enforce constraint
        if penalty_strength is None:
            obj_scale = max(
                lam * np.max(np.abs(cov_hat)),
                (1.0 - lam) * np.max(np.abs(mu_hat)),
                1e-6,
            )
            A = 2.0 * obj_scale * n + 1.0
        else:
            A = penalty_strength

        Q = np.zeros((n, n))

        # --- Objective terms ---
        # Risk diagonal:    λ · σ̂ᵢᵢ
        # Return diagonal: −(1−λ) · μ̂ᵢ
        for i in range(n):
            Q[i, i] += lam * cov_hat[i, i] - (1.0 - lam) * mu_hat[i]

        # Risk off-diagonal: 2λ · σ̂ᵢⱼ  (factor-2 from symmetric product)
        for i in range(n):
            for j in range(i + 1, n):
                Q[i, j] += 2.0 * lam * cov_hat[i, j]

        # --- Cardinality constraint: A · (Σxᵢ − K)² ---
        # Expanding: A·[(1−2K)·Σxᵢ + 2·Σᵢ<ⱼ xᵢxⱼ + K²]
        for i in range(n):
            Q[i, i] += A * (1.0 - 2.0 * K)
            for j in range(i + 1, n):
                Q[i, j] += 2.0 * A

        offset = A * K ** 2

        names = asset_names or [f"asset_{i:03d}" for i in range(n)]

        return QUBOProblem(
            Q=Q,
            offset=offset,
            n_variables=n,
            asset_names=names,
            metadata={
                "risk_aversion": lam,
                "num_select": K,
                "penalty_strength": A,
                "mu_scale": mu_scale,
                "cov_scale": cov_scale,
                "n_assets_candidate": n,
            },
        )

    def to_ising(self, problem: QUBOProblem) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Convert QUBO  xᵀQx  to Ising  hᵀz + zᵀJz  via  xᵢ = (1−zᵢ)/2.

        Returns:
            h:      Linear Ising coefficients, shape (N,)
            J:      Upper-triangular quadratic coefficients, shape (N, N)
            offset: Total constant offset (including QUBO offset)
        """
        Q = problem.Q
        n = problem.n_variables

        h = np.zeros(n)
        J = np.zeros((n, n))
        constant = problem.offset

        # Diagonal QUBO term Q_ii * x_i  →  Q_ii * (1−z_i)/2
        for i in range(n):
            h[i] -= Q[i, i] / 2.0
            constant += Q[i, i] / 2.0

        # Off-diagonal QUBO term Q_ij * x_i * x_j  →  Q_ij*(1−zᵢ)(1−zⱼ)/4
        for i in range(n):
            for j in range(i + 1, n):
                q = Q[i, j]
                J[i, j] += q / 4.0
                h[i] -= q / 4.0
                h[j] -= q / 4.0
                constant += q / 4.0

        return h, J, constant
