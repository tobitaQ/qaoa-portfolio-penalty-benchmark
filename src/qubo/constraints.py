# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Reusable constraint penalty terms for QUBO formulation.

Each constraint is added as a quadratic penalty to the QUBO matrix Q:
    H_total = H_objective + Σ A_k · H_constraint_k(x)

Import and compose these when extending the base PortfolioQUBO formulation
for Paper ② (ESG, liquidity, sector constraints).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CardinalityConstraint:
    """
    Enforce exactly K assets selected: (Σ xᵢ − K)² = 0.

    Adds to Q:
        Q_ii += A * (1 − 2K)
        Q_ij += 2A  for i < j
        offset += A * K²
    """

    num_select: int    # K
    penalty: float     # A

    def apply(self, Q: np.ndarray) -> float:
        """Mutates Q in-place and returns constant offset."""
        n = Q.shape[0]
        K, A = self.num_select, self.penalty
        for i in range(n):
            Q[i, i] += A * (1.0 - 2.0 * K)
            for j in range(i + 1, n):
                Q[i, j] += 2.0 * A
        return A * K ** 2


@dataclass
class SectorConstraint:
    """
    Soft sector-concentration penalty (not an exact "at most M per sector").

    sector_map[i] = sector_id for asset i (integer label). For every sector whose
    *universe* holds more than ``max_per_sector`` assets, every pair of assets in
    that sector is coupled with

        Q_ij += 2 · A · (|sector| − max_per_sector),   i < j in the sector,

    so the penalty grows quadratically with the number of selected assets from the
    sector and is charged from the second selected asset on. It does not vanish
    for selections that stay within ``max_per_sector``, and a sector whose universe
    already fits within the bound is left unpenalised — so it discourages
    concentration rather than enforcing a hard cap. An exact "at most M" would
    need slack variables; this term was written for the constraint-extension
    follow-up and is not used by any experiment of the penalty-benchmark paper.
    """

    sector_map: list[int]       # sector label per asset
    max_per_sector: int         # sectors larger than this are penalised
    penalty: float              # A

    def apply(self, Q: np.ndarray) -> float:
        """Mutates Q in-place and returns constant offset (0 for this constraint)."""
        n = Q.shape[0]
        sectors: dict[int, list[int]] = {}
        for i, s in enumerate(self.sector_map):
            sectors.setdefault(s, []).append(i)

        for members in sectors.values():
            if len(members) <= self.max_per_sector:
                continue
            # Couple every pair within an oversized sector (soft penalty; see
            # the class docstring for what this does and does not enforce)
            excess = len(members) - self.max_per_sector
            pair_penalty = self.penalty * excess
            for idx_a, i in enumerate(members):
                for j in members[idx_a + 1 :]:
                    hi, lo = (i, j) if i < j else (j, i)
                    Q[lo, hi] += 2.0 * pair_penalty

        return 0.0


@dataclass
class BudgetConstraint:
    """
    Enforce budget: Σ cᵢ xᵢ = B  (weighted sum equals budget B).

    Useful when assets have different costs / lot sizes.
    Penalty: A · (Σ cᵢ xᵢ − B)²
    """

    costs: np.ndarray   # cost cᵢ per asset
    budget: float       # target B
    penalty: float      # A

    def apply(self, Q: np.ndarray) -> float:
        """Mutates Q in-place and returns constant offset A·B²."""
        n = Q.shape[0]
        c, B, A = self.costs, self.budget, self.penalty
        for i in range(n):
            Q[i, i] += A * c[i] * (c[i] - 2.0 * B)
            for j in range(i + 1, n):
                Q[i, j] += 2.0 * A * c[i] * c[j]
        return A * B ** 2
