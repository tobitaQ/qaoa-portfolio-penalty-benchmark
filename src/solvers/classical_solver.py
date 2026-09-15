# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Classical QUBO solver for benchmark comparison.

Provides three strategies:
  - exact:        exact solution by enumerating the feasible set C(N,K)
  - brute_force:  exact solution by enumerating all 2^N bitstrings (N ≤ 20)
  - simulated_annealing: heuristic when neither exact method is affordable

All are used as baselines in Paper ① experiments.

Why ``exact`` exists
--------------------
``brute_force`` walks all 2^N assignments and then *discards* every one whose
cardinality is not K — for N=30, K=6 that is 1.07e9 candidates evaluated to
keep 5.9e5 of them. Enumerating the K-subsets directly visits exactly the
feasible set and is 1808x smaller at that size, which moves the exact-optimum
frontier from N=20 to N=30 (seconds rather than days). The reported optimality
gap for N=24/28/30 then compares QAOA against a *proven* optimum instead of
against an annealing heuristic.
"""

from __future__ import annotations

import itertools
from math import comb
from typing import Literal

import numpy as np
from scipy.optimize import differential_evolution

from src.qubo.portfolio import QUBOProblem
from src.solvers.base import AbstractSolver, SolverResult

#: Largest feasible set ``exact`` will enumerate under ``method="auto"``.
#: Measured on the paper's instances (A6000 host, 2026-08): ~11.5 us per
#: candidate, so N=30/K=6 (593,775 candidates) takes ~7 s and this ceiling
#: caps a single solve at roughly a minute. N=50/K=10 is 1.0e10 candidates —
#: far above any workable ceiling, so it stays on simulated annealing.
MAX_EXACT_CANDIDATES = 5_000_000


class ClassicalSolver(AbstractSolver):
    """
    Classical baseline solver.

    Args:
        method: "exact" | "brute_force" | "simulated_annealing" | "auto".
            "auto" prefers ``exact`` whenever the problem carries a cardinality
            constraint whose feasible set fits under ``max_exact_candidates``,
            falls back to ``brute_force`` for small unconstrained problems, and
            uses simulated annealing otherwise.
        max_exact_candidates: Ceiling on C(N,K) for the "auto" choice above.
            Ignored when ``method="exact"`` is requested explicitly.
    """

    def __init__(
        self,
        method: Literal[
            "exact", "brute_force", "simulated_annealing", "auto"
        ] = "auto",
        max_exact_candidates: int = MAX_EXACT_CANDIDATES,
    ) -> None:
        self.method = method
        self.max_exact_candidates = max_exact_candidates
        super().__init__(backend=f"classical_{method}")

    def solve(self, problem: QUBOProblem, seed: int = 42) -> SolverResult:
        n = problem.n_variables
        effective = self.method
        if effective == "auto":
            effective = self._auto_method(problem)

        t0 = self._timer()
        if effective == "exact":
            bitstring, meta = self._exact_cardinality(problem)
        elif effective == "brute_force":
            bitstring, meta = self._brute_force(problem)
        else:
            bitstring, meta = self._simulated_annealing(problem, seed=seed)
        runtime = self._timer() - t0

        meta["method"] = effective
        return self._make_result(problem, bitstring, runtime, meta)

    def _auto_method(self, problem: QUBOProblem) -> str:
        """Pick the cheapest strategy that still proves optimality, if any."""
        n = problem.n_variables
        K = problem.metadata.get("num_select", -1)
        if 0 < K <= n and comb(n, K) <= self.max_exact_candidates:
            return "exact"
        # No cardinality constraint to exploit: fall back to the 2^N walk while
        # it is affordable, then to the heuristic.
        return "brute_force" if n <= 20 else "simulated_annealing"

    # ------------------------------------------------------------------
    # Internal strategies
    # ------------------------------------------------------------------

    def _exact_cardinality(self, problem: QUBOProblem) -> tuple[np.ndarray, dict]:
        """Enumerate the feasible set {x : sum(x) = K} and return its minimum.

        Exhaustive over the *feasible* set, so the result is a proven global
        optimum of the constrained problem — the same answer ``_brute_force``
        gives, reached without visiting the 2^N - C(N,K) infeasible assignments.

        Two details keep this bit-for-bit interchangeable with ``_brute_force``,
        which matters because Table I is published from these numbers:

        * the energy is evaluated as ``x @ Q @ x`` on a full-length float64
          vector, not as a sum over the selected submatrix, so the
          floating-point summation order is unchanged;
        * ties are broken toward the smaller bitstring read as a binary number
          with index 0 as the most significant bit. ``_brute_force`` gets that
          ordering for free from ``itertools.product`` and keeps the first
          strict minimum; ``itertools.combinations`` enumerates in a different
          order, so the rule is applied explicitly here.

        Returns:
            The optimal bitstring and metadata recording the search size.
        """
        n = problem.n_variables
        K = problem.metadata.get("num_select", -1)
        Q = problem.Q
        if not 0 < K <= n:
            raise ValueError(
                f"exact enumeration needs a cardinality constraint; "
                f"problem has num_select={K!r} for N={n}"
            )

        best_energy = np.inf
        best_mask = None
        best_x = np.zeros(n)

        for combo in itertools.combinations(range(n), K):
            x = np.zeros(n)
            x[list(combo)] = 1.0
            e = float(x @ Q @ x)
            if e > best_energy:
                continue
            mask = sum(1 << (n - 1 - i) for i in combo)
            if e < best_energy or mask < best_mask:
                best_energy = e
                best_mask = mask
                best_x = x

        return best_x, {
            "n_evaluated": comb(n, K),
            "search_space": 2**n,
            "exhaustive": True,
        }

    def _brute_force(self, problem: QUBOProblem) -> tuple[np.ndarray, dict]:
        """Enumerate all 2^N bitstrings; return the minimum-energy feasible one."""
        n = problem.n_variables
        K = problem.metadata.get("num_select", -1)
        Q = problem.Q

        best_energy = np.inf
        best_x = np.zeros(n)

        for bits in itertools.product([0, 1], repeat=n):
            x = np.array(bits, dtype=float)
            if K > 0 and int(x.sum()) != K:
                continue
            e = float(x @ Q @ x)
            if e < best_energy:
                best_energy = e
                best_x = x.copy()

        return best_x, {
            "n_evaluated": 2**n,
            "search_space": 2**n,
            "exhaustive": True,
        }

    def _simulated_annealing(
        self,
        problem: QUBOProblem,
        seed: int,
        n_steps: int = 50_000,
        t_init: float = 2.0,
        t_final: float = 0.01,
    ) -> tuple[np.ndarray, dict]:
        """
        Simulated annealing with feasibility-preserving bit swaps.

        When K is specified, moves are 2-opt swaps (flip one 0→1 and one 1→0
        simultaneously) to keep the cardinality fixed throughout.
        """
        rng = np.random.default_rng(seed)
        n = problem.n_variables
        Q = problem.Q
        K = problem.metadata.get("num_select", -1)

        # Initialise: select K random assets
        x = np.zeros(n)
        if K > 0:
            idx = rng.choice(n, size=K, replace=False)
            x[idx] = 1.0
        else:
            x = rng.integers(0, 2, size=n).astype(float)

        energy = float(x @ Q @ x)
        best_x = x.copy()
        best_energy = energy

        temps = np.linspace(t_init, t_final, n_steps)
        accepted = 0

        for step, T in enumerate(temps):
            if K > 0:
                # Cardinality-preserving swap
                ones = np.where(x == 1)[0]
                zeros = np.where(x == 0)[0]
                if len(ones) == 0 or len(zeros) == 0:
                    continue
                i_out = rng.choice(ones)
                i_in = rng.choice(zeros)
                x[i_out] = 0.0
                x[i_in] = 1.0
                new_energy = float(x @ Q @ x)
                delta = new_energy - energy
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    energy = new_energy
                    accepted += 1
                    if energy < best_energy:
                        best_energy = energy
                        best_x = x.copy()
                else:
                    x[i_out] = 1.0
                    x[i_in] = 0.0
            else:
                i = rng.integers(n)
                x[i] = 1.0 - x[i]
                new_energy = float(x @ Q @ x)
                delta = new_energy - energy
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    energy = new_energy
                    accepted += 1
                    if energy < best_energy:
                        best_energy = energy
                        best_x = x.copy()
                else:
                    x[i] = 1.0 - x[i]

        return best_x, {
            "n_steps": n_steps,
            "t_init": t_init,
            "t_final": t_final,
            "acceptance_rate": accepted / n_steps,
            # Heuristic: the returned energy is an upper bound on the optimum,
            # so an optimality gap measured against it is a lower bound.
            "exhaustive": False,
        }
