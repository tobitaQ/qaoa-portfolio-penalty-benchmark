# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Tests for ClassicalSolver (exact C(N,K), brute force, simulated annealing)."""

import math

import numpy as np
import pytest

from src.qubo.portfolio import PortfolioQUBO
from src.solvers.classical_solver import ClassicalSolver


@pytest.fixture
def toy_problem():
    """5-asset problem — small enough for brute force."""
    np.random.seed(1)
    n = 5
    returns = np.random.uniform(0.05, 0.20, n)
    cov = np.eye(n) * 0.04 + np.random.uniform(0.001, 0.005, (n, n))
    cov = (cov + cov.T) / 2
    return PortfolioQUBO().formulate(returns, cov, num_select=2)


class TestClassicalSolver:
    def test_brute_force_feasible(self, toy_problem):
        solver = ClassicalSolver(method="brute_force")
        result = solver.solve(toy_problem, seed=42)
        assert result.feasible(2), f"Expected K=2, got {result.num_selected}"

    def test_simulated_annealing_feasible(self, toy_problem):
        solver = ClassicalSolver(method="simulated_annealing")
        result = solver.solve(toy_problem, seed=42)
        assert result.feasible(2), f"Expected K=2, got {result.num_selected}"

    def test_auto_prefers_exact_when_cardinality_is_constrained(self, toy_problem):
        solver = ClassicalSolver(method="auto")
        result = solver.solve(toy_problem, seed=42)
        assert result.metadata["method"] == "exact"

    def test_auto_falls_back_when_the_feasible_set_is_too_large(self, toy_problem):
        """A ceiling below C(N,K) must not silently run an unaffordable search."""
        solver = ClassicalSolver(method="auto", max_exact_candidates=1)
        result = solver.solve(toy_problem, seed=42)
        assert result.metadata["method"] == "brute_force"

    def test_result_has_runtime(self, toy_problem):
        solver = ClassicalSolver(method="brute_force")
        result = solver.solve(toy_problem, seed=42)
        assert result.runtime_seconds > 0

    def test_energy_is_finite(self, toy_problem):
        solver = ClassicalSolver(method="brute_force")
        result = solver.solve(toy_problem, seed=42)
        assert np.isfinite(result.energy_no_offset)

    def test_deterministic_with_same_seed(self, toy_problem):
        solver = ClassicalSolver(method="simulated_annealing")
        r1 = solver.solve(toy_problem, seed=99)
        r2 = solver.solve(toy_problem, seed=99)
        assert np.allclose(r1.bitstring, r2.bitstring)

    def test_exact_matches_brute_force_bit_for_bit(self):
        """The published tables switch to ``exact``; it must not move a digit.

        Both methods are exhaustive over the same feasible set, so agreement on
        the *energy* is expected. The stronger claim tested here is that the
        returned bitstring and the float64 energy are identical, i.e. the
        summation order and the tie-break rule survived the rewrite.
        """
        rng = np.random.default_rng(7)
        for n, K in ((6, 2), (8, 3), (10, 4), (12, 5)):
            returns = rng.uniform(0.05, 0.30, n)
            cov = rng.uniform(0.001, 0.005, (n, n))
            cov = (cov + cov.T) / 2 + np.eye(n) * 0.04
            problem = PortfolioQUBO().formulate(returns, cov, num_select=K)

            fast = ClassicalSolver(method="exact").solve(problem, seed=42)
            slow = ClassicalSolver(method="brute_force").solve(problem, seed=42)

            assert fast.energy_no_offset == slow.energy_no_offset, (
                f"N={n} K={K}: energies differ by "
                f"{fast.energy_no_offset - slow.energy_no_offset:.3e}"
            )
            assert np.array_equal(fast.bitstring, slow.bitstring), (
                f"N={n} K={K}: different optimum selected"
            )

    def test_exact_breaks_ties_the_same_way_as_brute_force(self):
        """Fully degenerate instance: every K-subset has identical energy.

        With identical returns and a scalar covariance, all C(N,K) feasible
        assignments tie exactly, so the answer is decided purely by the
        tie-break rule. ``itertools.combinations`` walks the feasible set in a
        different order than ``itertools.product``, so this fails if the rule
        is left implicit.
        """
        n, K = 8, 3
        returns = np.full(n, 0.1)
        cov = np.eye(n) * 0.04
        problem = PortfolioQUBO().formulate(returns, cov, num_select=K)

        fast = ClassicalSolver(method="exact").solve(problem, seed=42)
        slow = ClassicalSolver(method="brute_force").solve(problem, seed=42)
        assert np.array_equal(fast.bitstring, slow.bitstring)

    def test_exact_is_never_worse_than_annealing(self):
        """Sanity check on the heuristic this replaces for N in 24..30."""
        rng = np.random.default_rng(11)
        n, K = 14, 5
        returns = rng.uniform(0.05, 0.30, n)
        cov = rng.uniform(0.001, 0.005, (n, n))
        cov = (cov + cov.T) / 2 + np.eye(n) * 0.04
        problem = PortfolioQUBO().formulate(returns, cov, num_select=K)

        exact = ClassicalSolver(method="exact").solve(problem, seed=42)
        annealed = ClassicalSolver(method="simulated_annealing").solve(problem, seed=42)
        assert exact.energy_no_offset <= annealed.energy_no_offset

    def test_exact_reports_the_feasible_set_size(self, toy_problem):
        """Metadata must show the search was over C(N,K), not 2^N."""
        result = ClassicalSolver(method="exact").solve(toy_problem, seed=42)
        assert result.metadata["n_evaluated"] == math.comb(5, 2)
        assert result.metadata["search_space"] == 2**5
        assert result.metadata["exhaustive"] is True

    def test_annealing_is_flagged_as_not_exhaustive(self, toy_problem):
        """Downstream needs to tell a proven optimum from an upper bound."""
        result = ClassicalSolver(method="simulated_annealing").solve(toy_problem, seed=42)
        assert result.metadata["exhaustive"] is False

    def test_exact_requires_a_cardinality_constraint(self):
        """Without K there is no feasible set to enumerate — fail, don't guess."""
        rng = np.random.default_rng(3)
        n = 6
        returns = rng.uniform(0.05, 0.30, n)
        cov = np.eye(n) * 0.04
        problem = PortfolioQUBO().formulate(returns, cov, num_select=2)
        problem.metadata["num_select"] = -1

        with pytest.raises(ValueError, match="cardinality"):
            ClassicalSolver(method="exact").solve(problem, seed=42)

    def test_brute_force_finds_global_optimum(self):
        """For a trivially structured problem, verify the known optimum is found."""
        # Asset 0 has the best return and lowest variance — must be selected.
        returns = np.array([1.0, 0.01, 0.01, 0.01])
        cov = np.eye(4) * 0.001
        problem = PortfolioQUBO().formulate(returns, cov, num_select=1, risk_aversion=0.0)
        solver = ClassicalSolver(method="brute_force")
        result = solver.solve(problem, seed=42)
        assert 0 in result.selected_assets, "Asset 0 (highest return) should be selected"
