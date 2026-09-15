# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Tests for the near-degeneracy census behind Fig. 9.

The measurement rests on two rewrites of code that already existed elsewhere:
an exhaustive walk of the feasible set, and a vectorised Sharpe ratio. Both are
checked here against the implementations the paper's tables are computed with,
because a silent disagreement would look like a finding.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from experiments.paper01_qubo_baseline.run_degeneracy import (
    band_summary,
    enumerate_feasible,
    minimum_feasible_penalty,
    range_normalised_ratio,
    sharpe_for_all,
)
from src.finance.metrics import PortfolioMetrics
from src.qubo.portfolio import PortfolioQUBO
from src.solvers.classical_solver import ClassicalSolver


@pytest.fixture
def instance():
    """A 9-asset, K=3 problem: 84 feasible subsets, exhaustive in milliseconds."""
    rng = np.random.default_rng(5)
    n, k = 9, 3
    returns = rng.uniform(0.05, 0.30, n)
    cov = rng.uniform(0.001, 0.005, (n, n))
    cov = (cov + cov.T) / 2 + np.eye(n) * 0.04
    problem = PortfolioQUBO().formulate(returns, cov, num_select=k)
    return problem, returns, cov, n, k


class TestEnumerateFeasible:
    def test_covers_exactly_the_feasible_set(self, instance):
        problem, _, _, n, k = instance
        combos, energies = enumerate_feasible(problem, n, k)
        assert len(combos) == len(energies) == 84
        assert all(len(c) == k for c in combos)
        assert len({frozenset(c) for c in combos}) == 84

    def test_optimum_is_bit_identical_to_the_solver(self, instance):
        """Fig. 9 quotes gaps against Table I's optimum; it must be that value."""
        problem, _, _, n, k = instance
        _, energies = enumerate_feasible(problem, n, k)
        solved = ClassicalSolver(method="exact").solve(problem, seed=42)
        assert float(energies.min()) == solved.energy_no_offset


class TestSharpeForAll:
    def test_agrees_with_portfolio_metrics_on_every_subset(self, instance):
        """The vectorised form is an optimisation, not a second definition."""
        problem, returns, cov, n, k = instance
        combos, _ = enumerate_feasible(problem, n, k)
        metrics = PortfolioMetrics(risk_free_rate=0.001)

        fast = sharpe_for_all(combos, returns, cov, k)
        slow = np.array([
            metrics.evaluate(list(c), returns, cov, target_k=k).sharpe_ratio
            for c in combos
        ])
        assert np.allclose(fast, slow, rtol=1e-12, atol=0.0)

    def test_handles_a_degenerate_zero_volatility_portfolio(self):
        """Guard the division: a zero-variance subset must not produce inf/nan."""
        n, k = 4, 2
        returns = np.full(n, 0.1)
        cov = np.zeros((n, n))
        combos = list(itertools.combinations(range(n), k))
        assert np.all(sharpe_for_all(combos, returns, cov, k) == 0.0)


class TestBandSummary:
    def test_counts_and_quantiles_describe_the_selected_band(self):
        sharpe = np.array([1.0, 1.2, 1.4, 1.6, 1.8])
        within = np.array([True, True, True, False, False])

        row = band_summary(10, 3, sharpe, within, "reported_gap", 0.1, 1.0)

        assert row["n_within"] == 3
        assert row["feasible_set_size"] == 5
        assert row["frac_within"] == pytest.approx(0.6)
        assert row["sharpe_min"] == pytest.approx(1.0)
        assert row["sharpe_median"] == pytest.approx(1.2)
        assert row["sharpe_max"] == pytest.approx(1.4)
        # Spread is quoted against the optimum's Sharpe, here 1.0.
        assert row["sharpe_spread_pct"] == pytest.approx(40.0)

    def test_a_band_holding_only_the_optimum_has_zero_spread(self):
        sharpe = np.array([1.5, 0.2, 0.3])
        within = np.array([True, False, False])

        row = band_summary(10, 3, sharpe, within, "objective_range", 1.0, 1.5)

        assert row["n_within"] == 1
        assert row["sharpe_spread_pct"] == pytest.approx(0.0)


class TestTheFindingItself:
    def test_the_gap_denominator_is_the_penalty_constant(self, instance):
        """Fig. 9's right panel claims |E*| is A*K^2, fixed by the formulation.

        If this ever stops holding — a reformulation that folds the constant in
        differently, say — the "understatement factor" loses its meaning and the
        figure has to be re-derived rather than merely re-rendered.
        """
        problem, _, _, n, k = instance
        a = problem.metadata["penalty_strength"]
        assert problem.offset == pytest.approx(a * k**2)

        _, energies = enumerate_feasible(problem, n, k)
        # For feasible x the penalty contributes exactly -A*K^2, so adding the
        # offset back recovers an O(1) portfolio objective out of an O(A*K^2)
        # reported energy.
        objective = energies + problem.offset
        assert abs(objective).max() < abs(energies).min()


class TestMinimumFeasiblePenalty:
    """A_min is what makes the penalty overshoot measurable rather than asserted."""

    @staticmethod
    def _objective_only(returns, cov, k):
        return PortfolioQUBO().formulate(
            returns, cov, num_select=k, risk_aversion=0.5, penalty_strength=0.0
        ).Q

    def test_matches_a_full_2n_search(self, instance):
        """Cross-check the window+bound search against brute force over 2^N.

        The production path enumerates only cardinalities near K and bounds the
        rest. This asserts the shortcut lands on the same A_min that walking
        every assignment would give.
        """
        problem, returns, cov, n, k = instance
        q_obj = self._objective_only(returns, cov, k)

        bits = ((np.arange(1 << n)[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
        objective = ((bits @ q_obj) * bits).sum(axis=1)
        cardinality = bits.sum(axis=1).astype(int)
        best = {m: float(objective[cardinality == m].min()) for m in range(n + 1)}
        expected = max((best[k] - best[m]) / (m - k) ** 2 for m in range(n + 1) if m != k)

        a_min, _, _ = minimum_feasible_penalty(q_obj, n, k)
        assert a_min == pytest.approx(expected, rel=1e-12)

    def test_a_min_is_the_feasibility_threshold(self, instance):
        """Just above A_min the optimum is feasible; just below it is not."""
        problem, returns, cov, n, k = instance
        q_obj = self._objective_only(returns, cov, k)
        a_min, _, _ = minimum_feasible_penalty(q_obj, n, k)

        bits = ((np.arange(1 << n)[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
        objective = ((bits @ q_obj) * bits).sum(axis=1)
        cardinality = bits.sum(axis=1).astype(int)

        def optimal_cardinality(a):
            return int(cardinality[np.argmin(objective + a * (cardinality - k) ** 2)])

        assert optimal_cardinality(a_min * 1.01) == k
        assert optimal_cardinality(a_min * 0.99) != k

    def test_the_heuristic_penalty_overshoots(self, instance):
        """The claim Table VII rests on: A_used is orders of magnitude above A_min."""
        problem, returns, cov, n, k = instance
        q_obj = self._objective_only(returns, cov, k)
        a_min, _, _ = minimum_feasible_penalty(q_obj, n, k)
        assert problem.metadata["penalty_strength"] / a_min > 10

    def test_the_bound_never_understates_a_min(self, instance):
        """A narrower exact window may only raise A_min, never lower it.

        Understating would overstate the overshoot factor and flatter the
        finding, so the direction of the error matters more than its size.
        """
        problem, returns, cov, n, k = instance
        q_obj = self._objective_only(returns, cov, k)
        wide, _, _ = minimum_feasible_penalty(q_obj, n, k, exact_window=n)
        narrow, _, _ = minimum_feasible_penalty(q_obj, n, k, exact_window=0)
        assert narrow >= wide

    def test_the_reported_gap_scales_inversely_with_the_penalty(self, instance):
        """Two feasible solutions, one penalty knob: the gap is not a property
        of the solutions."""
        problem, returns, cov, n, k = instance
        combos, energies = enumerate_feasible(problem, n, k)
        order = np.argsort(energies)
        e_star, e_other = float(energies[order[0]]), float(energies[order[5]])
        objective_star = e_star + problem.offset
        numerator = abs(e_other - e_star)

        def gap(a):
            return numerator / abs(objective_star - a * k**2) * 100.0

        a = problem.metadata["penalty_strength"]
        # Doubling the penalty roughly halves the number reported for the very
        # same pair of portfolios.
        assert gap(2 * a) == pytest.approx(gap(a) / 2, rel=0.02)


class TestRangeNormalisedRatio:
    """The metric Abbas et al. prescribe, measured rather than argued about.

    Table XIII's claim is that on a penalty-encoded QUBO this ratio does not
    degrade gracefully as the penalty grows — it is already collapsed at the
    weight the formulation uses, because C_max is attained by a maximally
    violating assignment and scales with the penalty while the feasible spread
    does not.
    """

    def test_matches_the_definition(self):
        obj = np.array([-1.0, 0.0, 2.0])
        ratio = range_normalised_ratio(obj, c_max=100.0, offset=10.0, value=-9.0)
        # C_min = -1 - 10 = -11; (100 - (-9)) / (100 - (-11)) = 109/111
        assert ratio == pytest.approx(109.0 / 111.0)

    def test_the_best_feasible_solution_scores_one(self):
        obj = np.array([-1.0, 0.0, 2.0])
        assert range_normalised_ratio(obj, 100.0, 10.0, -11.0) == pytest.approx(1.0)

    def test_a_larger_penalty_compresses_the_whole_feasible_set(self):
        """Raising A must push best and worst feasible closer together, not
        separate them — the mechanism Table XIII reports."""
        obj = np.array([-1.0, 2.0])                       # spread 3, independent of A
        k2 = 4
        def spread(a):
            offset = a * k2
            c_max = 50.0 * a                              # violating assignment scales with A
            best = range_normalised_ratio(obj, c_max, offset, float(obj.min()) - offset)
            worst = range_normalised_ratio(obj, c_max, offset, float(obj.max()) - offset)
            return best - worst
        assert spread(10.0) > spread(100.0) > 0
        assert spread(100.0) < 1e-3


class TestRankOfEnergy:
    def test_an_ulp_above_the_census_value_does_not_shift_the_rank(self):
        from experiments.paper01_qubo_baseline.run_degeneracy import rank_of_energy

        census = np.array([-3.0, -2.0, -1.0, 0.0])
        assert rank_of_energy(census, -1.0) == 3
        assert rank_of_energy(census, np.nextafter(-1.0, 0.0)) == 3
        assert rank_of_energy(census, np.nextafter(-1.0, -np.inf)) == 3
        # A plain searchsorted on the nudged value says 4: the bug being fixed.
        assert int(np.searchsorted(census, np.nextafter(-1.0, 0.0), side="left")) + 1 == 4
