# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""E0: the independent evaluator agrees with the QUBO path, on every state.

The 2026-09-11 plan (§4.1) asks for a scorer that never reads ``Q`` and for
the identities below to hold on all 2^N states of small real instances. The
tolerances are absolute, on objectives of magnitude O(K^2) ≤ 16 here, and are
far below anything the paper reports to.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.paper01_qubo_baseline.run_degeneracy import minimum_feasible_penalty
from experiments.paper01_qubo_baseline.run_experiment import (
    DATA_END_DATE,
    LOOKBACK_YEARS,
    RISK_AVERSION,
)
from experiments.paper01_qubo_baseline.run_instances import instance_tickers
from src.finance.data_loader import FinanceDataLoader
from src.qubo import scoring
from src.qubo.portfolio import PortfolioQUBO

#: Agreement between the two evaluation paths, absolute, in objective units.
TOL = 1e-9


def _instance(n: int, instance: int = 0, penalty=None):
    loader = FinanceDataLoader()
    data = loader.load(instance_tickers(n, instance), end_date=DATA_END_DATE,
                       lookback_years=LOOKBACK_YEARS)
    k = max(2, n // 5)
    problem = PortfolioQUBO().formulate(
        data.returns, data.covariance, num_select=k,
        risk_aversion=RISK_AVERSION, penalty_strength=penalty)
    return problem, data, k


@pytest.fixture(scope="module")
def n8():
    problem, data, k = _instance(8)
    obj = scoring.Objective.from_problem(problem, data.returns, data.covariance)
    return problem, data, k, obj, scoring.enumerate_all(obj)


@pytest.fixture(scope="module")
def n12():
    problem, data, k = _instance(12)
    obj = scoring.Objective.from_problem(problem, data.returns, data.covariance)
    return problem, data, k, obj, scoring.enumerate_all(obj)


class TestBitOrder:
    def test_round_trip(self):
        idx = np.arange(64)
        X = scoring.bitstrings_from_indices(idx, 6)
        assert np.array_equal(scoring.indices_from_bitstrings(X), idx)

    def test_wire_zero_is_the_most_significant_bit(self):
        X = scoring.bitstrings_from_indices(np.array([1 << 5]), 6)
        assert X[0].tolist() == [1, 0, 0, 0, 0, 0]

    def test_cardinality_is_the_popcount(self):
        card = scoring.cardinality_of_indices(10)
        idx = np.arange(1 << 10)
        expected = np.array([bin(i).count("1") for i in idx])
        assert np.array_equal(card, expected)


class TestIdentities:
    @pytest.mark.parametrize("fixture", ["n8", "n12"])
    def test_offset_energy_matches_xQx_on_every_state(self, fixture, request):
        problem, _, _, obj, states = request.getfixturevalue(fixture)
        n = problem.n_variables
        A = problem.metadata["penalty_strength"]
        X = scoring.bitstrings_from_indices(np.arange(1 << n), n)
        via_q = np.einsum("ij,jk,ik->i", X, problem.Q, X)
        via_f = obj.offset_energy(X, A)
        assert np.max(np.abs(via_q - via_f)) < TOL

    @pytest.mark.parametrize("fixture", ["n8", "n12"])
    def test_penalised_equals_offset_plus_constant(self, fixture, request):
        problem, _, k, obj, _ = request.getfixturevalue(fixture)
        n = problem.n_variables
        A = problem.metadata["penalty_strength"]
        X = scoring.bitstrings_from_indices(np.arange(1 << n), n)
        assert np.max(np.abs(obj.penalised(X, A) - obj.offset_energy(X, A)
                             - A * k * k)) < TOL
        assert problem.offset == pytest.approx(A * k * k)

    @pytest.mark.parametrize("fixture", ["n8", "n12"])
    def test_penalty_vanishes_on_the_feasible_set(self, fixture, request):
        problem, _, k, obj, states = request.getfixturevalue(fixture)
        n = problem.n_variables
        A = problem.metadata["penalty_strength"]
        feas = scoring.enumerate_feasible(obj)
        X = scoring.bitstrings_from_indices(feas.indices, n)
        assert np.max(np.abs(obj.penalised(X, A) - obj.objective(X))) == 0.0
        assert np.array_equal(states.cardinality[feas.indices], np.full(feas.size, k))
        assert np.max(np.abs(states.f[feas.indices] - feas.f)) < TOL

    def test_penalty_is_independent_of_the_objective(self, n8):
        """Changing A must move only the violating states."""
        problem, data, k, obj, _ = n8
        n = problem.n_variables
        X = scoring.bitstrings_from_indices(np.arange(1 << n), n)
        d = obj.penalised(X, 3.0) - obj.penalised(X, 1.0)
        assert np.max(np.abs(d - 2.0 * obj.violation(X))) < TOL

    def test_the_feasible_optimum_is_the_exact_solver_optimum(self, n12):
        from src.solvers.classical_solver import ClassicalSolver

        problem, _, _, obj, _ = n12
        feas = scoring.enumerate_feasible(obj)
        exact = ClassicalSolver(method="exact").solve(problem)
        e_star = obj.offset_energy(exact.bitstring, problem.metadata["penalty_strength"])
        assert abs(float(e_star[0]) - exact.energy_no_offset) < TOL
        assert abs(feas.f_star - (exact.energy_no_offset + problem.offset)) < TOL
        assert feas.optimum_index == int(scoring.indices_from_bitstrings(exact.bitstring)[0])


class TestThresholds:
    @pytest.mark.parametrize("n", [8, 12, 16])
    def test_exact_acrit_is_never_above_the_bounded_amin(self, n):
        """The shipped A_min bounds cardinalities outside K±2, which can only
        overstate; the exact threshold must sit at or below it, and coincide
        when the binding cardinality was inside the enumerated window."""
        problem, data, k = _instance(n)
        obj = scoring.Objective.from_problem(problem, data.returns, data.covariance)
        a_m = scoring.enumerate_all(obj).cardinality_minima()
        a_crit, binding = scoring.critical_penalty(a_m, k)
        q_obj = PortfolioQUBO().formulate(data.returns, data.covariance, num_select=k,
                                          risk_aversion=RISK_AVERSION,
                                          penalty_strength=0.0).Q
        a_min, binding_min, exact = minimum_feasible_penalty(q_obj, n, k)
        assert a_crit <= a_min + 1e-12
        if exact:
            assert a_crit == pytest.approx(a_min, abs=1e-10)
            assert binding == binding_min

    def test_at_acrit_an_infeasible_state_ties_and_above_it_none_does(self, n12):
        problem, _, k, obj, states = n12
        n = problem.n_variables
        a_m = states.cardinality_minima()
        a_crit, binding = scoring.critical_penalty(a_m, k)
        assert a_crit > 0.0
        X = scoring.bitstrings_from_indices(np.arange(1 << n), n)
        feasible = states.cardinality == k
        f_star = a_m[k]
        # Exactly at A_crit the binding cardinality's best state ties x*.
        h = obj.penalised(X, a_crit)
        assert h[~feasible].min() == pytest.approx(f_star, abs=TOL)
        assert states.cardinality[np.argmin(np.where(feasible, np.inf, h))] == binding
        # Strictly above, every infeasible state costs more than x*.
        h = obj.penalised(X, a_crit * 1.1)
        assert h[~feasible].min() > f_star + 1e-6

    def test_amargin_puts_the_best_infeasible_state_at_the_target(self, n12):
        problem, _, k, obj, states = n12
        n = problem.n_variables
        a_m = states.cardinality_minima()
        feas = scoring.enumerate_feasible(obj)
        a_margin, target = scoring.margin_penalty(a_m, k, feas.f_bar, eps=0.0)
        assert target == pytest.approx(0.5 * (feas.f_star + feas.f_bar))
        X = scoring.bitstrings_from_indices(np.arange(1 << n), n)
        h = obj.penalised(X, a_margin)
        assert h[states.cardinality != k].min() == pytest.approx(target, abs=TOL)
        a_crit, _ = scoring.critical_penalty(a_m, k)
        assert a_margin >= a_crit

    def test_acrit_clamps_at_zero_when_the_optimum_needs_no_help(self):
        a_m = np.array([5.0, 3.0, 1.0, 2.0, 4.0])  # K = 2 is already the minimum
        a_crit, _ = scoring.critical_penalty(a_m, 2)
        assert a_crit == 0.0


class TestGaps:
    def test_deflation_links_the_two_gaps(self, n12):
        problem, _, k, obj, _ = n12
        feas = scoring.enumerate_feasible(obj)
        A = problem.metadata["penalty_strength"]
        f_x = feas.f_sorted[10]
        g_off = scoring.gap_offset(f_x, feas.f_star, A, k)
        g_f = float(feas.gap_range(f_x))
        d = scoring.deflation_factor(feas.f_star, A, k, feas.delta)
        assert g_f == pytest.approx(d * g_off, rel=1e-12)
        # The conventional gap is percent-of-|E*|; the paper's Table VI quoted
        # 80x deflation for this instance, so D must be of that order.
        assert 50 < d < 120

    def test_rank_counts_strictly_better_solutions(self, n12):
        _, _, _, obj, _ = n12
        feas = scoring.enumerate_feasible(obj)
        assert feas.rank(feas.f_star) == (1, 1)
        rank, ties = feas.rank(feas.f_sorted[3])
        assert rank == 4 and ties >= 1
        assert feas.rank(feas.f_max + 1.0)[0] == feas.size + 1

    def test_within_and_optimal_masks(self, n12):
        _, _, _, obj, _ = n12
        feas = scoring.enumerate_feasible(obj)
        assert feas.within(1.0).all()
        assert feas.within(0.0).sum() == 1
        assert feas.is_optimal(1e-9).sum() == 1

    def test_success_probability(self):
        assert scoring.success_probability(0.0, 1000) == 0.0
        assert scoring.success_probability(1.0, 1) == 1.0
        assert scoring.success_probability(0.01, 100) == pytest.approx(1 - 0.99 ** 100)


class TestProbabilityHelpers:
    def test_cardinality_mass_sums_to_one(self):
        n = 6
        probs = np.random.default_rng(0).random(1 << n)
        probs /= probs.sum()
        mass = scoring.cardinality_mass(probs, scoring.cardinality_of_indices(n), n)
        assert mass.shape == (n + 1,)
        assert mass.sum() == pytest.approx(1.0)
        # The uniform distribution puts C(n, m)/2^n on each cardinality.
        uniform = np.full(1 << n, 1.0 / (1 << n))
        mass = scoring.cardinality_mass(uniform, scoring.cardinality_of_indices(n), n)
        from math import comb
        assert np.allclose(mass, [comb(n, m) / (1 << n) for m in range(n + 1)])

    def test_shot_counts(self):
        samples = np.array([[0, 1, 1], [0, 1, 1], [1, 0, 0]], dtype=float)
        idx, counts = scoring.shot_counts(samples)
        assert idx.tolist() == [3, 4]
        assert counts.tolist() == [2, 1]


class TestPosition:
    def test_position_finds_each_feasible_state(self, n12):
        _, _, _, obj, _ = n12
        feas = scoring.enumerate_feasible(obj)
        for pos in (0, 7, feas.size - 1):
            assert feas.position(int(feas.indices[pos])) == pos

    def test_position_rejects_an_infeasible_state(self, n12):
        _, _, _, obj, _ = n12
        feas = scoring.enumerate_feasible(obj)
        with pytest.raises(KeyError):
            feas.position(0)   # the empty portfolio, cardinality 0

    def test_rank_from_own_value_is_ulp_safe(self, n12):
        """Evaluating the same bitstring elsewhere can land one ulp off; the
        rank via position() must not move."""
        problem, _, _, obj, states = n12
        feas = scoring.enumerate_feasible(obj)
        pos = 30
        own = float(feas.f[feas.position(int(feas.indices[pos]))])
        assert feas.rank(own)[0] == int((feas.f < own).sum()) + 1
        nudged = np.nextafter(own, np.inf)
        # The nudged value would count the solution itself as strictly below.
        assert feas.rank(nudged)[0] == feas.rank(own)[0] + 1


def test_success_probability_does_not_cancel_for_small_p():
    """1 − (1 − P)^S loses the leading digits when P is small.

    The naive form agreed with the exact value to eleven figures at P = 10^-6,
    and which digits survived depended on the platform's ``pow``: the same
    analysis wrote different CSVs on the Mac and on the GPU host. The exact
    value here is computed in 50-digit decimal, independent of the formula
    under test.
    """
    from decimal import Decimal, getcontext

    from src.qubo.scoring import success_probability

    getcontext().prec = 50
    for p in (1e-9, 1e-6, 1.9073322619855837e-05, 0.06, 0.336, 0.9):
        for shots in (10, 100, 1000, 10000):
            exact = 1 - (1 - Decimal(repr(p))) ** shots
            got = success_probability(p, shots)
            naive = 1.0 - (1.0 - p) ** shots
            assert abs(Decimal(repr(got)) - exact) <= abs(exact) * Decimal("1e-15")
            # and it is the naive form, not this one, that is wrong
            if p <= 1e-6:
                assert abs(Decimal(repr(naive)) - exact) > abs(exact) * Decimal("1e-13")


def test_success_probability_clamps_the_ends():
    from src.qubo.scoring import success_probability

    assert success_probability(0.0, 1000) == 0.0
    assert success_probability(1.0, 1000) == 1.0
    assert success_probability(-0.5, 10) == 0.0
    assert success_probability(1.5, 10) == 1.0


class TestCriticalPenaltyBoundaryCases:
    """A_crit = 0 covers three situations, and they are not the same (review M8).

    The draft said "at A = A_crit an infeasible assignment ties x*", which holds
    only when the inner maximum is positive. When the clamp is active the
    feasible optimum is already strictly better at A = 0 and nothing ties at any
    A >= 0 -- which is the case all nine of the N = 12 instances with A_crit = 0
    turn out to be in.
    """

    @staticmethod
    def _a_crit(a_m, k):
        from src.qubo import scoring

        return scoring.critical_penalty(np.asarray(a_m, dtype=float), k)[0]

    def test_positive_inner_max_gives_a_tie_at_the_threshold(self):
        # K = 1; cardinality 0 is better than the best feasible, so the penalty
        # has to buy the difference back.
        a_m = [-2.0, -1.0, 0.0]
        a_crit = self._a_crit(a_m, 1)
        assert a_crit > 0
        # at A = a_crit the two are level, and a larger A separates them
        assert a_m[0] + a_crit * (0 - 1) ** 2 == pytest.approx(a_m[1])
        assert a_m[0] + (a_crit + 0.5) * (0 - 1) ** 2 > a_m[1]

    def test_zero_inner_max_is_a_tie_at_zero(self):
        a_m = [-1.0, -1.0, 0.0]
        assert self._a_crit(a_m, 1) == 0.0
        assert a_m[0] == a_m[1]          # already tied with no penalty at all

    def test_negative_inner_max_never_ties(self):
        a_m = [0.5, -1.0, 0.0]
        assert self._a_crit(a_m, 1) == 0.0
        # strictly better at A = 0, and the gap only widens with A
        assert a_m[1] < min(a_m[0], a_m[2])

    def test_a_crit_zero_does_not_imply_a_unique_minimiser(self):
        """It says the minimum over all assignments is attained at K."""
        a_m = [-1.0, -1.0, -1.0]
        assert self._a_crit(a_m, 1) == 0.0
