# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Regression tests for the two inequalities the paper states in closed form.

Review round 3 asked that both be pinned by small numerical tests, and that
the coupling bound not be read as a disagreement rate between independent
batches (M2). Neither test touches a circuit.
"""

from __future__ import annotations

import numpy as np

from src.qubo.scoring import selected_output_tv_bound, success_probability


# --- E4-A: TV(selected output) ≤ 1 − (1 − d)^S ≤ S·d -------------------------

def test_coupling_bound_is_below_the_linear_bound_and_monotone():
    for d in (0.0, 1e-9, 6.1e-7, 1e-3, 0.2):
        for S in (1, 10, 1000, 10000):
            b = selected_output_tv_bound(d, S)
            assert 0.0 <= b <= min(1.0, S * d) + 1e-15
            assert b >= selected_output_tv_bound(d, S - 1) - 1e-15 if S > 1 else True
    # The two bounds coincide to first order where the paper uses them
    # (Table VI: SV1 2.6e-14 → 2.6e-11, single precision 6.1e-7 → 6.1e-4).
    assert abs(selected_output_tv_bound(2.6e-14, 1000) / 2.6e-11 - 1) < 1e-6
    assert abs(selected_output_tv_bound(6.1e-7, 1000) / 6.1e-4 - 1) < 1e-3


def test_coupling_bound_holds_for_a_selection_rule_on_two_close_samplers():
    """Empirical TV of the *selected* output stays under 1 − (1 − d)^S."""
    rng = np.random.default_rng(1)
    # Two distributions over 6 outcomes at TV distance d; outcome 0 plays ⊥.
    p = np.array([0.30, 0.25, 0.20, 0.15, 0.07, 0.03])
    q = p.copy(); q[1] -= 0.02; q[2] += 0.02          # d = 0.02
    d = 0.5 * np.abs(p - q).sum()
    S, reps = 20, 40000
    select = lambda batch: (batch[batch > 0].min() if (batch > 0).any() else 0)  # best feasible = lowest index
    out_p = np.array([select(rng.choice(6, S, p=p)) for _ in range(reps)])
    out_q = np.array([select(rng.choice(6, S, p=q)) for _ in range(reps)])
    hp = np.bincount(out_p, minlength=6) / reps
    hq = np.bincount(out_q, minlength=6) / reps
    tv_selected = 0.5 * np.abs(hp - hq).sum()
    assert tv_selected <= selected_output_tv_bound(d, S) + 0.01   # 0.01: Monte Carlo slack


def test_the_bound_is_not_a_disagreement_rate_between_independent_batches():
    """d = 0, S = 1, two fair coins: bound 0, yet independent draws differ half the time."""
    assert selected_output_tv_bound(0.0, 1) == 0.0
    rng = np.random.default_rng(2)
    x, y = rng.integers(0, 2, 20000), rng.integers(0, 2, 20000)
    assert abs(np.mean(x != y) - 0.5) < 0.02


# --- E3 multistart: Q_pool ≥ Q_single by AM–GM ------------------------------

def _q_pool_and_single(p: np.ndarray, S: int) -> tuple[float, float]:
    L = p.size
    pool = 1.0 - np.prod([1.0 - success_probability(pk, S) for pk in p])
    single = float(np.mean([success_probability(pk, L * S) for pk in p]))
    return float(pool), single


def test_pooled_starts_never_lose_to_a_random_single_start_at_equal_shots():
    rng = np.random.default_rng(3)
    for _ in range(200):
        p = rng.uniform(0, 1e-2, size=5) ** rng.uniform(0.5, 3)
        pool, single = _q_pool_and_single(p, 1000)
        assert pool >= single - 1e-12
    # Equality exactly when the five success probabilities coincide.
    pool, single = _q_pool_and_single(np.full(5, 3e-4), 1000)
    assert abs(pool - single) < 1e-12
