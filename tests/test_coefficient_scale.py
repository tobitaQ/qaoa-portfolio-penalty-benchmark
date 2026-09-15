# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""The "0.5 % of the largest coefficient" statement, made precise (review round 3, M6)."""

import numpy as np
import pytest

from experiments.paper01_qubo_baseline import coefficient_scale as cs
from experiments.paper01_qubo_baseline.run_e1_cross import DATA_END_DATE, InstanceReference
from src.finance.data_loader import FinanceDataLoader
from src.qubo.portfolio import PortfolioQUBO


@pytest.fixture(scope="module")
def ref():
    return InstanceReference(12, 0, FinanceDataLoader(), PortfolioQUBO(), DATA_END_DATE)


def test_penalty_only_ising_scale_is_the_closed_form(ref):
    """h_i^pen = A(K − N/2), J_ij^pen = A/2: the penalty's own scale is max of the two."""
    row = cs.scales_for(ref, PortfolioQUBO(), "Aheur", ref.a_heur)
    A, n, k = ref.a_heur, ref.n, ref.k
    assert row["s_H_penalty_only"] == pytest.approx(max(abs(A * (k - n / 2)), A / 2), rel=1e-12)
    # The QUBO's largest penalty entry is the diagonal A(1 − 2K), a different number.
    assert row["max_abs_Q_penalty"] == pytest.approx(abs(A * (1 - 2 * k)), rel=1e-12)


def test_the_two_ratios_differ_and_epsilon_F_is_delta_over_s_H(ref):
    row = cs.scales_for(ref, PortfolioQUBO(), "Aheur", ref.a_heur)
    assert row["qubo_ratio_objective_over_max"] != pytest.approx(row["ising_ratio_objective_over_s_H"], rel=1e-3)
    assert row["epsilon_F"] == pytest.approx(ref.feasible.delta / row["s_H"], rel=1e-12)
    # At A = 0 the scale is the objective's own, so ε_F is Δ_F / s_obj and O(1).
    zero = cs.scales_for(ref, PortfolioQUBO(), "A0", 0.0)
    assert zero["s_H"] == pytest.approx(zero["s_H_objective_only"], rel=1e-12)
    assert zero["epsilon_F"] > 0.1


def test_summary_carries_median_and_range_per_setting():
    rows = [{"n_assets": 12, "penalty_label": "Aheur", "qubo_ratio_objective_over_max": v,
             "ising_ratio_objective_over_s_H": 2 * v, "epsilon_F": 3 * v} for v in (0.1, 0.2, 0.4)]
    [s] = cs.summarise(rows)
    assert s["instances"] == 3 and s["epsilon_F_median"] == pytest.approx(0.6)
    assert (s["qubo_ratio_objective_over_max_min"], s["qubo_ratio_objective_over_max_max"]) == (0.1, 0.4)
