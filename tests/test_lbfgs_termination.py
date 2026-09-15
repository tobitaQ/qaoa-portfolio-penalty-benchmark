# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""The L-BFGS-B path must say why it stopped (review M7-a).

E3 called its runs converged because they used fewer than the 100 gradient
evaluations they were allowed. That is also what a failed line search looks
like, and the E3 runs were single precision while SciPy's default ftol of
2.2e-9 sits below a single-precision expectation's resolution -- so "relative
reduction of f below tolerance" can mean the objective stopped resolving rather
than that a minimum was reached. Re-running a small block at both precisions
found line-search failures and budget exhaustion in single precision and none
in double. None of that was recoverable from what the runs recorded.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.qubo.portfolio import PortfolioQUBO


def _problem(n: int = 6, k: int = 2):
    rng = np.random.default_rng(0)
    returns = rng.normal(0.08, 0.02, size=n)
    a = rng.normal(size=(n, n))
    cov = (a @ a.T) / n * 0.04
    return PortfolioQUBO().formulate(returns, cov, num_select=k, risk_aversion=0.5)


@pytest.fixture(scope="module")
def lbfgs_result():
    pytest.importorskip("pennylane")
    pytest.importorskip("scipy")
    from src.solvers.braket_solver import BraketSolver

    solver = BraketSolver(backend="lightning_cpu", p_layers=1, n_shots=64,
                          n_optimizer_steps=12, optimizer="lbfgs")
    return solver.solve(_problem(), seed=42).metadata


def test_the_termination_reason_is_recorded(lbfgs_result):
    m = lbfgs_result
    assert m["scipy_message"], "no stopping reason recorded"
    # -1 is the run's own budget cut; 0/1/2 are SciPy's own statuses.
    assert m["scipy_status"] in (-1, 0, 1, 2)
    if m["scipy_status"] == -1:
        assert "budget" in m["scipy_message"]
        assert np.isnan(m["scipy_final_grad_norm"])
    else:
        assert m["scipy_iterations"] >= 0
        assert m["scipy_final_grad_norm"] >= 0.0


def test_the_tolerances_and_precision_travel_with_it(lbfgs_result):
    """A stopping reason is not interpretable without them."""
    m = lbfgs_result
    assert m["scipy_ftol"] == pytest.approx(2.220446049250313e-09)
    assert m["scipy_gtol"] == pytest.approx(1e-05)
    assert m["scipy_version"]
    assert m["precision"] in ("single", "double")


def test_adam_runs_carry_no_scipy_fields(lbfgs_result):
    """None must mean "not applicable", never "converged"."""
    pytest.importorskip("pennylane")
    from src.solvers.braket_solver import BraketSolver

    m = BraketSolver(backend="lightning_cpu", p_layers=1, n_shots=64,
                     n_optimizer_steps=3).solve(_problem(), seed=42).metadata
    assert m["scipy_status"] is None
    assert m["scipy_message"] is None


def test_acquisition_scores_come_from_the_exact_final_state():
    """Review round 3 (M7): the precision diagnostic scores P_τ / Q_τ, not only ⟨H⟩."""
    from experiments.paper01_qubo_baseline import run_lbfgs_termination as diag
    from experiments.paper01_qubo_baseline.run_degeneracy import RISK_FREE_RATE
    from experiments.paper01_qubo_baseline.run_e1_cross import DATA_END_DATE, InstanceReference, RISK_AVERSION
    from src.finance.data_loader import FinanceDataLoader
    from src.finance.metrics import PortfolioMetrics
    from src.solvers.braket_solver import BraketSolver

    ref = InstanceReference(12, 0, FinanceDataLoader(), PortfolioQUBO(), DATA_END_DATE)
    problem = PortfolioQUBO().formulate(ref.data.returns, ref.data.covariance, num_select=ref.k,
                                        risk_aversion=RISK_AVERSION, penalty_strength=ref.a_heur)
    res = BraketSolver(backend="lightning_cpu", p_layers=1, n_shots=200, n_optimizer_steps=3,
                       optimizer="lbfgs", record_state=True).solve(problem, seed=42)
    row = diag.acquisition_scores(res, ref, ref.a_heur, PortfolioMetrics(risk_free_rate=RISK_FREE_RATE))
    assert 0.0 < row["final_P_F"] <= 1.0
    assert row["final_P_opt"] <= row["final_P_tau_main"] <= row["final_P_F"]
    assert row["final_Q_tau_main_1000"] == pytest.approx(1 - (1 - row["final_P_tau_main"]) ** 1000)
    assert 0.0 <= row["final_D_cond"] <= 1.0
    assert row["shots_feasible"] <= 200


def test_precision_pairs_compare_the_same_run_at_both_precisions():
    from experiments.paper01_qubo_baseline import run_lbfgs_termination as diag

    def r(precision, q, msg):
        return {"n_assets": 12, "instance": 0, "penalty_label": "Aheur", "qaoa_seed": 42,
                "precision": precision, "best_expectation": -1.0, "final_P_F": 0.2,
                "final_P_tau_main": 0.01, "final_Q_tau_main_1000": q, "final_D_cond": 0.002,
                "within_tau_main": True, "scipy_message": msg}
    rows = [r("single", 0.30, "ftol"), r("double", 0.31, "gtol"),
            {**r("single", 0.5, "x"), "qaoa_seed": 43}]        # no double arm: dropped
    [p] = diag.precision_pairs(rows)
    assert p["final_Q_tau_main_1000_diff"] == pytest.approx(-0.01)
    assert p["within_tau_main_agree"] is True and p["same_stop_reason"] is False
