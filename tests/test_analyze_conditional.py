# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from experiments.paper01_qubo_baseline import analyze_conditional as ac


def test_conditional_tv_is_zero_for_uniform_and_one_minus_1_over_f_for_a_point_mass():
    p_f, d = ac.conditional_tv(np.full(4, 0.05))
    assert p_f == pytest.approx(0.2) and d == pytest.approx(0.0)
    p_f, d = ac.conditional_tv(np.array([0.0, 0.0, 0.3, 0.0]))
    assert p_f == pytest.approx(0.3) and d == pytest.approx(1 - 1 / 4)


def test_the_reviewers_counterexample_has_r_tau_one_but_is_not_uniform():
    # F has four elements, the first two form F_tau. (1/2, 0, 1/2, 0) lands in
    # F_tau at the uniform rate 1/2 -- R_tau = 1 -- and is not uniform on F.
    p = np.array([0.5, 0.0, 0.5, 0.0]) * 0.4       # P_F = 0.4
    p_f, d = ac.conditional_tv(p)
    r_tau = (p[:2].sum() / p_f) / (2 / 4)
    assert r_tau == pytest.approx(1.0)
    assert d == pytest.approx(0.5)


def test_conditional_tv_is_undefined_without_feasible_mass():
    p_f, d = ac.conditional_tv(np.zeros(6))
    assert p_f == 0.0 and np.isnan(d)
    p_f, d = ac.conditional_tv(np.full(6, 1e-12))
    assert np.isnan(d)


def test_per_run_reads_artifacts_and_summarise_counts_undefined(tmp_path):
    idx = np.arange(5)
    np.savez(tmp_path / "N12_i03_Aheur_s42.npz", feasible_indices=idx,
             final_probs_feasible=np.full(5, 0.04), initial_probs_feasible=np.array([0.1, 0, 0, 0, 0]))
    np.savez(tmp_path / "N12_i04_Aheur_s42.npz", feasible_indices=idx,
             final_probs_feasible=np.zeros(5), initial_probs_feasible=np.full(5, 0.04))
    (tmp_path / "notes.txt").write_text("ignored")
    rows = ac.per_run(tmp_path)
    assert [(r["n_assets"], r["instance"], r["penalty_label"], r["qaoa_seed"]) for r in rows] \
        == [(12, 3, "Aheur", 42), (12, 4, "Aheur", 42)]
    assert rows[0]["final_D_cond"] == pytest.approx(0.0)
    assert rows[0]["initial_D_cond"] == pytest.approx(0.8)
    assert np.isnan(rows[1]["final_D_cond"])
    [s] = ac.summarise(rows)
    assert s["runs"] == 2 and s["runs_undefined"] == 1
    assert s["final_D_cond_median"] == pytest.approx(0.0)
    assert s["point_mass_D_cond_median"] == pytest.approx(0.8)
    # The finite-shot reference is what uniform looks like at 1,000 shots: positive, well under 1.
    assert 0.0 < s["uniform_F_1000_shot_D_cond"] < 0.1
