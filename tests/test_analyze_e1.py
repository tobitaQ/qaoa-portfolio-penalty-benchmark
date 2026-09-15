# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""The E1 manifest table (plan §12.2 Table 1) counts what ran, not what worked."""

import math

import pytest
from experiments.paper01_qubo_baseline import analyze_e1
from experiments.paper01_qubo_baseline.run_e1_cross import DEFAULT_INSTANCES, DEFAULT_SEEDS, MULTIPLES


def _ref(n, inst, zero):
    return {"n_assets": str(n), "n_select": "2", "instance": str(inst),
            "a_crit_is_zero": str(zero), "feasible_set_size": "66",
            "reference_seconds": "0.01"}


def _run(n, inst, label, seed, status="ok", g_f="0.0", best_feasible="True"):
    return {"n_assets": str(n), "instance": str(inst), "penalty_label": label,
            "qaoa_seed": str(seed), "status": status, "g_F": g_f,
            "best_all_feasible": best_feasible, "runtime_s": "10.0"}


def test_a_crit_zero_instances_shrink_the_expected_count():
    refs = [_ref(12, 0, False), _ref(12, 1, True)]
    rows = []
    for seed in (42, 43):
        for _, label in MULTIPLES:
            rows.append(_run(12, 0, label, seed))
        rows.append(_run(12, 1, "A0", seed))
        for inst in (0, 1):
            rows += [_run(12, inst, "Aheur", seed), _run(12, inst, "Amargin", seed)]
    [m] = analyze_e1.manifest_table(rows, refs)
    assert m["runs_nominal"] == len(DEFAULT_INSTANCES) * len(DEFAULT_SEEDS) * 6
    # Instance 0: 6 conditions; instance 1: 3 (A0 + two heuristics); 2 seeds.
    assert m["runs_expected"] == (6 + 3) * 2 == m["runs_done"]
    assert m["instances_a_crit_zero"] == 1
    assert m["seeds"] == "42 43"


def test_failures_and_missing_feasible_shots_are_counted_not_dropped():
    refs = [_ref(16, 0, False)]
    rows = [_run(16, 0, "Aheur", 42),
            _run(16, 0, "Aheur", 43, status="error:RuntimeError"),
            _run(16, 0, "Aheur", 44, status="error:RuntimeError"),
            _run(16, 0, "Amargin", 42, g_f="", best_feasible="False")]
    [m] = analyze_e1.manifest_table(rows, refs)
    assert (m["runs_done"], m["runs_ok"], m["runs_failed"]) == (4, 2, 2)
    assert m["failure_kinds"] == "error:RuntimeError×2"
    assert m["runs_without_feasible_shot"] == 1
    assert m["runs_best_shot_infeasible"] == 1
    assert abs(m["gpu_hours_total"] - 40.0 / 3600) < 1e-12


def test_expected_follows_the_conditions_actually_run():
    # E3 runs only the two heuristics on the same instances (plan §7.1).
    refs = [_ref(12, 0, False), _ref(12, 1, True)]
    rows = [_run(12, inst, label, seed)
            for inst in (0, 1) for label in ("Aheur", "Amargin") for seed in (42, 43, 44)]
    [m] = analyze_e1.manifest_table(rows, refs)
    assert m["penalty_conditions_run"] == 2
    assert m["runs_expected"] == 2 * 2 * 3 == m["runs_done"]


def test_initial_final_pairs_within_the_run_at_a_fixed_penalty():
    """Review round 3, M4: the optimisation effect is initial → final of the
    same run, not A_heur → A_margin after optimisation."""
    refs = [{"n_assets": "12", "instance": "0", "P_tau_main_uniform_F": "0.05"},
            {"n_assets": "12", "instance": "1", "P_tau_main_uniform_F": "0.05"}]
    rows, tv = [], []
    # Two instances × two seeds; P_F doubles in every run, P_τ/P_F stays at the
    # uniform rate (R_τ = 1 before and after), D_cond drops from 0.3 to 0.1.
    for inst in ("0", "1"):
        for seed, pf in (("42", 0.10), ("43", 0.20)):
            rid = f"N12_i0{inst}_Aheur_s{seed}"
            rows.append({"run_id": rid, "n_assets": "12", "instance": inst, "penalty_label": "Aheur",
                         "qaoa_seed": seed, "status": "ok",
                         "initial_P_F": str(pf), "final_P_F": str(2 * pf),
                         "initial_P_tau_main": str(0.05 * pf), "final_P_tau_main": str(0.05 * 2 * pf),
                         "initial_Q_tau_main_1000": "0.1", "final_Q_tau_main_1000": "0.3",
                         "initial_expected_violation": "20", "final_expected_violation": "10"})
            tv.append({"run_id": rid, "initial_D_cond": "0.3", "final_D_cond": "0.1"})
    out = {r["field"]: r for r in analyze_e1.initial_final(rows, refs, tv)}
    assert out["P_F"]["runs"] == 4 and out["P_F"]["instances"] == 2
    # Seed-median within instance (0.10, 0.20 → 0.15), then the instance median.
    assert out["P_F"]["initial_median"] == pytest.approx(0.15)
    assert out["P_F"]["final_median"] == pytest.approx(0.30)
    assert out["P_F"]["paired_diff_median"] == pytest.approx(0.15)
    assert out["P_F"]["instances_increased"] == 2
    assert out["R_tau"]["initial_median"] == pytest.approx(1.0)
    assert out["R_tau"]["paired_diff_median"] == pytest.approx(0.0)
    assert out["D_cond"]["paired_diff_median"] == pytest.approx(-0.2)
    assert out["expected_violation"]["instances_decreased"] == 2
    # Without the conditional-TV file the D_cond rows are simply absent.
    assert "D_cond" not in {r["field"] for r in analyze_e1.initial_final(rows, refs, None)}


def test_acquisition_factors_decompose_exactly_over_the_same_pairs():
    """Review round 4, C02: P_τ ratio = P_F ratio × R_τ ratio, pair by pair,
    so the geometric means multiply exactly; a zero on either side is counted."""
    def row(label, seed, pf, pt):
        return {"n_assets": "20", "instance": "0", "penalty_label": label,
                "qaoa_seed": seed, "status": "ok",
                "final_P_F": str(pf), "final_P_tau_main": str(pt)}
    rows = [row("Aheur", "42", 0.10, 0.001), row("Amargin", "42", 0.40, 0.002),   # P_F ×4, R ×0.5
            row("Aheur", "43", 0.20, 0.002), row("Amargin", "43", 0.20, 0.008),   # P_F ×1, R ×4
            row("Aheur", "44", 0.10, 0.0), row("Amargin", "44", 0.30, 0.003)]     # zero: skipped
    [d] = analyze_e1.acquisition_factors(rows)
    assert d["pairs"] == 2 and d["skipped_zero"] == 1
    assert d["P_F_ratio_geomean"] == pytest.approx(2.0)      # sqrt(4 × 1)
    assert d["R_tau_ratio_geomean"] == pytest.approx(math.sqrt(2.0))  # sqrt(0.5 × 4)
    assert d["P_tau_ratio_geomean"] == pytest.approx(
        d["P_F_ratio_geomean"] * d["R_tau_ratio_geomean"])
    assert d["pairs_P_F_ratio_above_1"] == 1 and d["pairs_R_tau_ratio_above_1"] == 1
    assert d["pairs_P_tau_ratio_above_1"] == 2


def test_optimum_hits_are_counted_on_every_completed_run():
    """Review round 4, P02: the zero-gap share Fig. 2 draws, on Fig. 2's population."""
    def row(label, seed, g_f, opt):
        return {"n_assets": "16", "instance": "0", "penalty_label": label,
                "qaoa_seed": seed, "status": "ok", "g_F": g_f, "is_optimal": opt}
    rows = [row("Aheur", "42", "0.0", "True"), row("Aheur", "43", "0.02", "False"),
            row("Aheur", "44", "", ""),                       # no feasible shot
            row("Aheur", "45", "0.0", "True", ) | {"status": "error:X"}]
    [h] = analyze_e1.optimum_hits(rows)
    assert (h["runs"], h["runs_with_feasible_shot"], h["optimum_hits"]) == (3, 2, 1)
    assert h["share_of_runs_with_feasible_shot"] == pytest.approx(0.5)
    assert h["share_of_runs"] == pytest.approx(1 / 3)
