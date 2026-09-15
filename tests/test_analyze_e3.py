# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""E3 pairs each control run with the E1 run that shares (N, instance, A, seed)."""

import pytest
import numpy as np

from experiments.paper01_qubo_baseline import analyze_e3


def _run(n, inst, label, seed, best_e, p_tau, opt="adam_0.1", penalty="13.0",
         steps="50", grad="", by50="", is_optimal="False", seconds="10.0"):
    return {"n_assets": str(n), "instance": str(inst), "penalty_label": label,
            "qaoa_seed": str(seed), "status": "ok", "penalty": penalty,
            "optimizer": opt, "steps": steps, "best_expectation": str(best_e),
            "best_expectation_by_50": by50, "best_expectation_by_100": by50,
            "expectation_at_final_angles": str(best_e),
            "objective_evaluations": grad, "gradient_evaluations": grad,
            "final_P_F": "0.2", "final_P_tau_main": str(p_tau),
            "final_Q_tau_main_1000": str(1 - (1 - p_tau) ** 1000), "final_P_opt": str(p_tau),
            "g_F": "0.01", "rank_best_feasible": "2", "is_optimal": is_optimal,
            "within_tau_main": "False", "optimisation_seconds": seconds,
            "runtime_s": seconds, "shots": "1000"}


def test_baseline_budget_columns_are_filled_from_the_step_count():
    [r] = analyze_e3._normalise_baseline([_run(12, 0, "Aheur", 42, -2.0, 1e-3)])
    # E1 ran exactly 50 ADAM steps = 50 objective and 50 gradient evaluations,
    # and its best within 50 evaluations is its best overall.
    assert (r["objective_evaluations"], r["gradient_evaluations"]) == ("50", "50")
    assert r["best_expectation_by_50"] == "-2.0"
    assert r["optimizer"] == analyze_e3.BASELINE_LABEL


def test_contrast_refuses_to_pair_runs_with_a_different_penalty():
    base = analyze_e3._normalise_baseline([_run(12, 0, "Aheur", 42, -2.0, 1e-3)])
    ctrl = [_run(12, 0, "Aheur", 42, -2.1, 1e-3, opt="lbfgs_100grad",
                 penalty="14.0", steps="100", grad="60", by50="-2.05")]
    groups = {analyze_e3.BASELINE_LABEL: base, "adam_0.03": [], "lbfgs_100grad": ctrl}
    with pytest.raises(RuntimeError, match="A differs"):
        analyze_e3.contrasts(groups)


def test_contrast_is_instance_level_and_signed_by_direction():
    base, ctrl = [], []
    for inst in (0, 1, 2):
        for seed in (42, 43):
            base.append(_run(16, inst, "Amargin", seed, -5.0, 1e-3))
            # L-BFGS reaches a lower expectation on every run, same P_tau.
            ctrl.append(_run(16, inst, "Amargin", seed, -5.01, 1e-3, opt="lbfgs_100grad",
                             steps="100", grad="60", by50="-5.0"))
    groups = {analyze_e3.BASELINE_LABEL: analyze_e3._normalise_baseline(base),
              "adam_0.03": [], "lbfgs_100grad": ctrl}
    rows = {r["field"]: r for r in analyze_e3.contrasts(groups)
            if r["treatment"] == "lbfgs_100grad" and r["penalty_label"] == "Amargin"}
    e = rows["best_expectation"]
    assert e["pairs"] == 6 and e["instances"] == 3
    assert abs(e["instance_median_diff"] + 0.01) < 1e-12
    assert e["direction"] == "better" and e["instances_treatment_better"] == 3
    assert rows["final_P_tau_main"]["direction"] == "equal"
    # The extra budget is reported as "worse", never hidden.
    assert rows["gradient_evaluations"]["instance_median_diff"] == 10
    assert rows["gradient_evaluations"]["direction"] == "worse"


def test_multistart_sums_the_budget_over_seeds():
    rows = [_run(16, 0, "Aheur", s, -5.0, p, is_optimal=o, seconds="17.0")
            for s, p, o in ((42, 1e-3, "False"), (43, 5e-3, "True"), (44, 2e-3, "False"))]
    [m] = analyze_e3.multistart(rows)
    assert m["seeds_per_instance"] == 3
    assert m["single_seed_P_tau_median"] == 2e-3 and m["best_of_seeds_P_tau_median"] == 5e-3
    assert m["single_seed_optimal_rate"] == pytest.approx(1 / 3) and m["any_seed_optimal_rate"] == 1.0
    assert m["single_seed_optimisation_seconds_median"] == 17.0
    assert m["all_seeds_optimisation_seconds_median"] == 51.0
    assert m["all_seeds_shots_median"] == 3000.0


def test_multistart_compares_two_runnable_strategies_at_equal_shots():
    """Random single start (Q of each seed, then the mean) against pooled
    batches, per instance; the median seed's state is not a strategy."""
    p = {42: 1e-4, 43: 5e-3, 44: 2e-3}          # skewed over seeds on purpose
    rows = [_run(16, 0, "Aheur", s, -5.0, q, seconds="17.0") for s, q in p.items()]
    [m] = analyze_e3.multistart(rows)
    shots = 1000 * len(p)
    q_single = np.mean([1 - (1 - q) ** shots for q in p.values()])
    q_pooled = 1 - np.prod([(1 - q) ** 1000 for q in p.values()])
    assert m["random_seed_Q_tau_5000_median"] == pytest.approx(q_single, rel=1e-9)
    assert m["pooled_5x1000_Q_tau_median"] == pytest.approx(q_pooled, rel=1e-9)
    assert m["pooled_minus_random_Q_tau_median"] == pytest.approx(q_pooled - q_single, rel=1e-9)
    # Q first, then the mean: the mean p put into Q is a different number.
    q_of_mean_p = 1 - (1 - np.mean(list(p.values()))) ** shots
    assert abs(m["random_seed_Q_tau_5000_median"] - q_of_mean_p) > 1e-3
    # ... and so is the median seed's state, which is carried under its own name.
    assert m["median_seed_Q_tau_5000_median"] == pytest.approx(1 - (1 - 2e-3) ** shots, rel=1e-9)
    assert m["median_seed_Q_tau_5000_median"] != m["random_seed_Q_tau_5000_median"]
    # AM–GM: pooling never loses at equal shots, so the count is of material gains.
    assert m["pooled_minus_random_Q_tau_median"] >= 0
    assert m["instances_pooled_gains_over_0.01_Q_tau"] == 1
    assert m["extra_optimisation_seconds_median"] == pytest.approx(34.0)
    # One instance: the bootstrap CI is undefined, not fabricated.
    assert np.isnan(m["pooled_minus_random_Q_tau_ci95_low"])


def test_multistart_gain_is_zero_when_every_seed_is_the_same():
    rows = [_run(12, i, "Amargin", s, -5.0, 3e-3) for i in (0, 1) for s in (42, 43)]
    [m] = analyze_e3.multistart(rows)
    # Equal p on every seed: mean_k(a^2) = a^2 = prod_k a, the AM–GM bound is tight.
    assert m["pooled_minus_random_Q_tau_median"] == pytest.approx(0.0, abs=1e-12)
    assert m["instances_pooled_gains_over_0.01_Q_tau"] == 0


def test_summary_predicts_the_batch_counts_from_the_recorded_states():
    """The i.i.d. check that exposed the L-BFGS-B handoff defect.

    A batch drawn at the recorded final angles hits the optimum in about
    Σ_j [1 − (1 − P_0,j)^S] runs and finds no feasible shot in about
    Σ_j (1 − P_F,j)^S; a batch drawn somewhere else does not. The summary
    prints both expectations beside the observed counts so the comparison is
    a table cell rather than a claim.
    """
    rows = []
    for inst, p_opt, p_f, hit in ((0, 0.005, 0.5, "True"), (1, 1e-6, 1e-4, "False"),
                                  (2, 0.002, 0.3, "True")):
        r = _run(16, inst, "Aheur", 42, -5.0, p_opt, is_optimal=hit)
        r["final_P_F"] = str(p_f)
        rows.append(r)
    [s] = analyze_e3.summarise({analyze_e3.BASELINE_LABEL: analyze_e3._normalise_baseline(rows),
                               "adam_0.03": [], "lbfgs_100grad": []})
    assert s["optimum_hits"] == 2
    exp_hit = sum(1 - (1 - p) ** 1000 for p in (0.005, 1e-6, 0.002))
    exp_nf = sum((1 - p) ** 1000 for p in (0.5, 1e-4, 0.3))
    assert abs(s["expected_optimum_hits"] - exp_hit) < 1e-12
    assert abs(s["expected_no_feasible_shot_runs"] - exp_nf) < 1e-12
    assert 0.9 < s["expected_no_feasible_shot_runs"] < 1.0   # the P_F = 1e-4 run
