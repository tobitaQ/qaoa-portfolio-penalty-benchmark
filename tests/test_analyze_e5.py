# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""E5 compares the E1 block with its re-run on the earlier window, row by row."""

import numpy as np

from experiments.paper01_qubo_baseline import analyze_e5


def _row(n, inst, label):
    return {"n_assets": str(n), "instance": str(inst), "penalty_label": label, "status": "ok"}


def test_block_keeps_only_the_pre_fixed_instances_and_penalties():
    rows = [_row(16, i, lab) for i in range(30) for lab in ("Aheur", "Amargin", "Acrit5")]
    kept = analyze_e5.block(rows)
    assert {int(r["instance"]) for r in kept} == set(range(10))
    assert {r["penalty_label"] for r in kept} == {"Aheur", "Amargin"}
    assert len(kept) == 20


def test_reference_stats_skip_the_zero_threshold_instances_in_the_overshoot():
    refs = [{"n_assets": "12", "instance": "0", "a_crit_is_zero": "True", "a_heur_over_crit": "",
             "delta_F": "1.0", "f_star": "-1.0", "P_tau_main_uniform_F": "0.02",
             "data_start": "2020-08-05", "data_end": "2023-08-05"},
            {"n_assets": "12", "instance": "1", "a_crit_is_zero": "False", "a_heur_over_crit": "100",
             "delta_F": "2.0", "f_star": "-1.5", "P_tau_main_uniform_F": "0.04",
             "data_start": "2020-08-05", "data_end": "2023-08-05"},
            {"n_assets": "12", "instance": "25", "a_crit_is_zero": "False", "a_heur_over_crit": "5",
             "delta_F": "9.0", "f_star": "-9.0", "P_tau_main_uniform_F": "0.9",
             "data_start": "2020-08-05", "data_end": "2023-08-05"}]
    st = analyze_e5._ref_stats(refs, 12)
    assert st["instances"] == 2                 # instance 25 is outside the block
    assert st["a_crit_zero"] == 1
    assert st["overshoot_median"] == 100        # the A_crit = 0 instance has no ratio
    assert st["delta_F_median"] == 1.5
    assert st["data_start"] == "2020-08-05"


def test_compare_places_each_window_in_its_own_column():
    ref = {"n_assets": "16", "instance": "0", "a_crit_is_zero": "False", "a_heur_over_crit": "100",
           "delta_F": "1.0", "f_star": "-1.0", "P_tau_main_uniform_F": "0.02",
           "data_start": "2023-08-06", "data_end": "2026-08-05"}
    summ = {"n_assets": "16", "penalty_label": "Aheur", "final_P_F_median": 0.2,
            "final_P_tau_main_median": 1e-3, "final_Q_tau_main_1000_median": 0.6,
            "g_off_median": 1e-4, "g_F_median": 0.02, "optimal_rate": 0.3,
            "no_feasible_shot_runs": 1, "g_off_success_but_not_tau_rate": 0.4,
            "deflation_D_median": 100.0}
    per_window = {
        "E1": {"refs": [ref], "summary": [summ], "contrasts": [], "shots": [], "decomposition": []},
        "E5": {"refs": [dict(ref, data_start="2020-08-05", data_end="2023-08-05", delta_F="2.0")],
               "summary": [dict(summ, final_Q_tau_main_1000_median=0.7)],
               "contrasts": [], "shots": [], "decomposition": []},
    }
    table = {r["quantity"]: r for r in analyze_e5.compare(per_window)}
    assert table["window"]["E1"] == "2023-08-06 – 2026-08-05"
    assert table["window"]["E5"] == "2020-08-05 – 2023-08-05"
    assert (table["Δ_F, median"]["E1"], table["Δ_F, median"]["E5"]) == (1.0, 2.0)
    assert (table["Aheur: Q_τ(1000) median"]["E1"], table["Aheur: Q_τ(1000) median"]["E5"]) == (0.6, 0.7)
    assert table["Aheur: g_off median (%)"]["E1"] == 1e-2   # written in percent
