# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""E4-A aggregation: numerics are distances between exact states, sampling is
what ten batches of one state return, and the two are never mixed."""

import numpy as np

from experiments.paper01_qubo_baseline import analyze_e4


def test_mc_batches_counts_distinct_best_feasible_per_replicate():
    rng = np.random.default_rng(0)
    # Two feasible states; the first is far more likely and has the lower f,
    # so every batch that holds any feasible shot names it as best.
    p = np.array([0.30, 0.01])
    f = np.array([-1.0, 0.0])
    out = analyze_e4._mc_batches(rng, p, f, shots=1000, n_batches=10, replicates=50)
    assert out["mc_distinct_best_median"] == 1.0
    assert out["mc_distinct_best_p975"] <= 2.0
    assert out["mc_no_feasible_batches_mean"] == 0.0


def test_mc_batches_with_no_feasible_mass_reports_all_batches_empty():
    rng = np.random.default_rng(0)
    out = analyze_e4._mc_batches(rng, np.zeros(3), np.zeros(3), shots=10, n_batches=4, replicates=5)
    assert out["mc_distinct_best_median"] == 0
    assert out["mc_no_feasible_batches_mean"] == 4.0


def test_cross_arm_reports_pooled_and_common_best_portfolios():
    def row(arm, best):
        return {"n_assets": "16", "instance": "0", "qaoa_seed": "42",
                "best_feasible_index": best}
    batches = {"gpu_double": [row("gpu_double", "7"), row("gpu_double", "9")],
               "sv1": [row("sv1", "7"), row("sv1", "11"), row("sv1", "")]}
    [c] = analyze_e4.cross_arm(batches)
    assert c["distinct_best_feasible_pooled"] == 3
    assert c["best_feasible_common_to_all_arms"] == 1
    assert c["distinct_gpu_double"] == 2 and c["distinct_sv1"] == 2
