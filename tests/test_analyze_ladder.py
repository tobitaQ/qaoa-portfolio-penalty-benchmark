# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""The ladder is the same comparison as E2, at the five weights E2 skipped.

What has to hold is the join and the counting rule. Every run is scored against
*its own* instance's floor: scoring a run against another instance's reference
row is the fault that once turned 0 of 900 into 7 of 900, so the first test
makes the two instances' floors differ and checks that swapping them changes
the answer. The count is of runs strictly above the floor, because the median
product P_F * R_tau stays below one even where 41 % of runs exceed it.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from experiments.paper01_qubo_baseline import analyze_ladder as al


def _write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _fixture(tmp_path: Path, floors=(0.01, 0.10)) -> Path:
    """Two instances at one size and two weights, with different floors."""
    runs = []
    for instance, p_tau in ((0, 0.02), (1, 0.02)):          # the same P_tau ...
        for label in ("Aheur", "1.1xAcrit"):
            runs.append({"n_assets": 12, "instance": instance, "penalty_label": label,
                         "qaoa_seed": 42, "status": "ok", "final_P_F": 0.2,
                         "final_P_tau_main": p_tau, "final_Q_tau_main_1000": 0.5,
                         "g_F": 0.01, "g_off": 0.001, "best_all_feasible": "True"})
    _write(tmp_path / al.CROSS, runs)
    _write(tmp_path / al.REFERENCE,
           [{"n_assets": 12, "instance": 0, "P_tau_main_uniform_F": floors[0]},
            {"n_assets": 12, "instance": 1, "P_tau_main_uniform_F": floors[1]}])
    return tmp_path


def test_each_run_is_scored_against_its_own_instance_floor(tmp_path):
    rows = al.ladder(_fixture(tmp_path))
    # P_tau = 0.02 is above instance 0's floor of 0.01 and below instance 1's
    # 0.10, so exactly one of the two runs in each cell crosses.
    assert {r["penalty_label"]: r["runs_above_floor"] for r in rows} == {"1.1xAcrit": 1, "Aheur": 1}
    swapped = al.ladder(_fixture(tmp_path, floors=(0.10, 0.10)))
    assert {r["runs_above_floor"] for r in swapped} == {0}


def test_r_tau_is_the_conditional_rate_over_the_floors_own_rate(tmp_path):
    rows = al.ladder(_fixture(tmp_path, floors=(0.05, 0.05)))
    # R_tau = (P_tau / P_F) / u_tau = (0.02 / 0.2) / 0.05 = 2, and the ratio to
    # the floor is P_F * R_tau = 0.4: a sampler can prefer good solutions inside
    # F and still sit below a draw that is always feasible.
    for r in rows:
        assert r["R_tau_median"] == pytest.approx(2.0)
        assert r["P_tau_over_floor_median"] == pytest.approx(0.4)
        assert r["P_tau_over_floor_median"] == pytest.approx(r["P_F_median"] * r["R_tau_median"])


def test_the_ladder_reads_weakest_weight_first_and_keeps_every_cell(tmp_path):
    rows = al.ladder(_fixture(tmp_path))
    assert [r["penalty_label"] for r in rows] == ["1.1xAcrit", "Aheur"]
    assert all(r["runs"] == 2 and r["instances"] == 2 for r in rows)


def test_failed_runs_and_batches_without_a_feasible_shot_are_counted_not_dropped(tmp_path):
    d = _fixture(tmp_path)
    runs = list(csv.DictReader(open(d / al.CROSS, newline="", encoding="utf-8")))
    runs[0]["status"] = "failed"
    runs[1]["g_F"] = ""                       # no feasible shot in the batch
    runs[1]["best_all_feasible"] = "False"
    _write(d / al.CROSS, runs)
    rows = {r["penalty_label"]: r for r in al.ladder(d)}
    assert rows["Aheur"]["runs"] == 1         # the failed run is not scored
    assert rows["1.1xAcrit"]["no_feasible_shot_runs"] == 1
    assert rows["1.1xAcrit"]["infeasible_batch_minimum_runs"] == 1


def test_the_committed_ladder_matches_what_the_script_computes():
    """The CSV in the repository is the script's output, not an edited copy.

    Every row the file carries has to equal what the script computes now. A
    block still being measured is allowed to be *missing* from the file -- the
    run writes its own manifest first and the ladder is regenerated when it
    ends -- but a row that is there and disagrees means the file was edited or
    the inputs moved, and the paper's Table III is checked against this file
    cell by cell, so a missing row cannot reach the manuscript unnoticed.
    """
    out = al.RESULTS_DIR / al.OUT
    if not out.exists():                                     # pragma: no cover
        pytest.skip("ladder not generated on this host")
    committed = list(csv.DictReader(open(out, newline="", encoding="utf-8")))
    fresh = {(r["n_assets"], r["penalty_label"]): r for r in al.ladder(al.RESULTS_DIR)}
    for c in committed:
        f = fresh[(int(c["n_assets"]), c["penalty_label"])]
        assert int(c["runs_above_floor"]) == f["runs_above_floor"]
        assert int(c["runs"]) == f["runs"]
        assert float(c["R_tau_median"]) == pytest.approx(f["R_tau_median"], rel=1e-9)
        assert float(c["P_tau_over_floor_median"]) == pytest.approx(f["P_tau_over_floor_median"], rel=1e-9)


def test_the_xy_block_joins_the_ladder_as_one_row_per_size(tmp_path):
    """E6 rows are scored against E6's own reference table, like every block.

    Its eps_F is measured at the A = 0 scales, over the instances the block
    ran, and is not 1: with no penalty, s_H is the objective's own largest
    Ising coefficient, and the objective's feasible range can be wider than it.
    """
    runs = [{"n_assets": 16, "instance": i, "mixer": "xy_ring", "qaoa_seed": 42,
             "status": "ok", "final_P_F": 1.0, "final_P_tau_main": 0.08,
             "final_Q_tau_main_1000": 1.0, "g_F": 0.0, "g_off": 0.0,
             "best_all_feasible": "True"} for i in (0, 1)]
    _write(tmp_path / al.XY_CROSS, runs)
    _write(tmp_path / al.XY_REFERENCE,
           [{"n_assets": 16, "instance": 0, "P_tau_main_uniform_F": 0.02},
            {"n_assets": 16, "instance": 1, "P_tau_main_uniform_F": 0.04}])
    _write(tmp_path / al.SCALES,
           [{"n_assets": 16, "instance": 0, "penalty_label": "A0", "epsilon_F": 1.5,
             "ising_ratio_objective_over_s_H": 1.0},
            {"n_assets": 16, "instance": 1, "penalty_label": "A0", "epsilon_F": 2.5,
             "ising_ratio_objective_over_s_H": 1.0},
            {"n_assets": 16, "instance": 2, "penalty_label": "A0", "epsilon_F": 99.0,
             "ising_ratio_objective_over_s_H": 1.0}])
    rows = al.xy_rows(tmp_path)
    assert len(rows) == 1
    r = rows[0]
    assert r["penalty_label"] == "XY" and r["runs"] == 2 and r["n_assets"] == 16
    # P_F = 1, so R_tau and the ratio to the floor are the same number, and the
    # median is over the two instances' own floors: 0.08/0.02 = 4, 0.08/0.04 = 2.
    assert r["P_tau_over_floor_median"] == pytest.approx(3.0)
    assert r["R_tau_median"] == pytest.approx(3.0)
    assert r["runs_above_floor"] == 2
    # The median is over the two instances that ran, not the three the scale
    # file holds: a cell's eps_F describes the problems its runs used.
    assert r["epsilon_F_median"] == pytest.approx(2.0)


def test_a_cell_without_a_scale_row_reports_no_epsilon_rather_than_one(tmp_path):
    """A missing scale file leaves eps_F undefined; it never defaults to 1."""
    import math
    rows = al.ladder(_fixture(tmp_path))
    assert all(math.isnan(r["epsilon_F_median"]) for r in rows)


def test_the_ladder_is_the_penalty_block_alone_when_e6_has_not_run(tmp_path):
    assert al.xy_rows(tmp_path) == []
    rows = al.ladder(_fixture(tmp_path))
    assert {r["penalty_label"] for r in rows} == {"Aheur", "1.1xAcrit"}
