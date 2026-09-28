# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT
"""The aggregation refuses a record that fails a relation, and the guards around it.

A review of the drivers (2026-09-28) showed three ways a record could reach a
table without anything objecting: ``analyze_e1`` checked the job's exit status
and not the record; ``--results-tag ""`` reached the canonical file names past
the overwrite guard; and ``--resume`` skipped cells by key without asking
whether the manifest had been written under the same arguments. Each is pinned
here as "a correct record passes, an altered one stops".
"""

from __future__ import annotations

import shutil

import pandas as pd
import pytest

from experiments.paper01_qubo_baseline import analyze_e1, run_e1_cross
from experiments.paper01_qubo_baseline.run_experiment import checked_tag, results_paths
from src import record_checks

E1 = run_e1_cross.RESULTS_DIR / "paper01_e1_cross.csv"
REF = run_e1_cross.RESULTS_DIR / "paper01_e1_reference.csv"
pytestmark = pytest.mark.skipif(not E1.exists(), reason="the committed E1 record is not present")


@pytest.fixture(scope="module")
def block():
    """One instance of the committed record: the smallest whole thing to check."""
    rows = pd.read_csv(E1)
    refs = pd.read_csv(REF)
    rows = rows[(rows.n_assets == 12) & (rows.instance == 0)]
    refs = refs[(refs.n_assets == 12) & (refs.instance == 0)]
    assert len(rows) > 0 and len(refs) == 1
    return rows.reset_index(drop=True), refs.reset_index(drop=True)


def test_a_correct_record_passes_every_relation(block):
    rows, refs = block
    assert record_checks.check_record(rows, refs) == {}


@pytest.mark.parametrize("column,value,expected", [
    ("final_P_F", 0.99, "state-batch"),      # the review's own example
    ("penalty", 1.5, "penalty"),
    ("f_star", -0.5, "reference-row"),
    ("feasible_shots", 99999, "ranges"),
])
def test_one_edited_cell_stops_the_record(block, column, value, expected):
    rows, refs = block
    bad = rows.copy()
    bad.loc[0, column] = value
    found = record_checks.check_record(bad, refs)
    assert expected in found, (column, sorted(found))


def _stage(tmp_path, monkeypatch, rows, refs, tag="gate"):
    monkeypatch.setattr(analyze_e1, "RESULTS_DIR", tmp_path)
    rows.to_csv(tmp_path / f"paper01_e1_cross_{tag}.csv", index=False)
    refs.to_csv(tmp_path / f"paper01_e1_reference_{tag}.csv", index=False)
    return ["--results-tag", tag, "--allow-partial"]


def test_the_aggregation_runs_on_a_correct_record(tmp_path, monkeypatch, block, capsys):
    rows, refs = block
    argv = _stage(tmp_path, monkeypatch, rows, refs)
    assert analyze_e1.main(argv) == 0
    assert "none failing" in capsys.readouterr().out


def test_the_aggregation_refuses_an_edited_record(tmp_path, monkeypatch, block, capsys):
    rows, refs = block
    bad = rows.copy()
    bad.loc[0, "final_P_F"] = 0.99
    argv = _stage(tmp_path, monkeypatch, bad, refs)
    assert analyze_e1.main(argv) == 1
    err = capsys.readouterr().err
    assert "refusing to aggregate" in err and "state-batch" in err
    assert not (tmp_path / "paper01_e1_summary_gate.csv").exists(), "it wrote a summary anyway"


def test_the_refusal_can_be_overridden_deliberately(tmp_path, monkeypatch, block):
    rows, refs = block
    bad = rows.copy()
    bad.loc[0, "final_P_F"] = 0.99
    argv = _stage(tmp_path, monkeypatch, bad, refs) + ["--allow-inconsistent"]
    assert analyze_e1.main(argv) == 0


@pytest.mark.parametrize("tag", ["", " ", "a/b", "a b"])
def test_a_tag_that_is_not_a_filename_is_refused(tag):
    with pytest.raises(ValueError):
        checked_tag(tag)
    with pytest.raises(ValueError):
        results_paths(tag)


def test_an_empty_tag_does_not_reach_the_canonical_names(tmp_path, monkeypatch, capsys):
    """``--results-tag ""`` used to be falsy for the suffix and not-None for the guard."""
    monkeypatch.setattr(run_e1_cross, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_e1_cross, "ARTIFACT_DIR", tmp_path / "e1_runs")
    canonical = tmp_path / "paper01_e1_cross.csv"
    canonical.write_text("kept\n", encoding="utf-8")
    assert run_e1_cross.main(["--results-tag", "", "--n", "12"]) == 2
    assert canonical.read_text(encoding="utf-8") == "kept\n"
    assert "non-empty" in capsys.readouterr().err


def test_resume_accepts_the_same_conditions_and_refuses_changed_ones():
    recorded = [{"spec_version": "e1-v2", "backend": "lightning_cpu", "precision": "double",
                 "p_layers": "2", "steps": "2", "optimizer": "adam_0.1", "shots": "50",
                 "data_end": "2026-08-05"}]
    same = {"spec_version": "e1-v2", "backend": "lightning_cpu", "precision": "double",
            "p_layers": 2, "steps": 2, "optimizer": "adam_0.1", "shots": 50,
            "data_end": "2026-08-05"}
    assert run_e1_cross.resume_conflicts(recorded, same) == []
    for column, other in (("steps", 3), ("shots", 100), ("precision", "single"),
                          ("optimizer", "lbfgs_100grad"), ("data_end", "2023-08-05")):
        changed = dict(same, **{column: other})
        conflicts = run_e1_cross.resume_conflicts(recorded, changed)
        assert conflicts and conflicts[0].startswith(column), (column, conflicts)


def test_an_empty_manifest_has_no_conflicts():
    """A first run is not a resume conflict with itself."""
    assert run_e1_cross.resume_conflicts([], {"steps": 50}) == []


def test_the_two_sided_rejection_region_is_within_twice_the_threshold():
    """The threshold is applied to one tail, so a correct row fails at up to 2 x TAIL_P.

    The paper states the union bound that follows from this, not from TAIL_P
    alone; this pins the factor so the bound cannot go stale in the other
    direction.
    """
    from scipy.stats import binom
    import numpy as np
    n, p = 1000, 0.488
    x = np.arange(n + 1)
    tail = np.minimum(binom.cdf(x, n, p), binom.sf(x - 1, n, p))
    rejected = binom.pmf(x[tail < record_checks.TAIL_P], n, p).sum()
    assert record_checks.TAIL_P < rejected <= 2 * record_checks.TAIL_P
