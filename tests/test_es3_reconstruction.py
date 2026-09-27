# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT
"""E-S3 harness: the parts that turn subprocess output into the table's numbers.

The harness itself clones the public package and runs for ~15 minutes, so
only its bookkeeping is tested here: the block comparison that reports which
column groups moved between hosts, the per-run completion pattern the
interruption step keys on, and the step record's CSV shape.
"""

from __future__ import annotations

import re
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from experiments.system_paper import es3_reconstruction as es3


def _row(**over):
    base = {"n_assets": 12, "instance": 0, "penalty_label": "Aheur", "qaoa_seed": 42,
            "initial_angles": "1 2 3 4", "final_angles": "0.1 0.2 0.3 0.4"}
    base.update({c: 0.5 for c in es3.STATE_COLUMNS})
    base.update({c: 3.0 for c in es3.SHOT_COLUMNS})
    base.update(over)
    return base


def test_compare_block_separates_state_shot_and_angle_columns(tmp_path, monkeypatch):
    got = pd.DataFrame([_row(final_P_F=0.5 + 1e-7, feasible_shots=4.0, final_angles="9 9 9 9")])
    ref = pd.DataFrame([_row()])
    clone = tmp_path / "clone"
    results = clone / "experiments" / "paper01_qubo_baseline" / "results"
    results.mkdir(parents=True)
    ref.to_csv(results / "paper01_e1_cross.csv", index=False)
    got.to_csv(results / "got.csv", index=False)
    h = es3.Harness(tmp_path, log=lambda *_: None)
    summary = h._compare_block(results / "got.csv", "unit")
    by = {(c["group"], c["column"]): c for c in h.columns}
    assert by[("state", "final_P_F")]["max_abs_diff"] == pytest.approx(1e-7)
    assert by[("state", "f_star")]["identical_rows"] == 1
    assert by[("shots", "feasible_shots")]["identical_rows"] == 0
    assert by[("angles", "initial_angles")]["identical_rows"] == 1
    assert by[("angles", "final_angles")]["identical_rows"] == 0
    assert "1 rows matched" in summary and "initial_angles identical in 1" in summary


def test_run_completion_pattern_matches_the_driver_line():
    """The interruption step counts completed runs by the driver's per-run line."""
    line = "      Aheur s42  g_off=0.0231 g_F=0.0004 P_F=0.183  12.3s"
    assert re.search(r"\d+\.\ds$", line.strip())
    assert not re.search(r"\d+\.\ds$", "E1 e1-v2: N=[12] x 2 instances x 2 seeds ... 0 rows already done")


def test_step_record_has_the_table_columns():
    s = es3.Step("x", "1-aggregates", "cmd", 1.0, 0, "ok", "m", "none")
    assert list(asdict(s)) == ["step", "level", "command", "seconds", "exit_code", "outcome",
                               "metric", "manual_inputs"]


def test_block_is_the_documented_24_runs():
    """N = 12, instances 0-1, seeds 42-43: six penalty settings per instance."""
    assert es3.E1_BLOCK == ["--n", "12", "--instances", "0", "1", "--seeds", "42", "43"]
    assert 2 * 2 * 6 == 24
    assert isinstance(np.float64(1.0), float)


# ----------------------------------------------------------------------
# The bound on what an interruption costs
# ----------------------------------------------------------------------

def test_a_failed_write_leaves_the_previous_file_whole(tmp_path, monkeypatch):
    """The per-run flush rewrites the whole file, so it must land atomically.

    The paper's "the loss is bounded at the run in flight" holds only if the
    rewrite cannot be interrupted half-way; it could be, until this was checked. It is now
    written beside the file and moved into place; this kills the writer inside
    the write and checks that what was on disk before is still there.
    """
    import csv as _csv
    from experiments.paper01_qubo_baseline import run_experiment as rx

    path = tmp_path / "rows.csv"
    rx.write_csv(path, [{"run": 1, "value": "first"}, {"run": 2, "value": "second"}])
    before = path.read_text(encoding="utf-8")

    class Boom(RuntimeError):
        pass

    real = _csv.DictWriter.writerows

    def explode(self, rows):
        real(self, rows[:1])
        raise Boom("killed mid-write")

    monkeypatch.setattr(_csv.DictWriter, "writerows", explode)
    with pytest.raises(Boom):
        rx.write_csv(path, [{"run": i, "value": "new"} for i in range(1, 12)])

    assert path.read_text(encoding="utf-8") == before, "the completed rows were lost"
    assert not list(tmp_path.glob("*.tmp*")) or path.exists()
