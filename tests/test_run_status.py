# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT
"""The job status record: written beside the manifest, read before aggregating.

Origin: 2026-09-16, a two-hour harness crashed in its first minute, the shell
script around it printed its completion marker, and the failure went unnoticed
for nine hours. The end state of a job has to be a written fact.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from src import run_status


def test_lifecycle_running_then_exit(tmp_path):
    out = tmp_path / "paper01_e1_cross_x.csv"
    st = run_status.JobStatus(out, total=3, spec_version="e1-v2", command=["cmd"])
    rec = run_status.read(out)
    assert rec["state"] == "RUNNING" and rec["done"] == 0 and rec["total"] == 3
    st.heartbeat(2, last="N12_i00_Aheur_s43")
    assert run_status.read(out)["last"] == "N12_i00_Aheur_s43"
    st.finish(0)
    rec = run_status.read(out)
    assert rec["state"] == "EXIT" and rec["exit_code"] == 0
    assert "EXIT 0" in run_status.describe(rec)


def test_failure_and_stall_are_described(tmp_path):
    out = tmp_path / "m.csv"
    st = run_status.JobStatus(out, total=10)
    st.heartbeat(4, last="r4")
    st.finish(1)
    assert run_status.describe(run_status.read(out)).startswith("FAILED (exit 1), 4/10")
    st2 = run_status.JobStatus(tmp_path / "n.csv", total=10)
    st2.heartbeat(6)
    rec = run_status.read(tmp_path / "n.csv")
    later = datetime.now(timezone.utc) + timedelta(hours=2)
    assert "STALLED" in run_status.describe(rec, now=later)
    assert "STALLED" not in run_status.describe(rec)


def test_require_complete_refuses_running_failed_and_partial(tmp_path):
    out = tmp_path / "m.csv"
    st = run_status.JobStatus(out, total=10)
    st.heartbeat(6)
    with pytest.raises(RuntimeError, match="not complete"):
        run_status.require_complete(out)
    assert run_status.require_complete(out, allow_partial=True).startswith("RUNNING")
    st.finish(1)
    with pytest.raises(RuntimeError, match="FAILED"):
        run_status.require_complete(out)
    st.finish(0)  # EXIT 0 but done != total
    with pytest.raises(RuntimeError, match="not complete"):
        run_status.require_complete(out)
    st.heartbeat(10)
    st.finish(0)
    assert run_status.require_complete(out).startswith("EXIT 0, 10/10")


def test_a_manifest_without_a_record_is_let_through_with_a_note(tmp_path):
    text = run_status.require_complete(tmp_path / "legacy.csv")
    assert "predates" in text


def test_the_record_is_written_atomically(tmp_path):
    out = tmp_path / "m.csv"
    run_status.JobStatus(out, total=1)
    p = run_status.status_path(out)
    assert p.exists() and not p.with_suffix(".json.tmp").exists()
    json.loads(p.read_text(encoding="utf-8"))


def _manifest(path, run_ids):
    """A manifest with the one column the status check reads."""
    path.write_text("run_id,value\n" + "".join(f"{r},1\n" for r in run_ids), encoding="utf-8")


def test_a_complete_record_beside_the_file_it_describes_is_let_through(tmp_path):
    out = tmp_path / "paper01_e1_cross_x.csv"
    _manifest(out, [f"r{i}" for i in range(30)])
    st = run_status.JobStatus(out, total=30)
    st.heartbeat(30, last="r29")
    st.finish(0)
    assert run_status.require_complete(out).startswith("EXIT 0, 30/30")
    assert run_status.disagreements(out, run_status.read(out)) == []


def test_a_row_deleted_after_the_job_ended_stops_the_aggregation(tmp_path):
    """The review case: EXIT 0, 30/30 beside a file holding twenty-nine rows."""
    out = tmp_path / "paper01_e1_cross_x.csv"
    ids = [f"r{i}" for i in range(30)]
    _manifest(out, ids)
    st = run_status.JobStatus(out, total=30)
    st.heartbeat(30, last="r29")
    st.finish(0)
    _manifest(out, ids[1:])                      # one row gone, record untouched
    with pytest.raises(RuntimeError, match="30 row.* and the file holds 29"):
        run_status.require_complete(out)
    # --allow-partial is for a run known to be unfinished, not for a file that
    # is not the one the record describes.
    with pytest.raises(RuntimeError, match="do not describe the same run"):
        run_status.require_complete(out, allow_partial=True)


def test_a_duplicated_row_stops_the_aggregation(tmp_path):
    out = tmp_path / "paper01_e1_cross_x.csv"
    ids = [f"r{i}" for i in range(30)]
    _manifest(out, ids)
    st = run_status.JobStatus(out, total=30)
    st.heartbeat(30, last="r29")
    st.finish(0)
    _manifest(out, ids + ["r0"])                 # thirty-one rows, one of them twice
    with pytest.raises(RuntimeError, match="appear more than once"):
        run_status.require_complete(out)


def test_a_last_run_the_file_does_not_hold_stops_the_aggregation(tmp_path):
    out = tmp_path / "paper01_e1_cross_x.csv"
    _manifest(out, ["r0", "r1"])
    st = run_status.JobStatus(out, total=3)
    st.heartbeat(3, last="r2")                   # the row r2 never landed
    st.finish(0)
    with pytest.raises(RuntimeError, match="not in the file"):
        run_status.require_complete(out)


def test_a_manifest_without_run_ids_is_not_second_guessed(tmp_path):
    """Nothing to compare: the check says nothing rather than inventing a rule."""
    out = tmp_path / "other.csv"
    out.write_text("a,b\n1,2\n", encoding="utf-8")
    st = run_status.JobStatus(out, total=1)
    st.heartbeat(1)
    st.finish(0)
    assert run_status.disagreements(out, run_status.read(out)) == []
    assert run_status.require_complete(out).startswith("EXIT 0")


def test_a_kill_between_the_write_and_the_beat_is_not_a_disagreement(tmp_path):
    """The driver writes the row, then beats; a kill in between is that gap."""
    out = tmp_path / "paper01_e1_cross_x.csv"
    st = run_status.JobStatus(out, total=30)
    st.heartbeat(5, last="r4")
    _manifest(out, [f"r{i}" for i in range(6)])   # the sixth row landed, no beat followed
    assert run_status.disagreements(out, run_status.read(out)) == []
    # Still refused, but for being unfinished -- which --allow-partial answers.
    with pytest.raises(RuntimeError, match="not complete"):
        run_status.require_complete(out)
    assert run_status.require_complete(out, allow_partial=True).startswith("RUNNING")
    # Two rows ahead is not a kill between a write and a beat.
    _manifest(out, [f"r{i}" for i in range(7)])
    assert run_status.disagreements(out, run_status.read(out))


def test_the_failure_count_is_recorded_and_printed(tmp_path):
    out = tmp_path / "m.csv"
    st = run_status.JobStatus(out, total=4)
    st.heartbeat(4, last="r3", failed=2)
    st.finish(0)
    rec = run_status.read(out)
    assert rec["failed"] == 2
    assert run_status.describe(rec).startswith("EXIT 0, 4/4 (2 failed)")
    st2 = run_status.JobStatus(tmp_path / "n.csv", total=2)
    st2.heartbeat(2, last="r1")
    st2.finish(0)
    assert "failed" not in run_status.describe(run_status.read(tmp_path / "n.csv"))
