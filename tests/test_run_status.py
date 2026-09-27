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
