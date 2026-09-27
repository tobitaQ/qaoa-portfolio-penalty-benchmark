# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT
"""
The end state of a job as a written record beside its output.

A driver that flushes after every run bounds the loss on interruption to one
run; it does not say how the job ended, and a reader (or the author) who
aggregates a manifest infers that from the log — "it printed done", "the
file is there" — which is exactly the inference the record is meant to
replace. Found the hard way on 2026-09-16: a two-hour harness crashed in its
first minute inside a shell script without ``set -e``, the script printed its
completion marker, and the failure went unnoticed for nine hours.

``JobStatus`` writes a one-object JSON file next to the output it describes
(``<output>.status.json``), atomically, on every change:

    {"state": "RUNNING" | "EXIT" | "FAILED", "exit_code": null | int,
     "pid": ..., "host": ..., "started_utc": ..., "updated_utc": ...,
     "done": <rows written>, "total": <rows planned>, "last": <last run id>,
     "spec_version": ..., "command": [...]}

A job killed by a signal leaves ``RUNNING`` with a stale ``updated_utc``; that
staleness is the stall, and ``describe()`` says so. ``require_complete()`` is
what an aggregation calls before it reads a manifest: it refuses (with the
state written in the record) unless the job ended with ``EXIT 0`` and
``done == total``, or the caller passes ``allow_partial``. A manifest without
a status file predates this module; ``require_complete`` says that too and
lets it through, because the committed E1 record is one.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

STALE_SECONDS = 1800


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def status_path(output: Path) -> Path:
    return Path(str(output) + ".status.json")


class JobStatus:
    """Status record for one driver invocation writing ``output``."""

    def __init__(self, output: Path, total: int | None = None, spec_version: str = "",
                 command: list[str] | None = None) -> None:
        self.path = status_path(output)
        self.record = {
            "state": "RUNNING", "exit_code": None, "pid": os.getpid(),
            "host": socket.gethostname(), "started_utc": _now(), "updated_utc": _now(),
            "done": 0, "total": total, "last": None, "spec_version": spec_version,
            "command": command if command is not None else sys.argv,
        }
        self._write()

    def _write(self) -> None:
        self.record["updated_utc"] = _now()
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.record, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def heartbeat(self, done: int, last: str | None = None, total: int | None = None) -> None:
        self.record["done"] = done
        if last is not None:
            self.record["last"] = last
        if total is not None:
            self.record["total"] = total
        self._write()

    def finish(self, exit_code: int = 0) -> None:
        self.record["state"] = "EXIT" if exit_code == 0 else "FAILED"
        self.record["exit_code"] = exit_code
        self._write()


def read(output: Path) -> dict | None:
    p = status_path(output)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def describe(record: dict | None, now: datetime | None = None) -> str:
    """One line a human or a log can read: state, progress, staleness."""
    if record is None:
        return "no status record (the manifest predates status files)"
    state = record.get("state")
    done, total = record.get("done"), record.get("total")
    progress = f"{done}/{total}" if total else f"{done} rows"
    if state == "RUNNING":
        updated = datetime.fromisoformat(record["updated_utc"])
        age = ((now or datetime.now(timezone.utc)) - updated).total_seconds()
        stale = f", heartbeat {age / 60:.0f} min old (STALLED)" if age > STALE_SECONDS else ""
        return f"RUNNING pid {record.get('pid')} on {record.get('host')}, {progress}{stale}"
    if state == "EXIT":
        return f"EXIT 0, {progress}, finished {record.get('updated_utc')}"
    return f"FAILED (exit {record.get('exit_code')}), {progress}, at {record.get('last')}"


def require_complete(output: Path, allow_partial: bool = False) -> str:
    """Refuse to aggregate a manifest whose job did not end complete.

    Returns the description; raises ``RuntimeError`` when the job is RUNNING,
    FAILED, or EXIT with ``done != total``, unless ``allow_partial``.
    """
    record = read(output)
    text = describe(record)
    if record is None:
        return text
    complete = (record.get("state") == "EXIT" and record.get("exit_code") == 0
                and (record.get("total") is None or record.get("done") == record.get("total")))
    if not complete and not allow_partial:
        raise RuntimeError(
            f"{output.name}: the job that wrote this manifest is not complete -- {text}. "
            f"Aggregating it would report a partial cross as if it were whole; pass "
            f"--allow-partial to do so knowingly, or resume the run.")
    return text
