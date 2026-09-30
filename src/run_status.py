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

A status record is itself a claim about a file, and a claim nothing compares
to its subject is the inference this module exists to remove. So
``require_complete`` also reads the manifest and refuses when the two
disagree -- the row count against ``done``, a run id that appears twice, a
``last`` the file does not hold. Without that, ``EXIT 0, 30/30`` beside a
twenty-nine-row file aggregated without a word.
"""

from __future__ import annotations

import csv
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
            "done": 0, "failed": 0, "total": total, "last": None, "spec_version": spec_version,
            "command": command if command is not None else sys.argv,
        }
        self._write()

    def _write(self) -> None:
        self.record["updated_utc"] = _now()
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.record, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def heartbeat(self, done: int, last: str | None = None, total: int | None = None,
                  failed: int | None = None) -> None:
        """``done`` rows are on disk, ``failed`` of them recording a failure."""
        self.record["done"] = done
        if last is not None:
            self.record["last"] = last
        if total is not None:
            self.record["total"] = total
        if failed is not None:
            self.record["failed"] = failed
        self._write()

    def finish(self, exit_code: int = 0) -> None:
        """How the *driver* ended, not how its runs went.

        ``EXIT 0`` says the loop reached its end and every planned run was
        attempted and recorded. Runs that raised are recorded as failures and
        counted in ``failed``; they do not make the job ``FAILED``, which is
        reserved for the driver itself stopping. ``describe()`` prints the
        failure count so that the two are never read as one.
        """
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
    # A job that attempted everything and recorded failures is EXIT 0 with a
    # non-zero count here. Printing the count beside the progress is what keeps
    # "ran to the end" from being read as "every run succeeded".
    if record.get("failed"):
        progress += f" ({record['failed']} failed)"
    if state == "RUNNING":
        updated = datetime.fromisoformat(record["updated_utc"])
        age = ((now or datetime.now(timezone.utc)) - updated).total_seconds()
        stale = f", heartbeat {age / 60:.0f} min old (STALLED)" if age > STALE_SECONDS else ""
        return f"RUNNING pid {record.get('pid')} on {record.get('host')}, {progress}{stale}"
    if state == "EXIT":
        return f"EXIT 0, {progress}, finished {record.get('updated_utc')}"
    return f"FAILED (exit {record.get('exit_code')}), {progress}, at {record.get('last')}"


def manifest_rows(output: Path) -> tuple[int, list[str]] | None:
    """The data rows of a CSV manifest and the run ids in them, or ``None``.

    ``None`` when there is no file or it carries no ``run_id`` column; the
    caller then has nothing to compare the status record against.
    """
    if not output.exists():
        return None
    with output.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "run_id" not in reader.fieldnames:
            return None
        rows = list(reader)
    # A manifest emptied to its header is zero rows, not "nothing to compare":
    # the header is what says this is a manifest.
    return len(rows), [r["run_id"] for r in rows]


def disagreements(output: Path, record: dict | None) -> list[str]:
    """Where the status record and the file beside it do not describe the same run.

    A status record is a claim about a file, and until something compares the
    two the claim is the same inference the record was written to replace: a
    review of this module deleted one row from a thirty-row manifest, left
    ``EXIT 0, 30/30`` in place, and the aggregation printed ``30/30`` and
    ``done 29`` on adjacent lines and published the medians of twenty-nine.
    """
    if record is None:
        return []
    seen = manifest_rows(output)
    if seen is None:
        return []
    n_rows, ids = seen
    out = []
    done = record.get("done")
    ended = record.get("state") == "EXIT" and record.get("exit_code") == 0
    if done is not None:
        # The driver writes the row and then beats: a job killed between the
        # two leaves the file one row ahead of the record, and that lag is the
        # design working, not a disagreement. A job that ended normally beat
        # once more on its way out, so there the two must match exactly.
        lag = n_rows - done
        if lag < 0 or (lag > 0 and ended) or lag > 1:
            out.append(f"the record says {done} row(s) were written and the file holds {n_rows}")
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        out.append(f"{len(duplicates)} run id(s) appear more than once, first {duplicates[0]}")
    last = record.get("last")
    if last is not None and last not in ids:
        out.append(f"the last run the record names, {last}, is not in the file")
    return out


def require_complete(output: Path, allow_partial: bool = False) -> str:
    """Refuse to aggregate a manifest whose job did not end complete.

    Returns the description; raises ``RuntimeError`` when the job is RUNNING,
    FAILED, or EXIT with ``done != total``, unless ``allow_partial`` -- and,
    whatever ``allow_partial`` says, when the record and the file disagree
    about what was written. ``allow_partial`` is for aggregating a run that is
    knowingly unfinished; it is not a licence to aggregate a file that is not
    the one the record describes.
    """
    record = read(output)
    text = describe(record)
    if record is None:
        return text
    off = disagreements(output, record)
    if off:
        raise RuntimeError(
            f"{output.name}: its status record and the file do not describe the same run -- "
            + "; ".join(off) + ". The record says the job ended; the file says otherwise, "
            f"and which of the two is right is not something this check can decide. "
            f"Re-run, or resume, rather than aggregating either.")
    complete = (record.get("state") == "EXIT" and record.get("exit_code") == 0
                and (record.get("total") is None or record.get("done") == record.get("total")))
    if not complete and not allow_partial:
        raise RuntimeError(
            f"{output.name}: the job that wrote this manifest is not complete -- {text}. "
            f"Aggregating it would report a partial cross as if it were whole; pass "
            f"--allow-partial to do so knowingly, or resume the run.")
    return text
