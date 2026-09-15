# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Per-run task ledger of the SV1 series, from the archived task records.

Every Braket task writes a ``results.json`` whose ``taskMetadata`` carries the
task id, its creation time and its shot count (0 for an analytic task). The
earlier SV1 series (Sec. V-D, Supplementary Sec. S-VII) did not record task
counts per run in its result CSVs; the records themselves do. This groups the
records into runs by submission time — a gap of more than ``GAP_SECONDS``
between consecutive tasks starts a new run, since one optimisation submits a
task every few seconds — and counts analytic against sampling tasks in each.

The first run in the ledger is the N = 8 pilot of 2026-08-09: 50 analytic
tasks and 177 sampling tasks, 227 in all, which is the number Sec. VI-B
quotes. The run that follows it, after the angles were frozen, is 4 × 51.

Reads the archive directory (one sub-directory per task, each holding a
``results.json``; the S3 bucket's layout, or the local mirror of it) and writes

    paper01_sv1_task_ledger.csv   one row per run: start, end, task counts

Run:
    python -m experiments.paper01_qubo_baseline.sv1_task_ledger --archive ~/qfr-archive/sv1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from experiments.paper01_qubo_baseline.run_e1_cross import RESULTS_DIR
from experiments.paper01_qubo_baseline.run_experiment import write_csv

GAP_SECONDS = 120.0


def read_tasks(archive: Path) -> list[tuple[datetime, int, str]]:
    tasks = []
    for path in sorted(archive.glob("*/results.json")):
        with open(path) as f:
            meta = json.load(f).get("taskMetadata", {})
        created = meta.get("createdAt")
        if not created:
            continue
        tasks.append((datetime.fromisoformat(created.replace("Z", "+00:00")),
                      int(meta.get("shots", 0) or 0), str(meta.get("id", ""))))
    tasks.sort()
    return tasks


def ledger(tasks: list[tuple[datetime, int, str]], gap: float = GAP_SECONDS) -> list[dict]:
    runs: list[list[tuple[datetime, int, str]]] = []
    for t in tasks:
        if runs and (t[0] - runs[-1][-1][0]).total_seconds() <= gap:
            runs[-1].append(t)
        else:
            runs.append([t])
    out = []
    for k, r in enumerate(runs, 1):
        analytic = sum(1 for t in r if t[1] == 0)
        sampling = len(r) - analytic
        out.append({
            "run": k,
            "first_task_utc": r[0][0].isoformat().replace("+00:00", "Z"),
            "last_task_utc": r[-1][0].isoformat().replace("+00:00", "Z"),
            "tasks": len(r), "analytic_tasks": analytic, "sampling_tasks": sampling,
            "shots_total": sum(t[1] for t in r),
            # The account id inside the ARN is masked, as in every committed record.
            "first_task_id": re.sub(r":\d{12}:", ":000000000000:", r[0][2]),
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--gap-seconds", type=float, default=GAP_SECONDS)
    args = ap.parse_args(argv)
    tasks = read_tasks(args.archive)
    if not tasks:
        print(f"no task records under {args.archive}", file=sys.stderr)
        return 1
    rows = ledger(tasks, args.gap_seconds)
    write_csv(RESULTS_DIR / "paper01_sv1_task_ledger.csv", rows)
    print(f"{len(tasks)} tasks in {len(rows)} runs")
    for r in rows:
        print(f"run {r['run']:2d} {r['first_task_utc']} → {r['last_task_utc'][11:19]}  "
              f"tasks {r['tasks']:4d} = analytic {r['analytic_tasks']:4d} + sampling {r['sampling_tasks']:3d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
