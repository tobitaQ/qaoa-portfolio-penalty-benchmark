# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT
"""Rebuild the repetition table of the account-level cost control from the run records.

`es2_observe_cost_control.py` writes one summary per run while it runs, against the live
service. Two of its fields were added after the first repetition, so run 1's summary lacks
them, and the paper would otherwise report a number no committed file holds. Every field
except the alarm's own timestamp is a function of the per-task records, so this script
fills a missing field by recomputing it from them.

Where a live value exists it is kept, because the live code subtracted full-precision
timestamps while the file stores them truncated to the second; the recomputation is
therefore accurate to about a second, which this script checks against every field the
live summaries do hold.

    python -m experiments.system_paper.es2_cost_control_aggregate      # seconds, no AWS
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from statistics import median

import pandas as pd

from experiments.paper01_qubo_baseline.run_experiment import write_csv

RESULTS = Path(__file__).parent / "results"
LIMIT_USD = 0.30
SV1_TASK_USD = 3.0 / 60 * 0.075
RUNS = (1, 2, 3, 4, 5)
#: The recomputation reads timestamps truncated to the second, so it may differ by this much.
TOLERANCE = {"seconds_limit_to_alarm": 1, "tasks_created_after_alarm": 1,
             "seconds_alarm_to_denial": 1, "tasks_created_after_first_denial": 1}


def recomputed(run: int, alarm: datetime, first_denial: datetime) -> dict:
    tasks = pd.read_csv(RESULTS / f"paper02_es2_cost_control_tasks_run{run}.csv")
    created = tasks[tasks.outcome == "created"]
    when = {int(r["index"]): datetime.fromisoformat(r.created_utc) for _, r in tasks.iterrows()}
    over = next(i for i in created["index"] if i * SV1_TASK_USD > LIMIT_USD)
    return {
        "tasks_created": len(created),
        "first_task_over_limit": int(over),
        "tasks_created_after_limit_crossed": int((created["index"] >= over).sum()),
        "seconds_limit_to_alarm": round((alarm - when[int(over)]).total_seconds()),
        "tasks_created_after_alarm": int(sum(when[int(i)] >= alarm for i in created["index"])),
        "seconds_alarm_to_denial": round((first_denial - alarm).total_seconds()),
        "tasks_created_after_first_denial": int(sum(when[int(i)] > first_denial for i in created["index"])),
        "denials_observed": int((tasks.outcome == "denied").sum()),
        "list_price_usd": round(len(created) * SV1_TASK_USD, 4),
    }


def one(run: int, report: list) -> dict:
    live = pd.read_csv(RESULTS / f"paper02_es2_cost_control_summary_run{run}.csv").iloc[0]
    alarm = datetime.fromisoformat(live.alarm_to_ALARM_utc)
    first_denial = datetime.fromisoformat(live.first_denial_utc)
    rec = recomputed(run, alarm, first_denial)
    row = {"run": run}
    for field, value in rec.items():
        have = live[field] if field in live.index else None
        if have is None or pd.isna(have) or have == "":
            row[field] = value
            report.append(f"run {run}: {field} = {value} (recomputed; the live summary has none)")
        else:
            row[field] = type(value)(have)
            if abs(row[field] - value) > TOLERANCE.get(field, 0):
                report.append(f"run {run}: {field} live {row[field]} vs recomputed {value} — OUT OF TOLERANCE")
    row["alarm_to_ALARM_utc"] = live.alarm_to_ALARM_utc
    row["first_denial_utc"] = live.first_denial_utc
    return row


def main() -> int:
    report: list[str] = []
    rows = [one(r, report) for r in RUNS]
    write_csv(RESULTS / "paper02_es2_cost_control_repetitions.csv", rows)
    quantities = ["tasks_created", "tasks_created_after_limit_crossed", "seconds_limit_to_alarm",
                  "tasks_created_after_alarm", "seconds_alarm_to_denial",
                  "tasks_created_after_first_denial", "list_price_usd"]
    summary = [{"quantity": q, "median": median(r[q] for r in rows),
                "min": min(r[q] for r in rows), "max": max(r[q] for r in rows)} for q in quantities]
    write_csv(RESULTS / "paper02_es2_cost_control_repetitions_summary.csv", summary)
    for line in report:
        print("  " + line)
    for s in summary:
        print(f"  {s['quantity']}: median {s['median']} ({s['min']}–{s['max']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
