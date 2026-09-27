# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Relation R14: the tasks a metered invocation was acknowledged to cost equal the tasks the
provider records for it.

Inputs: the pre-submission estimate the driver persisted (``paper01_metered_estimate_*.json``)
and the task ledger built from the provider's task records (``paper01_sv1_task_ledger.csv``,
one row per invocation with the tasks actually created). A mismatch means the framework decided
a different task count than the invocation expected — the 227-versus-51 pilot of paper ①, which
was found by reading a bill, becomes a checked relation.

    python -m experiments.paper01_qubo_baseline.check_metered --estimate FILE --ledger-run N
    python -m experiments.paper01_qubo_baseline.check_metered --expected 51 --ledger-run 1   # the historical pilot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).parent / "results"


def check(expected_tasks: int, ledger_tasks: int) -> dict:
    return {"expected_tasks": expected_tasks, "billed_tasks": ledger_tasks,
            "status": "pass" if expected_tasks == ledger_tasks else "fail",
            "detail": ("the provider recorded the expected number of tasks" if expected_tasks == ledger_tasks else
                       f"the provider recorded {ledger_tasks} tasks for an invocation acknowledged at {expected_tasks}: "
                       f"{ledger_tasks - expected_tasks:+d}")}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--estimate", help="estimate JSON written by run_experiment")
    p.add_argument("--expected", type=int, help="expected task count (in place of --estimate)")
    p.add_argument("--ledger", default=str(RESULTS / "paper01_sv1_task_ledger.csv"))
    p.add_argument("--ledger-run", type=int, required=True, help="row (run number) in the ledger")
    args = p.parse_args(argv)
    expected = json.loads(Path(args.estimate).read_text(encoding="utf-8"))["expected_tasks"] if args.estimate else args.expected
    if expected is None:
        p.error("--estimate or --expected is required")
    ledger = pd.read_csv(args.ledger)
    row = ledger[ledger.run == args.ledger_run].iloc[0]
    r = check(int(expected), int(row.tasks))
    print(f"R14 metered: {r['status']} — {r['detail']} (ledger run {args.ledger_run}: "
          f"{int(row.analytic_tasks)} analytic + {int(row.sampling_tasks)} sampling)")
    return 0 if r["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
