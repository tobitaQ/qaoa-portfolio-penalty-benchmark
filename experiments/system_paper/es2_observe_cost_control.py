# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
E-S2, third kind of control: how many tasks slip past an account-level detective
control before its denial takes effect.

AWS's cost-control solution for Amazon Braket (May 2023; deployed here on
2026-09-17 with a $0.30 monthly limit and the deny targeted at the project's
IAM user) meters every task from its state-change events, raises a CloudWatch
alarm when the running cost crosses the limit, and a Lambda then attaches a
managed policy that denies ``braket:CreateQuantumTask``. The documentation says
"a few" tasks are created after the limit is exceeded. This script measures the
few: it submits the cheapest SV1 task there is (a Bell circuit, 10 shots, billed
at the 3-second minimum, $0.00375) one after another, as a variational loop
would, records every creation with its time, and stops at the first denial (or
at ``--max-tasks``). Afterwards it reads the alarm's state history and reports
how many tasks were created after the metric crossed, and after the alarm
fired.

Expected bill: about 80 tasks to reach $0.30, plus the ones that slip through;
at most ``--max-tasks`` × $0.00375. Refuses to run without ``--yes-bill-me``.

Run:
    python -m experiments.system_paper.es2_observe_cost_control --s3-bucket <bucket> --yes-bill-me
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.paper01_qubo_baseline.run_experiment import write_csv
from src.solvers.braket_solver import BRAKET_REGION, SV1_ARN

RESULTS_DIR = Path(__file__).parent / "results"
TASKS_CSV = RESULTS_DIR / "paper02_es2_cost_control_tasks.csv"
SUMMARY_CSV = RESULTS_DIR / "paper02_es2_cost_control_summary.csv"
SV1_TASK_USD = 3.0 / 60 * 0.075
ALARM_NAME = "Quantum Task Cost Monthly Aggregate"
DENY_POLICY_NAME_HINT = "braket-cost-control"


def now() -> datetime:
    return datetime.now(timezone.utc)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--s3-bucket", required=True)
    p.add_argument("--s3-prefix", default="system_paper_es2_cost_control")
    p.add_argument("--max-tasks", type=int, default=400)
    p.add_argument("--confirm-denials", type=int, default=3,
                   help="after the first denial, retry this many times over two minutes")
    p.add_argument("--yes-bill-me", action="store_true")
    p.add_argument("--tag", default="", help="suffix for the output files (repetitions)")
    args = p.parse_args(argv)
    suffix = f"_{args.tag}" if args.tag else ""
    tasks_csv = TASKS_CSV.with_name(TASKS_CSV.stem + suffix + ".csv")
    summary_csv = SUMMARY_CSV.with_name(SUMMARY_CSV.stem + suffix + ".csv")
    print(f"Estimate: up to {args.max_tasks} SV1 tasks at ${SV1_TASK_USD:.5f} = "
          f"${args.max_tasks * SV1_TASK_USD:.2f}; the monthly limit of the deployed control is $0.30.")
    if not args.yes_bill_me:
        print("pass --yes-bill-me to proceed")
        return 2

    import boto3
    from braket.aws import AwsDevice, AwsSession
    from braket.circuits import Circuit

    session = AwsSession(boto_session=boto3.Session(region_name=BRAKET_REGION))
    device = AwsDevice(SV1_ARN, aws_session=session)
    circuit = Circuit().h(0).cnot(0, 1)
    rows: list[dict] = []
    t_start = now()
    denied_at = None
    denials = 0
    i = 0
    while i < args.max_tasks:
        i += 1
        t0 = now()
        try:
            task = device.run(circuit, (args.s3_bucket, args.s3_prefix), shots=10)
            state = task.state()
            # Wait for completion so the control's metering (at COMPLETED for
            # simulators) can keep pace, exactly as a sequential loop would.
            result = task.result()
            final = task.state()
            rows.append({"index": i, "created_utc": t0.isoformat(timespec="seconds"),
                         "completed_utc": now().isoformat(timespec="seconds"), "task_arn": task.id,
                         "outcome": "created", "state": final, "error": ""})
            print(f"  [{i:>3}] created {task.id.rsplit('/', 1)[-1]} -> {final} "
                  f"(cum. list price ${i * SV1_TASK_USD:.3f})", flush=True)
        except Exception as exc:
            name = type(exc).__name__
            text = str(exc)
            denied = "AccessDenied" in text or "AccessDenied" in name or "not authorized" in text
            rows.append({"index": i, "created_utc": t0.isoformat(timespec="seconds"),
                         "completed_utc": "", "task_arn": "", "outcome": "denied" if denied else "error",
                         "state": "", "error": f"{name}: {text[:200]}"})
            print(f"  [{i:>3}] {'DENIED' if denied else 'ERROR'}: {name}: {text[:120]}", flush=True)
            if denied:
                denials += 1
                denied_at = denied_at or t0
                if denials > args.confirm_denials:
                    break
                time.sleep(40)
            else:
                time.sleep(5)
    write_csv(tasks_csv, rows)

    # The alarm's own record of when it fired.
    cw = boto3.client("cloudwatch", region_name=BRAKET_REGION)
    hist = cw.describe_alarm_history(AlarmName=ALARM_NAME, HistoryItemType="StateUpdate",
                                     StartDate=t_start, EndDate=now(), MaxRecords=50)
    alarm_at = None
    for item in sorted(hist.get("AlarmHistoryItems", []), key=lambda h: h["Timestamp"]):
        if "to ALARM" in item.get("HistorySummary", ""):
            alarm_at = item["Timestamp"]
            break
    # Time the denial was attached, from the alarm-action Lambda's log.
    attached_at = None
    try:
        logs = boto3.client("logs", region_name=BRAKET_REGION)
        ev = logs.filter_log_events(logGroupName="/aws/lambda/braket-cost-control-cost-control-action",
                                    startTime=int(t_start.timestamp() * 1000),
                                    filterPattern="attach", limit=5)
        if ev.get("events"):
            attached_at = datetime.fromtimestamp(ev["events"][0]["timestamp"] / 1000, tz=timezone.utc)
    except Exception:
        pass
    created = [r for r in rows if r["outcome"] == "created"]
    after_alarm = [r for r in created if alarm_at and datetime.fromisoformat(r["created_utc"]) >= alarm_at]
    over_limit_index = next((r["index"] for r in created if r["index"] * SV1_TASK_USD > 0.30), None)
    after_limit = [r for r in created if over_limit_index and r["index"] >= over_limit_index]
    summary = [{
        "started_utc": t_start.isoformat(timespec="seconds"),
        "tasks_created": len(created), "list_price_usd": round(len(created) * SV1_TASK_USD, 4),
        "monthly_limit_usd": 0.30,
        "first_task_over_limit": over_limit_index,
        "tasks_created_after_limit_crossed": len(after_limit),
        "alarm_to_ALARM_utc": alarm_at.isoformat(timespec="seconds") if alarm_at else "",
        "tasks_created_after_alarm": len(after_alarm),
        "first_denial_utc": denied_at.isoformat(timespec="seconds") if denied_at else "",
        "seconds_alarm_to_denial": (round((denied_at - alarm_at).total_seconds()) if (alarm_at and denied_at) else ""),
        "policy_attached_utc": attached_at.isoformat(timespec="seconds") if attached_at else "",
        "tasks_created_after_first_denial": len([r for r in created if denied_at and datetime.fromisoformat(r["created_utc"]) > denied_at]),
        "seconds_limit_to_alarm": (round((alarm_at - datetime.fromisoformat(next(r["created_utc"] for r in created if r["index"] == over_limit_index))).total_seconds()) if (alarm_at and over_limit_index) else ""),
        "denials_observed": denials,
        "stopped_by": "denial" if denied_at else "max-tasks",
    }]
    write_csv(summary_csv, summary)
    for k, v in summary[0].items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
