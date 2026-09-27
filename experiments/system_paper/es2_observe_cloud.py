# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
E-S2 on the service: observe the two cells that the harness only modelled.

``es2_guard_stages.py`` never talks to AWS, so two of its inputs were taken from
documentation and from one old bill rather than observed:

1. **The spending limit.** Amazon Braket's per-QPU spending limit (November
   2025) is documented to reject, at ``CreateQuantumTask``, a task whose
   list-price estimate exceeds the remaining cap, and not to exist for
   simulators. Here a $1.00 limit is created on IonQ Forte-1, a 100-shot
   circuit ($0.30 + 100 x $0.08 = $8.30) is submitted against it, and the
   response is recorded; the same limit is requested for SV1 and the refusal
   recorded. Expected bill: $0. Worst case, if the limit does not act: one
   $8.30 task, which this script cancels at once if it was ever created.
2. **The sampling fallback.** The 177-task sampling call of the 2026-08-09
   pilot (``paper01_sv1_task_ledger.csv``, run 1) was inferred from the
   ledger; the metered path was not re-run under instrumentation. Here the
   same N = 8, p = 2 sampling QNode is sent to SV1 twice -- once with the
   angles marked trainable, once frozen -- and the tasks the service created
   in each window are counted through ``SearchQuantumTasks``. Expected:
   177 and 1 tasks, about $0.66 + $0.004 at the 3-second minimum.

Everything is written to ``paper02_es2_cloud_observation.csv``. The script
refuses to run without ``--yes-bill-me`` and prints the estimate first.

Run:
    python -m experiments.system_paper.es2_observe_cloud --s3-bucket <bucket> --yes-bill-me
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.paper01_qubo_baseline.run_experiment import write_csv
from experiments.system_paper.es2_guard_stages import toy_problem
from src.solvers.braket_solver import (
    BRAKET_REGION, IONQ_ARN, SV1_ARN, BraketSolver, frozen_angles,
)

RESULTS_DIR = Path(__file__).parent / "results"
OUT_CSV = RESULTS_DIR / "paper02_es2_cloud_observation.csv"
CAP_USD = "1.00"
PROBE_SHOTS = 100
SAMPLING_SHOTS = 1000
N = 8
P = 2


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def braket_client():
    import boto3
    return boto3.client("braket", region_name=BRAKET_REGION)


def observe_spending_limit(client, rows: list[dict]) -> None:
    """Create (or find) the $1 cap on IonQ; ask for one on SV1; record both."""
    existing = client.search_spending_limits(filters=[{"name": "deviceArn", "operator": "EQUAL", "values": [IONQ_ARN]}])
    limits = existing.get("spendingLimits", [])
    if limits:
        arn = limits[0]["spendingLimitArn"]
        rows.append({"step": "spending-limit-ionq", "time_utc": now(), "device": "IonQ Forte-1",
                     "outcome": "already present", "detail": f"{limits[0].get('spendingLimit')} USD, {arn}"})
    else:
        resp = client.create_spending_limit(deviceArn=IONQ_ARN, spendingLimit=CAP_USD,
                                            clientToken=f"es2-{int(time.time())}")
        rows.append({"step": "spending-limit-ionq", "time_utc": now(), "device": "IonQ Forte-1",
                     "outcome": "created", "detail": f"{CAP_USD} USD, {resp.get('spendingLimitArn')}"})
    try:
        client.create_spending_limit(deviceArn=SV1_ARN, spendingLimit=CAP_USD,
                                     clientToken=f"es2-sv1-{int(time.time())}")
        rows.append({"step": "spending-limit-sv1", "time_utc": now(), "device": "SV1",
                     "outcome": "UNEXPECTED: created", "detail": "documentation says simulators are unsupported"})
    except Exception as exc:  # the documented refusal
        rows.append({"step": "spending-limit-sv1", "time_utc": now(), "device": "SV1",
                     "outcome": "refused", "detail": f"{type(exc).__name__}: {exc}"[:300]})


def observe_probe(client, bucket: str, prefix: str, rows: list[dict]) -> None:
    """One 100-shot Bell circuit against the $1 cap: expected rejection at creation."""
    from braket.aws import AwsDevice, AwsSession
    from braket.circuits import Circuit
    import boto3

    session = AwsSession(boto_session=boto3.Session(region_name=BRAKET_REGION))
    device = AwsDevice(IONQ_ARN, aws_session=session)
    status = device.status
    circuit = Circuit().h(0).cnot(0, 1)
    t0 = now()
    try:
        task = device.run(circuit, (bucket, prefix), shots=PROBE_SHOTS)
        # Not rejected: cancel before it can run, and say so loudly.
        try:
            task.cancel()
            cancelled = "cancelled"
        except Exception as exc:
            cancelled = f"cancel failed: {exc}"
        rows.append({"step": "probe-ionq-100-shots", "time_utc": t0, "device": f"IonQ Forte-1 ({status})",
                     "outcome": "UNEXPECTED: task created", "detail": f"{task.id}; {cancelled}; "
                     f"list price ${0.30 + PROBE_SHOTS * 0.08:.2f} if it ran"})
    except Exception as exc:
        rows.append({"step": "probe-ionq-100-shots", "time_utc": t0, "device": f"IonQ Forte-1 ({status})",
                     "outcome": "rejected at creation", "detail": f"{type(exc).__name__}: {exc}"[:400]})
    # Independent confirmation: no task for this device since t0.
    found = client.search_quantum_tasks(filters=[
        {"name": "deviceArn", "operator": "EQUAL", "values": [IONQ_ARN]},
        {"name": "createdAt", "operator": "GTE", "values": [t0]}])
    rows.append({"step": "probe-ionq-tasks-since", "time_utc": now(), "device": "IonQ Forte-1",
                 "outcome": f"{len(found.get('quantumTasks', []))} task(s) created since the probe",
                 "detail": ";".join(t["quantumTaskArn"] for t in found.get("quantumTasks", []))})


def count_tasks(client, device_arn: str, since_iso: str) -> list[dict]:
    tasks, token = [], None
    while True:
        kwargs = dict(filters=[{"name": "deviceArn", "operator": "EQUAL", "values": [device_arn]},
                               {"name": "createdAt", "operator": "GTE", "values": [since_iso]}],
                      maxResults=100)
        if token:
            kwargs["nextToken"] = token
        resp = client.search_quantum_tasks(**kwargs)
        tasks += resp.get("quantumTasks", [])
        token = resp.get("nextToken")
        if not token:
            return tasks


def observe_fallback(client, bucket: str, prefix: str, rows: list[dict]) -> None:
    """The N = 8 sampling QNode on SV1 with trainable, then frozen, angles."""
    import pennylane as qml
    from pennylane import numpy as pnp
    from src.qubo.portfolio import PortfolioQUBO

    problem = toy_problem(N)
    h, J, _ = PortfolioQUBO().to_ising(problem)
    scale = float(max(np.max(np.abs(h)), np.max(np.abs(J)), 1e-12))
    solver = BraketSolver(backend="lightning_cpu", p_layers=P, n_shots=SAMPLING_SHOTS,
                          s3_bucket=bucket, s3_prefix=prefix)
    cost_h = solver._build_cost_hamiltonian(h / scale, J / scale, N, qml)
    mixer_h = solver._build_mixer_hamiltonian(N, qml)
    dev = solver._make_device(N, qml, "braket_sv1")

    @qml.qnode(dev, shots=SAMPLING_SHOTS)
    def sample_circuit(params):
        gammas, betas = params[:P], params[P:]
        for i in range(N):
            qml.Hadamard(wires=i)
        for gamma, beta in zip(gammas, betas):
            qml.ApproxTimeEvolution(cost_h, gamma, 1)
            qml.ApproxTimeEvolution(mixer_h, beta, 1)
        return [qml.sample(qml.PauliZ(i)) for i in range(N)]

    trainable = pnp.array([0.3, 0.6, 0.9, 1.2], requires_grad=True)
    for label, params in (("trainable", trainable), ("frozen", frozen_angles(trainable))):
        t0 = now()
        t_start = time.perf_counter()
        raw = sample_circuit(params)
        seconds = time.perf_counter() - t_start
        time.sleep(5)   # let the task index catch up
        tasks = count_tasks(client, SV1_ARN, t0)
        shots_ok = np.atleast_2d(np.asarray(raw, dtype=float)).size == N * SAMPLING_SHOTS
        rows.append({"step": f"sv1-sampling-{label}", "time_utc": t0, "device": "SV1",
                     "outcome": f"{len(tasks)} task(s) created",
                     "detail": f"diff_method={sample_circuit.diff_method}; requires_grad="
                               f"{bool(qml.math.requires_grad(params))}; {seconds:.0f} s wall; "
                               f"samples shape ok={shots_ok}; list price at 3 s minimum "
                               f"${len(tasks) * 0.00375:.3f}"})
        print(f"  {label}: {len(tasks)} tasks in {seconds:.0f} s", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--s3-bucket", required=True)
    p.add_argument("--s3-prefix", default="system_paper_es2")
    p.add_argument("--yes-bill-me", action="store_true")
    p.add_argument("--skip-fallback", action="store_true")
    p.add_argument("--skip-probe", action="store_true")
    args = p.parse_args(argv)

    est = 177 * 0.00375 + 0.00375
    print(f"Estimate: spending limit + probe $0.00 (worst case $8.30 if the cap does not act); "
          f"SV1 fallback about ${est:.2f} (178 tasks at the 3 s minimum).")
    if not args.yes_bill_me:
        print("pass --yes-bill-me to proceed")
        return 2

    client = braket_client()
    rows: list[dict] = []
    observe_spending_limit(client, rows)
    if not args.skip_probe:
        observe_probe(client, args.s3_bucket, args.s3_prefix, rows)
    for r in rows:
        print(f"  {r['step']}: {r['outcome']} -- {r['detail'][:160]}", flush=True)
    if not args.skip_fallback:
        observe_fallback(client, args.s3_bucket, args.s3_prefix, rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Merge by step: the stages run at different times (the QPU has execution
    # windows), and a later stage must not erase an earlier observation.
    merged: dict[str, dict] = {}
    if OUT_CSV.exists():
        import csv
        with open(OUT_CSV, newline="") as f:
            for r in csv.DictReader(f):
                merged[r["step"]] = r
    for r in rows:
        merged[r["step"]] = r
    write_csv(OUT_CSV, list(merged.values()))
    print(f"wrote {OUT_CSV} ({len(merged)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
