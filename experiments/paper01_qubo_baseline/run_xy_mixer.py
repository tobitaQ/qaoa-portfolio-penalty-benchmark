# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""E6: the same measurement with the constraint taken out of the cost.

Sec. V-C measures the preference inside the feasible set against eps_F, the
share of the cost Hamiltonian the objective carries, and finds it absent where
the textbook penalty weight puts it (eps_F = 1.5 %) and present where the weight
approaches the feasibility threshold. Every one of those runs keeps the
constraint *in* the cost: a quadratic penalty, a transverse-field mixer, and an
initial state uniform over all 2^N bitstrings.

This driver runs the other end of that design choice. The constraint is carried
by the ansatz instead:

    initial state   |D^N_K>, equal amplitude on every feasible bitstring
    mixer           ring XX+YY, which commutes with the Hamming weight [45], [46]
    cost            the portfolio objective alone (A = 0)

so eps_F = 1 by construction, P_F = 1 at every angle, and the only thing
interference can act on is the objective. Everything else is held at the
published setting -- the same instances, seeds, depth, optimizer, budget, shots
and evaluator -- so the comparison against the penalty runs is of the encoding
and nothing else.

``--mixer transverse`` runs the control that separates the two halves of the
change: the cost without its penalty, but the published ansatz. There the
feasible set is a 0.5-2 % slice of the uniform initial state and no term pushes
the optimiser toward it, which is what the penalty was for.

Writes (one row per run, plus the per-instance reference the rows are scored
against, because a block scored against another block's reference table is the
fault the uniform-feasible comparison tripped on once already):

    paper01_xy_cross.csv        one row per (N, instance, seed)
    paper01_xy_reference.csv    one row per (N, instance)
    xy_runs/<run_id>.npz        angles, trajectory, feasible-set probabilities

Run (GPU host, no AWS, $0):
    python -m experiments.paper01_qubo_baseline.run_xy_mixer --resume
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import numpy as np

from experiments.paper01_qubo_baseline.run_e1_cross import (
    DATA_END_DATE,
    DEFAULT_INSTANCES,
    DEFAULT_N,
    DEFAULT_SEEDS,
    InstanceReference,
    LOOKBACK_YEARS,  # noqa: F401  (documents the window the loader uses)
    RESULTS_DIR,
    RISK_AVERSION,
    SPEC_VERSION,
    input_digest,
    instance_digest,
    read_manifest,
    score_run,
)
from experiments.paper01_qubo_baseline.run_experiment import write_csv
from src.finance.data_loader import FinanceDataLoader
from src.finance.metrics import PortfolioMetrics
from src.qubo.portfolio import PortfolioQUBO
from src import run_status

ARTIFACT_DIR = RESULTS_DIR / "xy_runs"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E6: constraint in the ansatz, not in the cost")
    p.add_argument("--n", type=int, nargs="+", default=DEFAULT_N)
    p.add_argument("--instances", type=int, nargs="+", default=DEFAULT_INSTANCES)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--mixer", default="xy_ring", choices=["xy_ring", "transverse"],
                   help="xy_ring: Dicke state + weight-preserving mixer. "
                        "transverse: the published ansatz on the penalty-free cost")
    p.add_argument("--backend", default="lightning_gpu",
                   choices=["lightning_cpu", "lightning_gpu"])
    p.add_argument("--precision", default="single", choices=["single", "double"])
    p.add_argument("--p-layers", type=int, default=2)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--stepsize", type=float, default=0.1)
    p.add_argument("--shots", type=int, default=1000)
    p.add_argument("--end-date", default=DATA_END_DATE)
    p.add_argument("--results-tag", default=None)
    p.add_argument("--resume", action="store_true",
                   help="Skip (N, instance, seed, mixer) rows already recorded")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from src.solvers.braket_solver import BraketSolver

    suffix = f"_{args.results_tag}" if args.results_tag else ""
    manifest_path = RESULTS_DIR / f"paper01_xy_cross{suffix}.csv"
    reference_path = RESULTS_DIR / f"paper01_xy_reference{suffix}.csv"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(manifest_path) if args.resume else []
    done = {(r["n_assets"], r["instance"], r["qaoa_seed"], r.get("mixer", "")) for r in rows}
    reference_rows = read_manifest(reference_path) if args.resume else []
    reference_done = {(r["n_assets"], r["instance"]) for r in reference_rows}

    loader, formulator, metrics = FinanceDataLoader(), PortfolioQUBO(), PortfolioMetrics()
    total = len(args.n) * len(args.instances) * len(args.seeds)
    status = run_status.JobStatus(manifest_path, total=total)
    status.heartbeat(len(rows))
    failed = 0
    t_start = time.perf_counter()
    try:
        for n in sorted(args.n):
            for instance in args.instances:
                ref = InstanceReference(n, instance, loader, formulator, args.end_date)
                if (str(n), str(instance)) not in reference_done:
                    reference_rows.append(ref.row())
                    reference_done.add((str(n), str(instance)))
                    write_csv(reference_path, reference_rows)
                # A = 0: with the constraint in the ansatz the penalty term has
                # nothing to do, and with the published ansatz it is the control
                # that shows what the penalty was carrying.
                problem = formulator.formulate(
                    ref.data.returns, ref.data.covariance, num_select=ref.k,
                    risk_aversion=RISK_AVERSION, penalty_strength=0.0)
                solver = BraketSolver(
                    backend=args.backend, p_layers=args.p_layers, n_shots=args.shots,
                    n_optimizer_steps=args.steps, precision=args.precision,
                    record_state=True, optimizer="adam", stepsize=args.stepsize,
                    mixer=args.mixer, n_select=ref.k if args.mixer == "xy_ring" else None)
                print(f"\nN={n} K={ref.k} inst {instance:>2} |F|={ref.feasible.size:,} "
                      f"Δ_F={ref.feasible.delta:.4f} mixer={args.mixer} A=0", flush=True)
                for seed in args.seeds:
                    key = (str(n), str(instance), str(seed), args.mixer)
                    if key in done:
                        continue
                    base = {
                        "spec_version": SPEC_VERSION,
                        "run_id": f"N{n}_i{instance:02d}_{args.mixer}_s{seed}",
                        "n_assets": n, "n_select": ref.k, "instance": instance,
                        "instance_digest": instance_digest(ref.tickers),
                        "input_digest": input_digest(ref.data.returns, ref.data.covariance),
                        "mixer": args.mixer, "penalty_label": "A0", "penalty": 0.0,
                        "qaoa_seed": seed, "backend": args.backend,
                        "precision": args.precision, "p_layers": args.p_layers,
                        "steps": args.steps, "optimizer": "adam",
                        "stepsize": args.stepsize, "shots": args.shots,
                        "f_star": ref.feasible.f_star, "delta_F": ref.feasible.delta,
                        "feasible_set_size": ref.feasible.size,
                        "data_start": ref.data.start_date, "data_end": ref.data.end_date,
                    }
                    try:
                        res = solver.solve(problem, seed=seed)
                        scored, artifact = score_run(res, problem, ref, 0.0, metrics)
                        np.savez_compressed(ARTIFACT_DIR / f"{base['run_id']}.npz", **artifact)
                        row = {**base, "status": "ok", "error": "", **scored}
                        print(f"  s{seed}  P_F={row['final_P_F']:.3f} "
                              f"P_τ={row['final_P_tau_main']:.4f} "
                              f"P_opt={row['final_P_opt']:.4f}  "
                              f"g_F={'none' if row['g_F'] == '' else format(row['g_F'], '.4f')} "
                              f"rank={row['rank_best_feasible']}  {row['runtime_s']:.1f}s",
                              flush=True)
                    except Exception as exc:                      # noqa: BLE001
                        traceback.print_exc()
                        row = {**base, "status": f"error:{type(exc).__name__}",
                               "error": str(exc)[:200]}
                        failed += 1
                        print(f"  s{seed}  FAILED {type(exc).__name__}", flush=True)
                    rows.append(row)
                    done.add(key)
                    write_csv(manifest_path, rows)
                    status.heartbeat(len(rows), last=base["run_id"], failed=failed)
    except BaseException as exc:                                   # noqa: BLE001
        write_csv(manifest_path, rows)
        status.finish(130 if isinstance(exc, KeyboardInterrupt) else 1)
        raise
    write_csv(manifest_path, rows)
    status.heartbeat(len(rows), total=len(rows), failed=failed)
    status.finish(0)
    print(f"\n{len(rows)} run(s), {failed} failed, "
          f"{(time.perf_counter() - t_start) / 60:.1f} min -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
