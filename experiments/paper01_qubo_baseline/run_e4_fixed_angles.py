# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Paper ① / TQE plan E4-A: the same QAOA state measured on several backends.

§V-D of the draft reports that SV1 and the local GPU return different
portfolios at N ≥ 16 from the same configuration, and that SV1 disagrees with
itself. The 2026-08 "shot-RNG control" that was meant to rule out sampling
noise re-drew identical shots (see ``BraketSolver._seed_sampler``), so the
attribution between *numerics* (a different state) and *sampling* (different
shots from the same state) is open. This experiment separates the two by
fixing the angles:

    12 circuits  = N ∈ {12, 16, 20} × instances {0, 1} × seeds {42, 43},
                   at A_heur, angles taken from the stored E1 runs
    per arm      = the exact probability vector of the state (one analytic
                   circuit; one task on SV1) + ``--batches`` independent
                   1,000-shot batches (one task each on SV1)
    arms         = gpu_single, gpu_double, cpu_double (1 thread), sv1

Every batch is scored by the E1 scorer (``score_shots``) against the same
enumerated feasible set, and every exact state by ``_state_scores``. Nothing
is optimised here.

Cost guard: the ``sv1`` arm refuses to run without ``--results-tag``,
``--s3-bucket`` and ``--yes-bill-me``, prints the task count and the list-price
estimate first, and records each task's ARN and billed execution time.

Run (local arms are free):
    python -m experiments.paper01_qubo_baseline.run_e4_fixed_angles --arm gpu_double
    python -m experiments.paper01_qubo_baseline.run_e4_fixed_angles --arm gpu_single
    python -m experiments.paper01_qubo_baseline.run_e4_fixed_angles --arm cpu_double
    # pilot on SV1: one circuit, one batch (2 tasks, ~$0.01)
    python -m experiments.paper01_qubo_baseline.run_e4_fixed_angles --arm sv1 \\
        --n 12 --instances 0 --seeds 42 --batches 1 --results-tag e4a_pilot \\
        --s3-bucket <bucket> --yes-bill-me
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from experiments.paper01_qubo_baseline.run_e1_cross import (
    ARTIFACT_DIR,
    RESULTS_DIR,
    RISK_AVERSION,
    RISK_FREE_RATE,
    SHOT_BUDGETS,
    InstanceReference,
    _state_scores,
    read_manifest,
    score_shots,
)
from experiments.paper01_qubo_baseline.run_experiment import DATA_END_DATE, write_csv
from experiments.paper01_qubo_baseline.run_instances import instance_digest
from src.finance.data_loader import FinanceDataLoader
from src.finance.metrics import PortfolioMetrics
from src.qubo.portfolio import PortfolioQUBO

SPEC_VERSION = "e4a-v1"

#: The pre-fixed design (plan §8.1). Chosen before any E4 result existed.
DEFAULT_N = [12, 16, 20]
DEFAULT_INSTANCES = [0, 1]
DEFAULT_SEEDS = [42, 43]
PENALTY_LABEL = "Aheur"
DEFAULT_BATCHES = 10
SHOTS = 1000

#: arm -> (optimizer backend, sampling backend, precision)
ARMS = {
    "gpu_single": ("lightning_gpu", "lightning_gpu", "single"),
    "gpu_double": ("lightning_gpu", "lightning_gpu", "double"),
    "cpu_double": ("lightning_cpu", "lightning_cpu", "double"),
    "sv1": ("lightning_cpu", "braket_sv1", "double"),
}
CLOUD_ARMS = {"sv1"}

#: SV1 list price and billing floor, for the estimate printed before a run.
SV1_USD_PER_MINUTE = 0.075
SV1_MIN_SECONDS = 3.0

E4_ARTIFACT_DIR = RESULTS_DIR / "e4_runs"

STATE_KEY = ("arm", "n_assets", "instance", "qaoa_seed")


def load_e1_angles(n: int, instance: int, seed: int) -> dict:
    """Final angles, penalty and Hamiltonian scale of the stored E1 run."""
    path = ARTIFACT_DIR / f"N{n}_i{instance:02d}_{PENALTY_LABEL}_s{seed}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path}: the E1 artifact is needed for the angles")
    d = np.load(path)
    return {"final_angles": np.asarray(d["final_angles"], dtype=float),
            "penalty": float(d["penalty"]),
            "hamiltonian_scale": float(d["hamiltonian_scale"]),
            "run_id": path.stem}


def feasible_scores(pf: np.ndarray, ref: InstanceReference, prefix: str) -> dict:
    """The ``_state_scores`` quantities that need only the feasible set's mass."""
    from src.qubo import scoring

    out = {
        f"{prefix}_probs_sum": "",
        f"{prefix}_P_F": float(pf.sum()),
        f"{prefix}_P_opt": float(pf[ref.optimal].sum()),
        f"{prefix}_P_tau_main": float(pf[ref.within_main].sum()),
        f"{prefix}_P_tau_alt": float(pf[ref.within_alt].sum()),
        f"{prefix}_expected_f": "",
        f"{prefix}_expected_violation": "",
    }
    for s in SHOT_BUDGETS:
        out[f"{prefix}_Q_tau_main_{s}"] = scoring.success_probability(out[f"{prefix}_P_tau_main"], s)
        out[f"{prefix}_Q_opt_{s}"] = scoring.success_probability(out[f"{prefix}_P_opt"], s)
    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E4-A: fixed angles on several backends")
    p.add_argument("--arm", required=True, choices=sorted(ARMS))
    p.add_argument("--n", type=int, nargs="+", default=DEFAULT_N)
    p.add_argument("--instances", type=int, nargs="+", default=DEFAULT_INSTANCES)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--batches", type=int, default=DEFAULT_BATCHES)
    p.add_argument("--shots", type=int, default=SHOTS)
    p.add_argument("--p-layers", type=int, default=2)
    p.add_argument("--no-probabilities", action="store_true",
                   help="skip the analytic circuit (only the shot batches)")
    p.add_argument("--shot-seed", type=int, default=20260912,
                   help="local arms: batch b re-seeds the device sampler with shot_seed + b")
    p.add_argument("--end-date", default=DATA_END_DATE)
    p.add_argument("--results-tag", default=None)
    p.add_argument("--s3-bucket", default=None)
    p.add_argument("--yes-bill-me", action="store_true",
                   help="required for the sv1 arm: acknowledges the printed estimate")
    p.add_argument("--resume", action="store_true",
                   help="skip circuits already in the states file for this arm")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from src.solvers.braket_solver import BraketSolver

    backend, sample_backend, precision = ARMS[args.arm]
    n_circuits = len(args.n) * len(args.instances) * len(args.seeds)
    tasks = n_circuits * (args.batches + (0 if args.no_probabilities else 1))
    if args.arm in CLOUD_ARMS:
        if not (args.results_tag and args.s3_bucket and args.yes_bill_me):
            print("ABORT: the sv1 arm needs --results-tag, --s3-bucket and --yes-bill-me",
                  file=sys.stderr)
            return 1
        est = tasks * SV1_MIN_SECONDS / 60.0 * SV1_USD_PER_MINUTE
        print(f"SV1: {n_circuits} circuits x ({args.batches} batches"
              f"{'' if args.no_probabilities else ' + 1 analytic'}) = {tasks} tasks; "
              f"at the {SV1_MIN_SECONDS:.0f} s billing floor that is ${est:.2f} "
              f"(more if an analytic task runs longer than the floor)", flush=True)

    suffix = f"_{args.results_tag}" if args.results_tag else ""
    states_path = RESULTS_DIR / f"paper01_e4_states{suffix}.csv"
    batches_path = RESULTS_DIR / f"paper01_e4_batches{suffix}.csv"
    E4_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    state_rows = read_manifest(states_path) if args.resume else []
    batch_rows = read_manifest(batches_path) if args.resume else []
    for r in state_rows + batch_rows:
        if r.get("spec_version") != SPEC_VERSION:
            print(f"ABORT: existing rows carry spec {r.get('spec_version')!r}, "
                  f"this code is {SPEC_VERSION!r}", file=sys.stderr)
            return 1
    done = {tuple(r[k] for k in STATE_KEY) for r in state_rows}

    loader, formulator = FinanceDataLoader(), PortfolioQUBO()
    metrics = PortfolioMetrics(risk_free_rate=RISK_FREE_RATE)
    solver = BraketSolver(backend=backend, sample_backend=sample_backend,
                          precision=precision, p_layers=args.p_layers,
                          n_shots=args.shots, s3_bucket=args.s3_bucket)
    print(f"E4-A {SPEC_VERSION}: arm {args.arm} ({sample_backend}/{precision}), "
          f"N={args.n} x instances {args.instances} x seeds {args.seeds}, "
          f"{args.batches} x {args.shots} shots; {len(done)} circuits already done", flush=True)

    t_start = time.perf_counter()
    for n in sorted(args.n):
        for instance in args.instances:
            ref = InstanceReference(n, instance, loader, formulator, args.end_date)
            problem = formulator.formulate(
                ref.data.returns, ref.data.covariance, num_select=ref.k,
                risk_aversion=RISK_AVERSION, penalty_strength=ref.a_heur)
            for seed in args.seeds:
                key = (args.arm, str(n), str(instance), str(seed))
                if key in done:
                    continue
                e1 = load_e1_angles(n, instance, seed)
                if abs(e1["penalty"] - ref.a_heur) > 1e-9 * max(1.0, ref.a_heur):
                    raise RuntimeError(f"{e1['run_id']}: stored A {e1['penalty']!r} is not "
                                       f"this instance's A_heur {ref.a_heur!r}")
                base = {
                    "spec_version": SPEC_VERSION, "arm": args.arm,
                    "backend": sample_backend, "precision": precision,
                    "n_assets": n, "n_select": ref.k, "instance": instance,
                    "instance_digest": instance_digest(ref.tickers),
                    "qaoa_seed": seed, "e1_run_id": e1["run_id"],
                    "penalty_label": PENALTY_LABEL, "penalty": ref.a_heur,
                    "p_layers": args.p_layers,
                    "angles": " ".join(f"{v:.17g}" for v in e1["final_angles"]),
                    "f_star": ref.feasible.f_star, "delta_F": ref.feasible.delta,
                    "feasible_set_size": ref.feasible.size,
                    "data_start": ref.data.start_date, "data_end": ref.data.end_date,
                }
                t0 = time.perf_counter()
                out = solver.evaluate_fixed_angles(
                    problem, e1["final_angles"], batches=args.batches,
                    shot_seed=args.shot_seed, probabilities=not args.no_probabilities,
                    state_indices=ref.feasible.indices)
                if abs(out["hamiltonian_scale"] - e1["hamiltonian_scale"]) > 1e-12 * e1["hamiltonian_scale"]:
                    raise RuntimeError(f"{e1['run_id']}: Hamiltonian scale {out['hamiltonian_scale']!r} "
                                       f"differs from E1's {e1['hamiltonian_scale']!r}; not the same circuit")
                arns = list(out["task_arns"])
                durations = list(out["execution_ms"])
                # The analytic task, if any, is the first ARN.
                state_arn = arns.pop(0) if (arns and not args.no_probabilities) else ""
                state_ms = durations.pop(0) if (durations and not args.no_probabilities) else ""

                srow = dict(base)
                srow.update({
                    "status": "ok", "hamiltonian_scale": out["hamiltonian_scale"],
                    "shot_seed": out["shot_seed"] if out["shot_seed"] is not None else "",
                    "shot_seed_applied": out["shot_seed_applied"] if out["shot_seed_applied"] is not None else "",
                    "batches": len(out["batches"]), "shots": args.shots,
                    "task_arn": state_arn, "execution_ms": state_ms,
                    "seconds": out["seconds"],
                })
                P = out["probabilities"]
                pf = out["state_probabilities"]
                if P is not None:
                    srow.update(_state_scores(P, ref, "final"))
                elif pf is not None:
                    # Cloud: amplitudes of the feasible set only, so the
                    # feasible-conditioned quantities exist and the full-state
                    # expectations do not.
                    srow.update(feasible_scores(pf, ref, "final"))
                if pf is not None:
                    np.savez_compressed(E4_ARTIFACT_DIR / f"{args.arm}_{e1['run_id']}.npz",
                                        probabilities=(P if P is not None else np.zeros(0)),
                                        feasible_probabilities=pf, angles=e1["final_angles"],
                                        feasible_indices=ref.feasible.indices)
                state_rows.append(srow)

                for b, samples in enumerate(out["batches"]):
                    brow = dict(base)
                    brow.update({"batch": b, "task_arn": arns[b] if arns else "",
                                 "execution_ms": durations[b] if durations else ""})
                    brow.update(score_shots(samples, ref, ref.a_heur, metrics))
                    batch_rows.append(brow)
                done.add(key)
                write_csv(states_path, state_rows)
                write_csv(batches_path, batch_rows)
                fs = [float(r["feasible_shot_fraction"]) for r in batch_rows
                      if tuple(str(r[k]) for k in STATE_KEY) == key]
                pf_txt = f"P_F={srow['final_P_F']:.4f}" if pf is not None else "P_F=n/a"
                print(f"  {args.arm:10s} {e1['run_id']:22s} {pf_txt} feasible shots "
                      f"{min(fs):.3f}-{max(fs):.3f} over {len(fs)} batches "
                      f"[{time.perf_counter() - t0:.1f}s"
                      f"{'' if not durations else f', billed {sum(durations) / 1000:.1f}s'}]",
                      flush=True)
    if args.arm in CLOUD_ARMS:
        billed = [float(r["execution_ms"]) for r in state_rows + batch_rows
                  if r.get("arm") == args.arm and r.get("execution_ms") not in ("", None)]
        floor = sum(max(ms / 1000.0, SV1_MIN_SECONDS) for ms in billed)
        print(f"\nSV1 tasks this file: {len(billed)}; execution {sum(billed) / 1000:.1f} s, "
              f"billable at the floor {floor:.1f} s = ${floor / 60 * SV1_USD_PER_MINUTE:.3f}")
    print(f"done in {(time.perf_counter() - t_start) / 60:.1f} min -> {states_path.name}, "
          f"{batches_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
