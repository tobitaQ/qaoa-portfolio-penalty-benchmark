# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""E3 diagnostic: why L-BFGS-B stopped, at single and at double precision.

Sec. V-C called the E3 quasi-Newton runs converged on the evidence that 273 of
300 used fewer than the 100 gradient evaluations they were allowed. That is
also what a failed line search looks like, and it is what running out of
budget looks like from the outside. The runs recorded neither SciPy's status
nor its message, so the claim could not be checked -- and the E3 block ran in
single precision, where SciPy's default ftol of 2.2e-9 sits two orders of
magnitude below the resolution of the expectation being minimised, so
"relative reduction of f below tolerance" can mean the objective stopped
resolving rather than that a minimum was reached.

This re-runs a small fixed block at both precisions and records what SciPy
reports. It is a diagnostic of the stopping criterion, not a new experiment:
no claim in the paper rests on its optimised values.

Review round 3 (M7) added the acquisition side: a similar final expectation
at two precisions does not imply similar P_τ or Q_τ, since two distributions
with the same mean can put different mass on the success set. Each run now
also scores its final state (P_F, P_τ, Q_τ(1000), D_cond) and its 1,000-shot
batch, from the exact probabilities the local simulator returns, so the
precision comparison is made on what the paper reports rather than on ⟨H⟩.
The stopping-reason columns are unchanged and must reproduce the earlier CSV.

    python -m experiments.paper01_qubo_baseline.run_lbfgs_termination
    python -m experiments.paper01_qubo_baseline.run_lbfgs_termination -n 20

Local simulator only, no billing. N = 12 takes about a minute, N = 16 about
ten on a CPU; N = 20 wants the GPU host.

Outputs are named by the sizes run: ``paper01_e3_lbfgs_termination_n12_16.csv``
and ``_n20.csv`` (one row per run) and ``paper01_e3_precision_pairs_n*.csv``
(one row per single/double pair). The paper's Table S36 reads the pair rows of
every size concatenated into ``paper01_e3_precision_pairs.csv``. The
stopping-reason columns are host-sensitive at single precision: the first
N = 12/16 block, run on a Mac, is kept as ``_n12_16_mac.csv``; its double-
precision rows reproduce on the Linux host, 14 of its 16 single-precision rows
do not (Supplementary Sec. S-VI).

Batch columns (``shots_feasible``, ``g_F``, ``within_tau_main``, ``is_optimal``
and the pairs' ``within_tau_main_agree``) written before 2026-09-13 describe the
*initial* state: the solver's L-BFGS-B path handed the sampler the initial
angles (``BraketSolver.solve``, fixed that day). The state columns were never
affected. The block was re-run with the fix (review round 4).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from experiments.paper01_qubo_baseline.analyze_conditional import conditional_tv
from experiments.paper01_qubo_baseline.run_degeneracy import RISK_FREE_RATE
from experiments.paper01_qubo_baseline.run_e1_cross import (
    DATA_END_DATE,
    InstanceReference,
    RESULTS_DIR,
    RISK_AVERSION,
    _state_scores,
    score_shots,
)
from experiments.paper01_qubo_baseline.run_experiment import write_csv
from src.finance.metrics import PortfolioMetrics
from src.finance.data_loader import FinanceDataLoader
from src.qubo.portfolio import PortfolioQUBO
from src.solvers.braket_solver import BraketSolver

#: The block is small and fixed before looking: two instances, two seeds, both
#: penalty settings, at each size asked for.
INSTANCES = (0, 1)
SEEDS = (42, 43)
PRECISIONS = ("single", "double")
P_LAYERS = 2
GRADIENT_BUDGET = 100
N_SHOTS = 1000


def run(sizes: list[int], backend: str) -> list[dict]:
    loader, formulator = FinanceDataLoader(), PortfolioQUBO()
    metrics = PortfolioMetrics(risk_free_rate=RISK_FREE_RATE)
    out: list[dict] = []
    for n in sizes:
        for instance in INSTANCES:
            ref = InstanceReference(n, instance, loader, formulator, DATA_END_DATE)
            for label, penalty in (("Aheur", ref.a_heur), ("Amargin", ref.a_margin)):
                problem = formulator.formulate(
                    ref.data.returns, ref.data.covariance, num_select=ref.k,
                    risk_aversion=RISK_AVERSION, penalty_strength=penalty)
                for seed in SEEDS:
                    for precision in PRECISIONS:
                        solver = BraketSolver(
                            backend=backend, precision=precision,
                            p_layers=P_LAYERS, n_shots=N_SHOTS,
                            n_optimizer_steps=GRADIENT_BUDGET,
                            optimizer="lbfgs", record_state=True)
                        t0 = time.perf_counter()
                        res = solver.solve(problem, seed=seed)
                        m = res.metadata
                        row = {
                            "n_assets": n, "instance": instance,
                            "penalty_label": label, "penalty": penalty,
                            "qaoa_seed": seed, "precision": precision,
                            "gradient_evaluations": m["gradient_evaluations"],
                            "objective_evaluations": m["objective_evaluations"],
                            "scipy_status": m["scipy_status"],
                            "scipy_message": m["scipy_message"],
                            "scipy_iterations": m["scipy_iterations"],
                            "scipy_final_grad_norm": m["scipy_final_grad_norm"],
                            "scipy_version": m["scipy_version"],
                            "scipy_ftol": m["scipy_ftol"],
                            "scipy_gtol": m["scipy_gtol"],
                            "best_expectation": float(np.min(m["convergence"])),
                            "seconds": time.perf_counter() - t0,
                        }
                        row.update(acquisition_scores(res, ref, penalty, metrics))
                        out.append(row)
                        print(f"N={n} i{instance} {label:8s} s{seed} {precision:6s} "
                              f"∇{row['gradient_evaluations']:>4} "
                              f"status {row['scipy_status']:>3} "
                              f"⟨H⟩ {row['best_expectation']:+.4f}  "
                              f"{row['scipy_message']}", flush=True)
    return out


def acquisition_scores(res, ref: InstanceReference, penalty: float,
                       metrics: PortfolioMetrics) -> dict:
    """What the paper reports, from the run's exact final state and its batch."""
    m = res.metadata
    state = _state_scores(np.asarray(m["final_probabilities"], dtype=float), ref, "final")
    p_f, d_cond = conditional_tv(np.asarray(m["final_probabilities"], dtype=float)[ref.feasible.indices])
    batch = score_shots(m["last_samples"], ref, penalty, metrics,
                        solver_bitstring=res.bitstring, solver_energy=res.energy_no_offset)
    return {
        "final_P_F": state["final_P_F"],
        "final_P_tau_main": state["final_P_tau_main"],
        "final_P_opt": state["final_P_opt"],
        "final_Q_tau_main_1000": state["final_Q_tau_main_1000"],
        "final_expected_violation": state["final_expected_violation"],
        "final_D_cond": d_cond,
        "shots_feasible": batch["feasible_shots"],
        "g_F": batch["g_F"],
        "within_tau_main": batch["within_tau_main"],
        "is_optimal": batch["is_optimal"],
    }


#: Columns compared between the two precisions of the same (N, instance, A, seed).
PRECISION_FIELDS = ("best_expectation", "final_P_F", "final_P_tau_main",
                    "final_Q_tau_main_1000", "final_D_cond")


def precision_pairs(rows: list[dict]) -> list[dict]:
    """Single − double, run by run, for the expectation and the acquisition scores.

    One row per (N, instance, penalty, seed) that ran at both precisions. The
    question is whether the two arms agree on what the paper reports; the
    expectation alone cannot answer it.
    """
    by = {}
    for r in rows:
        by.setdefault((r["n_assets"], r["instance"], r["penalty_label"], r["qaoa_seed"]), {})[r["precision"]] = r
    out = []
    for (n, inst, label, seed), arms in sorted(by.items()):
        if not {"single", "double"} <= set(arms):
            continue
        row = {"n_assets": n, "instance": inst, "penalty_label": label, "qaoa_seed": seed}
        for f in PRECISION_FIELDS:
            a, b = float(arms["single"][f]), float(arms["double"][f])
            row[f"{f}_single"], row[f"{f}_double"], row[f"{f}_diff"] = a, b, a - b
        row["within_tau_main_agree"] = arms["single"]["within_tau_main"] == arms["double"]["within_tau_main"]
        row["same_stop_reason"] = arms["single"]["scipy_message"] == arms["double"]["scipy_message"]
        out.append(row)
    return out


def summarise(rows: list[dict]) -> None:
    for n in sorted({r["n_assets"] for r in rows}):
        for precision in PRECISIONS:
            g = [r for r in rows if r["n_assets"] == n and r["precision"] == precision]
            if not g:
                continue
            msg = [str(r["scipy_message"]) for r in g]
            norms = [r["scipy_final_grad_norm"] for r in g
                     if not np.isnan(r["scipy_final_grad_norm"])]
            print(f"N={n} {precision:>6} ({len(g)} runs): "
                  f"ftol {sum('RELATIVE' in m for m in msg)} / "
                  f"gtol {sum('PROJECTED' in m for m in msg)} / "
                  f"line-search failure {sum(m.startswith('ABNORMAL') for m in msg)} / "
                  f"budget {sum('budget' in m for m in msg)} | "
                  f"|g|max {min(norms):.1e}-{max(norms):.1e}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--sizes", type=int, nargs="+", default=[12, 16])
    ap.add_argument("--backend", default="lightning_cpu")
    args = ap.parse_args(argv)
    rows = run(args.sizes, args.backend)
    tag = "_".join(str(n) for n in args.sizes)
    path = RESULTS_DIR / f"paper01_e3_lbfgs_termination_n{tag}.csv"
    write_csv(path, rows)
    print(f"\nwrote {path}\n")
    pairs = precision_pairs(rows)
    pairs_path = RESULTS_DIR / f"paper01_e3_precision_pairs_n{tag}.csv"
    write_csv(pairs_path, pairs)
    print(f"wrote {pairs_path}\n")
    summarise(rows)
    for f in PRECISION_FIELDS:
        d = np.asarray([abs(p[f"{f}_diff"]) for p in pairs], dtype=float)
        if d.size:
            print(f"|single − double| {f:>22}: median {np.nanmedian(d):.2e}, max {np.nanmax(d):.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
