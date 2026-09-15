# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Paper ① / TQE plan E1: instance × initial-seed × penalty cross experiment.

Why
---
The manuscript's penalty sweep (Table XIII) is two sizes at one seed, and its
30-instance QAOA sweep is seed 42 only. Neither can say whether what happens
when A changes depends on the particular asset subset or the particular
initial angles. This script crosses the three factors on the local GPU:

    N ∈ {12, 16, 20}  ×  30 instances  ×  seeds {42..46}  ×  6 penalty settings

Penalty settings, per instance (plan §5.2–5.4):

    1.1·A_crit, 2·A_crit, 5·A_crit, 10·A_crit, A_heur, A_margin

where A_crit is the *exact* threshold from the all-states enumeration
(``scoring.critical_penalty``), A_heur is what ``PortfolioQUBO.formulate``
picks by default (≈ N+1, the setting every published table used), and
A_margin follows Brandhofer et al.'s feasible-mean criterion. When A_crit = 0
the four multiples collapse to A = 0 and are run once, labelled ``A0``.

What each run records (plan §5.5)
---------------------------------
One manifest row in ``results/paper01_e1_cross{_tag}.csv`` and one artifact
``results/e1_runs/N{N}_i{inst}_{label}_s{seed}.npz`` with the initial and
final angles, the full angle/expectation trajectory, the final and initial
states' probability over the feasible set (from the exact statevector, not
from shots), the per-cardinality probability mass, and the shot counts.

The scores are all from ``src.qubo.scoring`` -- the evaluator that never reads
``Q`` -- so a run is judged by f(x), g_F, P_F, P_τ, Q_τ(S) and its rank in
the feasible set, with the conventional g_off recorded alongside for the
re-scoring analysis of plan §4.3.

E2 comes for free: the initial state's probabilities are the "unoptimised
QAOA" control, and the uniform controls need only the feasible-set census.

Run
---
    # Pilot (plan §13.2): 3 sizes × 2 instances × 1 seed × 6 A = 36 nominal
    python -m experiments.paper01_qubo_baseline.run_e1_cross \
        --backend lightning_gpu --precision single \
        --instances 0 1 --seeds 42 --results-tag pilot

    # Full E1, resumable
    python -m experiments.paper01_qubo_baseline.run_e1_cross \
        --backend lightning_gpu --precision single --resume

    # E3 (plan §7): the same runs under a different optimizer setting, on a
    # pre-fixed block of 10 instances x 5 seeds x {Aheur, Amargin}, so each
    # row pairs with an E1 row on (N, instance, penalty, seed).
    python -m experiments.paper01_qubo_baseline.run_e1_cross \
        --backend lightning_gpu --precision single --resume \
        --instances 0 1 2 3 4 5 6 7 8 9 --penalties Aheur Amargin \
        --stepsize 0.03 --steps 100 --results-tag e3_adam003
    python -m experiments.paper01_qubo_baseline.run_e1_cross \
        --backend lightning_gpu --precision single --resume \
        --instances 0 1 2 3 4 5 6 7 8 9 --penalties Aheur Amargin \
        --optimizer lbfgs --steps 100 --results-tag e3_lbfgs
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import csv
import sys
import time
import traceback
from math import comb
from pathlib import Path

import numpy as np

from experiments.paper01_qubo_baseline.run_degeneracy import RISK_FREE_RATE
from experiments.paper01_qubo_baseline.run_experiment import (
    DATA_END_DATE,
    LOOKBACK_YEARS,
    RESULTS_DIR,
    RISK_AVERSION,
    canonical_overwrite_refusal,
    write_csv,
)
from experiments.paper01_qubo_baseline.run_instances import (
    instance_digest,
    instance_tickers,
)
from src.finance.data_loader import FinanceDataLoader
from src.finance.metrics import PortfolioMetrics
from src.qubo import scoring
from src.qubo.portfolio import PortfolioQUBO

#: Bump whenever the run specification changes; rows from another version
#: are refused by --resume rather than mixed (plan §13.2).
SPEC_VERSION = "e1-v2"  # v2: rank from the feasible set's own value (ulp-safe)

DEFAULT_N = [12, 16, 20]
DEFAULT_INSTANCES = list(range(30))
DEFAULT_SEEDS = [42, 43, 44, 45, 46]

#: Multiples of A_crit (plan §5.2). Labels are the manifest's ``penalty_label``.
MULTIPLES = [(1.1, "1.1xAcrit"), (2.0, "2xAcrit"), (5.0, "5xAcrit"), (10.0, "10xAcrit")]

#: Safety margin added to the A_margin threshold, in objective units (the
#: objective is O(K^2) ≤ 16 here, so this is ~1e-7 relative). Recorded per row.
EPS_A = 1e-6

#: Quality thresholds on g_F (fractions of the feasible range, plan §4.2).
TAU_MAIN = 0.01
TAU_ALT = 0.05

#: "Exact optimum": f − f* ≤ OPT_TOL · Δ_F, so ties count as optimal.
OPT_TOL = 1e-9

#: Shot budgets for Q_τ(S).
SHOT_BUDGETS = [10, 100, 1000, 10000]

ARTIFACT_DIR = RESULTS_DIR / "e1_runs"

RUN_KEY = ("n_assets", "instance", "penalty_label", "qaoa_seed")


class InstanceReference:
    """Everything about one (N, instance) that does not depend on A or seed."""

    def __init__(self, n: int, instance: int, loader, formulator, end_date: str):
        self.n, self.instance = n, instance
        self.tickers = instance_tickers(n, instance)
        self.data = loader.load(self.tickers, end_date=end_date,
                                lookback_years=LOOKBACK_YEARS)
        self.k = min(max(2, n // 5), self.data.n_assets)
        t0 = time.perf_counter()
        # A_heur is the formulator's default; the scales in the metadata are
        # the same for every A, so one formulation defines the objective.
        heur = formulator.formulate(self.data.returns, self.data.covariance,
                                    num_select=self.k, risk_aversion=RISK_AVERSION)
        self.a_heur = float(heur.metadata["penalty_strength"])
        self.objective = scoring.Objective.from_problem(
            heur, self.data.returns, self.data.covariance)
        self.states = scoring.enumerate_all(self.objective)
        self.feasible = scoring.enumerate_feasible(self.objective)
        self.a_m = self.states.cardinality_minima()
        self.a_crit, self.binding = scoring.critical_penalty(self.a_m, self.k)
        self.a_margin, self.target = scoring.margin_penalty(
            self.a_m, self.k, self.feasible.f_bar, EPS_A)
        self.within_main = self.feasible.within(TAU_MAIN)
        self.within_alt = self.feasible.within(TAU_ALT)
        self.optimal = self.feasible.is_optimal(OPT_TOL)
        self.reference_seconds = time.perf_counter() - t0

    def conditions(self) -> list[tuple[str, float]]:
        """Penalty settings for this instance, duplicates removed (plan §5.4)."""
        if self.a_crit > 0.0:
            out = [(label, m * self.a_crit) for m, label in MULTIPLES]
        else:
            out = [("A0", 0.0)]
        out += [("Aheur", self.a_heur), ("Amargin", self.a_margin)]
        return out

    def row(self) -> dict:
        feas = self.feasible
        size = feas.size
        return {
            "spec_version": SPEC_VERSION,
            "n_assets": self.n,
            "n_select": self.k,
            "instance": self.instance,
            "instance_digest": instance_digest(self.tickers),
            "tickers": " ".join(self.tickers),
            "feasible_set_size": size,
            "f_star": feas.f_star,
            "f_max_feasible": feas.f_max,
            "delta_F": feas.delta,
            "f_bar_feasible": feas.f_bar,
            "optimum_index": feas.optimum_index,
            "n_optimal_ties": int(self.optimal.sum()),
            "a_crit": self.a_crit,
            "a_crit_binding_m": self.binding,
            "a_crit_is_zero": self.a_crit == 0.0,
            "a_heur": self.a_heur,
            "a_heur_over_crit": (self.a_heur / self.a_crit if self.a_crit > 0 else ""),
            "a_margin": self.a_margin,
            "a_margin_target": self.target,
            "eps_A": EPS_A,
            "cardinality_minima": " ".join(f"{v:.17g}" for v in self.a_m),
            # Uniform controls (plan §6.2): over all 2^N states and over F.
            "count_within_tau_main": int(self.within_main.sum()),
            "count_within_tau_alt": int(self.within_alt.sum()),
            "P_F_uniform_all": size / 2 ** self.n,
            "P_tau_main_uniform_all": self.within_main.sum() / 2 ** self.n,
            "P_tau_main_uniform_F": self.within_main.mean(),
            "P_opt_uniform_all": self.optimal.sum() / 2 ** self.n,
            "P_opt_uniform_F": self.optimal.mean(),
            "reference_seconds": self.reference_seconds,
            "data_start": self.data.start_date,
            "data_end": self.data.end_date,
        }


def _state_scores(P: np.ndarray, ref: InstanceReference, prefix: str) -> dict:
    """P_F, P_opt, P_τ and expectations of f and V from a probability vector."""
    feas = ref.feasible
    pf = P[feas.indices]
    card = ref.states.cardinality.astype(float)
    out = {
        f"{prefix}_probs_sum": float(P.sum()),
        f"{prefix}_P_F": float(pf.sum()),
        f"{prefix}_P_opt": float(pf[ref.optimal].sum()),
        f"{prefix}_P_tau_main": float(pf[ref.within_main].sum()),
        f"{prefix}_P_tau_alt": float(pf[ref.within_alt].sum()),
        f"{prefix}_expected_f": float(P @ ref.states.f),
        f"{prefix}_expected_violation": float(P @ (card - ref.k) ** 2),
    }
    for s in SHOT_BUDGETS:
        out[f"{prefix}_Q_tau_main_{s}"] = scoring.success_probability(out[f"{prefix}_P_tau_main"], s)
        out[f"{prefix}_Q_opt_{s}"] = scoring.success_probability(out[f"{prefix}_P_opt"], s)
    return out


def score_shots(samples: np.ndarray, ref: InstanceReference, a_value: float,
                metrics: PortfolioMetrics, solver_bitstring=None,
                solver_energy: float | None = None) -> dict:
    """Score one batch of shots against the instance's enumerated feasible set.

    Shared by E1 (one batch per run, from the solver) and E4 (several batches
    of one fixed state, from several backends). Plan §5.5: the lowest-energy
    shot is scored whether or not it is feasible; the best *feasible* shot only
    when one exists; a batch without one is a counted outcome.

    Args:
        samples: 0/1 array of shape (shots, N).
        ref: The instance's reference (feasible set, thresholds, data).
        a_value: Penalty weight the circuit was built with.
        metrics: Portfolio metrics evaluator.
        solver_bitstring: If given, the solver's own best shot, recorded as
            agreeing or not with this scorer's choice.
        solver_energy: If given, the solver's x@Q@x for that shot; the
            scorer's E_A must agree with it.

    Returns:
        The manifest row for the batch.
    """
    n, k = ref.n, ref.k
    feas = ref.feasible
    shot_idx, shot_cnt = scoring.shot_counts(samples)
    shot_card = ref.states.cardinality[shot_idx]
    shot_f = ref.states.f[shot_idx]
    shot_E = shot_f + a_value * (shot_card.astype(float) - k) ** 2 - a_value * k * k
    n_shots = int(shot_cnt.sum())

    # Minimum-E_A shot, feasible or not: what the solver returns.
    best_all = int(np.argmin(shot_E))
    best_all_idx = int(shot_idx[best_all])
    best_all_x = scoring.bitstrings_from_indices([best_all_idx], n)[0]
    # The solver picked its best shot by x@Q@x; this path re-derives it from
    # f and V. Disagreement (a tie broken differently, or a real bug) is
    # recorded rather than hidden, and the energy itself must agree.
    scorer_agrees = (bool(np.array_equal(best_all_x, solver_bitstring))
                     if solver_bitstring is not None else "")
    if solver_energy is not None and abs(float(shot_E[best_all]) - solver_energy) > 1e-6 * max(1.0, abs(solver_energy)):
        raise RuntimeError(f"E_A {shot_E[best_all]!r} disagrees with x@Q@x {solver_energy!r}")

    row = {
        "shots": n_shots,
        "feasible_shots": int(shot_cnt[shot_card == k].sum()),
        "distinct_shots": int(shot_idx.size),
        "best_all_index": best_all_idx,
        "best_all_cardinality": int(shot_card[best_all]),
        "best_all_feasible": bool(shot_card[best_all] == k),
        "best_all_E": float(shot_E[best_all]),
        "best_all_f": float(shot_f[best_all]),
        "energy_no_offset_solver": solver_energy if solver_energy is not None else "",
        "best_shot_scorer_agrees": scorer_agrees,
    }
    row["feasible_shot_fraction"] = row["feasible_shots"] / n_shots

    # Best feasible shot (plan §5.5): only when one exists.
    feas_mask = shot_card == k
    if feas_mask.any():
        j = int(np.flatnonzero(feas_mask)[np.argmin(shot_f[feas_mask])])
        # Score with the feasible set's own value of this solution, so the
        # rank is exactly consistent with the census (see FeasibleSet.position).
        f_bf = float(feas.f[feas.position(int(shot_idx[j]))])
        rank, ties = feas.rank(f_bf)
        x_bf = scoring.bitstrings_from_indices([int(shot_idx[j])], n)[0]
        stats = metrics.evaluate([i for i in range(n) if x_bf[i] > 0.5],
                                 ref.data.returns, ref.data.covariance, target_k=k)
        row.update({
            "best_feasible_index": int(shot_idx[j]),
            "f_best_feasible": f_bf,
            "g_F": float(feas.gap_range(f_bf)),
            "g_off": scoring.gap_offset(f_bf, feas.f_star, a_value, k),
            "rank_best_feasible": rank,
            "rank_ties": ties,
            "is_optimal": bool(f_bf - feas.f_star <= OPT_TOL * feas.delta),
            "within_tau_main": bool(feas.gap_range(f_bf) <= TAU_MAIN),
            "within_tau_alt": bool(feas.gap_range(f_bf) <= TAU_ALT),
            "sharpe_best_feasible": stats.sharpe_ratio,
            "return_best_feasible": stats.expected_return,
            "volatility_best_feasible": stats.volatility,
        })
        # Empirical shot-level success (the finite-sample counterpart of P_τ).
        gF_shots = feas.gap_range(shot_f)
        row["shots_within_tau_main"] = int(shot_cnt[feas_mask & (gF_shots <= TAU_MAIN)].sum())
        row["shots_optimal"] = int(shot_cnt[feas_mask & (shot_f - feas.f_star <= OPT_TOL * feas.delta)].sum())
    else:
        for key in ("best_feasible_index", "f_best_feasible", "g_F", "g_off",
                    "rank_best_feasible", "rank_ties", "is_optimal", "within_tau_main",
                    "within_tau_alt", "sharpe_best_feasible", "return_best_feasible",
                    "volatility_best_feasible"):
            row[key] = ""
        row["shots_within_tau_main"] = 0
        row["shots_optimal"] = 0

    return row


def score_run(res, problem, ref: InstanceReference, a_value: float,
              metrics: PortfolioMetrics) -> tuple[dict, dict]:
    """Turn one SolverResult into a manifest row and an artifact payload."""
    md = res.metadata
    n, k = ref.n, ref.k
    feas = ref.feasible
    obj = ref.objective
    t0 = time.perf_counter()

    samples = md["last_samples"]
    row = score_shots(samples, ref, a_value, metrics,
                      solver_bitstring=res.bitstring, solver_energy=res.energy_no_offset)
    shot_idx, shot_cnt = scoring.shot_counts(samples)

    opt_x = scoring.bitstrings_from_indices([feas.optimum_index], n)[0]
    row["sharpe_optimum"] = metrics.evaluate(
        [i for i in range(n) if opt_x[i] > 0.5], ref.data.returns,
        ref.data.covariance, target_k=k).sharpe_ratio
    row["deflation_D"] = scoring.deflation_factor(feas.f_star, a_value, k, feas.delta)
    row["abs_E_star"] = abs(feas.f_star - a_value * k * k)

    # Exact-state scores, final and initial (plan §6.2 controls).
    P_final = md["final_probabilities"]
    P_init = md["initial_probabilities"]
    row.update(_state_scores(P_final, ref, "final"))
    row.update(_state_scores(P_init, ref, "initial"))

    conv = np.asarray(md["convergence"], dtype=float)
    row.update({
        "hamiltonian_scale": md["hamiltonian_scale"],
        "final_expectation": md["final_expectation"],
        "expectation_at_final_angles": md["expectation_at_final_angles"],
        "best_expectation": md["best_expectation"],
        "best_expectation_step": md["best_expectation_step"],
        # Plan §7.2: the state of the optimisation at fixed evaluation counts,
        # so optimizers with different per-step costs compare at equal budget.
        "best_expectation_by_50": float(conv[:50].min()) if conv.size else None,
        "best_expectation_by_100": float(conv[:100].min()) if conv.size else None,
        "objective_evaluations": md.get("objective_evaluations"),
        "gradient_evaluations": md.get("gradient_evaluations"),
        "initial_angles": " ".join(f"{v:.17g}" for v in md["initial_angles"]),
        "final_angles": " ".join(f"{v:.17g}" for v in md["final_angles"]),
        "optimisation_seconds": md["optimisation_seconds"],
        "sampling_seconds": md["sampling_seconds"],
        "state_seconds": md["state_seconds"],
        "runtime_s": res.runtime_seconds,
    })
    row["evaluation_seconds"] = time.perf_counter() - t0

    card_final = scoring.cardinality_mass(P_final, ref.states.cardinality, n)
    card_init = scoring.cardinality_mass(P_init, ref.states.cardinality, n)
    artifact = {
        "initial_angles": md["initial_angles"],
        "final_angles": md["final_angles"],
        "angles_trajectory": md["angles_trajectory"],
        "convergence": np.asarray(md["convergence"], dtype=float),
        "feasible_indices": feas.indices,
        "feasible_f": feas.f,
        "final_probs_feasible": P_final[feas.indices],
        "initial_probs_feasible": P_init[feas.indices],
        "final_cardinality_mass": card_final,
        "initial_cardinality_mass": card_init,
        "shot_indices": shot_idx,
        "shot_counts": shot_cnt,
        "penalty": np.float64(a_value),
        "hamiltonian_scale": np.float64(md["hamiltonian_scale"]),
    }
    return row, artifact


def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E1: instance x seed x penalty cross")
    p.add_argument("--n", type=int, nargs="+", default=DEFAULT_N)
    p.add_argument("--instances", type=int, nargs="+", default=DEFAULT_INSTANCES,
                   help="Instance indices (0 is the published subset)")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--backend", default="lightning_gpu",
                   choices=["lightning_cpu", "lightning_gpu"],
                   help="Local simulators only; this is thousands of runs")
    p.add_argument("--precision", default="single", choices=["single", "double"])
    p.add_argument("--p-layers", type=int, default=2)
    p.add_argument("--steps", type=int, default=50,
                   help="ADAM steps, or the gradient-evaluation budget for L-BFGS")
    p.add_argument("--optimizer", default="adam", choices=["adam", "lbfgs"])
    p.add_argument("--stepsize", type=float, default=0.1, help="ADAM step size")
    p.add_argument("--penalties", nargs="+", default=None,
                   help="Restrict to these penalty labels (e.g. Aheur Amargin)")
    p.add_argument("--shots", type=int, default=1000)
    p.add_argument("--end-date", default=DATA_END_DATE)
    p.add_argument("--results-tag", default=None)
    p.add_argument("--write-canonical", action="store_true",
                   help="Allow an untagged, non-resumed run to replace the "
                        "committed paper01_e1_cross.csv / paper01_e1_reference.csv "
                        "and the e1_runs/ artifacts; refused otherwise")
    p.add_argument("--resume", action="store_true",
                   help="Skip (N, instance, label, seed) rows already in the manifest")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from src.solvers.braket_solver import BraketSolver

    suffix = f"_{args.results_tag}" if args.results_tag else ""
    manifest_path = RESULTS_DIR / f"paper01_e1_cross{suffix}.csv"
    reference_path = RESULTS_DIR / f"paper01_e1_reference{suffix}.csv"
    artifact_dir = ARTIFACT_DIR if not args.results_tag else ARTIFACT_DIR.with_name(
        f"e1_runs_{args.results_tag}")
    refusal = canonical_overwrite_refusal(
        (manifest_path, reference_path), args.results_tag, args.write_canonical,
        resume=args.resume)
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    artifact_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(manifest_path) if args.resume else []
    for r in rows:
        if r.get("spec_version") != SPEC_VERSION:
            print(f"ABORT: {manifest_path} holds spec {r.get('spec_version')!r}, "
                  f"this code is {SPEC_VERSION!r}; do not mix", file=sys.stderr)
            return 1
    done = {tuple(r[k] for k in RUN_KEY) for r in rows}
    reference_rows = read_manifest(reference_path) if args.resume else []
    reference_done = {(r["n_assets"], r["instance"]) for r in reference_rows}

    loader, formulator = FinanceDataLoader(), PortfolioQUBO()
    metrics = PortfolioMetrics(risk_free_rate=RISK_FREE_RATE)
    solver = BraketSolver(backend=args.backend, precision=args.precision,
                          p_layers=args.p_layers, n_shots=args.shots,
                          n_optimizer_steps=args.steps, record_state=True,
                          optimizer=args.optimizer, stepsize=args.stepsize)
    optimizer_label = (f"adam_{args.stepsize:g}" if args.optimizer == "adam"
                       else f"lbfgs_{args.steps}grad")

    print(f"E1 {SPEC_VERSION}: N={args.n} x {len(args.instances)} instances x "
          f"{len(args.seeds)} seeds x <=6 penalties on {args.backend}/{args.precision}, "
          f"optimizer {optimizer_label}; "
          f"{len(done)} rows already done", flush=True)
    t_start = time.perf_counter()
    n_new = 0
    for n in sorted(args.n):
        for instance in args.instances:
            ref = InstanceReference(n, instance, loader, formulator, args.end_date)
            if (str(n), str(instance)) not in reference_done:
                reference_rows.append(ref.row())
                reference_done.add((str(n), str(instance)))
                write_csv(reference_path, reference_rows)
            conds = ref.conditions()
            if args.penalties:
                conds = [c for c in conds if c[0] in args.penalties]
            print(f"\nN={n} K={ref.k} inst {instance:>2} |F|={ref.feasible.size:,} "
                  f"A_crit={ref.a_crit:.5f} (m={ref.binding}) A_margin={ref.a_margin:.5f} "
                  f"A_heur={ref.a_heur:.3f} Δ_F={ref.feasible.delta:.4f} "
                  f"[{ref.reference_seconds:.1f}s] conditions={[c[0] for c in conds]}",
                  flush=True)
            for label, a_value in conds:
                problem = formulator.formulate(
                    ref.data.returns, ref.data.covariance, num_select=ref.k,
                    risk_aversion=RISK_AVERSION, penalty_strength=a_value)
                for seed in args.seeds:
                    key = (str(n), str(instance), label, str(seed))
                    if key in done:
                        continue
                    base = {
                        "spec_version": SPEC_VERSION,
                        "run_id": f"N{n}_i{instance:02d}_{label}_s{seed}",
                        "n_assets": n, "n_select": ref.k, "instance": instance,
                        "instance_digest": instance_digest(ref.tickers),
                        "penalty_label": label, "penalty": a_value,
                        "a_crit": ref.a_crit,
                        "a_over_crit": (a_value / ref.a_crit if ref.a_crit > 0 else ""),
                        "a_margin": ref.a_margin, "a_heur": ref.a_heur,
                        "qaoa_seed": seed,
                        "backend": args.backend, "precision": args.precision,
                        "p_layers": args.p_layers, "steps": args.steps,
                        "optimizer": optimizer_label,
                        "stepsize": args.stepsize if args.optimizer == "adam" else "",
                        "f_star": ref.feasible.f_star, "delta_F": ref.feasible.delta,
                        "feasible_set_size": ref.feasible.size,
                        "data_start": ref.data.start_date, "data_end": ref.data.end_date,
                    }
                    try:
                        res = solver.solve(problem, seed=seed)
                        scored, artifact = score_run(res, problem, ref, a_value, metrics)
                        np.savez_compressed(artifact_dir / f"{base['run_id']}.npz", **artifact)
                        row = {**base, "status": "ok", "error": "", **scored}
                        g_f = "none" if row["g_F"] == "" else f"{row['g_F']:.4f}"
                        flag = "" if row["best_all_feasible"] else "best-shot INFEASIBLE "
                        print(f"  {label:>9s} s{seed}  A={a_value:8.4f}  "
                              f"P_F={row['final_P_F']:.3f} P_τ={row['final_P_tau_main']:.3f} "
                              f"P_opt={row['final_P_opt']:.4f}  best_feas g_F={g_f} "
                              f"rank={row['rank_best_feasible']}  {flag}"
                              f"{row['runtime_s']:.1f}s", flush=True)
                    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
                        traceback.print_exc()
                        row = {**base, "status": f"error:{type(exc).__name__}",
                               "error": str(exc)[:200]}
                        print(f"  {label:>9s} s{seed}  FAILED {type(exc).__name__}", flush=True)
                    rows.append(row)
                    done.add(key)
                    n_new += 1
                    # Rows may lack keys when a run failed; keep one header.
                    fields = list(rows[0].keys())
                    for r in rows:
                        for k_ in r:
                            if k_ not in fields:
                                fields.append(k_)
                    with open(manifest_path, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=fields)
                        w.writeheader()
                        w.writerows(rows)
    elapsed = time.perf_counter() - t_start
    print(f"\n{n_new} new runs in {elapsed/60:.1f} min; {len(rows)} rows → {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
