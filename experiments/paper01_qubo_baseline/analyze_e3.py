# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Paper ① / TQE plan §7: summarise the E3 optimizer controls against E1.

E3 re-ran a pre-fixed block of E1 (instances 0-9 × seeds 42-46 × {Aheur,
Amargin}, N ∈ {12,16,20}; 300 runs) under two other optimizer settings:

    e3_adam003   ADAM, step 0.03, 100 steps          (control A)
    e3_lbfgs     L-BFGS-B on the analytic expectation with adjoint
                 gradients, ≤ 100 gradient evaluations (control B)

Reads ``results/paper01_e1_cross.csv`` (baseline, ADAM 0.1 × 50) and
``results/paper01_e1_cross_e3_{adam003,lbfgs}.csv`` and writes

    paper01_e3_summary.csv     per (N, penalty, optimizer): medians/IQR of the
                               optimised expectation, its value after 50
                               evaluations, P_F / P_τ / Q_τ(1000), g_F,
                               evaluation counts, wall-clock, plus the
                               within-instance seed spread of the same
                               quantities (plan §7.2: budget and variation
                               side by side), and the two batch counts
                               (optimum hits, runs without a feasible shot)
                               beside what the recorded exact states predict
                               for them under independent draws
    paper01_e3_contrasts.csv   paired treatment − baseline on (N, instance,
                               penalty, seed), instance-level bootstrap CI
    paper01_e3_multistart.csv  plan §7.3 re-analysis of E1: five starts with
                               1,000 shots each, pooled, against a random
                               single start with 5,000 shots -- the gain per
                               instance, and what it costs

Statistics follow plan §10: pairs are first reduced within an instance
(median over seeds), then instances are bootstrapped. ⟨H⟩ is compared only
between runs that share the same A (the pairing guarantees this); it is
never compared across penalty conditions.

Run:
    python -m experiments.paper01_qubo_baseline.analyze_e3
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np

from experiments.paper01_qubo_baseline.analyze_e1 import (
    BOOTSTRAP,
    RNG_SEED,
    _bootstrap_ci,
    _f,
    _instance_level,
    _q,
    load,
)
from src.qubo.scoring import success_probability
from experiments.paper01_qubo_baseline.run_e1_cross import RESULTS_DIR
from experiments.paper01_qubo_baseline.run_experiment import write_csv

#: (tag, label) of the E3 controls; the baseline is E1 itself.
CONTROLS = [("e3_adam003", "adam_0.03"), ("e3_lbfgs", "lbfgs_100grad")]
BASELINE_LABEL = "adam_0.1"

#: The E3 block (plan §7.1).
E3_INSTANCES = set(range(10))
E3_PENALTIES = ("Aheur", "Amargin")

#: Quantities contrasted, and whether larger is better.
CONTRAST_FIELDS = [("best_expectation", False), ("best_expectation_by_50", False),
                   ("expectation_at_final_angles", False),
                   ("final_P_F", True), ("final_P_tau_main", True),
                   ("final_Q_tau_main_1000", True), ("final_P_opt", True),
                   ("g_F", False), ("rank_best_feasible", False),
                   ("objective_evaluations", False), ("gradient_evaluations", False),
                   ("optimisation_seconds", False)]

PENALTY_ORDER = ["Aheur", "Amargin"]


def _key(r: dict) -> tuple:
    return (r["n_assets"], r["instance"], r["penalty_label"], r["qaoa_seed"])


def _normalise_baseline(rows: list[dict]) -> list[dict]:
    """Fill the budget columns E1 did not record.

    E1 ran ADAM for exactly 50 steps; ``step_and_cost`` evaluates the
    objective and its gradient once per step, so both counts are 50 and the
    best expectation within the first 50 evaluations is the best overall.
    """
    out = []
    for r in rows:
        r = dict(r)
        if r.get("objective_evaluations", "") == "":
            r["objective_evaluations"] = r["steps"]
            r["gradient_evaluations"] = r["steps"]
        if r.get("best_expectation_by_50", "") == "":
            r["best_expectation_by_50"] = r["best_expectation"]
        if r.get("best_expectation_by_100", "") == "":
            r["best_expectation_by_100"] = r["best_expectation"]
        r["optimizer"] = BASELINE_LABEL
        out.append(r)
    return out


def e3_block(rows: list[dict]) -> list[dict]:
    return [r for r in rows
            if int(r["instance"]) in E3_INSTANCES and r["penalty_label"] in E3_PENALTIES]


def _seed_spread(g: list[dict], field: str) -> float:
    """Median over instances of (max − min over seeds) of ``field``."""
    by_inst: dict[str, list[float]] = defaultdict(list)
    for r in g:
        v = _f(r.get(field, ""))
        if not np.isnan(v):
            by_inst[r["instance"]].append(v)
    spreads = [max(v) - min(v) for v in by_inst.values() if len(v) >= 2]
    return float(np.median(spreads)) if spreads else np.nan


def summarise(groups: dict[str, list[dict]]) -> list[dict]:
    out = []
    for n in sorted({int(r["n_assets"]) for g in groups.values() for r in g}):
        for label in PENALTY_ORDER:
            for opt in [BASELINE_LABEL] + [lab for _, lab in CONTROLS]:
                g = [r for r in groups[opt]
                     if int(r["n_assets"]) == n and r["penalty_label"] == label]
                ok = [r for r in g if r["status"] == "ok"]
                feas = [r for r in ok if r["g_F"] != ""]
                if not g:
                    continue
                row = {
                    "n_assets": n, "penalty_label": label, "optimizer": opt,
                    "runs": len(g), "runs_ok": len(ok), "runs_failed": len(g) - len(ok),
                    "instances": len({r["instance"] for r in g}),
                    "no_feasible_shot_runs": len(ok) - len(feas),
                    "optimum_hits": sum(r["is_optimal"] == "True" for r in ok),
                    # What the recorded exact states predict for the two batch
                    # counts under independent draws: Σ_j (1 − P_F,j)^S runs
                    # with no feasible shot and Σ_j [1 − (1 − P_0,j)^S] runs
                    # whose best feasible shot is the exact optimum. Not a
                    # binomial with one shared rate -- every run has its own.
                    # This is the check that exposed the L-BFGS-B handoff
                    # defect (batches drawn at the initial angles): the ADAM
                    # cells sit within a few units of these, the pre-fix
                    # L-BFGS-B cells did not (6 hits against 14.8 expected at
                    # N = 16, A_heur; 9 runs without a feasible shot against
                    # 0.2 at N = 20, A_heur).
                    "expected_no_feasible_shot_runs": float(sum(
                        (1.0 - _f(r["final_P_F"])) ** int(float(r["shots"])) for r in ok)),
                    "expected_optimum_hits": float(sum(
                        1.0 - (1.0 - _f(r["final_P_opt"])) ** int(float(r["shots"])) for r in ok)),
                    "optimal_rate": np.mean([r["is_optimal"] == "True" for r in feas]) if feas else np.nan,
                    "within_tau_main_rate": np.mean([r["within_tau_main"] == "True" for r in feas]) if feas else np.nan,
                    # The same two rates over every completed run, counting a
                    # run that produced no feasible shot as a failure to hit
                    # rather than as absent. That is the operational rate: at
                    # N = 20, A_heur, L-BFGS-B nine runs of fifty found no
                    # feasible shot, and conditioning them away turned 0.04
                    # into 0.05 and an equal rate into an advantage.
                    "optimal_rate_unconditional": np.mean(
                        [r["is_optimal"] == "True" for r in ok]) if ok else np.nan,
                    "within_tau_main_rate_unconditional": np.mean(
                        [r["within_tau_main"] == "True" for r in ok]) if ok else np.nan,
                    # Where the optimizer stopped relative to its cap: a run
                    # that used fewer gradient evaluations than the cap
                    # converged by SciPy's own criterion.
                    "gradient_evaluations_below_cap_rate": np.mean(
                        [_f(r["gradient_evaluations"]) < _f(r["steps"]) for r in ok]) if ok else np.nan,
                }
                for field in ("best_expectation", "best_expectation_by_50",
                              "expectation_at_final_angles", "final_P_F",
                              "final_P_tau_main", "final_Q_tau_main_1000", "final_P_opt",
                              "g_F", "rank_best_feasible", "objective_evaluations",
                              "gradient_evaluations", "optimisation_seconds", "runtime_s"):
                    vals = [_f(r.get(field, "")) for r in ok]
                    row[f"{field}_p25"] = _q(vals, 25)
                    row[f"{field}_median"] = _q(vals, 50)
                    row[f"{field}_p75"] = _q(vals, 75)
                for field in ("best_expectation", "final_P_tau_main", "final_Q_tau_main_1000"):
                    row[f"{field}_seed_spread_median"] = _seed_spread(ok, field)
                out.append(row)
    return out


def contrasts(groups: dict[str, list[dict]]) -> list[dict]:
    rng = np.random.default_rng(RNG_SEED)
    base = {_key(r): r for r in groups[BASELINE_LABEL] if r["status"] == "ok"}
    out = []
    for n in sorted({int(r["n_assets"]) for r in groups[BASELINE_LABEL]}):
        for _, opt in CONTROLS:
            for label in PENALTY_ORDER:
                for field, larger_better in CONTRAST_FIELDS:
                    pairs: dict[str, list[float]] = defaultdict(list)
                    n_pairs = n_dropped = 0
                    for r in groups[opt]:
                        if r["status"] != "ok" or int(r["n_assets"]) != n \
                                or r["penalty_label"] != label:
                            continue
                        b = base.get(_key(r))
                        if b is None:
                            continue
                        if abs(_f(r["penalty"]) - _f(b["penalty"])) > 1e-9 * max(1.0, abs(_f(b["penalty"]))):
                            raise RuntimeError(f"A differs between E1 and E3 for {_key(r)}")
                        a, c = _f(r.get(field, "")), _f(b.get(field, ""))
                        if np.isnan(a) or np.isnan(c):
                            n_dropped += 1
                            continue
                        pairs[r["instance"]].append(a - c)
                        n_pairs += 1
                    inst = _instance_level(pairs)
                    if inst.size == 0:
                        continue
                    lo, hi = _bootstrap_ci(inst, rng)
                    med = float(np.median(inst))
                    sign = "equal" if med == 0 else (
                        "better" if (med > 0) == larger_better else "worse")
                    out.append({
                        "n_assets": n, "treatment": opt, "control": BASELINE_LABEL,
                        "penalty_label": label, "field": field,
                        "pairs": n_pairs, "pairs_dropped": n_dropped,
                        "instances": int(inst.size),
                        "instance_median_diff": med, "ci95_low": lo, "ci95_high": hi,
                        "instances_treatment_better": int(((inst > 0) if larger_better else (inst < 0)).sum()),
                        "instances_worse": int(((inst < 0) if larger_better else (inst > 0)).sum()),
                        "instances_tied": int((inst == 0).sum()),
                        # A signed median can be small because the changes are
                        # small or because they cancel. These separate the two:
                        # the typical size of a change, its spread, and its
                        # tail. Reporting only the signed median was how the
                        # draft came to describe an optimizer setting that moves
                        # one cell by 0.32 as moving things by 1e-3 to 1e-2.
                        "instance_median_abs_diff": float(np.median(np.abs(inst))),
                        "instance_iqr_diff": float(np.percentile(inst, 75)
                                                   - np.percentile(inst, 25)),
                        "instance_p95_abs_diff": float(np.percentile(np.abs(inst), 95)),
                        "instance_max_abs_diff": float(np.max(np.abs(inst))),
                        "direction": sign,
                    })
    return out


def multistart(rows: list[dict], refs: list[dict] | None = None) -> list[dict]:
    """Plan §7.3: five starts against one, priced in optimisations and in shots.

    The comparison is between two strategies that can both be run without
    knowing the answer, at the same 5,000 shots:

    * **random single start, 5,000 shots** -- optimise once from a seed drawn
      uniformly from the five, sample that state 5,000 times. Per instance i,
      Q_single,i = (1/5) Σ_k [1 − (1 − p_ik)^5000]: the Q of each seed, then
      the mean over seeds. (The mean *p* over seeds put into Q would be a
      different, non-implementable quantity, and so would the median seed's
      p: nobody knows before optimising which seed will turn out to be the
      median one. An earlier draft compared against the median seed.)
    * **five starts, 1,000 shots each, batches pooled** -- optimise five
      times, draw 1,000 shots from each state, keep the best feasible
      candidate over all 5,000. Q_multi,i = 1 − ∏_k (1 − p_ik)^1000.

    With a_k = (1 − p_ik)^1000 the two are mean_k(a_k^5) and ∏_k a_k, so by
    AM–GM Q_multi,i ≥ Q_single,i on every instance, identically. The question
    is therefore not the sign but the size: the increment per instance,
    Q_multi,i − Q_single,i, with an instance-level bootstrap CI and the count
    of instances where it exceeds 0.01, set against the four extra
    optimisations it costs (``all_seeds`` − ``single_seed`` seconds).

    Also carried, per cell: the median and mean seed at 1,000 shots (the E1
    baseline, and what one draw of a seed returns), the best seed's exact
    P_τ and Q_τ(1000) (an oracle, since choosing it needs the answer), the
    median seed sampled 5,000 times (a summary, not a strategy; kept so the
    earlier table can be reproduced), and uniform sampling from the feasible
    set at 5,000 shots as the no-optimisation control at the same candidate
    count. The same pair of strategies is evaluated for reaching the exact
    optimum, from ``final_P_opt``.
    """
    rng = np.random.default_rng(RNG_SEED)
    ref_by = {(r["n_assets"], r["instance"]): r for r in (refs or [])}
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["status"] == "ok":
            groups[(int(r["n_assets"]), r["penalty_label"], r["instance"])].append(r)
    per_cell: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (n, label, _inst), g in groups.items():
        cell = per_cell[(n, label)]
        p_tau = [_f(r["final_P_tau_main"]) for r in g]
        q1000 = [_f(r["final_Q_tau_main_1000"]) for r in g]
        opt = [r["is_optimal"] == "True" for r in g]
        g_f = [_f(r["g_F"]) for r in g if r["g_F"] != ""]
        cell["seeds"].append(len(g))
        cell["single_P_tau"].append(float(np.median(p_tau)))
        cell["best_P_tau"].append(float(np.max(p_tau)))
        cell["single_Q1000"].append(float(np.median(q1000)))
        cell["random_Q1000"].append(float(np.mean(q1000)))
        cell["best_Q1000"].append(float(np.max(q1000)))
        # Equal-shot strategies, from the exact one-shot probabilities rather
        # than from the Q values, which are already compounded over 1,000
        # draws. Q first, then the mean over seeds.
        q5000 = [success_probability(p, 1000 * len(g)) for p in p_tau]
        pooled = float(1.0 - np.prod([1.0 - success_probability(p, 1000) for p in p_tau]))
        cell["median_seed_Q5000"].append(success_probability(float(np.median(p_tau)), 1000 * len(g)))
        cell["random_Q5000"].append(float(np.mean(q5000)))
        cell["pooled_Q5000"].append(pooled)
        cell["gain_Q"].append(pooled - float(np.mean(q5000)))
        ref = ref_by.get((str(n), _inst))
        if ref is not None:
            cell["uniform_F_Q5000"].append(
                success_probability(_f(ref["P_tau_main_uniform_F"]), 1000 * len(g)))
        # Union of the seeds' shot batches: optimum found by at least one.
        cell["single_optimal"].append(float(np.mean(opt)))
        cell["any_optimal"].append(float(any(opt)))
        # The same equal-budget question for the exact optimum.
        p_opt = [_f(r["final_P_opt"]) for r in g]
        o5000 = [success_probability(q, 1000 * len(g)) for q in p_opt]
        o_pooled = float(1.0 - np.prod([1.0 - success_probability(q, 1000) for q in p_opt]))
        cell["median_seed_P_opt_5000"].append(success_probability(float(np.median(p_opt)), 1000 * len(g)))
        cell["random_P_opt_5000"].append(float(np.mean(o5000)))
        cell["pooled_P_opt_5000"].append(o_pooled)
        cell["gain_opt"].append(o_pooled - float(np.mean(o5000)))
        if g_f:
            cell["single_g_F"].append(float(np.median(g_f)))
            cell["best_g_F"].append(float(np.min(g_f)))
        single_s = float(np.median([_f(r["optimisation_seconds"]) for r in g]))
        total_s = float(np.sum([_f(r["optimisation_seconds"]) for r in g]))
        cell["single_seconds"].append(single_s)
        cell["total_seconds"].append(total_s)
        cell["extra_seconds"].append(total_s - single_s)
        cell["total_shots"].append(float(np.sum([_f(r["shots"]) for r in g])))
    out = []
    for (n, label), c in sorted(per_cell.items(), key=lambda kv: (kv[0][0], PENALTY_ORDER.index(kv[0][1]) if kv[0][1] in PENALTY_ORDER else 9)):
        gain_q = np.asarray(c["gain_Q"], dtype=float)
        gain_o = np.asarray(c["gain_opt"], dtype=float)
        if np.any(gain_q < -1e-12) or np.any(gain_o < -1e-12):
            raise RuntimeError(f"pooled < random single at N={n} {label}: AM-GM violated")
        q_lo, q_hi = _bootstrap_ci(gain_q, rng)
        o_lo, o_hi = _bootstrap_ci(gain_o, rng)
        out.append({
            "n_assets": n, "penalty_label": label,
            "instances": len(c["seeds"]), "seeds_per_instance": int(np.median(c["seeds"])),
            "single_seed_P_tau_median": float(np.median(c["single_P_tau"])),
            "best_of_seeds_P_tau_median": float(np.median(c["best_P_tau"])),
            "single_seed_Q_tau_1000_median": float(np.median(c["single_Q1000"])),
            "random_seed_Q_tau_1000_median": float(np.median(c["random_Q1000"])),
            "best_of_seeds_Q_tau_1000_median": float(np.median(c["best_Q1000"])),
            "median_seed_Q_tau_5000_median": float(np.median(c["median_seed_Q5000"])),
            "random_seed_Q_tau_5000_median": float(np.median(c["random_Q5000"])),
            "pooled_5x1000_Q_tau_median": float(np.median(c["pooled_Q5000"])),
            "pooled_minus_random_Q_tau_median": float(np.median(gain_q)),
            "pooled_minus_random_Q_tau_ci95_low": q_lo,
            "pooled_minus_random_Q_tau_ci95_high": q_hi,
            "pooled_minus_random_Q_tau_max": float(np.max(gain_q)),
            "instances_pooled_gains_over_0.01_Q_tau": int((gain_q > 0.01).sum()),
            "uniform_F_Q_tau_5000_median": (float(np.median(c["uniform_F_Q5000"]))
                                            if c["uniform_F_Q5000"] else np.nan),
            "single_seed_optimal_rate": float(np.mean(c["single_optimal"])),
            "any_seed_optimal_rate": float(np.mean(c["any_optimal"])),
            "median_seed_P_opt_5000_median": float(np.median(c["median_seed_P_opt_5000"])),
            "random_seed_P_opt_5000_median": float(np.median(c["random_P_opt_5000"])),
            "pooled_5x1000_P_opt_median": float(np.median(c["pooled_P_opt_5000"])),
            "pooled_minus_random_P_opt_median": float(np.median(gain_o)),
            "pooled_minus_random_P_opt_ci95_low": o_lo,
            "pooled_minus_random_P_opt_ci95_high": o_hi,
            "pooled_minus_random_P_opt_max": float(np.max(gain_o)),
            "instances_pooled_gains_over_0.01_P_opt": int((gain_o > 0.01).sum()),
            "single_seed_g_F_median": float(np.median(c["single_g_F"])) if c["single_g_F"] else np.nan,
            "best_of_seeds_g_F_median": float(np.median(c["best_g_F"])) if c["best_g_F"] else np.nan,
            "single_seed_optimisation_seconds_median": float(np.median(c["single_seconds"])),
            "all_seeds_optimisation_seconds_median": float(np.median(c["total_seconds"])),
            "extra_optimisation_seconds_median": float(np.median(c["extra_seconds"])),
            "all_seeds_shots_median": float(np.median(c["total_shots"])),
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args(argv)

    base_path = RESULTS_DIR / "paper01_e1_cross.csv"
    if not base_path.exists():
        print(f"missing {base_path}", file=sys.stderr)
        return 1
    e1 = _normalise_baseline(load(base_path))
    ref_path = RESULTS_DIR / "paper01_e1_reference.csv"
    e1_refs = load(ref_path) if ref_path.exists() else []
    groups = {BASELINE_LABEL: e3_block(e1)}
    for tag, label in CONTROLS:
        p = RESULTS_DIR / f"paper01_e1_cross_{tag}.csv"
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            return 1
        rows = load(p)
        bad = {r["optimizer"] for r in rows} - {label}
        if bad:
            raise RuntimeError(f"{p} holds optimizer labels {bad}, expected {label}")
        groups[label] = e3_block(rows)
    for label, g in groups.items():
        print(f"{label:14s} {len(g)} runs in the E3 block "
              f"({sum(r['status'] != 'ok' for r in g)} failed)")

    write_csv(RESULTS_DIR / "paper01_e3_summary.csv", summarise(groups))
    write_csv(RESULTS_DIR / "paper01_e3_contrasts.csv", contrasts(groups))
    write_csv(RESULTS_DIR / "paper01_e3_multistart.csv", multistart(e1, e1_refs))

    print("\n=== E3: treatment − baseline (instance medians, 95 % bootstrap CI) ===")
    for c in contrasts(groups):
        if c["field"] in ("best_expectation", "final_P_tau_main", "final_Q_tau_main_1000",
                          "gradient_evaluations", "optimisation_seconds"):
            print(f"N={c['n_assets']:2d} {c['treatment']:13s} {c['penalty_label']:7s} "
                  f"{c['field']:22s} {c['instance_median_diff']:+.3e} "
                  f"[{c['ci95_low']:+.3e}, {c['ci95_high']:+.3e}] "
                  f"better {c['instances_treatment_better']}/{c['instances']} {c['direction']}")
    print("\n=== within-instance seed spread (baseline), median over instances ===")
    for s in summarise(groups):
        if s["optimizer"] == BASELINE_LABEL:
            print(f"N={s['n_assets']:2d} {s['penalty_label']:7s} "
                  f"⟨H⟩ spread {s['best_expectation_seed_spread_median']:.3f}  "
                  f"P_τ spread {s['final_P_tau_main_seed_spread_median']:.2e}  "
                  f"Q_τ(1000) spread {s['final_Q_tau_main_1000_seed_spread_median']:.3f}")
    print("\n=== multistart re-analysis of E1, Q_τ at 5,000 shots (median over instances) ===")
    print(f"{'N':>2} {'A':>8} {'1×1k med':>9} {'rand 1×5k':>10} {'5×1k pooled':>12} "
          f"{'gain [CI]':>24} {'>0.01':>5} {'unif-F 5k':>10} {'opt rand→pool':>14} {'sec 1→5':>11}")
    for m in multistart(e1, e1_refs):
        if m["penalty_label"] in PENALTY_ORDER:
            print(f"{m['n_assets']:>2} {m['penalty_label']:>8} "
                  f"{m['single_seed_Q_tau_1000_median']:>9.3f} "
                  f"{m['random_seed_Q_tau_5000_median']:>10.3f} "
                  f"{m['pooled_5x1000_Q_tau_median']:>12.3f} "
                  f"{m['pooled_minus_random_Q_tau_median']:>+7.3f} "
                  f"[{m['pooled_minus_random_Q_tau_ci95_low']:+.3f}, {m['pooled_minus_random_Q_tau_ci95_high']:+.3f}] "
                  f"{m['instances_pooled_gains_over_0.01_Q_tau']:>3d}/{m['instances']:<2d} "
                  f"{m['uniform_F_Q_tau_5000_median']:>10.3f} "
                  f"{m['random_seed_P_opt_5000_median']:>6.3f}→{m['pooled_5x1000_P_opt_median']:<6.3f} "
                  f"{m['single_seed_optimisation_seconds_median']:>5.0f}→{m['all_seeds_optimisation_seconds_median']:<5.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
