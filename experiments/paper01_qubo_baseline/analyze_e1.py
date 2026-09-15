# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Paper ① / TQE plan: summarise the E1 cross experiment and the E2 controls.

Reads ``results/paper01_e1_cross{_tag}.csv`` (one row per run) and
``results/paper01_e1_reference{_tag}.csv`` (one row per instance) and writes

    paper01_e1_summary{_tag}.csv    per (N, penalty): medians/IQR and rates
    paper01_e1_contrasts{_tag}.csv  paired contrasts, instance-level bootstrap CI
    paper01_e1_shots{_tag}.csv      E2: Q_τ(S) for optimised / unoptimised /
                                    uniform-all / uniform-F, per N and S
    paper01_e1_decomposition{_tag}.csv  g_off ratio split into solution part
                                    and denominator part (plan §5.6)
    paper01_e1_manifest{_tag}.csv   plan §12.2 Table 1: per N, the nominal
                                    design, the A = 0 collapse, runs done,
                                    failures, runs without a feasible shot,
                                    GPU time
    paper01_e1_acquisition_factors{_tag}.csv  P_τ(A_margin)/P_τ(A_heur) split
                                    into the P_F ratio and the R_τ ratio,
                                    geometric mean over the same pairs
    paper01_e1_optimum_hits{_tag}.csv  per (N, setting): runs whose best
                                    feasible shot is the exact optimum

Statistics follow plan §10: the unit of generalisation is the instance. A
paired difference is first summarised within an instance (median over the
seeds that share it), then the instance-level values are bootstrapped
(resampling instances) for a 95 % percentile interval. Failed runs and runs
without a feasible shot are counted, not dropped; a contrast on g_F is only
formed where both sides have a feasible shot, and the number of such pairs is
reported next to the estimate.

Run:
    python -m experiments.paper01_qubo_baseline.analyze_e1 [--results-tag pilot]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.qubo.scoring import success_probability
from experiments.paper01_qubo_baseline.run_e1_cross import (
    DEFAULT_INSTANCES,
    DEFAULT_SEEDS,
    MULTIPLES,
    RESULTS_DIR,
    SHOT_BUDGETS,
    TAU_MAIN,
)
from experiments.paper01_qubo_baseline.run_experiment import write_csv

#: Paired contrasts to report: (treatment, control, restrict to A_crit > 0?).
CONTRASTS = [("Amargin", "Aheur", False), ("5xAcrit", "Aheur", True),
             ("10xAcrit", "Aheur", True), ("1.1xAcrit", "Aheur", True)]

#: Quantities contrasted, and whether larger is better (for the sign column).
CONTRAST_FIELDS = [("final_P_F", True), ("final_P_tau_main", True),
                   ("final_Q_tau_main_1000", True), ("final_P_opt", True),
                   ("g_F", False), ("g_off", False), ("rank_best_feasible", False),
                   ("feasible_shot_fraction", True), ("runtime_s", False)]

#: The conventional "success" threshold the paper's tables used: 0.1 % of |E*|.
G_OFF_SUCCESS = 0.001

BOOTSTRAP = 2000
RNG_SEED = 20260911

PENALTY_ORDER = ["A0", "1.1xAcrit", "2xAcrit", "5xAcrit", "10xAcrit", "Amargin", "Aheur"]


def _f(v: str) -> float:
    return float(v) if v not in ("", None) else np.nan


def load(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _q(values, q):
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    return float(np.percentile(v, q)) if v.size else np.nan


def summarise(rows: list[dict]) -> list[dict]:
    """Per (N, penalty_label): rates and median/IQR of the main quantities."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(int(r["n_assets"]), r["penalty_label"])].append(r)
    out = []
    for (n, label), g in sorted(groups.items(),
                                key=lambda kv: (kv[0][0], PENALTY_ORDER.index(kv[0][1]))):
        ok = [r for r in g if r["status"] == "ok"]
        feas = [r for r in ok if r["g_F"] != ""]
        row = {
            "n_assets": n, "penalty_label": label,
            "runs": len(g), "runs_ok": len(ok), "runs_failed": len(g) - len(ok),
            "instances": len({r["instance"] for r in g}),
            "a_over_crit_median": _q([_f(r["a_over_crit"]) for r in ok], 50),
            "best_shot_feasible_rate": np.mean([r["best_all_feasible"] == "True" for r in ok]) if ok else np.nan,
            "no_feasible_shot_runs": sum(1 for r in ok if r["g_F"] == ""),
            "optimal_rate": np.mean([r["is_optimal"] == "True" for r in feas]) if feas else np.nan,
            "within_tau_main_rate": np.mean([r["within_tau_main"] == "True" for r in feas]) if feas else np.nan,
            "g_off_success_rate": np.mean([_f(r["g_off"]) <= G_OFF_SUCCESS for r in feas]) if feas else np.nan,
            "g_off_success_but_not_tau_rate": np.mean(
                [(_f(r["g_off"]) <= G_OFF_SUCCESS) and (_f(r["g_F"]) > TAU_MAIN) for r in feas]) if feas else np.nan,
        }
        for field in ("final_P_F", "final_P_tau_main", "final_P_opt",
                      "final_Q_tau_main_1000", "final_Q_opt_1000", "initial_P_F",
                      "initial_P_tau_main", "feasible_shot_fraction", "g_F", "g_off",
                      "rank_best_feasible", "deflation_D", "final_expected_violation",
                      "runtime_s"):
            vals = [_f(r.get(field, "")) for r in ok]
            row[f"{field}_p25"] = _q(vals, 25)
            row[f"{field}_median"] = _q(vals, 50)
            row[f"{field}_p75"] = _q(vals, 75)
        out.append(row)
    return out


def _instance_level(pairs: dict[tuple, list[float]]) -> np.ndarray:
    """Median over seeds within each instance → one value per instance."""
    return np.asarray([float(np.median(v)) for v in pairs.values() if v], dtype=float)


def _bootstrap_ci(values: np.ndarray, rng) -> tuple[float, float]:
    if values.size < 2:
        return np.nan, np.nan
    meds = np.empty(BOOTSTRAP)
    for b in range(BOOTSTRAP):
        meds[b] = np.median(rng.choice(values, size=values.size, replace=True))
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def contrasts(rows: list[dict]) -> list[dict]:
    """Paired treatment − control per (N, instance, seed), instance-level CI."""
    rng = np.random.default_rng(RNG_SEED)
    index = {(r["n_assets"], r["instance"], r["penalty_label"], r["qaoa_seed"]): r
             for r in rows if r["status"] == "ok"}
    out = []
    for n in sorted({int(r["n_assets"]) for r in rows}):
        for treat, control, positive_only in CONTRASTS:
            for field, larger_better in CONTRAST_FIELDS:
                pairs: dict[str, list[float]] = defaultdict(list)
                n_pairs = 0
                n_dropped = 0
                for key, r in index.items():
                    if key[0] != str(n) or key[2] != treat:
                        continue
                    c = index.get((key[0], key[1], control, key[3]))
                    if c is None:
                        continue
                    if positive_only and _f(r["a_crit"]) <= 0.0:
                        continue
                    a, b = _f(r.get(field, "")), _f(c.get(field, ""))
                    if np.isnan(a) or np.isnan(b):
                        n_dropped += 1
                        continue
                    pairs[key[1]].append(a - b)
                    n_pairs += 1
                inst = _instance_level(pairs)
                if inst.size == 0:
                    continue
                lo, hi = _bootstrap_ci(inst, rng)
                sign = "better" if (np.median(inst) > 0) == larger_better else "worse"
                if np.median(inst) == 0:
                    sign = "equal"
                out.append({
                    "n_assets": n, "treatment": treat, "control": control, "field": field,
                    "pairs": n_pairs, "pairs_without_feasible_shot": n_dropped,
                    "instances": int(inst.size),
                    "instance_median_diff": float(np.median(inst)),
                    "ci95_low": lo, "ci95_high": hi,
                    "instances_treatment_better": int(((inst > 0) if larger_better else (inst < 0)).sum()),
                    "instances_tied": int((inst == 0).sum()),
                    "direction": sign,
                })
    return out


#: Quantities the optimiser can move at a fixed penalty weight, as
#: (name, initial column, final column). D_cond comes from the conditional-TV
#: file (``analyze_conditional.py``), joined on run_id; R_τ = q_τ / u_τ is
#: derived from P_τ / P_F against the instance's uniform-over-F rate.
INITIAL_FINAL_FIELDS = [
    ("P_F", "initial_P_F", "final_P_F"),
    ("P_tau_main", "initial_P_tau_main", "final_P_tau_main"),
    ("Q_tau_main_1000", "initial_Q_tau_main_1000", "final_Q_tau_main_1000"),
    ("expected_violation", "initial_expected_violation", "final_expected_violation"),
    ("R_tau", "initial_R_tau", "final_R_tau"),
    ("D_cond", "initial_D_cond", "final_D_cond"),
]


def initial_final(rows: list[dict], refs: list[dict], tv_rows: list[dict] | None = None) -> list[dict]:
    """Initial → final state of the same run, at a fixed penalty weight.

    Review round 3 (M4): "optimisation at A_heur buys feasible mass" had been
    supported with P_F at A_heur against P_F at A_margin, which is the
    difference between two penalty settings after optimisation, not the
    effect of optimisation. This is the latter: for every completed run, the
    final state's P_F, P_τ, Q_τ(1000), E[V], R_τ and D_cond against the
    initial state's at the same angles' starting point, paired within the
    run, the median over seeds within an instance, and the median and
    bootstrap CI over instances (Sec. IV-E). Marginal medians of the initial
    and final levels are carried beside the paired difference; they need not
    subtract to it.
    """
    rng = np.random.default_rng(RNG_SEED)
    ref_by = {(r["n_assets"], r["instance"]): r for r in refs}
    tv_by = {r["run_id"]: r for r in (tv_rows or [])}
    out = []
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["status"] == "ok":
            groups[(int(r["n_assets"]), r["penalty_label"])].append(r)
    for (n, label), g in sorted(groups.items(), key=lambda kv: (kv[0][0], PENALTY_ORDER.index(kv[0][1]) if kv[0][1] in PENALTY_ORDER else 9)):
        for name, col0, col1 in INITIAL_FINAL_FIELDS:
            init: dict[str, list[float]] = defaultdict(list)
            fin: dict[str, list[float]] = defaultdict(list)
            diff: dict[str, list[float]] = defaultdict(list)
            n_runs = n_undefined = 0
            for r in g:
                if name == "R_tau":
                    ref = ref_by.get((r["n_assets"], r["instance"]))
                    u = _f(ref["P_tau_main_uniform_F"]) if ref else np.nan
                    pf0, pf1 = _f(r["initial_P_F"]), _f(r["final_P_F"])
                    a = (_f(r["initial_P_tau_main"]) / pf0 / u) if pf0 > 0 and u > 0 else np.nan
                    b = (_f(r["final_P_tau_main"]) / pf1 / u) if pf1 > 0 and u > 0 else np.nan
                elif name == "D_cond":
                    t = tv_by.get(r["run_id"])
                    if t is None:
                        continue
                    a, b = _f(t[col0]), _f(t[col1])
                else:
                    a, b = _f(r.get(col0, "")), _f(r.get(col1, ""))
                if np.isnan(a) or np.isnan(b):
                    n_undefined += 1
                    continue
                n_runs += 1
                init[r["instance"]].append(a); fin[r["instance"]].append(b)
                diff[r["instance"]].append(b - a)
            inst = _instance_level(diff)
            if inst.size == 0:
                continue
            lo, hi = _bootstrap_ci(inst, rng)
            out.append({
                "n_assets": n, "penalty_label": label, "field": name,
                "runs": n_runs, "runs_undefined": n_undefined, "instances": int(inst.size),
                "initial_median": float(np.median(_instance_level(init))),
                "final_median": float(np.median(_instance_level(fin))),
                "paired_diff_median": float(np.median(inst)),
                "paired_diff_ci95_low": lo, "paired_diff_ci95_high": hi,
                "paired_diff_p25": float(np.percentile(inst, 25)),
                "paired_diff_p75": float(np.percentile(inst, 75)),
                "instances_increased": int((inst > 0).sum()),
                "instances_decreased": int((inst < 0).sum()),
            })
    return out


def acquisition_factors(rows: list[dict], treat: str = "Amargin",
                        control: str = "Aheur") -> list[dict]:
    """P_τ(treat) / P_τ(control) = P_F ratio × R_τ ratio, pair by pair.

    Review round 4 (C02): Table IV's medians of R_τ at the two weights are
    separate levels and do not say which factor carried the paired P_τ gain.
    With P_τ = P_F · u_τ · R_τ and u_τ an instance constant, the ratio of
    the two weights' P_τ on the same (instance, seed) factorises exactly into
    the P_F ratio and the R_τ ratio, and log turns that into a sum, so the
    geometric mean over one set of pairs is the decomposition (as in
    :func:`decomposition`). Pairs with a zero probability on either side are
    counted in ``skipped_zero`` rather than dropped silently; none occur in
    the E1 states. Medians of the three ratios and the count of pairs with
    each ratio above one are carried as levels.
    """
    index = {(r["n_assets"], r["instance"], r["penalty_label"], r["qaoa_seed"]): r
             for r in rows if r["status"] == "ok"}
    out = []
    for n in sorted({int(r["n_assets"]) for r in rows}):
        lf, lr, lt = [], [], []
        skipped_zero = 0
        for key, r in index.items():
            if key[0] != str(n) or key[2] != treat:
                continue
            c = index.get((key[0], key[1], control, key[3]))
            if c is None:
                continue
            pf_t, pf_c = _f(r["final_P_F"]), _f(c["final_P_F"])
            pt_t, pt_c = _f(r["final_P_tau_main"]), _f(c["final_P_tau_main"])
            if min(pf_t, pf_c, pt_t, pt_c) <= 0.0:
                skipped_zero += 1
                continue
            lf.append(np.log(pf_t / pf_c))
            lt.append(np.log(pt_t / pt_c))
            lr.append(lt[-1] - lf[-1])
        if not lt:
            continue
        lf, lr, lt = (np.asarray(v) for v in (lf, lr, lt))
        out.append({
            "n_assets": n, "treatment": treat, "control": control,
            "pairs": int(lt.size), "skipped_zero": skipped_zero,
            "P_tau_ratio_geomean": float(np.exp(lt.mean())),
            "P_F_ratio_geomean": float(np.exp(lf.mean())),
            "R_tau_ratio_geomean": float(np.exp(lr.mean())),
            "log_share_P_F": float(lf.mean() / lt.mean()) if lt.mean() != 0 else np.nan,
            "P_tau_ratio_median": float(np.exp(np.median(lt))),
            "P_F_ratio_median": float(np.exp(np.median(lf))),
            "R_tau_ratio_median": float(np.exp(np.median(lr))),
            "pairs_P_tau_ratio_above_1": int((lt > 0).sum()),
            "pairs_P_F_ratio_above_1": int((lf > 0).sum()),
            "pairs_R_tau_ratio_above_1": int((lr > 0).sum()),
        })
    return out


def optimum_hits(rows: list[dict]) -> list[dict]:
    """Per (N, penalty setting): runs whose best feasible shot is the exact optimum.

    Review round 4 (P02): Fig. 2's zero-gap share had been referred to
    Table VI (the E3 block) and to the seed-42 sweep, neither of which is the
    population Fig. 2 draws. This is that population -- every completed E1
    run -- with the share among runs that had a feasible shot (what the gap
    panels plot) and among all completed runs.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["status"] == "ok":
            groups[(int(r["n_assets"]), r["penalty_label"])].append(r)
    out = []
    for (n, label), g in sorted(groups.items(),
                                key=lambda kv: (kv[0][0], PENALTY_ORDER.index(kv[0][1]))):
        feas = [r for r in g if r["g_F"] != ""]
        hits = sum(r["is_optimal"] == "True" for r in feas)
        out.append({
            "n_assets": n, "penalty_label": label, "runs": len(g),
            "runs_with_feasible_shot": len(feas), "optimum_hits": int(hits),
            "share_of_runs_with_feasible_shot": hits / len(feas) if feas else np.nan,
            "share_of_runs": hits / len(g),
        })
    return out


def shot_budget_table(rows: list[dict], refs: list[dict]) -> list[dict]:
    """E2: Q_τ(S) for the optimised state, the initial state and two uniforms."""
    ref_by = {(r["n_assets"], r["instance"]): r for r in refs}
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["status"] == "ok":
            groups[(int(r["n_assets"]), r["penalty_label"])].append(r)
    out = []
    for (n, label), g in sorted(groups.items(),
                                key=lambda kv: (kv[0][0], PENALTY_ORDER.index(kv[0][1]))):
        for s in SHOT_BUDGETS:
            def q(p, s=s):
                # scoring.success_probability, not a second spelling of it:
                # the naive 1 − (1 − p)^S cancels for small p and drifted
                # between platforms (see that function's docstring).
                return success_probability(p, s)
            row = {"n_assets": n, "penalty_label": label, "shots": s, "runs": len(g)}
            for src in ("final", "initial"):
                row[f"Q_tau_{src}_median"] = _q([_f(r[f"{src}_Q_tau_main_{s}"]) for r in g], 50)
                row[f"Q_opt_{src}_median"] = _q([_f(r[f"{src}_Q_opt_{s}"]) for r in g], 50)
            # Uniform controls are instance properties: median over the
            # instances present in this group.
            insts = {(r["n_assets"], r["instance"]) for r in g}
            row["Q_tau_uniform_all_median"] = _q(
                [q(_f(ref_by[i]["P_tau_main_uniform_all"])) for i in insts], 50)
            row["Q_tau_uniform_F_median"] = _q(
                [q(_f(ref_by[i]["P_tau_main_uniform_F"])) for i in insts], 50)
            row["Q_opt_uniform_all_median"] = _q(
                [q(_f(ref_by[i]["P_opt_uniform_all"])) for i in insts], 50)
            row["Q_opt_uniform_F_median"] = _q(
                [q(_f(ref_by[i]["P_opt_uniform_F"])) for i in insts], 50)
            # How often the optimised state beats the strongest control.
            row["final_beats_uniform_F_rate"] = float(np.mean([
                _f(r[f"final_Q_tau_main_{s}"]) > q(_f(ref_by[(r["n_assets"], r["instance"])]["P_tau_main_uniform_F"]))
                for r in g]))
            # The same comparison at the two sensitivity thresholds (review
            # round 3, §6.6): monotonicity in S does not carry a result at
            # one τ over to another, so each τ is counted on its own. On
            # P directly, which decides every S at once.
            def beats(field, uniform):
                return int(sum(_f(r[field]) > uniform(ref_by[(r["n_assets"], r["instance"])]) for r in g))
            row["final_beats_uniform_F_runs_opt"] = beats(
                "final_P_opt", lambda ref: _f(ref["P_opt_uniform_F"]))
            row["final_beats_uniform_F_runs_alt"] = beats(
                "final_P_tau_alt", lambda ref: _f(ref["count_within_tau_alt"]) / _f(ref["feasible_set_size"]))
            out.append(row)
    return out


def _geomean(values: list[float]) -> float:
    """exp(mean(log x)). Every value here is a ratio of positive quantities."""
    return float(np.exp(np.mean(np.log(np.asarray(values, dtype=float)))))


def decomposition(rows: list[dict]) -> list[dict]:
    """g_off(x_A2, A2) / g_off(x_A1, A1) = solution part × denominator part.

    Formed only where both gaps are positive (plan §5.6); zero gaps and
    missing feasible shots are counted in the ``skipped_*`` columns.

    The identity holds pair by pair. It does **not** survive a median: the
    median of a product is not the product of the medians, and writing the
    three medians as an equation states an arithmetic falsehood (the N = 20
    columns read 29.0, 0.81 and 40.0, whose product is 32.4). The geometric
    mean is the summary that keeps the identity, because log turns the product
    into a sum and the mean of a sum is the sum of the means -- over the *same*
    pairs, which is why all three are accumulated in one loop. Both summaries
    are reported: the medians as robust levels, the geometric means as the
    decomposition.
    """
    index = {(r["n_assets"], r["instance"], r["penalty_label"], r["qaoa_seed"]): r
             for r in rows if r["status"] == "ok"}
    out = []
    for n in sorted({int(r["n_assets"]) for r in rows}):
        for treat, control, positive_only in CONTRASTS:
            sol, den, tot = [], [], []
            skipped_zero = skipped_missing = 0
            for key, r in index.items():
                if key[0] != str(n) or key[2] != treat:
                    continue
                c = index.get((key[0], key[1], control, key[3]))
                if c is None or (positive_only and _f(r["a_crit"]) <= 0.0):
                    continue
                if r["g_F"] == "" or c["g_F"] == "":
                    skipped_missing += 1
                    continue
                num_t = _f(r["f_best_feasible"]) - _f(r["f_star"])
                num_c = _f(c["f_best_feasible"]) - _f(c["f_star"])
                if num_t <= 0 or num_c <= 0:
                    skipped_zero += 1
                    continue
                sol.append(num_t / num_c)
                den.append(_f(c["abs_E_star"]) / _f(r["abs_E_star"]))
                tot.append(_f(r["g_off"]) / _f(c["g_off"]))
            if not tot:
                continue
            out.append({
                "n_assets": n, "treatment": treat, "control": control,
                "pairs": len(tot), "skipped_zero_gap": skipped_zero,
                "skipped_no_feasible_shot": skipped_missing,
                "g_off_ratio_median": float(np.median(tot)),
                "g_off_ratio_geomean": _geomean(tot),
                "solution_part_geomean": _geomean(sol),
                "denominator_part_geomean": _geomean(den),
                "solution_part_median": float(np.median(sol)),
                "solution_part_p25": float(np.percentile(sol, 25)),
                "solution_part_p75": float(np.percentile(sol, 75)),
                "denominator_part_median": float(np.median(den)),
                "solution_part_below_1_rate": float(np.mean(np.asarray(sol) < 1.0)),
            })
    return out


#: The two success rules are read at these thresholds rather than at one each,
#: so the headline disagreement rate can be seen as a point on a surface. The
#: primary pair (0.001, 0.01) is the plan's; the others are the ones a reader
#: would reach for.
G_OFF_THRESHOLDS = (0.0001, 0.0005, 0.001, 0.002)     # 0.01 %, 0.05 %, 0.1 %, 0.2 %
TAU_THRESHOLDS = (("tau_main", 0.01), ("tau_alt", 0.05), ("optimum", 0.0))


#: Fields the sensitivity re-runs the headline contrast on. The paper's three
#: directional claims at N = 20 are one improvement, one worsening and one
#: near-null, so all three are carried.
SENSITIVITY_FIELDS = ("final_Q_tau_main_1000", "g_off", "g_F")


def sensitivity(rows: list[dict], refs: list[dict]) -> list[dict]:
    """The A_margin - A_heur contrast under subsets of the design.

    The interval the paper quotes is over instances, with the same five seeds in
    every one and instance 0 -- the published instance -- deliberately included.
    Neither choice is wrong, and both are choices: a reader is entitled to ask
    whether the result is instance 0's, or one seed's, or the zero-penalty
    instances'. Each subset re-runs the same paired contrast, so the columns are
    comparable to the headline row and to each other.

    Subsets are of the *design*, not of the outcome: nothing here is selected on
    the value of the quantity being summarised.
    """
    a_crit_zero = {(r["n_assets"], r["instance"]) for r in refs
                   if r.get("a_crit_is_zero") == "True"}
    seeds = sorted({r["qaoa_seed"] for r in rows})
    subsets: list[tuple[str, callable]] = [
        ("all", lambda r: True),
        ("without instance 0", lambda r: r["instance"] != "0"),
        ("instance 0 only", lambda r: r["instance"] == "0"),
        ("A_crit > 0 instances", lambda r: (r["n_assets"], r["instance"]) not in a_crit_zero),
        ("A_crit = 0 instances", lambda r: (r["n_assets"], r["instance"]) in a_crit_zero),
    ]
    subsets += [(f"seed {sd} only", (lambda sd: lambda r: r["qaoa_seed"] == sd)(sd))
                for sd in seeds]
    subsets += [(f"without seed {sd}", (lambda sd: lambda r: r["qaoa_seed"] != sd)(sd))
                for sd in seeds]

    out = []
    for name, keep in subsets:
        subset = [r for r in rows if keep(r)]
        if not subset:
            continue
        for c in contrasts(subset):
            if c["treatment"] != "Amargin" or c["field"] not in SENSITIVITY_FIELDS:
                continue
            out.append({"subset": name, **c})
    return out


def threshold_grid(rows: list[dict]) -> list[dict]:
    """How often the two success rules disagree, over a grid of both thresholds.

    Sec. V-B reports one number -- 76 % of N = 20 heuristic-weight runs pass
    g_off <= 0.1 % and fail g_F <= 1 % -- which invites the question of whether
    the pair was chosen for it. Both rules are read here at four gap thresholds
    and three quality thresholds, on the same runs, so the reported figure can
    be located on the surface rather than taken on its own.

    "Disagree" is the asymmetric direction that matters: the gap rule passes and
    the feasible-range rule does not. The reverse direction is counted too,
    since a rule that is merely stricter is not the same as one that ranks runs
    differently.
    """
    out = []
    for n in sorted({int(r["n_assets"]) for r in rows}):
        for label in PENALTY_ORDER:
            g = [r for r in rows
                 if int(r["n_assets"]) == n and r["penalty_label"] == label
                 and r["status"] == "ok" and r["g_F"] != "" and r["g_off"] != ""]
            if not g:
                continue
            for gt in G_OFF_THRESHOLDS:
                for tname, tau in TAU_THRESHOLDS:
                    passes_gap = [abs(_f(r["g_off"])) <= gt for r in g]
                    passes_q = [_f(r["g_F"]) <= tau for r in g]
                    out.append({
                        "n_assets": n, "penalty_label": label,
                        "g_off_threshold": gt, "tau_name": tname, "tau": tau,
                        "runs": len(g),
                        "passes_gap_rule": sum(passes_gap),
                        "passes_quality_rule": sum(passes_q),
                        "gap_passes_quality_fails": sum(
                            a and not b for a, b in zip(passes_gap, passes_q)),
                        "quality_passes_gap_fails": sum(
                            b and not a for a, b in zip(passes_gap, passes_q)),
                        "disagreement_rate": (
                            sum(a != b for a, b in zip(passes_gap, passes_q)) / len(g)),
                    })
    return out


def feasibility_split(rows: list[dict], refs: list[dict]) -> list[dict]:
    """P_tau = P_F * Pr[g_F <= tau | x in F], and how selective that second factor is.

    The paper explains the penalty weight's effect on acquisition by saying a
    large A leaves little of the state on the objective. That predicts two
    separable things: how much mass lands in the feasible set at all, and
    whether the mass that does lands on the good part of it. This splits them.

    R_tau compares the conditional quality rate with what an indifferent draw
    from F would give:

        R_tau = (P_tau / P_F) / (|F_tau| / |F|),

    so R_tau = 1 is a state that finds the feasible set but sorts nothing
    within it, and R_tau > 1 is one that concentrates on the good part. It is
    computed per run and then summarised -- dividing the medians of P_tau and
    P_F would be a different quantity. Runs with P_F = 0 have no conditional
    rate and are counted separately.
    """
    ref_by = {(r["n_assets"], r["instance"]): r for r in refs}
    out = []
    for n in sorted({int(r["n_assets"]) for r in rows}):
        for label in PENALTY_ORDER:
            cond, enrich, undefined = [], [], 0
            p_f = []
            for r in rows:
                if (int(r["n_assets"]) != n or r["penalty_label"] != label
                        or r["status"] != "ok"):
                    continue
                ref = ref_by.get((r["n_assets"], r["instance"]))
                if ref is None:
                    continue
                pf, ptau = _f(r["final_P_F"]), _f(r["final_P_tau_main"])
                if not (pf > 0):
                    undefined += 1
                    continue
                share = _f(ref["count_within_tau_main"]) / _f(ref["feasible_set_size"])
                p_f.append(pf)
                cond.append(ptau / pf)
                if share > 0:
                    enrich.append((ptau / pf) / share)
            if not cond:
                continue
            out.append({
                "n_assets": n, "penalty_label": label, "runs": len(cond),
                "runs_without_feasible_mass": undefined,
                "P_F_median": float(np.median(p_f)),
                "conditional_quality_rate_median": float(np.median(cond)),
                "feasible_set_quality_share": float(np.median(
                    [_f(ref_by[(str(n), str(i))]["count_within_tau_main"])
                     / _f(ref_by[(str(n), str(i))]["feasible_set_size"])
                     for i in range(30) if (str(n), str(i)) in ref_by])),
                "enrichment_R_tau_median": float(np.median(enrich)) if enrich else np.nan,
                "enrichment_R_tau_p25": float(np.percentile(enrich, 25)) if enrich else np.nan,
                "enrichment_R_tau_p75": float(np.percentile(enrich, 75)) if enrich else np.nan,
                "runs_with_R_tau_above_1": int(np.sum(np.asarray(enrich) > 1.0)) if enrich else 0,
            })
    return out


def manifest_table(rows: list[dict], refs: list[dict]) -> list[dict]:
    """Plan §12.2 Table 1: what was planned, what collapsed, what ran, what failed.

    Nominal is the full cross (instances × seeds × (multiples + 2)); the
    A = 0 collapse is the plan §5.4 rule that the four multiples of a zero
    A_crit are one run, so "expected" is what the runner would execute for
    the instances present in the reference file. Failed runs are counted,
    not dropped (plan §10.3), and a completed run whose 1000 shots held no
    feasible bitstring is listed separately because it has P_F but no g_F.
    """
    n_conditions = len(MULTIPLES) + 2
    out = []
    for n in sorted({int(r["n_assets"]) for r in refs} | {int(r["n_assets"]) for r in rows}):
        r_n = [r for r in refs if int(r["n_assets"]) == n]
        g = [r for r in rows if int(r["n_assets"]) == n]
        ok = [r for r in g if r["status"] == "ok"]
        zero = sum(r["a_crit_is_zero"] == "True" for r in r_n)
        seeds = sorted({r["qaoa_seed"] for r in g}, key=int) or [str(s) for s in DEFAULT_SEEDS]
        failures = defaultdict(int)
        for r in g:
            if r["status"] != "ok":
                failures[r["status"]] += 1
        # Conditions actually in play at this N (E3 runs only Aheur/Amargin
        # under its own tag), so "expected" is what the runner would produce
        # for these instances and seeds, not the full E1 design.
        labels = {r["penalty_label"] for r in g} - {"A0"}
        multiples_seen = labels & {label for _, label in MULTIPLES}
        per_instance = len(labels) or n_conditions
        per_zero_instance = len(labels - multiples_seen) + (1 if multiples_seen else 0)
        out.append({
            "n_assets": n,
            "n_select": int(r_n[0]["n_select"]) if r_n else "",
            "instances_planned": len(DEFAULT_INSTANCES),
            "instances_with_reference": len(r_n),
            "instances_a_crit_zero": zero,
            "seeds": " ".join(seeds),
            "penalty_conditions_nominal": n_conditions,
            "runs_nominal": len(DEFAULT_INSTANCES) * len(DEFAULT_SEEDS) * n_conditions,
            # Each A_crit = 0 instance runs 3 conditions instead of 6.
            "penalty_conditions_run": len(labels),
            "runs_expected": ((len(r_n) - zero) * per_instance + zero * per_zero_instance) * len(seeds),
            "runs_done": len(g),
            "runs_ok": len(ok),
            "runs_failed": len(g) - len(ok),
            "failure_kinds": "; ".join(f"{k}×{v}" for k, v in sorted(failures.items())),
            "runs_without_feasible_shot": sum(1 for r in ok if r["g_F"] == ""),
            "runs_best_shot_infeasible": sum(1 for r in ok if r["best_all_feasible"] != "True"),
            "feasible_set_size": int(r_n[0]["feasible_set_size"]) if r_n else "",
            "reference_seconds_total": float(np.sum([_f(r["reference_seconds"]) for r in r_n])) if r_n else np.nan,
            "runtime_s_median": _q([_f(r["runtime_s"]) for r in ok], 50),
            "gpu_hours_total": float(np.nansum([_f(r["runtime_s"]) for r in g])) / 3600.0,
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-tag", default=None)
    args = ap.parse_args(argv)
    suffix = f"_{args.results_tag}" if args.results_tag else ""
    rows = load(RESULTS_DIR / f"paper01_e1_cross{suffix}.csv")
    refs = load(RESULTS_DIR / f"paper01_e1_reference{suffix}.csv")

    manifest = manifest_table(rows, refs)
    write_csv(RESULTS_DIR / f"paper01_e1_manifest{suffix}.csv", manifest)
    print(f"{'N':>2} {'inst':>4} {'A0':>3} {'nominal':>7} {'expected':>8} {'done':>5} "
          f"{'ok':>5} {'fail':>4} {'no-feas':>7} {'GPU h':>6}")
    for m in manifest:
        print(f"{m['n_assets']:>2} {m['instances_with_reference']:>4} {m['instances_a_crit_zero']:>3} "
              f"{m['runs_nominal']:>7} {m['runs_expected']:>8} {m['runs_done']:>5} "
              f"{m['runs_ok']:>5} {m['runs_failed']:>4} {m['runs_without_feasible_shot']:>7} "
              f"{m['gpu_hours_total']:>6.2f}")
    print()

    summary = summarise(rows)
    write_csv(RESULTS_DIR / f"paper01_e1_summary{suffix}.csv", summary)
    print(f"{len(rows)} runs, {sum(r['status'] == 'ok' for r in rows)} ok, "
          f"{len(refs)} instances (A_crit = 0 on "
          f"{sum(r['a_crit_is_zero'] == 'True' for r in refs)})\n")
    print(f"{'N':>2} {'penalty':>9} {'runs':>4} {'best-shot':>9} {'no-feas':>7} "
          f"{'opt':>5} {'≤τ':>5} {'P_F':>6} {'P_τ':>7} {'Q_τ(1k)':>8} {'g_F':>7} "
          f"{'g_off':>8} {'rank':>6} {'D':>6}")
    for s in summary:
        print(f"{s['n_assets']:>2} {s['penalty_label']:>9} {s['runs_ok']:>4} "
              f"{s['best_shot_feasible_rate']:>9.2f} {s['no_feasible_shot_runs']:>7} "
              f"{s['optimal_rate']:>5.2f} {s['within_tau_main_rate']:>5.2f} "
              f"{s['final_P_F_median']:>6.3f} {s['final_P_tau_main_median']:>7.4f} "
              f"{s['final_Q_tau_main_1000_median']:>8.3f} {s['g_F_median']:>7.4f} "
              f"{s['g_off_median']:>8.5f} {s['rank_best_feasible_median']:>6.0f} "
              f"{s['deflation_D_median']:>6.1f}")

    con = contrasts(rows)
    write_csv(RESULTS_DIR / f"paper01_e1_contrasts{suffix}.csv", con)
    print("\nPaired contrasts (instance-level median of treatment − control, 95% bootstrap CI):")
    for c in con:
        if c["field"] in ("final_P_F", "final_Q_tau_main_1000", "g_F", "rank_best_feasible"):
            print(f"  N={c['n_assets']:>2} {c['treatment']:>9} − {c['control']:<7} "
                  f"{c['field']:<22} {c['instance_median_diff']:>+9.4f} "
                  f"[{c['ci95_low']:>+8.4f}, {c['ci95_high']:>+8.4f}]  "
                  f"{c['instances_treatment_better']}/{c['instances']} instances better  "
                  f"({c['pairs']} pairs, {c['pairs_without_feasible_shot']} without feasible shot)")

    shots = shot_budget_table(rows, refs)
    write_csv(RESULTS_DIR / f"paper01_e1_shots{suffix}.csv", shots)
    print("\nE2: median Q_τ(S), τ = 1% of the feasible range:")
    print(f"{'N':>2} {'penalty':>9} {'S':>6} {'optimised':>9} {'initial':>9} "
          f"{'unif-all':>9} {'unif-F':>9} {'beats unif-F':>12}")
    for s in shots:
        if s["penalty_label"] in ("Aheur", "Amargin"):
            print(f"{s['n_assets']:>2} {s['penalty_label']:>9} {s['shots']:>6} "
                  f"{s['Q_tau_final_median']:>9.3f} {s['Q_tau_initial_median']:>9.3f} "
                  f"{s['Q_tau_uniform_all_median']:>9.3f} {s['Q_tau_uniform_F_median']:>9.3f} "
                  f"{s['final_beats_uniform_F_rate']:>12.2f}")

    sens = sensitivity(rows, refs)
    write_csv(RESULTS_DIR / f"paper01_e1_sensitivity{suffix}.csv", sens)
    print("\nA_margin − A_heur at N = 20 under subsets of the design "
          "(instance-level median [95 % CI]; instances better / of):")
    print(f"{'subset':>22} " + " ".join(f"{f:>28}" for f in SENSITIVITY_FIELDS))
    for name in dict.fromkeys(c["subset"] for c in sens):
        cells = {c["field"]: c for c in sens if c["subset"] == name and c["n_assets"] == 20}
        if not cells:
            continue
        line = f"{name:>22} "
        for f in SENSITIVITY_FIELDS:
            c = cells.get(f)
            if c is None:
                line += f"{'—':>28} "
                continue
            scale = 100.0 if f in ("g_off", "g_F") else 1.0
            line += (f"{c['instance_median_diff']*scale:>+8.3f} "
                     f"[{c['ci95_low']*scale:>+7.3f},{c['ci95_high']*scale:>+7.3f}] "
                     f"{c['instances_treatment_better']:>2}/{c['instances']:<2} ")
        print(line)

    grid = threshold_grid(rows)
    write_csv(RESULTS_DIR / f"paper01_e1_thresholds{suffix}.csv", grid)
    print("\nDisagreement between the two success rules, over both thresholds "
          "(share of runs where exactly one rule passes):")
    print(f"{'N':>2} {'penalty':>9} " + " ".join(f"{t*100:>7.2f}%" for t in G_OFF_THRESHOLDS)
          + "   (rows: g_off threshold; tau = 1 % of the feasible range)")
    for n in sorted({int(r["n_assets"]) for r in rows}):
        for label in ("Aheur", "Amargin"):
            cells = [c for c in grid if c["n_assets"] == n
                     and c["penalty_label"] == label and c["tau_name"] == "tau_main"]
            if not cells:
                continue
            by_t = {c["g_off_threshold"]: c for c in cells}
            print(f"{n:>2} {label:>9} " + " ".join(
                f"{by_t[t]['disagreement_rate']:>8.2f}" for t in G_OFF_THRESHOLDS))

    split = feasibility_split(rows, refs)
    write_csv(RESULTS_DIR / f"paper01_e1_feasibility_split{suffix}.csv", split)
    print("\nP_tau = P_F x Pr[quality | feasible], and enrichment over an "
          "indifferent draw from F:")
    print(f"{'N':>2} {'penalty':>9} {'P_F':>7} {'cond':>8} {'|F_t|/|F|':>10} "
          f"{'R_tau':>8} {'R>1':>6}")
    for c in split:
        if c["penalty_label"] in ("Aheur", "Amargin"):
            print(f"{c['n_assets']:>2} {c['penalty_label']:>9} {c['P_F_median']:>7.3f} "
                  f"{c['conditional_quality_rate_median']:>8.4f} "
                  f"{c['feasible_set_quality_share']:>10.4f} "
                  f"{c['enrichment_R_tau_median']:>8.2f} "
                  f"{c['runs_with_R_tau_above_1']:>3}/{c['runs']}")

    fac = acquisition_factors(rows)
    write_csv(RESULTS_DIR / f"paper01_e1_acquisition_factors{suffix}.csv", fac)
    print("\nP_tau(A_margin)/P_tau(A_heur) = P_F ratio x R_tau ratio, geometric mean over "
          "the same pairs (medians are levels):")
    for d in fac:
        print(f"  N={d['n_assets']:>2} {d['P_tau_ratio_geomean']:.2f} = "
              f"{d['P_F_ratio_geomean']:.2f} x {d['R_tau_ratio_geomean']:.2f}  "
              f"(P_F carries {100*d['log_share_P_F']:.0f}% of the log gain; medians "
              f"{d['P_tau_ratio_median']:.2f} / {d['P_F_ratio_median']:.2f} / "
              f"{d['R_tau_ratio_median']:.2f}; R>1 in {d['pairs_R_tau_ratio_above_1']}/{d['pairs']})")

    hits = optimum_hits(rows)
    write_csv(RESULTS_DIR / f"paper01_e1_optimum_hits{suffix}.csv", hits)
    print("\nExact-optimum share of the best feasible shot, per setting (Fig. 2's zero line):")
    for h in hits:
        print(f"  N={h['n_assets']:>2} {h['penalty_label']:>9} {h['optimum_hits']:>3}/"
              f"{h['runs_with_feasible_shot']:<3} of {h['runs']} runs "
              f"({100*h['share_of_runs_with_feasible_shot']:.0f}%)")

    tv_path = RESULTS_DIR / f"paper01_e1_conditional_tv{suffix}.csv"
    tv_rows = load(tv_path) if tv_path.exists() else None
    inifin = initial_final(rows, refs, tv_rows)
    write_csv(RESULTS_DIR / f"paper01_e1_initial_final{suffix}.csv", inifin)
    print("\nInitial → final at a fixed A (paired within run; medians over seeds, then instances)"
          + ("" if tv_rows else "  [no conditional-TV file: D_cond omitted]"))
    for r in inifin:
        if r["field"] in ("P_F", "D_cond", "R_tau"):
            print(f"  N={r['n_assets']:>2} {r['penalty_label']:>9} {r['field']:>7}: "
                  f"{r['initial_median']:.4f} → {r['final_median']:.4f}, paired "
                  f"{r['paired_diff_median']:+.4f} [{r['paired_diff_ci95_low']:+.4f}, "
                  f"{r['paired_diff_ci95_high']:+.4f}], up {r['instances_increased']}/{r['instances']}")

    dec = decomposition(rows)
    write_csv(RESULTS_DIR / f"paper01_e1_decomposition{suffix}.csv", dec)
    print("\ng_off ratio decomposition (treatment / control). Medians are levels "
          "and do not multiply; the geometric means are the decomposition:")
    for d in dec:
        print(f"  N={d['n_assets']:>2} {d['treatment']:>9}/{d['control']:<7} "
              f"medians: ratio {d['g_off_ratio_median']:>8.3f}, solution "
              f"{d['solution_part_median']:>7.3f} [{d['solution_part_p25']:.3f},"
              f"{d['solution_part_p75']:.3f}], denominator {d['denominator_part_median']:>8.3f}")
        print(f"  {'':>2}   {'':>9} {'':<7} "
              f"geometric: {d['g_off_ratio_geomean']:>8.3f} = "
              f"{d['solution_part_geomean']:.3f} × {d['denominator_part_geomean']:.3f}  "
              f"({d['pairs']} pairs; {d['skipped_zero_gap']} zero-gap, "
              f"{d['skipped_no_feasible_shot']} no-feasible skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
