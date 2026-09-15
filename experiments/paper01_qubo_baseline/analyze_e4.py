# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Paper ① / TQE plan E4-A: separate numerics from sampling across backends.

Reads the fixed-angle runs written by ``run_e4_fixed_angles.py`` -- the local
arms in ``paper01_e4_states.csv`` / ``paper01_e4_batches.csv`` and the SV1
arm under its results tag -- together with the per-circuit probability
artifacts in ``results/e4_runs/``, and writes

    paper01_e4_states_compare.csv   per circuit x arm: P_F, P_tau, and the
                                    distance of the arm's exact state from the
                                    GPU double-precision reference (max |dp|
                                    and total variation, on the feasible set
                                    for every arm and on all 2^N states where
                                    the full vector exists)
    paper01_e4_batches_summary.csv  per circuit x arm: what ten independent
                                    1,000-shot batches of the *same* state
                                    returned -- feasible fraction, best
                                    feasible shot, how many distinct "best"
                                    portfolios -- next to what i.i.d. sampling
                                    from the arm's own exact distribution
                                    predicts (Monte Carlo)
    paper01_e4_billing.csv          per cloud task: ARN, execution ms, floor

The question (plan §8.3): when two backends return different portfolios from
one configuration, is the *state* different (numerics) or are the *draws*
different (finite shots)? Here the state is fixed by its angles, so any
difference between arms' exact distributions is numerics, and any spread
between batches of one arm is sampling.

Run:
    python -m experiments.paper01_qubo_baseline.analyze_e4 [--sv1-tag e4a]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.paper01_qubo_baseline.analyze_e1 import _f, load
from experiments.paper01_qubo_baseline.run_e1_cross import RESULTS_DIR, TAU_MAIN
from experiments.paper01_qubo_baseline.run_e4_fixed_angles import (
    E4_ARTIFACT_DIR,
    SV1_MIN_SECONDS,
    SV1_USD_PER_MINUTE,
)
from experiments.paper01_qubo_baseline.run_experiment import write_csv

REFERENCE_ARM = "gpu_double"
ARM_ORDER = ["gpu_double", "cpu_double", "gpu_single", "sv1"]
MC_REPLICATES = 500
MC_SEED = 20260912

CIRCUIT_KEY = ("n_assets", "instance", "qaoa_seed")


def _key(r: dict) -> tuple:
    return tuple(str(r[k]) for k in CIRCUIT_KEY)


def _artifact(arm: str, r: dict) -> dict | None:
    p = E4_ARTIFACT_DIR / f"{arm}_{r['e1_run_id']}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    idx = d["feasible_indices"]
    full = d["probabilities"] if d["probabilities"].size else None
    feas = d["feasible_probabilities"] if "feasible_probabilities" in d.files else full[idx]
    return {"full": full, "feasible": feas, "indices": idx}


def compare_states(states: dict[str, list[dict]]) -> list[dict]:
    """Each arm's exact state against the GPU double-precision reference."""
    ref_rows = {_key(r): r for r in states.get(REFERENCE_ARM, [])}
    out = []
    for arm in ARM_ORDER:
        for r in states.get(arm, []):
            k = _key(r)
            a = _artifact(arm, r)
            row = {"arm": arm, "backend": r["backend"], "precision": r["precision"],
                   "n_assets": int(r["n_assets"]), "instance": int(r["instance"]),
                   "qaoa_seed": int(r["qaoa_seed"]), "e1_run_id": r["e1_run_id"],
                   "P_F": _f(r["final_P_F"]), "P_tau_main": _f(r["final_P_tau_main"]),
                   "P_opt": _f(r["final_P_opt"]), "Q_tau_main_1000": _f(r["final_Q_tau_main_1000"]),
                   "probs_sum": _f(r.get("final_probs_sum", ""))}
            ref = ref_rows.get(k)
            b = _artifact(REFERENCE_ARM, ref) if ref else None
            if a is not None and b is not None:
                d = a["feasible"] - b["feasible"]
                row.update({
                    "max_abs_dp_F": float(np.abs(d).max()),
                    "tv_F": float(0.5 * np.abs(d).sum()),
                    # tv_F alone is half the L1 distance of two *unnormalised*
                    # restrictions, which is not the total variation of a
                    # distribution and does not bound anything by itself. Fold
                    # every infeasible outcome into one symbol and it does: on
                    # F u {perp} with p_perp = 1 - P_F, this is the largest
                    # difference in probability the selection rule can see,
                    # since the rule only ever asks which feasible bitstring
                    # came up or whether none did.
                    "tv_F_perp": float(0.5 * np.abs(d).sum()
                                       + 0.5 * abs(row["P_F"] - _f(ref["final_P_F"]))),
                    "dP_F": row["P_F"] - _f(ref["final_P_F"]),
                    "dP_tau_main": row["P_tau_main"] - _f(ref["final_P_tau_main"]),
                    "rel_dP_tau_main": ((row["P_tau_main"] - _f(ref["final_P_tau_main"]))
                                        / _f(ref["final_P_tau_main"])),
                })
                if a["full"] is not None and b["full"] is not None:
                    dd = a["full"] - b["full"]
                    row["max_abs_dp_all"] = float(np.abs(dd).max())
                    row["tv_all"] = float(0.5 * np.abs(dd).sum())
                else:
                    row["max_abs_dp_all"] = row["tv_all"] = ""
            out.append(row)
    return out


def _mc_batches(rng, feasible_p: np.ndarray, feasible_f: np.ndarray, shots: int,
                n_batches: int, replicates: int) -> dict:
    """What i.i.d. draws from the arm's own state give for ``n_batches`` batches.

    Only the feasible set matters for the best feasible shot, so each batch is
    a Binomial(shots, P_F) number of feasible shots drawn from P conditioned on
    F; the best is the drawn state of lowest f.
    """
    p_f = float(feasible_p.sum())
    if p_f <= 0:
        return {"mc_distinct_best_median": 0, "mc_distinct_best_p025": 0, "mc_distinct_best_p975": 0,
                "mc_no_feasible_batches_mean": float(n_batches)}
    cond = feasible_p / p_f
    order = np.argsort(feasible_f, kind="stable")
    distinct, empty = [], []
    for _ in range(replicates):
        best = set(); n_empty = 0
        counts = rng.binomial(shots, p_f, size=n_batches)
        for c in counts:
            if c == 0:
                n_empty += 1
                continue
            draw = rng.choice(feasible_p.size, size=int(c), replace=True, p=cond)
            # lowest f among the drawn feasible states
            best.add(int(min(draw, key=lambda i: (feasible_f[i], i))))
        distinct.append(len(best)); empty.append(n_empty)
    distinct = np.asarray(distinct)
    return {"mc_distinct_best_median": float(np.median(distinct)),
            "mc_distinct_best_p025": float(np.percentile(distinct, 2.5)),
            "mc_distinct_best_p975": float(np.percentile(distinct, 97.5)),
            "mc_no_feasible_batches_mean": float(np.mean(empty))}


def summarise_batches(states: dict[str, list[dict]], batches: dict[str, list[dict]],
                      feasible_f_by_key: dict[tuple, np.ndarray]) -> list[dict]:
    rng = np.random.default_rng(MC_SEED)
    out = []
    for arm in ARM_ORDER:
        st = {_key(r): r for r in states.get(arm, [])}
        by: dict[tuple, list[dict]] = defaultdict(list)
        for r in batches.get(arm, []):
            by[_key(r)].append(r)
        for k, rows in sorted(by.items(), key=lambda kv: tuple(int(x) for x in kv[0])):
            s = st.get(k)
            shots = int(rows[0]["shots"])
            feas = np.asarray([_f(r["feasible_shot_fraction"]) for r in rows])
            g_f = np.asarray([_f(r["g_F"]) for r in rows if r["g_F"] != ""])
            best_feas = {r["best_feasible_index"] for r in rows if r["best_feasible_index"] != ""}
            best_all = {r["best_all_index"] for r in rows}
            p_f = _f(s["final_P_F"]) if s else np.nan
            sd = np.sqrt(shots * p_f * (1 - p_f)) if s else np.nan
            z = (feas * shots - shots * p_f) / sd if s and sd > 0 else np.full(len(rows), np.nan)
            row = {
                "arm": arm, "n_assets": int(k[0]), "instance": int(k[1]), "qaoa_seed": int(k[2]),
                "batches": len(rows), "shots": shots,
                "P_F_exact": p_f,
                "feasible_fraction_mean": float(feas.mean()),
                "feasible_fraction_min": float(feas.min()),
                "feasible_fraction_max": float(feas.max()),
                "feasible_count_z_max_abs": float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else "",
                "batches_within_2sd": int((np.abs(z) <= 2).sum()) if np.isfinite(z).any() else "",
                "no_feasible_shot_batches": sum(1 for r in rows if r["best_feasible_index"] == ""),
                "min_energy_shot_infeasible_batches": sum(1 for r in rows if r["best_all_feasible"] != "True"),
                "g_F_best_feasible_min": float(g_f.min()) if g_f.size else "",
                "g_F_best_feasible_median": float(np.median(g_f)) if g_f.size else "",
                "g_F_best_feasible_max": float(g_f.max()) if g_f.size else "",
                "batches_best_feasible_optimal": sum(1 for r in rows if r["is_optimal"] == "True"),
                "batches_best_feasible_within_tau": sum(1 for r in rows if r["within_tau_main"] == "True"),
                "distinct_best_feasible": len(best_feas),
                "distinct_min_energy_shot": len(best_all),
                "shots_within_tau_total": sum(int(r["shots_within_tau_main"]) for r in rows),
                "expected_shots_within_tau": (shots * len(rows) * _f(s["final_P_tau_main"])) if s else "",
            }
            a = _artifact(arm, s) if s else None
            if a is not None and k in feasible_f_by_key:
                row.update(_mc_batches(rng, a["feasible"], feasible_f_by_key[k], shots,
                                       len(rows), MC_REPLICATES))
            out.append(row)
    return out


def cross_arm(batches: dict[str, list[dict]]) -> list[dict]:
    """Pooled over arms: do the backends' best feasible shots overlap?"""
    pooled: dict[tuple, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for arm, rows in batches.items():
        for r in rows:
            if r["best_feasible_index"] != "":
                pooled[_key(r)][arm].add(r["best_feasible_index"])
    out = []
    for k, per_arm in sorted(pooled.items(), key=lambda kv: tuple(int(x) for x in kv[0])):
        union = set().union(*per_arm.values())
        common = set.intersection(*per_arm.values()) if per_arm else set()
        out.append({"n_assets": int(k[0]), "instance": int(k[1]), "qaoa_seed": int(k[2]),
                    "arms": len(per_arm),
                    "distinct_best_feasible_pooled": len(union),
                    "best_feasible_common_to_all_arms": len(common),
                    **{f"distinct_{arm}": len(per_arm.get(arm, set())) for arm in ARM_ORDER}})
    return out


def billing(states: dict[str, list[dict]], batches: dict[str, list[dict]]) -> list[dict]:
    out = []
    for arm in ("sv1",):
        for kind, rows in (("analytic", states.get(arm, [])), ("shots", batches.get(arm, []))):
            for r in rows:
                if r.get("task_arn"):
                    ms = _f(r["execution_ms"])
                    out.append({"arm": arm, "kind": kind, "n_assets": int(r["n_assets"]),
                                "instance": int(r["instance"]), "qaoa_seed": int(r["qaoa_seed"]),
                                "batch": r.get("batch", ""), "task_arn": r["task_arn"],
                                "execution_ms": ms,
                                "billable_seconds": max(ms / 1000.0, SV1_MIN_SECONDS),
                                "usd": max(ms / 1000.0, SV1_MIN_SECONDS) / 60.0 * SV1_USD_PER_MINUTE})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sv1-tag", default="e4a")
    args = ap.parse_args(argv)

    states: dict[str, list[dict]] = defaultdict(list)
    batches: dict[str, list[dict]] = defaultdict(list)
    for suffix in ("", f"_{args.sv1_tag}"):
        sp = RESULTS_DIR / f"paper01_e4_states{suffix}.csv"
        bp = RESULTS_DIR / f"paper01_e4_batches{suffix}.csv"
        if sp.exists():
            for r in load(sp):
                states[r["arm"]].append(r)
        if bp.exists():
            for r in load(bp):
                batches[r["arm"]].append(r)
    if not states:
        print("no E4 states found", file=sys.stderr)
        return 1
    for arm in ARM_ORDER:
        print(f"{arm:11s} {len(states.get(arm, [])):2d} circuits, {len(batches.get(arm, [])):3d} batches")

    # f over the feasible set, per circuit, from the E1 artifacts (any arm's
    # npz holds the same feasible indices; f comes from the E1 run file).
    from experiments.paper01_qubo_baseline.run_e1_cross import ARTIFACT_DIR
    feasible_f: dict[tuple, np.ndarray] = {}
    for arm, rows in states.items():
        for r in rows:
            k = _key(r)
            if k not in feasible_f:
                p = ARTIFACT_DIR / f"{r['e1_run_id']}.npz"
                if p.exists():
                    feasible_f[k] = np.load(p)["feasible_f"]

    cmp_rows = compare_states(states)
    sum_rows = summarise_batches(states, batches, feasible_f)
    cross_rows = cross_arm(batches)
    bill_rows = billing(states, batches)
    write_csv(RESULTS_DIR / "paper01_e4_states_compare.csv", cmp_rows)
    write_csv(RESULTS_DIR / "paper01_e4_batches_summary.csv", sum_rows)
    write_csv(RESULTS_DIR / "paper01_e4_cross_arm.csv", cross_rows)
    if bill_rows:
        write_csv(RESULTS_DIR / "paper01_e4_billing.csv", bill_rows)

    print("\n=== exact state vs gpu_double (numerics) ===")
    for r in cmp_rows:
        if r["arm"] == REFERENCE_ARM or "tv_F" not in r:
            continue
        print(f"N={r['n_assets']:2d} i{r['instance']} s{r['qaoa_seed']} {r['arm']:10s} "
              f"max|dp|_F={r['max_abs_dp_F']:.2e} TV_F={r['tv_F']:.2e} "
              f"dP_F={r['dP_F']:+.2e} dP_tau/P_tau={r['rel_dP_tau_main']:+.2e}")
    print("\n=== ten batches of the same state (sampling) ===")
    for r in sum_rows:
        zmax = r["feasible_count_z_max_abs"]
        zmax = "" if zmax == "" else f"{zmax:.2f}"
        print(f"N={r['n_assets']:2d} i{r['instance']} s{r['qaoa_seed']} {r['arm']:10s} "
              f"P_F={r['P_F_exact']:.4f} feas {r['feasible_fraction_min']:.3f}-{r['feasible_fraction_max']:.3f} "
              f"|z|max={zmax} "
              f"noF={r['no_feasible_shot_batches']} distinct best={r['distinct_best_feasible']} "
              f"(MC {r.get('mc_distinct_best_median', '')} [{r.get('mc_distinct_best_p025', '')}, {r.get('mc_distinct_best_p975', '')}]) "
              f"opt={r['batches_best_feasible_optimal']}/{r['batches']}")
    print("\n=== pooled over arms ===")
    for r in cross_rows:
        print(f"N={r['n_assets']:2d} i{r['instance']} s{r['qaoa_seed']} pooled distinct best={r['distinct_best_feasible_pooled']} "
              f"common to all {r['arms']} arms={r['best_feasible_common_to_all_arms']} "
              + " ".join(f"{a}={r[f'distinct_{a}']}" for a in ARM_ORDER))
    if bill_rows:
        usd = sum(r["usd"] for r in bill_rows)
        print(f"\nSV1: {len(bill_rows)} tasks, execution {sum(r['execution_ms'] for r in bill_rows) / 1000:.1f} s, "
              f"billable {sum(r['billable_seconds'] for r in bill_rows):.0f} s = ${usd:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
