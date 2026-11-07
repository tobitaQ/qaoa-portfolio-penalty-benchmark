# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""When does the circuit order solutions *inside* the feasible set?

Sec. V-B reported the comparison against a uniform draw from the feasible set
at one penalty weight, A_heur, where the answer is that the optimised state
never exceeds it: 0 of 450 runs. The cross ran six weights. Asking the same
question at the other five turns a negative result into a dose--response, and
the dose is how wide the objective's feasible range is next to the largest
coefficient of the Hamiltonian the optimiser descends:

    eps_F(A) = Delta_F / s_H(A),   s_H = max(max|h_i|, max|J_ij|)

computed per instance by ``coefficient_scale.py``. On the feasible set the
penalty vanishes and its constant cancels in differences, so eps_F is the
energy range the circuit can still resolve between feasible solutions once
H_C / s_H is what it descends. It is a ratio and not a share: it exceeds 1
wherever the objective's range is wider than that coefficient, which is what
the A = 0 rows measure (1.27, 1.67 and 2.26 in the instance median). At the
textbook weight it is 0.015 and the ordering is absent; at 1.1 times the
feasibility threshold it is 0.98 at 20 assets.

Nothing is re-run. Every quantity is read from the committed E1 files:

    paper01_e1_cross.csv                   one row per run
    paper01_e1_reference.csv               per instance, incl. the floor P_tau^uniform-F
    paper01_e1_conditional_tv_summary.csv  per (N, A), D_cond
    paper01_coefficient_scale.csv          per (N, instance, A), eps_F

and written to

    paper01_e1_penalty_ladder.csv          one row per (N, A), weakest weight first

The two factors of the identity P_tau / P_tau^uniform-F = P_F * R_tau are kept
side by side, because the floor is crossed only when their product exceeds one:
at 20 assets the largest median product is 0.80, so the honest statement is the
*share of runs* above the floor, not that the floor is beaten.

Run:
    python -m experiments.paper01_qubo_baseline.analyze_ladder
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.paper01_qubo_baseline.analyze_e1 import PENALTY_ORDER, _f, _q, load
from experiments.paper01_qubo_baseline.run_experiment import RESULTS_DIR, write_csv

#: Weakest penalty first, so the ladder reads in the direction of the dose.
LADDER_ORDER = ["A0", "1.1xAcrit", "2xAcrit", "5xAcrit", "10xAcrit", "Amargin", "Aheur"]

CROSS = "paper01_e1_cross.csv"
REFERENCE = "paper01_e1_reference.csv"
CONDITIONAL = "paper01_e1_conditional_tv_summary.csv"
#: Per instance, not per (N, A): the cells of this ladder cover different
#: instance sets -- 30 where A_crit > 0 is not required, 21 at N = 12 where it
#: is, 9 at the A = 0 cell of the cross -- so a median taken over a cell's own
#: instances is the only one that compares the same problems the runs used.
SCALES = "paper01_coefficient_scale.csv"
#: E6 runs the same measurement with the constraint in the ansatz instead of in
#: the cost (``run_xy_mixer.py``). Its rows belong on the same ladder: P_F = 1
#: holds at every angle by construction, while eps_F is measured like every
#: other row, at the A = 0 scales (the cost is the objective, so s_H is the
#: objective's own largest Ising coefficient). They are read from their own
#: block's reference table, like every other row here.
XY_CROSS = "paper01_xy_cross.csv"
XY_REFERENCE = "paper01_xy_reference.csv"
OUT = "paper01_e1_penalty_ladder.csv"
#: The same ladder on the instances a size ran at *every* weight. At N = 12 the
#: multiples of A_crit exist on 21 of the 30 instances, so the six cells of the
#: main file describe different instance sets and a share read across them mixes
#: the setting with the sample. At N = 16 and N = 20 the two files agree row for
#: row, which is what makes the N = 12 comparison readable.
OUT_COMMON = "paper01_e1_penalty_ladder_common.csv"


def _by_key(rows: list[dict], *keys: str) -> dict[tuple, dict]:
    return {tuple(r[k] for k in keys): r for r in rows}


def _scale_median(scale: dict[tuple, dict], n: int, label: str,
                  instances: set[int], field: str) -> float | None:
    """Median of one scale column over the instances this cell ran.

    Taking it from a per-(N, A) summary instead would mix instance sets: the
    multiples of A_crit exist on 21 of the 30 N = 12 instances, and the A = 0
    cell of the cross on 9.
    """
    v = [_f(scale[(n, i, label)][field]) for i in sorted(instances)
         if (n, i, label) in scale]
    return float(np.median(v)) if v else None


def _cell(n: int, label: str, g: list[dict], ref: dict[tuple, dict],
           eps: float | None, ising: float | None, d_cond: float | None) -> dict:
    """One (N, setting) row: the two factors, the floor and what crossed it."""
    floor, ratio, r_tau = [], [], []
    for r in g:
        # Each run is scored against its own instance's reference row. The
        # second window has its own reference file; joining across them is the
        # error the companion paper's reference-row relation catches.
        u = _f(ref[(r["n_assets"], r["instance"])]["P_tau_main_uniform_F"])
        p_tau, p_f = _f(r["final_P_tau_main"]), _f(r["final_P_F"])
        floor.append(u)
        ratio.append(p_tau / u if u > 0 else np.nan)
        r_tau.append((p_tau / p_f) / u if u > 0 and p_f > 0 else np.nan)
    ratio_a = np.asarray(ratio, dtype=float)
    return {
        "n_assets": n, "penalty_label": label, "runs": len(g),
        "instances": len({r["instance"] for r in g}),
        "epsilon_F_median": eps if eps is not None else np.nan,
        "ising_ratio_median": ising if ising is not None else np.nan,
        "D_cond_median": d_cond if d_cond is not None else np.nan,
        "P_F_median": _q([_f(r["final_P_F"]) for r in g], 50),
        "R_tau_p25": _q(r_tau, 25), "R_tau_median": _q(r_tau, 50), "R_tau_p75": _q(r_tau, 75),
        "P_tau_over_floor_median": _q(ratio, 50),
        "runs_above_floor": int(np.sum(ratio_a > 1.0)),
        "Q_tau_1000_median": _q([_f(r["final_Q_tau_main_1000"]) for r in g], 50),
        "g_F_median": _q([_f(r["g_F"]) for r in g if r["g_F"] != ""], 50),
        "no_feasible_shot_runs": sum(1 for r in g if r["g_F"] == ""),
        "infeasible_batch_minimum_runs": sum(1 for r in g if r["best_all_feasible"] == "False"),
    }


def _xy_d_cond(results_dir: Path) -> dict[tuple, float]:
    """Median distance from uniform on F per (mixer, N), from the E6 artifacts.

    The penalty block's D_cond comes from ``analyze_conditional.py``, which
    reads ``e1_runs``; E6 writes its own artifacts, and each one holds the same
    thing -- the exact probability of every feasible bitstring -- so the same
    quantity is computed here rather than left blank in the table.
    """
    import re
    from collections import defaultdict
    art = results_dir / "xy_runs"
    if not art.is_dir():
        return {}
    by: dict[tuple, list[float]] = defaultdict(list)
    for f in sorted(art.glob("*.npz")):
        m = re.match(r"N(\d+)_i\d+_(\w+?)_s\d+\.npz", f.name)
        if not m:
            continue
        with np.load(f) as z:
            p = np.asarray(z["final_probs_feasible"], dtype=float)
        p_f = p.sum()
        if p_f <= 0:
            continue
        by[(m.group(2), int(m.group(1)))].append(
            0.5 * float(np.abs(p / p_f - 1.0 / p.size).sum()))
    return {k: float(np.median(v)) for k, v in by.items()}


def xy_rows(results_dir: Path) -> list[dict]:
    """The E6 block as ladder rows, one per size, or nothing if it has not run."""
    cross, reference = results_dir / XY_CROSS, results_dir / XY_REFERENCE
    if not (cross.exists() and reference.exists()):
        return []
    runs = [r for r in load(cross) if r["status"] == "ok"]
    ref = _by_key(load(reference), "n_assets", "instance")
    d_cond = _xy_d_cond(results_dir)
    scale = {(int(r["n_assets"]), int(r["instance"]), r["penalty_label"]): r
             for r in (load(results_dir / SCALES) if (results_dir / SCALES).exists() else [])}
    out = []
    for mixer in sorted({r.get("mixer", "xy_ring") for r in runs}):
        for n in sorted({int(r["n_assets"]) for r in runs}):
            g = [r for r in runs if int(r["n_assets"]) == n and r.get("mixer") == mixer]
            if not g:
                continue
            # Both blocks put the whole cost on the objective, so their scales
            # are the A = 0 scales: eps_F = Delta_F / s_H(objective only), which
            # is measured per instance and is not 1.
            inst = {int(r["instance"]) for r in g}
            out.append(_cell(n, "XY" if mixer == "xy_ring" else "A0-transverse",
                             g, ref,
                             _scale_median(scale, n, "A0", inst, "epsilon_F"),
                             _scale_median(scale, n, "A0", inst,
                                           "ising_ratio_objective_over_s_H"),
                             d_cond.get((mixer, n))))
    return out


def ladder(results_dir: Path) -> list[dict]:
    """One row per (N, penalty setting), with both factors and the floor."""
    runs = [r for r in load(results_dir / CROSS) if r["status"] == "ok"]
    ref = _by_key(load(results_dir / REFERENCE), "n_assets", "instance")
    cond = _by_key(load(results_dir / CONDITIONAL), "n_assets", "penalty_label") \
        if (results_dir / CONDITIONAL).exists() else {}
    scale = {(int(r["n_assets"]), int(r["instance"]), r["penalty_label"]): r
             for r in (load(results_dir / SCALES) if (results_dir / SCALES).exists() else [])}

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        groups[(int(r["n_assets"]), r["penalty_label"])].append(r)

    out = []
    for (n, label), g in sorted(groups.items(),
                                key=lambda kv: (kv[0][0], LADDER_ORDER.index(kv[0][1]))):
        inst = {int(r["instance"]) for r in g}
        out.append(_cell(
            n, label, g, ref,
            _scale_median(scale, n, label, inst, "epsilon_F"),
            _scale_median(scale, n, label, inst, "ising_ratio_objective_over_s_H"),
            _f(cond[(str(n), label)]["final_D_cond_median"])
            if (str(n), label) in cond else None))
    out += xy_rows(results_dir)
    return out


def common_instance_ladder(results_dir: Path) -> list[dict]:
    """The penalty block restricted, per size, to instances that ran all six."""
    runs = [r for r in load(results_dir / CROSS) if r["status"] == "ok"]
    ref = _by_key(load(results_dir / REFERENCE), "n_assets", "instance")
    scale = {(int(r["n_assets"]), int(r["instance"]), r["penalty_label"]): r
             for r in (load(results_dir / SCALES) if (results_dir / SCALES).exists() else [])}
    weights = {w for w in LADDER_ORDER if w != "A0"}

    ran: dict[int, dict[int, set]] = defaultdict(lambda: defaultdict(set))
    for r in runs:
        ran[int(r["n_assets"])][int(r["instance"])].add(r["penalty_label"])

    out = []
    for n in sorted(ran):
        common = {i for i, labels in ran[n].items() if weights <= labels}
        for label in sorted(weights, key=LADDER_ORDER.index):
            g = [r for r in runs if int(r["n_assets"]) == n
                 and r["penalty_label"] == label and int(r["instance"]) in common]
            if not g:
                continue
            out.append(_cell(n, label, g, ref,
                             _scale_median(scale, n, label, common, "epsilon_F"),
                             _scale_median(scale, n, label, common,
                                           "ising_ratio_objective_over_s_H"),
                             None))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args(argv)

    rows = ladder(args.results_dir)
    write_csv(args.results_dir / OUT, rows)
    write_csv(args.results_dir / OUT_COMMON, common_instance_ladder(args.results_dir))
    print(f"{'N':>3} {'A':>9} {'runs':>5} {'eps_F':>7} {'D_cond':>7} {'P_F':>6} "
          f"{'R_tau':>6} {'P/floor':>8} {'>floor':>7} {'Q(1000)':>8} {'g_F':>7} {'infeas.min':>11}")
    for r in rows:
        print(f"{r['n_assets']:>3} {r['penalty_label']:>9} {r['runs']:>5} "
              f"{r['epsilon_F_median']:>7.3f} {r['D_cond_median']:>7.3f} {r['P_F_median']:>6.3f} "
              f"{r['R_tau_median']:>6.2f} {r['P_tau_over_floor_median']:>8.2f} "
              f"{str(r['runs_above_floor']) + '/' + str(r['runs']):>7} "
              f"{r['Q_tau_1000_median']:>8.3f} {r['g_F_median']:>7.4f} "
              f"{str(r['infeasible_batch_minimum_runs']) + '/' + str(r['runs']):>11}")
    print(f"\nwrote {args.results_dir / OUT} and {args.results_dir / OUT_COMMON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
