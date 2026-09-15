# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Paper ① / review round 2, C3: is the optimised state uniform *within* the
feasible set, or does it merely land in the τ-band at the uniform rate?

Sec. V-B explained E2's 0 of 450 by the enrichment ratio R_τ being 1.00 at
A_heur: conditioned on feasibility, the state finds the τ-band as often as an
indifferent draw from F would. That is one number about one subset. It is
consistent with the conditional distribution being uniform on F, and also with
it being anything else that happens to put the same mass in F_τ — the
distribution (½, 0, ½, 0) over a four-element F whose first two elements form
F_τ has R_τ = 1 and is not uniform. The E1 artifacts hold the exact
probability of every feasible bitstring, so the question can be answered
directly rather than inferred: the total-variation distance between the
conditional distribution p(x)/P_F on F and the uniform distribution on F,

    D_cond = ½ Σ_{x ∈ F} | p(x)/P_F − 1/|F| |,

which is 0 for uniform and approaches 1 − 1/|F| for a point mass. It is
undefined at P_F = 0, and at very small P_F the division amplifies whatever
error the probabilities carry, so runs under ``P_F_FLOOR`` are counted rather
than normalised. Two numbers are reported next to it for scale: the same
distance for the *initial* (unoptimised) state of the run, and the distance
a 1,000-shot empirical distribution drawn from uniform-on-F would show, which
is what "uniform" looks like at the experiment's own resolution.

Reads ``results/e1_runs/*.npz`` (``feasible_indices``, ``final_probs_feasible``,
``initial_probs_feasible``) and writes

    paper01_e1_conditional_tv.csv          one row per run
    paper01_e1_conditional_tv_summary.csv  per (N, A): median / IQR / range of
                                           D_cond, the initial-state and
                                           finite-shot reference distances,
                                           and the undefined count

Run (on the host that holds the artifacts):
    python -m experiments.paper01_qubo_baseline.analyze_conditional
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.paper01_qubo_baseline.analyze_e1 import PENALTY_ORDER
from experiments.paper01_qubo_baseline.run_e1_cross import ARTIFACT_DIR, RESULTS_DIR
from experiments.paper01_qubo_baseline.run_experiment import write_csv

#: Below this feasible mass the conditional distribution is not normalised.
P_F_FLOOR = 1e-9
#: Finite-shot reference: expected TV of a 1,000-shot empirical distribution
#: drawn from uniform-on-F, averaged over this many draws.
SHOTS = 1000
MC_REPLICATES = 200
MC_SEED = 20260912

_RUN_ID = re.compile(r"^N(?P<n>\d+)_i(?P<inst>\d+)_(?P<label>.+)_s(?P<seed>\d+)$")


def conditional_tv(probs_feasible: np.ndarray) -> tuple[float, float]:
    """(P_F, D_cond) for one state's probabilities on the feasible set.

    Returns ``D_cond = nan`` when P_F is under ``P_F_FLOOR``.
    """
    p = np.asarray(probs_feasible, dtype=float)
    p_f = float(p.sum())
    if not p_f > P_F_FLOOR:
        return p_f, float("nan")
    return p_f, float(0.5 * np.abs(p / p_f - 1.0 / p.size).sum())


def finite_shot_reference(size_f: int, shots: int, rng) -> float:
    """Mean TV between uniform-on-F and a ``shots``-draw empirical distribution."""
    if size_f <= 0:
        return float("nan")
    tv = np.empty(MC_REPLICATES)
    for b in range(MC_REPLICATES):
        counts = rng.multinomial(shots, np.full(size_f, 1.0 / size_f))
        tv[b] = 0.5 * np.abs(counts / shots - 1.0 / size_f).sum()
    return float(tv.mean())


def per_run(artifact_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(artifact_dir.glob("*.npz")):
        m = _RUN_ID.match(path.stem)
        if m is None:
            continue
        z = np.load(path)
        p_f, d_final = conditional_tv(z["final_probs_feasible"])
        p_f0, d_initial = conditional_tv(z["initial_probs_feasible"])
        size_f = int(z["feasible_indices"].size)
        rows.append({
            "run_id": path.stem, "n_assets": int(m["n"]), "instance": int(m["inst"]),
            "penalty_label": m["label"], "qaoa_seed": int(m["seed"]),
            "feasible_set_size": size_f,
            "final_P_F": p_f, "final_D_cond": d_final,
            "initial_P_F": p_f0, "initial_D_cond": d_initial,
            "point_mass_D_cond": 1.0 - 1.0 / size_f if size_f else float("nan"),
        })
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    rng = np.random.default_rng(MC_SEED)
    ref_cache: dict[int, float] = {}
    out = []
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[(r["n_assets"], r["penalty_label"])].append(r)
    order = {lab: i for i, lab in enumerate(PENALTY_ORDER)}
    for (n, label), g in sorted(by_cell.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 99))):
        d = np.asarray([r["final_D_cond"] for r in g], dtype=float)
        d0 = np.asarray([r["initial_D_cond"] for r in g], dtype=float)
        defined = d[~np.isnan(d)]
        sizes = sorted({r["feasible_set_size"] for r in g})
        ref = float(np.median([ref_cache.setdefault(s, finite_shot_reference(s, SHOTS, rng))
                               for s in sizes]))
        out.append({
            "n_assets": n, "penalty_label": label, "runs": len(g),
            "runs_undefined": int(np.isnan(d).sum()),
            "feasible_set_size_median": float(np.median([r["feasible_set_size"] for r in g])),
            "final_P_F_median": float(np.median([r["final_P_F"] for r in g])),
            "final_D_cond_median": float(np.median(defined)) if defined.size else float("nan"),
            "final_D_cond_p25": float(np.percentile(defined, 25)) if defined.size else float("nan"),
            "final_D_cond_p75": float(np.percentile(defined, 75)) if defined.size else float("nan"),
            "final_D_cond_min": float(defined.min()) if defined.size else float("nan"),
            "final_D_cond_max": float(defined.max()) if defined.size else float("nan"),
            "initial_D_cond_median": float(np.nanmedian(d0)) if np.any(~np.isnan(d0)) else float("nan"),
            "uniform_F_1000_shot_D_cond": ref,
            "point_mass_D_cond_median": float(np.median([r["point_mass_D_cond"] for r in g])),
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    args = ap.parse_args(argv)
    if not args.artifact_dir.is_dir() or not any(args.artifact_dir.glob("*.npz")):
        print(f"no run artifacts under {args.artifact_dir}", file=sys.stderr)
        return 1
    rows = per_run(args.artifact_dir)
    summary = summarise(rows)
    write_csv(RESULTS_DIR / "paper01_e1_conditional_tv.csv", rows)
    write_csv(RESULTS_DIR / "paper01_e1_conditional_tv_summary.csv", summary)
    print(f"{len(rows)} runs\n")
    print(f"{'N':>2} {'A':>9} {'runs':>4} {'undef':>5} {'P_F':>6} {'D_cond med (IQR)':>22} "
          f"{'min–max':>13} {'initial':>8} {'unif 1k':>8} {'point':>6}")
    for s in summary:
        print(f"{s['n_assets']:>2} {s['penalty_label']:>9} {s['runs']:>4} {s['runs_undefined']:>5} "
              f"{s['final_P_F_median']:>6.3f} {s['final_D_cond_median']:>7.3f} "
              f"({s['final_D_cond_p25']:.3f}–{s['final_D_cond_p75']:.3f}) "
              f"{s['final_D_cond_min']:>6.3f}–{s['final_D_cond_max']:<6.3f} "
              f"{s['initial_D_cond_median']:>8.3f} {s['uniform_F_1000_shot_D_cond']:>8.3f} "
              f"{s['point_mass_D_cond_median']:>6.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
