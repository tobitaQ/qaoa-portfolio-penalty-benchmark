# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Paper ① / TQE plan E0 §4.3: re-score the *same* solutions under different A.

No QAOA is run here. For each published instance (N = 8 … 30) a handful of
feasible solutions is fixed -- the optimum, solutions at chosen ranks, the
worst feasible one, and the QAOA solution Table I reports -- and each is
scored under every penalty setting E1 uses, with three denominators:

    g_off   (f − f*) / |f* − A·K²|         the conventional reported gap
    g_F     (f − f*) / Δ_F                 feasible-range gap (A-independent)
    r_all   (C_max − E_A) / (C_max − E_A*) Abbas-style ratio, C_max over all
                                           2^N states at this A

Because the solution is held fixed, every change in g_off across A is the
denominator and nothing else: g_F = D(A) · g_off with D(A) = |f* − A·K²|/Δ_F.
This is what separates "the number moved" from "the solver moved" in the E1
analysis (plan §5.6), and it is the figure the revision plan lists first.

Exactness is recorded per row. For N ≤ 20 the all-states enumeration gives the
exact A_crit and the exact C_max. For N > 20 the threshold is the shipped
bounded A_min (labelled ``A_safe``, an upper bound on A_crit) and C_max is the
energy of the all-ones bitstring, flagged ``c_max_proven = False`` (plan §3.5:
the maximum is not assumed to sit at the most violating string without proof).

Run:
    python -m experiments.paper01_qubo_baseline.run_e0_rescore
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import csv
import sys
import time

import numpy as np

from experiments.paper01_qubo_baseline.run_degeneracy import minimum_feasible_penalty
from experiments.paper01_qubo_baseline.run_e1_cross import EPS_A, MULTIPLES
from experiments.paper01_qubo_baseline.run_experiment import (
    DATA_END_DATE,
    LOOKBACK_YEARS,
    RESULTS_DIR,
    RISK_AVERSION,
    write_csv,
)
from experiments.paper01_qubo_baseline.run_instances import instance_tickers
from src.finance.data_loader import FinanceDataLoader
from src.qubo import scoring
from src.qubo.portfolio import PortfolioQUBO

DEFAULT_N = [8, 12, 16, 20, 24, 28, 30]

#: Ranks in the feasible set to hold fixed, besides optimum / worst / median.
FIXED_RANKS = [2, 5, 10, 50, 100, 1000]

#: Largest N for which the all-states enumeration is run (exact A_crit, C_max).
EXACT_N = 20


def published_qaoa_energy(n: int) -> float | None:
    path = RESULTS_DIR / "paper01_results.csv"
    if not path.exists():
        return None
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["n_assets"]) == n and not r["solver"].startswith("classical"):
                return float(r["energy_no_offset"])
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, nargs="+", default=DEFAULT_N)
    ap.add_argument("--end-date", default=DATA_END_DATE)
    args = ap.parse_args(argv)

    loader, formulator = FinanceDataLoader(), PortfolioQUBO()
    rows: list[dict] = []
    for n in sorted(args.n):
        t0 = time.perf_counter()
        data = loader.load(instance_tickers(n, 0), end_date=args.end_date,
                           lookback_years=LOOKBACK_YEARS)
        k = max(2, n // 5)
        heur = formulator.formulate(data.returns, data.covariance, num_select=k,
                                    risk_aversion=RISK_AVERSION)
        a_heur = float(heur.metadata["penalty_strength"])
        obj = scoring.Objective.from_problem(heur, data.returns, data.covariance)
        feas = scoring.enumerate_feasible(obj)
        order = np.argsort(feas.f, kind="stable")

        if n <= EXACT_N:
            states = scoring.enumerate_all(obj)
            a_m = states.cardinality_minima()
            a_crit, binding = scoring.critical_penalty(a_m, k)
            a_margin, target = scoring.margin_penalty(a_m, k, feas.f_bar, EPS_A)
            threshold_kind = "A_crit (exact, all states)"
        else:
            states = None
            q_obj = formulator.formulate(data.returns, data.covariance, num_select=k,
                                         risk_aversion=RISK_AVERSION,
                                         penalty_strength=0.0).Q
            a_crit, binding, _exact = minimum_feasible_penalty(q_obj, n, k)
            a_margin, target = np.nan, np.nan
            threshold_kind = "A_safe (bounded outside K±2; upper bound on A_crit)"

        # Which QAOA row of Table I is this, in the feasible ranking?
        e_pub = published_qaoa_energy(n)
        pub_pos = None
        if e_pub is not None:
            f_pub = e_pub + a_heur * k * k
            j = int(np.argmin(np.abs(feas.f - f_pub)))
            if abs(feas.f[j] - f_pub) < 1e-6:
                pub_pos = j

        fixed: list[tuple[str, int]] = [("optimum", int(order[0]))]
        for r in FIXED_RANKS:
            if r <= feas.size:
                fixed.append((f"rank_{r}", int(order[r - 1])))
        fixed.append(("median", int(order[feas.size // 2])))
        fixed.append(("worst_feasible", int(order[-1])))
        if pub_pos is not None:
            fixed.append(("published_qaoa", pub_pos))

        settings = [(label, m * a_crit) for m, label in MULTIPLES] if a_crit > 0 else [("A0", 0.0)]
        settings += [("Aheur", a_heur)]
        if not np.isnan(a_margin):
            settings.append(("Amargin", a_margin))

        for label, a in settings:
            if states is not None:
                c_max, c_max_m = scoring.max_penalised_energy(states, obj, a)
                c_max -= a * k * k          # to the E_A scale
                proven = True
            else:
                ones = np.ones((1, n))
                c_max = float(obj.offset_energy(ones, a)[0])
                c_max_m = n
                proven = False
            e_star = feas.f_star - a * k * k
            for sol_label, j in fixed:
                f_x = float(feas.f[j])
                rank, ties = feas.rank(f_x)
                e_x = f_x - a * k * k
                rows.append({
                    "n_assets": n, "n_select": k, "feasible_set_size": feas.size,
                    "solution": sol_label, "rank": rank, "rank_ties": ties,
                    "solution_index": int(feas.indices[j]),
                    "f": f_x, "f_star": feas.f_star, "delta_F": feas.delta,
                    "penalty_label": label, "penalty": a,
                    "threshold": a_crit, "threshold_kind": threshold_kind,
                    "threshold_binding_m": binding, "a_margin": a_margin,
                    "a_margin_target": target, "a_heur": a_heur,
                    "E_A": e_x, "E_A_star": e_star, "abs_E_star": abs(e_star),
                    "g_off": scoring.gap_offset(f_x, feas.f_star, a, k),
                    "g_F": float(feas.gap_range(f_x)),
                    "deflation_D": scoring.deflation_factor(feas.f_star, a, k, feas.delta),
                    "c_max_all_states": c_max, "c_max_cardinality": c_max_m,
                    "c_max_proven": proven,
                    "r_all": ((c_max - e_x) / (c_max - e_star)) if c_max != e_star else np.nan,
                    "r_all_of_worst_feasible":
                        ((c_max - (feas.f_max - a * k * k)) / (c_max - e_star)) if c_max != e_star else np.nan,
                })
        print(f"N={n:>2} K={k} |F|={feas.size:>8,} {threshold_kind.split(' ')[0]}={a_crit:.5f} "
              f"(m={binding}) A_heur={a_heur:.1f} A_margin={a_margin:.5f} Δ_F={feas.delta:.4f} "
              f"published QAOA rank={feas.rank(feas.f[pub_pos])[0] if pub_pos is not None else '-'} "
              f"[{time.perf_counter() - t0:.1f}s]", flush=True)
        worst = [r for r in rows if r["n_assets"] == n and r["solution"] == "worst_feasible"]
        for r in worst:
            print(f"    worst feasible at {r['penalty_label']:>9}: g_off={r['g_off']*100:9.4f}%  "
                  f"g_F=100%  r_all={r['r_all']:.6f}  D={r['deflation_D']:.1f}")

    out = RESULTS_DIR / "paper01_e0_rescore.csv"
    write_csv(out, rows)
    print(f"\nSaved {len(rows)} rows → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
