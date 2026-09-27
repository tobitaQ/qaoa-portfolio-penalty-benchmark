# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
A3 probe: what changes when a second constraint joins the cardinality constraint?

Go/no-go for the constraint-extension paper. Adds "at most one asset per
sector" to the K-cardinality constraint of the numerical paper, on the same
50-ticker universe, the same window and the same instances, and asks -- from
enumeration alone, no quantum run -- whether phenomena appear that the
single-constraint study could not show:

  1. Feasible-set collapse: |F| under K alone vs under K and the sector cap;
     the uniform-sampler baselines P_F and P_tau under both.
  2. Penalty competition: with H = f + A1 (sum x - K)^2 + A2 * (#same-sector
     pairs), the set of (A1, A2) for which the global minimum of H is feasible
     is a region bounded by a piecewise-linear frontier; the probe computes
     A1_crit as a function of A2 and A2's own critical value. In the
     single-constraint study A_crit was one number.
  3. Post-selection cost: the stored E1 QAOA states were optimised for K only;
     how much of their feasible mass also satisfies the sector cap, and what
     Q_tau(1000) under both constraints would be if the second constraint were
     enforced by filtering rather than by a penalty.
  4. Repair under two constraints: greedy swaps that keep both constraints,
     from random starts, against the enumerated optimum.

Sectors are a proxy: the first digit of the TSE code (2 foods, 3 textiles/
paper/retail, 4 chemicals/pharma, 5 glass/steel/metals, 6 machinery/electric,
7 transport equipment/precision, 8 finance/trading/real estate, 9 transport/
telecom/services/utilities). A real study would use the exchange's 33-sector
classification; the probe only needs a partition.

Run:
    python -m experiments.a3_constraints.probe_two_constraints      # ~10 min CPU
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.paper01_qubo_baseline.run_experiment import RISK_AVERSION, write_csv
from experiments.paper01_qubo_baseline.run_instances import instance_tickers
from src.finance.data_loader import FinanceDataLoader
from src.qubo import scoring
from src.qubo.portfolio import PortfolioQUBO

E1 = Path("experiments/paper01_qubo_baseline/results")
OUT = Path(__file__).parent / "results"
W1 = "2026-08-05"
TAU = 0.01
SHOTS = 1000
SIZES = (12, 16, 20)
INSTANCES = range(10)


def sector_of(ticker: str) -> int:
    return int(ticker[0])


def all_states(n: int) -> np.ndarray:
    """(2^n, n) 0/1 matrix, wire 0 = MSB (the project's bit order)."""
    idx = np.arange(1 << n, dtype=np.int64).reshape(-1, 1)
    shifts = (n - 1 - np.arange(n, dtype=np.int64)).reshape(1, -1)
    return ((idx >> shifts) & 1).astype(np.float64)


def sector_pairs(X: np.ndarray, sectors: np.ndarray) -> np.ndarray:
    """Number of selected pairs sharing a sector, per state: sum_s C(n_s, 2)."""
    out = np.zeros(X.shape[0])
    for s in np.unique(sectors):
        c = X[:, sectors == s].sum(axis=1)
        out += c * (c - 1) / 2
    return out


def critical_frontier(f, v1, v2, feasible, f_star, a2_grid):
    """A1_crit(A2): the smallest A1 such that every infeasible state costs at least f*."""
    inf = ~feasible
    need = f_star - f[inf]                      # how much penalty each infeasible state needs
    v1i, v2i = v1[inf], v2[inf]
    # States violating only the sector cap fix A2's own floor.
    only2 = (v1i == 0) & (v2i > 0)
    a2_alone = float(np.max(need[only2] / v2i[only2])) if only2.any() else 0.0
    a2_alone = max(a2_alone, 0.0)
    rows = []
    for a2 in a2_grid:
        m = v1i > 0
        a1 = float(np.max((need[m] - a2 * v2i[m]) / v1i[m])) if m.any() else 0.0
        rows.append((a2, max(a1, 0.0), a2 >= a2_alone))
    return a2_alone, rows


def repair_two(bits, f_of, sectors, k, max_iter=200):
    """Greedy swaps keeping K and the sector cap; returns (bits, moves)."""
    n = bits.size
    def ok(b):
        return b.sum() == k and sector_pairs(b.reshape(1, -1), sectors)[0] == 0
    cur = f_of(bits)
    moves = 0
    for _ in range(max_iter):
        best = None
        for i in np.flatnonzero(bits):
            for j in np.flatnonzero(bits == 0):
                cand = bits.copy(); cand[i], cand[j] = 0, 1
                if not ok(cand):
                    continue
                fv = f_of(cand)
                if fv < cur - 1e-15 and (best is None or fv < best[0]):
                    best = (fv, cand)
        if best is None:
            break
        cur, bits = best; moves += 1
    return bits, moves


def random_feasible(rng, sectors, k, n):
    """A random K-subset with at most one asset per sector (rejection)."""
    for _ in range(10000):
        pick = rng.choice(n, k, replace=False)
        if len(set(sectors[pick])) == k:
            b = np.zeros(n); b[pick] = 1
            return b
    return None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    loader = FinanceDataLoader()
    e1 = pd.read_csv(E1 / "paper01_e1_cross.csv", float_precision="round_trip")
    rng = np.random.default_rng(0)
    rows, frontier_rows, repair_rows = [], [], []
    for n in SIZES:
        X = all_states(n)
        for inst in INSTANCES:
            tickers = instance_tickers(n, inst)
            sectors = np.array([sector_of(t) for t in tickers])
            data = loader.load(tickers, end_date=W1, lookback_years=3)
            k = min(max(2, n // 5), n)
            prob = PortfolioQUBO().formulate(data.returns, data.covariance, num_select=k, risk_aversion=RISK_AVERSION)
            obj = scoring.Objective.from_problem(prob, data.returns, data.covariance)
            f = obj.objective(X)
            v1 = (X.sum(axis=1) - k) ** 2
            v2 = sector_pairs(X, sectors)
            feasK = v1 == 0
            feasKS = feasK & (v2 == 0)
            f_star_K, f_star_KS = float(f[feasK].min()), float(f[feasKS].min()) if feasKS.any() else np.nan
            fK, fKS = f[feasK], f[feasKS]
            dK = float(fK.max() - fK.min()); dKS = float(fKS.max() - fKS.min()) if feasKS.any() else np.nan
            opt_K = int(np.flatnonzero(feasK)[np.argmin(fK)])
            optK_sector_ok = bool(v2[opt_K] == 0)
            within_K = int(((fK - f_star_K) / dK <= TAU).sum())
            within_KS = int(((fKS - f_star_KS) / dKS <= TAU).sum()) if feasKS.any() and dKS > 0 else 0
            # single-constraint A_crit (paper 1's) and the two-constraint frontier
            a_crit_K = critical_frontier(f, v1, np.zeros_like(v2), feasK, f_star_K, [0.0])[1][0][1]
            a2_alone, front = critical_frontier(f, v1, v2, feasKS, f_star_KS, [0.0, 0.5 * a_crit_K, a_crit_K, 2 * a_crit_K, 5 * a_crit_K, 10 * a_crit_K]) if feasKS.any() else (np.nan, [])
            for a2, a1, ok2 in front:
                frontier_rows.append({"n": n, "instance": inst, "A2": a2, "A1_crit_given_A2": a1, "A2_meets_own_floor": ok2,
                                      "A2_over_AcritK": a2 / a_crit_K if a_crit_K > 0 else np.nan,
                                      "A1_crit_over_AcritK": a1 / a_crit_K if a_crit_K > 0 else np.nan})
            # post-selection cost from the stored E1 states (A_heur, K only)
            runs = e1[(e1.n_assets == n) & (e1.instance == inst) & (e1.penalty_label == "Aheur")]
            post = []
            for _, r in runs.iterrows():
                with np.load(E1 / "e1_runs" / f"{r.run_id}.npz") as z:
                    idx, pf = np.array(z["feasible_indices"]), np.array(z["final_probs_feasible"])
                ok = v2[idx] == 0
                p_both = float(pf[ok].sum())
                good = ok & ((f[idx] - f_star_KS) / dKS <= TAU) if feasKS.any() and dKS > 0 else np.zeros_like(ok)
                p_tau_both = float(pf[good].sum())
                post.append((float(pf.sum()), p_both, p_tau_both, scoring.success_probability(p_tau_both, SHOTS)))
            post = np.array(post) if post else np.full((1, 4), np.nan)
            # repair under both constraints from random starts
            hits, mv = 0, []
            if feasKS.any():
                opt_KS = int(np.flatnonzero(feasKS)[np.argmin(fKS)])
                f_of = lambda b: float(obj.objective(b)[0])
                for _ in range(20):
                    b = random_feasible(rng, sectors, k, n)
                    if b is None:
                        break
                    b2, m = repair_two(b, f_of, sectors, k)
                    hits += int(int("".join(str(int(x)) for x in b2), 2) == opt_KS); mv.append(m)
            repair_rows.append({"n": n, "instance": inst, "starts": len(mv), "reached_optimum": hits,
                                "moves_median": float(np.median(mv)) if mv else np.nan})
            rows.append({
                "n": n, "instance": inst, "k": k, "sectors_present": int(len(np.unique(sectors))),
                "largest_sector": int(np.bincount(sectors).max()),
                "F_K": int(feasK.sum()), "F_KS": int(feasKS.sum()), "F_ratio": float(feasKS.sum() / feasK.sum()),
                "P_F_uniform_K": float(feasK.mean()), "P_F_uniform_KS": float(feasKS.mean()),
                "within_tau_K": within_K, "within_tau_KS": within_KS,
                "f_star_K": f_star_K, "f_star_KS": f_star_KS, "optimum_K_satisfies_sector": optK_sector_ok,
                "delta_F_K": dK, "delta_F_KS": dKS,
                "A_crit_K": a_crit_K, "A2_crit_alone": a2_alone,
                "A1_crit_at_A2_0": front[0][1] if front else np.nan,
                "A1_crit_at_A2_10AcritK": front[-1][1] if front else np.nan,
                "stored_P_F_K_mean": float(np.nanmean(post[:, 0])), "stored_P_both_mean": float(np.nanmean(post[:, 1])),
                "stored_P_tau_both_mean": float(np.nanmean(post[:, 2])), "stored_Q_tau_both_1000_mean": float(np.nanmean(post[:, 3])),
                "repair_reached_optimum": hits, "repair_starts": len(mv),
            })
            print(f"N={n} inst {inst}: |F| {int(feasK.sum())}->{int(feasKS.sum())}, optK sector-ok={optK_sector_ok}, "
                  f"A_crit_K={a_crit_K:.4f} A1_crit(A2=0)={front[0][1] if front else float('nan'):.4f} "
                  f"A1_crit(A2=10A)={front[-1][1] if front else float('nan'):.4f} A2_alone={a2_alone:.4f} "
                  f"P_both(stored)={np.nanmean(post[:,1]):.3f} repair {hits}/{len(mv)}", flush=True)
    write_csv(OUT / "a3_probe_instances.csv", rows)
    write_csv(OUT / "a3_probe_frontier.csv", frontier_rows)
    write_csv(OUT / "a3_probe_repair.csv", repair_rows)
    df = pd.DataFrame(rows)
    print("\nSummary by N (median over instances):")
    print(df.groupby("n")[["F_ratio", "P_F_uniform_K", "P_F_uniform_KS", "within_tau_K", "within_tau_KS",
                            "stored_P_F_K_mean", "stored_P_both_mean", "stored_Q_tau_both_1000_mean"]].median().round(4).to_string())
    print("\noptimum under K alone already satisfies the sector cap:", df.groupby("n").optimum_K_satisfies_sector.mean().round(2).to_dict())
    print("A1_crit(A2=0)/A_crit_K median:", (df.A1_crit_at_A2_0 / df.A_crit_K).groupby(df.n).median().round(3).to_dict(),
          "| A1_crit(A2=10A_crit_K)/A_crit_K median:", (df.A1_crit_at_A2_10AcritK / df.A_crit_K).groupby(df.n).median().round(3).to_dict())
    print("A2_crit_alone / A_crit_K median:", (df.A2_crit_alone / df.A_crit_K).groupby(df.n).median().round(3).to_dict())
    print("repair reached optimum:", int(df.repair_reached_optimum.sum()), "/", int(df.repair_starts.sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
