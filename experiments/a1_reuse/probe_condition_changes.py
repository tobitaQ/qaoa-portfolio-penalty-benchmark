# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
A1 probe: when the conditions change, is there a decision to make?

Go/no-go for the paper "reuse, repair or recompute a stored optimisation
result when the user's conditions change". Everything is computed from the
committed E1/E5 records and the pinned statistics; no quantum run, no cloud.

For every stored E1 run (its best feasible portfolio under the 2023-26 window,
penalty A_heur or A_margin, N in {12, 16, 20}, instances 0-9 where E5 exists)
and for each kind of condition change, the probe measures the exact regret --
against the enumerated optimum under the new conditions -- of three choices:

  reuse      hand back the stored portfolio unchanged
  repair     one greedy pass of classical local moves from the stored portfolio
             (single-asset swaps; add/remove for a cardinality change)
  recompute  run the quantum sampler again: its distribution is in the record
             (for a window change, the E5 run with the same seed is the actual
             re-run; in general the stored feasible probabilities give the
             chance that a fresh 1,000-shot batch contains a solution at least
             as good as the stored one)

Condition changes:
  window     2023-08-06..2026-08-05 -> 2020-08-05..2023-08-05 (same tickers)
  lambda     risk aversion 0.5 -> 0.25 and 0.75 (same window)
  cardinality K -> K+1 and K-1 (same window)
  universe   the portfolio carried to another instance of the same N that
             contains all of its assets (instance overlaps)

Go criterion, fixed before running: for at least two kinds of change the
best choice varies across instances (reuse is best in 20-80 % of cases), and
the split is predictable from recorded features (Delta_F, near-degenerate
count, Q_tau of the stored state). No-go: one choice always wins.

Run:
    python -m experiments.a1_reuse.probe_condition_changes        # ~10 min CPU
"""

from __future__ import annotations

import itertools
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
W1 = "2026-08-05"   # E1 window end
W2 = "2023-08-05"   # E5 window end
SHOTS = 1000
TAU = 0.01


def bits_to_index(bits: np.ndarray) -> int:
    return int("".join(str(int(b)) for b in bits), 2)


def index_to_bits(index: int, n: int) -> np.ndarray:
    return scoring.bitstrings_from_indices([index], n)[0]


class Instance:
    """One (N, instance, window, lambda, K): objective and enumerated feasible set."""

    def __init__(self, n, inst, end_date, lam=RISK_AVERSION, k=None, loader=None):
        self.n, self.inst, self.end, self.lam = n, inst, end_date, lam
        self.tickers = instance_tickers(n, inst)
        self.data = (loader or FinanceDataLoader()).load(self.tickers, end_date=end_date, lookback_years=3)
        self.k = k or min(max(2, n // 5), n)
        prob = PortfolioQUBO().formulate(self.data.returns, self.data.covariance,
                                         num_select=self.k, risk_aversion=lam)
        self.obj = scoring.Objective.from_problem(prob, self.data.returns, self.data.covariance)
        self.feas = scoring.enumerate_feasible(self.obj)
        self.optimum = int(self.feas.indices[int(np.argmin(self.feas.f))])

    def f(self, index: int) -> float:
        return float(self.obj.objective(index_to_bits(index, self.n))[0])

    def regret(self, index: int) -> dict:
        """Exact regret of a feasible portfolio: g_F and rank in F."""
        fv = self.f(index)
        pos = np.flatnonzero(self.feas.indices == index)
        f_here = float(self.feas.f[pos[0]]) if pos.size else fv
        rank, _ = self.feas.rank(f_here)
        return {"f": f_here, "g_F": float(self.feas.gap_range(f_here)), "rank": int(rank),
                "within_tau": bool(self.feas.gap_range(f_here) <= TAU), "optimal": int(index == self.optimum)}

    def repair(self, index: int, target_k: int | None = None) -> tuple[int, int]:
        """Greedy local search: swaps at fixed K; add/remove first if K changes.
        Returns (index, number of moves)."""
        bits = index_to_bits(index, self.n).astype(int)
        moves = 0
        tk = target_k or self.k
        while bits.sum() < tk:            # add the best asset
            best = min((i for i in range(self.n) if not bits[i]),
                       key=lambda i: self.obj.objective(np.where(np.arange(self.n) == i, 1, bits))[0])
            bits[best] = 1; moves += 1
        while bits.sum() > tk:            # drop the worst asset
            best = min((i for i in range(self.n) if bits[i]),
                       key=lambda i: self.obj.objective(np.where(np.arange(self.n) == i, 0, bits))[0])
            bits[best] = 0; moves += 1
        improved = True
        cur = float(self.obj.objective(bits)[0])
        while improved:
            improved = False
            for i in range(self.n):
                if not bits[i]:
                    continue
                for j in range(self.n):
                    if bits[j]:
                        continue
                    cand = bits.copy(); cand[i], cand[j] = 0, 1
                    fv = float(self.obj.objective(cand)[0])
                    if fv < cur - 1e-15:
                        bits, cur, improved, moves = cand, fv, True, moves + 1
                        break
                if improved:
                    break
        return bits_to_index(bits), moves


def recompute_gamble(z: dict, feas_new: scoring.FeasibleSet, f_target: float) -> float:
    """P(a fresh 1,000-shot batch from the stored final state contains a
    feasible solution at least as good as f_target under the *new* objective)."""
    # Stored probabilities are over the old feasible set's indices; map to the new f.
    idx = z["feasible_indices"]
    probs = z["final_probs_feasible"]
    pos = {int(i): j for j, i in enumerate(feas_new.indices)}
    f_new = np.array([feas_new.f[pos[int(i)]] if int(i) in pos else np.inf for i in idx])
    p_good = float(probs[f_new <= f_target + 1e-15].sum())
    return scoring.success_probability(p_good, SHOTS)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    loader = FinanceDataLoader()
    e1 = pd.read_csv(E1 / "paper01_e1_cross.csv", float_precision="round_trip")
    e5 = pd.read_csv(E1 / "paper01_e1_cross_e5.csv", float_precision="round_trip")
    e1 = e1[(e1.instance < 10) & e1.penalty_label.isin(["Aheur", "Amargin"]) & e1.best_feasible_index.notna()]
    rows = []
    cache: dict[tuple, Instance] = {}

    def inst(n, i, end, lam=RISK_AVERSION, k=None):
        key = (n, i, end, lam, k)
        if key not in cache:
            cache[key] = Instance(n, i, end, lam, k, loader)
        return cache[key]

    for _, r in e1.iterrows():
        n, i, seed, lab = int(r.n_assets), int(r.instance), int(r.qaoa_seed), r.penalty_label
        stored = int(r.best_feasible_index)
        with np.load(E1 / "e1_runs" / f"{r.run_id}.npz") as zz:
            z = {k: np.array(zz[k]) for k in ("feasible_indices", "final_probs_feasible")}
        base = {"run_id": r.run_id, "n": n, "instance": i, "seed": seed, "penalty": lab,
                "stored_g_F_old": float(r.g_F), "stored_rank_old": int(r.rank_best_feasible),
                "delta_F_old": float(r.delta_F), "Q_tau_old": float(r.final_Q_tau_main_1000),
                "within_tau_count_old": None}
        old = inst(n, i, W1)
        base["within_tau_count_old"] = int(old.feas.within(TAU).sum())

        def record(change, new: Instance, target_k=None, rerun_f=None):
            reuse = new.regret(stored) if (target_k is None) else None
            rep_idx, moves = new.repair(stored, target_k)
            rep = new.regret(rep_idx)
            gamble = recompute_gamble(z, new.feas, rep["f"]) if target_k is None else float("nan")
            row = {**base, "change": change,
                   "reuse_g_F": reuse["g_F"] if reuse else np.nan, "reuse_rank": reuse["rank"] if reuse else np.nan,
                   "reuse_optimal": reuse["optimal"] if reuse else np.nan,
                   "repair_g_F": rep["g_F"], "repair_rank": rep["rank"], "repair_optimal": rep["optimal"], "repair_moves": moves,
                   "recompute_p_at_least_repair": gamble,
                   "new_feasible": int(new.feas.size), "new_within_tau": int(new.feas.within(TAU).sum()),
                   "new_delta_F": float(new.feas.delta)}
            if rerun_f is not None:
                row["rerun_g_F"] = float(new.feas.gap_range(rerun_f))
            rows.append(row)

        # window change: the E5 run with the same seed and penalty is the actual re-run
        new = inst(n, i, W2)
        m = e5[(e5.n_assets == n) & (e5.instance == i) & (e5.qaoa_seed == seed) & (e5.penalty_label == lab)]
        rerun_f = float(m.f_best_feasible.iloc[0]) if len(m) and pd.notna(m.f_best_feasible.iloc[0]) else None
        record("window", new, rerun_f=rerun_f)
        # lambda
        for lam in (0.25, 0.75):
            record(f"lambda_{lam}", inst(n, i, W1, lam))
        # cardinality
        for dk in (+1, -1):
            k2 = old.k + dk
            if 1 <= k2 <= n:
                record(f"K{dk:+d}", inst(n, i, W1, RISK_AVERSION, k2), target_k=k2)
        # universe: other instances of the same N containing all the portfolio's assets
        held = {old.tickers[b] for b in range(n) if index_to_bits(stored, n)[b] > 0.5}
        for j in range(10):
            if j == i:
                continue
            tj = instance_tickers(n, j)
            if held <= set(tj):
                other = inst(n, j, W1)
                idx_j = bits_to_index(np.array([1 if t in held else 0 for t in tj]))
                reg = other.regret(idx_j)
                rep_idx, moves = other.repair(idx_j)
                rep = other.regret(rep_idx)
                rows.append({**base, "change": f"universe_i{j}", "reuse_g_F": reg["g_F"], "reuse_rank": reg["rank"],
                             "reuse_optimal": reg["optimal"], "repair_g_F": rep["g_F"], "repair_rank": rep["rank"],
                             "repair_optimal": rep["optimal"], "repair_moves": moves,
                             "recompute_p_at_least_repair": np.nan, "new_feasible": int(other.feas.size),
                             "new_within_tau": int(other.feas.within(TAU).sum()), "new_delta_F": float(other.feas.delta)})
        if len(rows) % 200 < 6:
            print(f"  {len(rows)} rows ...", flush=True)

    df = pd.DataFrame(rows)
    write_csv(OUT / "a1_probe_rows.csv", df.to_dict("records"))
    df["kind"] = df.change.str.replace(r"_i\d+$", "", regex=True)
    summ = []
    for kind, g in df.groupby("kind"):
        s = {"kind": kind, "cases": len(g),
             "reuse_optimal_%": round(100 * g.reuse_optimal.mean(), 1) if g.reuse_optimal.notna().any() else "",
             "reuse_within_tau_%": round(100 * (g.reuse_g_F <= TAU).mean(), 1) if g.reuse_g_F.notna().any() else "",
             "reuse_g_F_median": round(g.reuse_g_F.median(), 4) if g.reuse_g_F.notna().any() else "",
             "repair_optimal_%": round(100 * g.repair_optimal.mean(), 1),
             "repair_within_tau_%": round(100 * (g.repair_g_F <= TAU).mean(), 1),
             "repair_moves_median": float(g.repair_moves.median()),
             "recompute_p>=repair_median": round(g.recompute_p_at_least_repair.median(), 3) if g.recompute_p_at_least_repair.notna().any() else "",
             "recompute_p>=repair_>0.5_%": round(100 * (g.recompute_p_at_least_repair > 0.5).mean(), 1) if g.recompute_p_at_least_repair.notna().any() else ""}
        if "rerun_g_F" in g and g.rerun_g_F.notna().any():
            s["rerun_within_tau_%"] = round(100 * (g.rerun_g_F <= TAU).mean(), 1)
            s["rerun_better_than_reuse_%"] = round(100 * (g.rerun_g_F < g.reuse_g_F - 1e-12).mean(), 1)
        summ.append(s)
    write_csv(OUT / "a1_probe_summary.csv", summ)
    print(pd.DataFrame(summ).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
