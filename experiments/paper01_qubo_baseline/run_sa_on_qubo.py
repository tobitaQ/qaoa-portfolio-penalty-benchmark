# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Paper ①  実験: 同じペナルティ符号化 QUBO の上で古典 SA を回す

なぜ必要か
----------
§VI-E は「この批判は solver が不完全であることに条件づく」を示すために焼きなましを
走らせたが、その SA は 2-opt swap で**実行可能領域に閉じ込められている**。つまり
ペナルティ地形を一度も歩いていない。査読者の指摘はここで、

    "SA は制約保存の move set を使っており、同じ penalty-encoded QUBO を解いていない。
     そのため『4経路は solver ではなく QUBO の性質』という主張が、実験では
     直接示されていない。"

は正当である。論文が主張しているのは

    「gap の分母・準縮退・目的関数とSharpeの乖離・インスタンス軸の4つは
      QUBO と符号化の性質であって、solver の性質ではない」

なので、**量子でない stochastic solver を同じ QUBO の上で走らせて同じ症状が出るか**を
測れば、その主張は実験で裏づけられる。出なければ主張を狭める必要がある。

この実験
--------
同一インスタンス・同一 Q（ペナルティ込み）に対し、単一ビット反転の SA を回す。
`num_select` を外した問題を渡すことで `ClassicalSolver` は制約保存 swap ではなく
ペナルティ地形上のランダムウォークに落ちる。測るもの:

  * 返した解が実行可能か（カーディナリティ K を満たすか）
  * 報告される最適性ギャップ  -- 分母は QAOA と同じ A·K² 支配
  * 実行可能集合での順位
  * Sharpe と厳密最適との差   -- 乖離が solver 非依存かどうか

実行例:
    python -m experiments.paper01_qubo_baseline.run_sa_on_qubo
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from itertools import combinations

import numpy as np

from experiments.paper01_qubo_baseline.run_experiment import (
    DATA_END_DATE,
    LOOKBACK_YEARS,
    RESULTS_DIR,
    RISK_AVERSION,
    SEED,
)
from experiments.paper01_qubo_baseline.run_degeneracy import rank_of_energy
from src.finance.data_loader import FinanceDataLoader
from src.finance.metrics import PortfolioMetrics
from src.qubo.portfolio import PortfolioQUBO
from src.solvers.classical_solver import ClassicalSolver

#: Sizes whose feasible set enumerates in seconds, so every run can be ranked.
SIZES = [12, 16, 20]

#: Seeds per size. The QAOA arm of Sec. VI-A uses the same five.
SEEDS = [42, 43, 44, 45, 46]


def rank_in_feasible(q: np.ndarray, x: np.ndarray, n: int, k: int) -> tuple[int, int]:
    """Position of x among feasible solutions ordered by energy, and the set size."""
    idx = list(combinations(range(n), k))
    xs = np.zeros((len(idx), n))
    for row, pick in enumerate(idx):
        xs[row, list(pick)] = 1.0
    energies = np.einsum("ij,jk,ik->i", xs, q, xs)
    order = np.sort(energies)
    e = float(x @ q @ x)
    # einsum and x @ q @ x accumulate differently; rank against the census's
    # own value of the nearest entry (see run_degeneracy.rank_of_energy).
    return rank_of_energy(order, e), len(idx)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, nargs="+", default=SIZES)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--end-date", default=DATA_END_DATE)
    args = ap.parse_args()

    loader, qubo, metrics = FinanceDataLoader(), PortfolioQUBO(), PortfolioMetrics()
    rows = []

    for n in args.n:
        data = loader.load_nikkei_subset(n_assets=n, end_date=args.end_date,
                                         lookback_years=LOOKBACK_YEARS,
                                         random_seed=SEED)
        k = min(max(2, n // 5), data.n_assets)
        problem = qubo.formulate(data.returns, data.covariance, num_select=k,
                                 risk_aversion=RISK_AVERSION)
        exact = ClassicalSolver(method="exact").solve(problem)
        opt = float(exact.bitstring @ problem.Q @ exact.bitstring)
        opt_sharpe = metrics.evaluate(exact.selected_assets, data.returns,
                                      data.covariance, target_k=k).sharpe_ratio

        # Drop num_select so the annealer walks the penalty landscape with
        # single-bit flips instead of being held inside the feasible set.
        free = replace(problem, metadata={**problem.metadata, "num_select": -1})

        print(f"\nN={n} K={k}  exact E={opt:.6f}  Sharpe={opt_sharpe:.4f}")
        for sd in args.seeds:
            r = ClassicalSolver(method="simulated_annealing").solve(free, seed=sd)
            x = r.bitstring
            e = float(x @ problem.Q @ x)
            card = int(x.sum())
            feasible = card == k
            gap = abs(e - opt) / abs(opt) * 100.0
            if feasible:
                rk, size = rank_in_feasible(problem.Q, x, n, k)
                sh = metrics.evaluate(list(np.flatnonzero(x)), data.returns,
                                      data.covariance, target_k=k).sharpe_ratio
                d_sh = (sh - opt_sharpe) / abs(opt_sharpe) * 100.0
            else:
                rk = size = None
                sh = d_sh = float("nan")
            rows.append(dict(n_assets=n, n_select=k, sa_seed=sd,
                             steps=int(r.metadata.get('n_steps', 50_000)),
                             energy_sa=e, energy_optimum=opt, reported_gap_pct=gap,
                             cardinality=card, feasible=feasible,
                             rank_in_feasible=rk, feasible_set_size=size,
                             sharpe_sa=sh, sharpe_optimum=opt_sharpe,
                             sharpe_delta_pct=d_sh,
                             data_start=str(data.start_date),
                             data_end=str(data.end_date)))
            flag = "OK" if feasible else f"INFEASIBLE (|x|={card})"
            print(f"  seed {sd}: gap={gap:8.4f}%  {flag:22s} "
                  f"rank={rk if rk else '-'}  dSharpe={d_sh:+7.2f}%")

    out = RESULTS_DIR / "paper01_sa_on_qubo.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
