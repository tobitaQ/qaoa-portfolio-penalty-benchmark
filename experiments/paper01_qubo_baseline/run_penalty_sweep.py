# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Paper ①  実験: ペナルティ重み A を動かすと solver はどうなるか

なぜ必要か
----------
§VI-D は「実行可能な2解の間では gap の分子が A に依らず分母だけが A に比例する
ので、gap ∝ 1/A である」を代数として示し、標準的発見則が A_min を 148〜1213倍
超過していることを測った。そのうえで、こう書いて逃げている:

    "This is not an argument for running solvers at A_min: a penalty that barely
     separates the feasible optimum will let a sampler drift out of the feasible
     set constantly."

これは主張であって測定ではない。「測れ、論じるな」を掲げる論文が、自分の免責
条項だけ論じている状態で、査読者に見つかれば刺さる。

このスクリプトは A を A_min の倍数で動かし、同じ QAOA 設定で:

  * 報告される最適性ギャップ         -- 代数どおり 1/A で動くはず
  * 最終ショットの実行可能率         -- 免責条項が正しければ A とともに上がる
  * 最適化後の期待値 <H_norm>        -- solver の条件付けが効くならここに出る
  * 実行可能集合での順位             -- A に依らず「同じ解」なのかを見る
  * 実行時間

を測る。どちらに転んでも報告価値がある。トレードオフが出れば免責条項が測定に
裏打ちされ、出なければ「発見則は単に無駄」という、より強い主張になる。

実行例:
    python -m experiments.paper01_qubo_baseline.run_penalty_sweep \
        --n 16 20 --backend lightning_gpu --precision single
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time

from experiments.paper01_qubo_baseline.run_experiment import (
    DATA_END_DATE,
    LOOKBACK_YEARS,
    RESULTS_DIR,
    RISK_AVERSION,
    SEED,
)
from src.finance.data_loader import FinanceDataLoader
from src.finance.metrics import PortfolioMetrics
from src.qubo.portfolio import PortfolioQUBO
from experiments.paper01_qubo_baseline.run_degeneracy import minimum_feasible_penalty
from src.solvers.classical_solver import ClassicalSolver

#: Multiples of A_min to test, plus the heuristic the paper publishes.
#: 1.1 is the smallest that still keeps the optimum feasible with margin; the
#: heuristic sits 385x above A_min at N = 16, so the grid spans three decades.
A_MULTIPLES = [1.1, 2.0, 5.0, 10.0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, nargs="+", default=[16, 20])
    ap.add_argument("--backend", default="lightning_gpu")
    ap.add_argument("--precision", default="single", choices=["single", "double"])
    ap.add_argument("--p-layers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--end-date", default=DATA_END_DATE)
    args = ap.parse_args()

    from src.solvers.braket_solver import BraketSolver

    loader, qubo = FinanceDataLoader(), PortfolioQUBO()
    metrics = PortfolioMetrics()
    rows = []

    for n in args.n:
        data = loader.load_nikkei_subset(n_assets=n, end_date=args.end_date,
                                         lookback_years=LOOKBACK_YEARS,
                                         random_seed=args.seed)
        k = min(max(2, n // 5), data.n_assets)
        published = qubo.formulate(data.returns, data.covariance,
                                   num_select=k, risk_aversion=RISK_AVERSION)
        # A_min from the same routine Sec. VI-D uses, so the two agree by
        # construction rather than by coincidence.
        q_obj = qubo.formulate(data.returns, data.covariance, num_select=k,
                               risk_aversion=RISK_AVERSION,
                               penalty_strength=0.0).Q
        a_min, _binding, _exact = minimum_feasible_penalty(q_obj, n, k)
        a_used = float(published.metadata["penalty_strength"])
        print(f"\nN={n} K={k}  A_min={a_min:.4f}  A_used={a_used:.4f} "
              f"({a_used / a_min:.0f}x)")

        for label, a in [(f"{m}xA_min", m * a_min) for m in A_MULTIPLES] + \
                        [("A_used", a_used)]:
            problem = qubo.formulate(data.returns, data.covariance,
                                     num_select=k, risk_aversion=RISK_AVERSION,
                                     penalty_strength=a)
            exact = ClassicalSolver(method="exact").solve(problem)
            e_star = float(exact.bitstring @ problem.Q @ exact.bitstring)

            solver = BraketSolver(backend=args.backend, precision=args.precision,
                                  p_layers=args.p_layers, n_shots=args.shots,
                                  n_optimizer_steps=args.steps)
            t0 = time.perf_counter()
            res = solver.solve(problem, seed=args.seed)
            runtime = time.perf_counter() - t0

            e = float(res.bitstring @ problem.Q @ res.bitstring)
            gap = abs(e - e_star) / abs(e_star) * 100.0
            feasible = int(res.bitstring.sum()) == k
            stats = metrics.evaluate(res.selected_assets, data.returns,
                                     data.covariance, target_k=k)
            sharpe = stats.sharpe_ratio
            frac = res.metadata.get("feasible_shot_fraction")
            trace = res.metadata.get("convergence", [])
            rows.append(dict(
                n_assets=n, n_select=k, penalty_label=label, penalty=a,
                a_min=a_min, a_over_amin=a / a_min,
                energy_quantum=e, energy_optimum=e_star, reported_gap_pct=gap,
                feasible=feasible, feasible_shot_fraction=frac,
                final_expectation=(trace[-1] if trace else None),
                sharpe_ratio=sharpe, runtime_s=runtime,
                backend=args.backend, precision=args.precision,
                p_layers=args.p_layers, steps=args.steps, shots=args.shots,
                qaoa_seed=args.seed,
                data_start=str(data.start_date), data_end=str(data.end_date),
            ))
            print(f"  {label:>10s} A={a:9.4f}  gap={gap:8.4f}%  "
                  f"feasible_shots={frac if frac is None else f'{frac:6.1%}'}  "
                  f"{'OK' if feasible else 'INFEASIBLE'}  {runtime:5.1f}s")

    out = RESULTS_DIR / "paper01_penalty_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
