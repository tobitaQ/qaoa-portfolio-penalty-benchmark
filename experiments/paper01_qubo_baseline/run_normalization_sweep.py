# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Paper ①  実験: 準縮退は正規化の産物か

なぜ必要か
----------
§III は μ と Σ をそれぞれ**そのインスタンス自身の**max-abs で割って [-1,1] に
収めている。敵対的査読者はここを突ける:

    "near-degeneracy の一部は、この per-instance scaling choice が作っている
     のではないか。目的関数のレンジを揃えてしまえば、0.1% 帯に入る解が多く
     見えるのは当たり前ではないか。"

これは正当な問いで、文章では答えられない。スケールの取り方を変えて同じ列挙を
回し、希釈率・0.1% 帯・Sharpe の散らばりが残るかどうかを見る。

比較する2つ
-----------
per-instance : 現行。各 N の部分集合ごとに max|μ|, max|Σ| で割る
global       : 50銘柄ユニバース全体から決めた固定スケールを全サイズに使う

global では部分集合の max は全体の max 以下なので μ̂, Σ̂ の絶対値が 1 より
小さくなり、A = 2·s·N + 1 も小さくなる。つまり目的関数とペナルティの両方が
動く — near-degeneracy がスケールの産物なら、ここで崩れるはず。

古典列挙のみ。QAOA は回さない。

実行例:
    python -m experiments.paper01_qubo_baseline.run_normalization_sweep
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations

import numpy as np

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

#: Sizes whose feasible sets enumerate in seconds. N = 50 is 1.0e10 subsets.
SIZES = [16, 20, 24, 28, 30]

#: The band the paper headlines, as a percentage of the reported gap.
BAND_PCT = 0.1


def census(q: np.ndarray, n: int, k: int, returns, cov, metrics):
    """Enumerate the feasible set once and summarise what the gap denominates."""
    idx = list(combinations(range(n), k))
    x = np.zeros((len(idx), n))
    for row, pick in enumerate(idx):
        x[row, list(pick)] = 1.0
    energy = np.einsum("ij,jk,ik->i", x, q, x)
    order = np.argsort(energy, kind="stable")
    energy = energy[order]
    e_star = energy[0]
    gap = np.abs(energy - e_star) / abs(e_star) * 100.0
    within = gap <= BAND_PCT
    sharpe = np.array([
        metrics.evaluate(list(idx[order[i]]), returns, cov, target_k=k).sharpe_ratio
        for i in np.flatnonzero(within)
    ])
    spread = float(energy[-1] - e_star)
    return {
        "feasible_set_size": len(idx),
        "objective_spread": spread,
        "abs_energy_optimum": float(abs(e_star)),
        "deflation_factor": float(abs(e_star) / spread) if spread else float("nan"),
        "n_within_band": int(within.sum()),
        "frac_within_band": float(within.mean()),
        "sharpe_min": float(sharpe.min()),
        "sharpe_max": float(sharpe.max()),
        "sharpe_spread_pct": float((sharpe.max() - sharpe.min()) / abs(sharpe.min()) * 100.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, nargs="+", default=SIZES)
    ap.add_argument("--end-date", default=DATA_END_DATE)
    args = ap.parse_args()

    loader, qubo, metrics = FinanceDataLoader(), PortfolioQUBO(), PortfolioMetrics()

    # The fixed scale: the widest universe the price cache carries.
    universe = loader.load_nikkei_subset(n_assets=50, end_date=args.end_date,
                                         lookback_years=LOOKBACK_YEARS,
                                         random_seed=SEED)
    g_mu = float(np.max(np.abs(universe.returns)) + 1e-12)
    g_cov = float(np.max(np.abs(universe.covariance)) + 1e-12)
    print(f"global scales from the 50-ticker universe: "
          f"max|mu|={g_mu:.6f}  max|Sigma|={g_cov:.6f}\n")

    rows = []
    for n in args.n:
        data = loader.load_nikkei_subset(n_assets=n, end_date=args.end_date,
                                         lookback_years=LOOKBACK_YEARS,
                                         random_seed=SEED)
        k = min(max(2, n // 5), data.n_assets)
        for label, scales in [("per-instance", None), ("global", (g_mu, g_cov))]:
            problem = qubo.formulate(data.returns, data.covariance, num_select=k,
                                     risk_aversion=RISK_AVERSION, scales=scales)
            stats = census(problem.Q, n, k, data.returns, data.covariance, metrics)
            rows.append(dict(n_assets=n, n_select=k, normalization=label,
                             penalty=float(problem.metadata["penalty_strength"]),
                             **stats,
                             data_start=str(data.start_date),
                             data_end=str(data.end_date)))
            print(f"  N={n:2d} {label:>12s}  A={rows[-1]['penalty']:7.3f}  "
                  f"deflation={stats['deflation_factor']:6.0f}x  "
                  f"band={stats['n_within_band']:7,d} "
                  f"({stats['frac_within_band']*100:5.1f}%)  "
                  f"Sharpe [{stats['sharpe_min']:.3f}, {stats['sharpe_max']:.3f}]")

    out = RESULTS_DIR / "paper01_normalization_sweep.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
