# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Paper ①  実験スクリプト: 準縮退解の測定

論文①は「QUBOの準縮退解がリターン/ボラ空間に散らばるので、最適性ギャップは
ポートフォリオ品質の代理指標にならない」と主張してきたが、その集合を一度も
数えていなかった。C(N,K) 厳密列挙（run_experiment の exact ソルバーと同じ）が
入ったので、N=30 まで実行可能解を全件順位づけして実測する。

測るもの
--------
1. 最適解の近傍に解が何個あるか（バンド別）
2. その集合のシャープレシオの分布
3. 報告している最適性ギャップの分母の正体

3 が中心的な発見になった。QUBO のペナルティ項 A(Σx − K)² は、実行可能な x に
対して定数 −A·K² を x^T Q x に寄与する（`PortfolioQUBO` はこれを `offset` に
積む）。つまり報告している energy は

    energy_no_offset = ポートフォリオ目的関数 − A·K²

であり、|energy| ≈ A·K² は **データにも解にも依存しない**。A は O(N)、K は N/5 で
増えるので分母は O(N³) で膨らむ一方、目的関数そのものは O(1) に留まる。
ギャップを |energy| で割ると、同じ解の劣化が N が大きいほど小さく見える。

出力
----
    results/paper01_degeneracy.csv           バンド別の解の個数とシャープ分布
    results/paper01_penalty.csv              ペナルティ重み A と報告ギャップの関係
    results/paper01_degeneracy_quantum.csv   QAOA解が全順位のどこにいるか
    results/paper01_degeneracy_top.csv       上位解（Fig.9 の描画用）

実行例:
    python -m experiments.paper01_qubo_baseline.run_degeneracy
    python -m experiments.paper01_qubo_baseline.run_degeneracy --n 8 12 16 20
"""

from __future__ import annotations

# Reproducibility: pin the OpenMP thread count before anything imports a
# lightning simulator. lightning.qubit reduces over the statevector in parallel,
# and the reduction order follows thread scheduling, so the *same* run repeated
# in the *same* process differs by ~1e-13 -- measured 2026-08-10 on this host at
# the default thread count, and deterministic at OMP_NUM_THREADS=1.
# lightning.gpu is unaffected (verified: three identical runs), which is why the
# published numbers -- all from the GPU -- were never at risk. But the default
# backend of these scripts is the CPU one, so the path a reader takes to
# reproduce from a clone was the non-deterministic path. Set before import
# because OpenMP reads this at library initialisation, and setdefault so an
# operator can still override it deliberately.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")


import argparse
import csv
import itertools
import sys
from math import comb
from pathlib import Path

import numpy as np

from experiments.paper01_qubo_baseline.run_experiment import (
    DATA_END_DATE,
    RESULTS_DIR,
    RISK_AVERSION,
    build_problem,
    write_csv,
)
from src.finance.data_loader import FinanceDataLoader
from src.qubo.portfolio import PortfolioQUBO

#: Sizes whose feasible set is small enough to enumerate exhaustively.
#: N=50/K=10 is 1.0e10 subsets and is deliberately absent.
DEFAULT_N = [8, 12, 16, 20, 24, 28, 30]

#: Risk-free rate. Matches PortfolioMetrics' default in run_experiment.
RISK_FREE_RATE = 0.001

#: Bands measured with the gap definition the paper currently reports,
#: (E - E*) / |E*| * 100 -- the one whose denominator carries the penalty.
#:
#: Six thresholds rather than three, because "why 0.1 %?" is the first thing a
#: reader asks of the headline count, and the answer has to be a curve rather
#: than an assurance. The band count is a step function of the threshold, so a
#: sweep shows whether 0.1 % sits on a plateau or on a cliff.
REPORTED_GAP_BANDS = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

#: Bands measured against the spread of the objective over the feasible set,
#: (E - E*) / (E_max - E*) * 100. Free of the penalty constant.
OBJECTIVE_RANGE_BANDS = [1.0, 5.0, 10.0]

#: How many of the best solutions to record per N for the figure.
TOP_SOLUTIONS = 2000


def enumerate_feasible(problem, n: int, k: int) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """Enumerate every feasible assignment and its QUBO energy.

    The energy is evaluated exactly as ``ClassicalSolver`` does -- ``x @ Q @ x``
    on a full-length float64 vector -- so the optimum found here is bit-for-bit
    the value published in Table I rather than merely close to it.

    Args:
        problem: The QUBO instance.
        n: Number of assets.
        k: Cardinality.

    Returns:
        The K-subsets in enumeration order and their energies.
    """
    Q = problem.Q
    combos = list(itertools.combinations(range(n), k))
    energies = np.empty(len(combos), dtype=float)
    for i, combo in enumerate(combos):
        x = np.zeros(n)
        x[list(combo)] = 1.0
        energies[i] = float(x @ Q @ x)
    return combos, energies


def sharpe_for_all(
    combos: list[tuple[int, ...]],
    returns: np.ndarray,
    covariance: np.ndarray,
    k: int,
    risk_free_rate: float = RISK_FREE_RATE,
) -> np.ndarray:
    """Equal-weight Sharpe ratio for every enumerated subset.

    Vectorised over the whole feasible set: at N=30 this is 593,775 portfolios,
    which one ``PortfolioMetrics.evaluate`` call at a time would take minutes.
    The formula is the same one ``PortfolioMetrics`` uses for equal weights --
    mu = mean(returns[S]), var = sum(Sigma[S,S]) / K^2 -- and the caller checks
    it against ``PortfolioMetrics`` on the optimum.

    Args:
        combos: The K-subsets.
        returns: Annualised expected returns per asset.
        covariance: Annualised covariance matrix.
        k: Cardinality (used as the equal weight 1/k).
        risk_free_rate: Subtracted from the portfolio return.

    Returns:
        Sharpe ratio per subset, in ``combos`` order.
    """
    idx = np.asarray(combos, dtype=np.intp)          # (m, k)
    mu = returns[idx].sum(axis=1) / k
    # Gather the k x k submatrix of Sigma for every subset and sum it. Held to
    # (m, k, k) rather than (m, k, n): at N=30 that is 171 MB instead of 855.
    sub = covariance[idx[:, :, None], idx[:, None, :]]
    var = sub.sum(axis=(1, 2)) / (k * k)
    vol = np.sqrt(np.maximum(var, 0.0))
    # np.where would still evaluate the division everywhere and warn on the
    # zero-volatility subsets; masking the divide itself leaves them at 0.0,
    # matching PortfolioMetrics.
    sharpe = np.zeros_like(vol)
    np.divide(mu - risk_free_rate, vol, out=sharpe, where=vol > 1e-10)
    return sharpe


def best_objective_of_cardinality(q_obj: np.ndarray, n: int, m: int) -> float:
    """Minimum penalty-free objective over all subsets of exactly ``m`` assets.

    Args:
        q_obj: The QUBO matrix with the cardinality penalty removed (A = 0).
        n: Number of assets.
        m: Subset size to enumerate.

    Returns:
        The minimum objective value over the C(n, m) subsets.
    """
    best = np.inf
    it = itertools.combinations(range(n), m)
    while True:
        chunk = list(itertools.islice(it, 400_000))
        if not chunk:
            break
        idx = np.asarray(chunk, dtype=np.intp)
        indicator = np.zeros((idx.shape[0], n))
        np.put_along_axis(indicator, idx, 1.0, axis=1)
        best = min(best, float((((indicator @ q_obj) * indicator).sum(1)).min()))
    return best


def objective_lower_bound(q_obj: np.ndarray, n: int, m: int) -> float:
    """A valid lower bound on the objective over any ``m``-asset subset.

    Every size-m subset contributes exactly m diagonal entries and C(m,2)
    off-diagonal entries, so taking the m smallest diagonals and the C(m,2)
    smallest off-diagonals bounds all of them at once. This is what makes the
    minimum-penalty search finite: the large cardinalities never have to be
    enumerated, only bounded.
    """
    diagonal = np.sort(np.diag(q_obj))
    upper = np.sort(q_obj[np.triu_indices(n, 1)])
    return float(diagonal[:m].sum() + upper[: comb(m, 2)].sum())


def minimum_feasible_penalty(
    q_obj: np.ndarray, n: int, k: int, exact_window: int = 2
) -> tuple[float, int, bool]:
    """Smallest penalty weight A that keeps the QUBO optimum feasible.

    The penalty makes an m-asset solution cost ``A(m-K)^2`` more than its
    objective, so the constrained optimum wins globally iff

        A > (best_obj(K) - best_obj(m)) / (m - K)^2   for every m != K,

    and the tightest such m defines A_min. Cardinalities within
    ``exact_window`` of K are enumerated; the rest are bounded, which is sound
    because a bound can only overstate A_min -- the direction that understates
    how far the conventional penalty heuristic overshoots.

    Args:
        q_obj: QUBO matrix with the penalty removed (A = 0).
        n: Number of assets.
        k: Required cardinality.
        exact_window: How far either side of K to enumerate exactly.

    Returns:
        ``(A_min, binding cardinality, whether that m was enumerated exactly)``.
    """
    top = min(n, k + exact_window)
    exact = {m: best_objective_of_cardinality(q_obj, n, m) for m in range(top + 1)}
    ratios = {
        m: (exact[k] - exact[m]) / (m - k) ** 2 for m in exact if m != k
    }
    for m in range(top + 1, n + 1):
        ratios[m] = (exact[k] - objective_lower_bound(q_obj, n, m)) / (m - k) ** 2
    binding = max(ratios, key=ratios.get)
    return max(ratios[binding], 0.0), binding, binding <= top


def rank_of_energy(sorted_energies: np.ndarray, energy: float) -> int:
    """1 + the number of feasible solutions strictly better than ``energy``.

    The energy is matched to the census entry nearest to it and the count is
    taken against *that* value, not against the float that arrived. A results
    row carries ``x @ Q @ x`` as computed by the solver's process, which can
    differ from the census's evaluation of the same bitstring by an ulp
    (7e-14 at N=24 and N=30, measured 2026-09-11), and a bare searchsorted on
    the arriving value then counts the solution as strictly below itself:
    Table VI reported 44/10,626 and 845/593,775 where the ranks are 43 and 844.
    """
    j = int(np.argmin(np.abs(sorted_energies - energy)))
    return int(np.searchsorted(sorted_energies, sorted_energies[j], side="left")) + 1


def range_normalised_ratio(
    objective_feasible: np.ndarray, c_max: float, offset: float, value: float
) -> float:
    """The approximation ratio prescribed by Abbas et al., (C_max − C)/(C_max − C_min).

    Implemented so the paper measures the field's one prescribed comparable metric
    rather than arguing about it. C_max is taken over all assignments, as the
    prescription specifies, so on a penalty-encoded QUBO it is attained by a
    maximally violating one and scales with the penalty weight — while the spread
    over the feasible set does not. The consequence is not a slow degradation: at
    the penalty weight this formulation already uses, the whole feasible set
    collapses into the fourth decimal place.

    Args:
        objective_feasible: Penalty-free objective over the feasible set.
        c_max: Maximum QUBO energy over all 2^N assignments, at this penalty.
        offset: The penalty constant A*K^2 subtracted from feasible energies.
        value: The energy being scored, on the same scale as ``c_max``.

    Returns:
        The ratio in [0, 1], 1 being best.
    """
    c_min = float(objective_feasible.min()) - offset
    return (c_max - value) / (c_max - c_min)


def band_summary(
    n: int,
    k: int,
    sharpe: np.ndarray,
    within: np.ndarray,
    band_kind: str,
    band_pct: float,
    sharpe_optimum: float,
) -> dict:
    """Summarise the Sharpe distribution over one near-optimal band."""
    sel = sharpe[within]
    total = sharpe.size
    return {
        "n_assets": n,
        "n_select": k,
        "feasible_set_size": total,
        "band_kind": band_kind,
        "band_pct": band_pct,
        "n_within": int(sel.size),
        "frac_within": sel.size / total,
        "sharpe_optimum": sharpe_optimum,
        "sharpe_min": float(sel.min()),
        "sharpe_p25": float(np.percentile(sel, 25)),
        "sharpe_median": float(np.median(sel)),
        "sharpe_p75": float(np.percentile(sel, 75)),
        "sharpe_max": float(sel.max()),
        # Width of the band's Sharpe spread, as a percentage of the optimum's
        # Sharpe: "solutions this close in objective differ by X% in Sharpe".
        "sharpe_spread_pct": float((sel.max() - sel.min()) / abs(sharpe_optimum) * 100)
        if abs(sharpe_optimum) > 1e-12 else float("nan"),
    }


def load_quantum_rows(results_csv: Path) -> dict[int, dict]:
    """Read the published QAOA rows, keyed by N."""
    if not results_csv.exists():
        return {}
    rows = {}
    with open(results_csv, newline="") as f:
        for r in csv.DictReader(f):
            if not r["solver"].startswith("classical"):
                rows[int(r["n_assets"])] = r
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure the near-degenerate solution set of the paper ① QUBOs"
    )
    p.add_argument("--n", type=int, nargs="+", default=DEFAULT_N,
                   help="Asset counts to enumerate (feasible set must fit in memory)")
    p.add_argument("--end-date", default=DATA_END_DATE,
                   help="Last day of the price window; must match the published runs")
    p.add_argument("--results-tag", default=None,
                   help="Suffix the output CSVs, as in run_experiment")
    p.add_argument("--top", type=int, default=TOP_SOLUTIONS,
                   help="Best solutions recorded per N for the figure")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.results_tag}" if args.results_tag else ""

    from src.finance.metrics import PortfolioMetrics

    loader = FinanceDataLoader()
    formulator = PortfolioQUBO()
    metrics = PortfolioMetrics(risk_free_rate=RISK_FREE_RATE)
    quantum = load_quantum_rows(RESULTS_DIR / "paper01_results.csv")

    band_rows: list[dict] = []
    quantum_rows: list[dict] = []
    top_rows: list[dict] = []
    penalty_rows: list[dict] = []

    for n in sorted(args.n):
        problem, data, k = build_problem(n, loader, formulator, args.end_date)
        size = comb(n, k)
        print(f"\n{'='*64}\nN={n}  K={k}  C(N,K)={size:,}\n{'='*64}", flush=True)

        combos, energies = enumerate_feasible(problem, n, k)
        best = int(energies.argmin())
        e_star = float(energies[best])
        e_range = float(energies.max() - e_star)

        sharpe = sharpe_for_all(combos, data.returns, data.covariance, k)

        # The vectorised Sharpe must agree with the class the paper reports
        # through; check it on the optimum rather than trusting the rewrite.
        reference = metrics.evaluate(
            list(combos[best]), data.returns, data.covariance, target_k=k
        )
        if not np.isclose(sharpe[best], reference.sharpe_ratio, rtol=1e-12, atol=0.0):
            print(f"  ABORT: vectorised Sharpe {sharpe[best]!r} disagrees with "
                  f"PortfolioMetrics {reference.sharpe_ratio!r}")
            return 1
        sharpe_optimum = float(sharpe[best])

        penalty_a = float(problem.metadata["penalty_strength"])
        print(f"  optimum E={e_star:.10f}  objective={e_star + problem.offset:.6f}  "
              f"Sharpe={sharpe_optimum:.6f}")
        print(f"  penalty A={penalty_a:.2f}  offset=A*K^2={problem.offset:.1f}  "
              f"objective range over feasible set={e_range:.6f}")
        # The number that explains contribution #9: how much a gap quoted
        # against |E*| is deflated relative to the objective's own spread.
        print(f"  |E*|/range = {abs(e_star) / e_range:.1f}x  "
              f"(a 1% reported gap spans {abs(e_star) * 0.01 / e_range:.1f}x the "
              f"whole feasible objective range)")

        reported_gap = (energies - e_star) / abs(e_star) * 100.0
        range_gap = (energies - e_star) / e_range * 100.0

        for band in REPORTED_GAP_BANDS:
            band_rows.append(band_summary(
                n, k, sharpe, reported_gap <= band,
                "reported_gap", band, sharpe_optimum))
        for band in OBJECTIVE_RANGE_BANDS:
            band_rows.append(band_summary(
                n, k, sharpe, range_gap <= band,
                "objective_range", band, sharpe_optimum))

        for r in band_rows[-len(REPORTED_GAP_BANDS) - len(OBJECTIVE_RANGE_BANDS):]:
            print(f"    {r['band_kind']:>16} <= {r['band_pct']:>5.2f}%  "
                  f"n={r['n_within']:>8,} ({r['frac_within']*100:6.2f}%)  "
                  f"Sharpe [{r['sharpe_min']:.3f}, {r['sharpe_max']:.3f}]  "
                  f"spread {r['sharpe_spread_pct']:.1f}%")

        # Where the published QAOA solution actually sits in the ranking.
        order = np.argsort(energies, kind="stable")
        if n in quantum:
            e_q = float(quantum[n]["energy_no_offset"])
            rank = rank_of_energy(energies[order], e_q)
            quantum_rows.append({
                "n_assets": n,
                "n_select": k,
                "feasible_set_size": size,
                "solver": quantum[n]["solver"],
                "energy_optimum": e_star,
                "energy_quantum": e_q,
                "reported_gap_pct": abs(e_q - e_star) / abs(e_star) * 100.0,
                "objective_range": e_range,
                "gap_of_objective_range_pct": abs(e_q - e_star) / e_range * 100.0,
                # How many times larger the honest number is than the quoted one.
                "deflation_factor": abs(e_star) / e_range,
                "rank": rank,
                "percentile": rank / size * 100.0,
                "sharpe_optimum": sharpe_optimum,
                "sharpe_quantum": float(quantum[n]["sharpe_ratio"]),
                "sharpe_delta_pct": (float(quantum[n]["sharpe_ratio"]) - sharpe_optimum)
                / abs(sharpe_optimum) * 100.0,
            })
            qr = quantum_rows[-1]
            print(f"  QAOA: reported gap {qr['reported_gap_pct']:.4f}% but "
                  f"{qr['gap_of_objective_range_pct']:.2f}% of the objective range "
                  f"({qr['deflation_factor']:.0f}x), rank {rank:,}/{size:,}, "
                  f"Sharpe {qr['sharpe_delta_pct']:+.1f}%")

        # How much of |E*| is a free parameter? The penalty only has to be
        # large enough to keep the optimum feasible; anything above that is
        # headroom the reporting metric silently divides by.
        q_obj = formulator.formulate(
            returns=data.returns,
            covariance=data.covariance,
            num_select=k,
            risk_aversion=RISK_AVERSION,
            penalty_strength=0.0,
        ).Q
        a_min, binding, binding_exact = minimum_feasible_penalty(q_obj, n, k)
        objective_star = e_star + problem.offset
        row = {
            "n_assets": n,
            "n_select": k,
            "penalty_used": penalty_a,
            "penalty_min": a_min,
            "penalty_ratio": penalty_a / a_min if a_min > 0 else float("inf"),
            "binding_cardinality": binding,
            "binding_is_exact": binding_exact,
            "objective_optimum": objective_star,
            "objective_range": e_range,
            # |E*| under each penalty; the numerator (E_q - E*) is the same
            # either way because both solutions are feasible.
            "abs_energy_optimum_used": abs(objective_star - penalty_a * k * k),
            "abs_energy_optimum_min": abs(objective_star - a_min * k * k),
        }
        if n in quantum:
            delta = abs(float(quantum[n]["energy_no_offset"]) - e_star)
            row["gap_at_penalty_used_pct"] = delta / row["abs_energy_optimum_used"] * 100.0
            row["gap_at_penalty_min_pct"] = delta / row["abs_energy_optimum_min"] * 100.0
            row["gap_of_objective_range_pct"] = delta / e_range * 100.0
        else:
            row["gap_at_penalty_used_pct"] = ""
            row["gap_at_penalty_min_pct"] = ""
            row["gap_of_objective_range_pct"] = ""
        penalty_rows.append(row)
        print(f"  penalty A={penalty_a:.2f} but A_min={a_min:.4f} "
              f"({penalty_a / a_min:.0f}x headroom, binding at m={binding}"
              f"{'' if binding_exact else ', bounded'})")
        if row["gap_at_penalty_used_pct"] != "":
            print(f"    same QAOA solution reads as "
                  f"{row['gap_at_penalty_used_pct']:.4f}% at A={penalty_a:.0f}, "
                  f"{row['gap_at_penalty_min_pct']:.2f}% at A_min, "
                  f"{row['gap_of_objective_range_pct']:.2f}% vs the objective range")

        for position, i in enumerate(order[: args.top], start=1):
            top_rows.append({
                "n_assets": n,
                "rank": position,
                "assets": " ".join(str(a) for a in combos[i]),
                "energy_no_offset": float(energies[i]),
                "reported_gap_pct": float(reported_gap[i]),
                "gap_of_objective_range_pct": float(range_gap[i]),
                "sharpe_ratio": float(sharpe[i]),
            })

    write_csv(RESULTS_DIR / f"paper01_penalty{suffix}.csv", penalty_rows)
    write_csv(RESULTS_DIR / f"paper01_degeneracy{suffix}.csv", band_rows)
    write_csv(RESULTS_DIR / f"paper01_degeneracy_quantum{suffix}.csv", quantum_rows)
    write_csv(RESULTS_DIR / f"paper01_degeneracy_top{suffix}.csv", top_rows)
    print(f"\nSaved {len(band_rows)} band rows, {len(quantum_rows)} quantum rows, "
          f"{len(top_rows)} top-solution rows → {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
