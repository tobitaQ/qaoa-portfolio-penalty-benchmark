# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Paper ①  実験スクリプト: インスタンス軸

論文①の全ての結果は、各サイズにつき**インスタンス1つ**で得られている。
`FinanceDataLoader.load_nikkei_subset` の `random_seed` は実装上使われておらず、
常に `NIKKEI225_TICKERS[:N]` を返すためである。したがって

    「N=20 でバックエンドが食い違った」
    「gap 0.1% のバンドに実行可能集合の1/4が入る」
    「発見則のペナルティは A_min を 148〜1213倍 超過する」

はいずれも**1インスタンスの観測**であって、分布ではない。最も近い先行研究
（Stopfer & Wagner, arXiv:2509.17876）は250インスタンスを回しているので、
査読者は必ずここを突く。

このスクリプトは50銘柄のキャッシュ済みユニバースから部分集合を引いてインスタンスを
作る。`_read_superset_cache` が効くので**ネットワークには出ない**＝clone しただけで
再現できるという論文の主張を壊さない。C(50,20) は 4.7e13 通りあるので、
インスタンス数はサンプリング設計だけの問題になる。

instance 0 は必ず `NIKKEI225_TICKERS[:N]`（＝Table I の公表インスタンス）にしてある。
分布の中で公表値がどこに位置するかが読めるようにするため。

出力
----
    results/paper01_instances.csv   インスタンスごとの1行

実行例:
    # 古典のみ（無料・数分）
    python -m experiments.paper01_qubo_baseline.run_instances --skip-quantum

    # QAOA も回す（GPU）
    python -m experiments.paper01_qubo_baseline.run_instances \
        --backend lightning_gpu --precision single --n 12 16 20 --instances 30
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
import hashlib
import sys
from math import comb

import numpy as np

from experiments.paper01_qubo_baseline.run_degeneracy import (
    RISK_FREE_RATE,
    enumerate_feasible,
    minimum_feasible_penalty,
    rank_of_energy,
    sharpe_for_all,
)
from experiments.paper01_qubo_baseline.run_experiment import (
    DATA_END_DATE,
    LOOKBACK_YEARS,
    RESULTS_DIR,
    RISK_AVERSION,
    SEED,
    write_csv,
)
from src.finance.data_loader import NIKKEI225_TICKERS, FinanceDataLoader
from src.finance.metrics import PortfolioMetrics
from src.qubo.portfolio import PortfolioQUBO
from src.solvers.braket_solver import BraketSolver
from src.solvers.classical_solver import ClassicalSolver

#: Sizes to sample. Bounded above by the feasible set we can enumerate.
DEFAULT_N = [12, 16, 20, 24, 28, 30]

#: Instances per size, including the published one at index 0.
DEFAULT_INSTANCES = 30

#: The band whose width Table V reports.
REPORTED_GAP_BAND = 0.1


def instance_tickers(n_assets: int, instance: int) -> list[str]:
    """Pick the asset subset for one instance.

    Instance 0 is the published subset, so the sweep contains the number the
    paper already reports rather than merely bracketing it. Later instances are
    drawn without replacement from the same 50-ticker cached universe, which
    keeps every instance offline-reproducible.

    Args:
        n_assets: Subset size N.
        instance: Instance index; 0 is the published subset.

    Returns:
        The tickers, sorted (the loader's column convention).
    """
    universe = NIKKEI225_TICKERS[:50]
    if n_assets > len(universe):
        raise ValueError(f"N={n_assets} exceeds the cached universe ({len(universe)})")
    if instance == 0:
        return sorted(universe[:n_assets])
    rng = np.random.default_rng(instance)
    picked = rng.choice(len(universe), size=n_assets, replace=False)
    return sorted(universe[i] for i in picked)


def instance_digest(tickers: list[str]) -> str:
    """Short content hash naming the instance, so a row can be traced back."""
    return hashlib.sha256("\x00".join(sorted(tickers)).encode()).hexdigest()[:12]


def measure(problem, data, k: int, metrics: PortfolioMetrics) -> dict:
    """Enumerate one instance's feasible set and summarise what a gap denominates.

    Args:
        problem: The QUBO instance.
        data: Its price data.
        k: Cardinality.
        metrics: Used to cross-check the vectorised Sharpe on the optimum.

    Returns:
        The instance's measured quantities, ready to become a CSV row.
    """
    n = problem.n_variables
    combos, energies = enumerate_feasible(problem, n, k)
    best = int(energies.argmin())
    e_star = float(energies[best])
    e_range = float(energies.max() - e_star)

    sharpe = sharpe_for_all(combos, data.returns, data.covariance, k)
    reference = metrics.evaluate(
        list(combos[best]), data.returns, data.covariance, target_k=k
    ).sharpe_ratio
    if not np.isclose(sharpe[best], reference, rtol=1e-12, atol=0.0):
        raise AssertionError(
            f"vectorised Sharpe {sharpe[best]!r} != PortfolioMetrics {reference!r}"
        )

    within = ((energies - e_star) / abs(e_star) * 100.0) <= REPORTED_GAP_BAND
    band = sharpe[within]
    return {
        "combos": combos,
        "energies": energies,
        "sharpe": sharpe,
        "row": {
            "energy_optimum": e_star,
            "objective_optimum": e_star + problem.offset,
            "objective_range": e_range,
            "abs_energy_optimum": abs(e_star),
            "deflation_factor": abs(e_star) / e_range,
            "sharpe_optimum": float(sharpe[best]),
            "band_pct": REPORTED_GAP_BAND,
            "band_count": int(band.size),
            "band_fraction": float(band.size / sharpe.size),
            "band_sharpe_min": float(band.min()),
            "band_sharpe_max": float(band.max()),
            "band_sharpe_spread_pct":
                float((band.max() - band.min()) / abs(sharpe[best]) * 100),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Repeat the paper ① measurements over many problem instances"
    )
    p.add_argument("--n", type=int, nargs="+", default=DEFAULT_N)
    p.add_argument("--instances", type=int, default=DEFAULT_INSTANCES,
                   help="Instances per size; index 0 is the published subset")
    p.add_argument("--backend", default="lightning_cpu",
                   choices=["lightning_cpu", "lightning_gpu", "braket_local"],
                   help="QAOA backend. Cloud backends are refused: this sweep is "
                        "hundreds of runs and would be billed accordingly")
    p.add_argument("--precision", default="double", choices=["single", "double"])
    p.add_argument("--p-layers", type=int, default=2)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--shots", type=int, default=1000)
    p.add_argument("--skip-quantum", action="store_true",
                   help="Classical measurements only (free, no GPU)")
    p.add_argument("--end-date", default=DATA_END_DATE)
    p.add_argument("--annealing-seeds", type=int, default=0,
                   help="Also run simulated annealing with this many seeds per "
                        "instance, into paper01_instances_annealing.csv. The "
                        "paper's critique is of benchmarking a stochastic solver "
                        "on a near-degenerate QUBO, not of QAOA specifically; "
                        "this measures whether a classical stochastic solver "
                        "shows the same decoupling on the same instances")
    p.add_argument("--results-tag", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.results_tag}" if args.results_tag else ""
    out = RESULTS_DIR / f"paper01_instances{suffix}.csv"

    loader = FinanceDataLoader()
    formulator = PortfolioQUBO()
    metrics = PortfolioMetrics(risk_free_rate=RISK_FREE_RATE)
    classical = ClassicalSolver(method="auto")
    quantum = None if args.skip_quantum else BraketSolver(
        backend=args.backend,
        p_layers=args.p_layers,
        n_shots=args.shots,
        n_optimizer_steps=args.steps,
        precision=args.precision,
    )

    print(f"Universe: {len(NIKKEI225_TICKERS[:50])} cached tickers, "
          f"window ends {args.end_date}")
    print(f"Sizes {sorted(args.n)} x {args.instances} instances"
          + ("" if quantum is None else
             f", QAOA on {args.backend} ({args.precision}), p={args.p_layers}"))

    rows: list[dict] = []
    annealing_rows: list[dict] = []
    annealer = ClassicalSolver(method="simulated_annealing")
    for n in sorted(args.n):
        for instance in range(args.instances):
            tickers = instance_tickers(n, instance)
            data = loader.load(
                tickers, end_date=args.end_date, lookback_years=LOOKBACK_YEARS
            )
            k = min(max(2, n // 5), data.n_assets)
            problem = formulator.formulate(
                returns=data.returns,
                covariance=data.covariance,
                num_select=k,
                risk_aversion=RISK_AVERSION,
            )

            measured = measure(problem, data, k, metrics)
            row = {
                "n_assets": n,
                "n_select": k,
                "instance": instance,
                "is_published_instance": instance == 0,
                "instance_digest": instance_digest(tickers),
                "feasible_set_size": comb(n, k),
                **measured["row"],
            }

            q_obj = formulator.formulate(
                returns=data.returns,
                covariance=data.covariance,
                num_select=k,
                risk_aversion=RISK_AVERSION,
                penalty_strength=0.0,
            ).Q
            a_min, binding, binding_exact = minimum_feasible_penalty(q_obj, n, k)
            row["penalty_used"] = float(problem.metadata["penalty_strength"])
            row["penalty_min"] = a_min
            row["penalty_overshoot"] = row["penalty_used"] / a_min if a_min > 0 else float("inf")
            row["penalty_binding_cardinality"] = binding
            row["penalty_binding_is_exact"] = binding_exact

            # The classical optimum is already known from the enumeration; solving
            # again would only re-derive it, so this is a consistency check that
            # the shipped solver agrees with the census on every instance.
            solved = classical.solve(problem, seed=SEED)
            if solved.energy_no_offset != row["energy_optimum"]:
                print(f"  ABORT: N={n} instance {instance}: solver "
                      f"{solved.energy_no_offset!r} != census {row['energy_optimum']!r}")
                return 1

            if quantum is not None:
                res = quantum.solve(problem, seed=SEED)
                stats = metrics.evaluate(
                    res.selected_assets, data.returns, data.covariance, target_k=k
                )
                e_q = res.energy_no_offset
                delta = abs(e_q - row["energy_optimum"])
                order = np.argsort(measured["energies"], kind="stable")
                row.update({
                    "quantum_backend": res.backend,
                    "energy_quantum": e_q,
                    "quantum_feasible": stats.feasible,
                    "quantum_runtime_s": res.runtime_seconds,
                    "reported_gap_pct": delta / row["abs_energy_optimum"] * 100.0,
                    "gap_of_objective_range_pct": delta / row["objective_range"] * 100.0,
                    "quantum_rank": rank_of_energy(measured["energies"][order], e_q),
                    "sharpe_quantum": stats.sharpe_ratio,
                    "sharpe_delta_pct":
                        (stats.sharpe_ratio - row["sharpe_optimum"])
                        / abs(row["sharpe_optimum"]) * 100.0,
                })
            else:
                for field in ("quantum_backend", "energy_quantum", "quantum_feasible",
                              "quantum_runtime_s", "reported_gap_pct",
                              "gap_of_objective_range_pct", "quantum_rank",
                              "sharpe_quantum", "sharpe_delta_pct"):
                    row[field] = ""

            # Same instance, same feasible-set ranking, a *classical* stochastic
            # solver. If the decoupling the paper reports is a property of the
            # landscape rather than of QAOA, it must show up here too.
            for annealing_seed in range(args.annealing_seeds):
                res = annealer.solve(problem, seed=1000 + annealing_seed)
                stats = metrics.evaluate(
                    res.selected_assets, data.returns, data.covariance, target_k=k)
                e_sa = res.energy_no_offset
                delta = abs(e_sa - row["energy_optimum"])
                order = np.argsort(measured["energies"], kind="stable")
                annealing_rows.append({
                    "n_assets": n,
                    "n_select": k,
                    "instance": instance,
                    "annealing_seed": 1000 + annealing_seed,
                    "energy_optimum": row["energy_optimum"],
                    "energy_annealing": e_sa,
                    "feasible": stats.feasible,
                    "reported_gap_pct": delta / row["abs_energy_optimum"] * 100.0,
                    "gap_of_objective_range_pct": delta / row["objective_range"] * 100.0,
                    "rank": rank_of_energy(measured["energies"][order], e_sa),
                    "feasible_set_size": comb(n, k),
                    "sharpe_optimum": row["sharpe_optimum"],
                    "sharpe_annealing": stats.sharpe_ratio,
                    "sharpe_delta_pct": (stats.sharpe_ratio - row["sharpe_optimum"])
                    / abs(row["sharpe_optimum"]) * 100.0,
                    "runtime_s": res.runtime_seconds,
                })

            rows.append(row)
            marker = " (published)" if instance == 0 else ""
            print(f"  N={n:>2} inst {instance:>2}{marker:<12} "
                  f"deflation {row['deflation_factor']:>6.0f}x  "
                  f"A_min {a_min:>7.4f} ({row['penalty_overshoot']:>5.0f}x)  "
                  f"band {row['band_count']:>7,} "
                  f"Sharpe [{row['band_sharpe_min']:.3f},{row['band_sharpe_max']:.3f}]"
                  + ("" if quantum is None else
                     f"  gap {row['reported_gap_pct']:.4f}% "
                     f"rank {row['quantum_rank']:,}"), flush=True)
            # Flush every row: the N=30 census plus A_min is ~20 s, so a full
            # sweep runs for tens of minutes and must survive an interruption.
            write_csv(out, rows)
            if annealing_rows:
                write_csv(RESULTS_DIR / f"paper01_instances_annealing{suffix}.csv",
                          annealing_rows)

    write_csv(out, rows)
    print(f"\nSaved {len(rows)} instance rows → {out}")
    if annealing_rows:
        sa_out = RESULTS_DIR / f"paper01_instances_annealing{suffix}.csv"
        write_csv(sa_out, annealing_rows)
        print(f"Saved {len(annealing_rows)} annealing rows → {sa_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
