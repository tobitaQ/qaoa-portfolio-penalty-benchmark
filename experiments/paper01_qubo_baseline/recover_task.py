# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Paper ①  実機タスクの結果回収

なぜ必要か
----------
IonQ Forte-1 の結果は `braket.task_result.program_set_task_result` という
入れ子スキーマで返る（実測 2026-08-09）:

    results.json                       -> programResults: ["programs/0/results.json"]
    programs/0/results.json            -> executableResults: ["executables/0.json"]
    programs/0/executables/0.json      -> measurementProbabilities, measuredQubits

一方 SV1 は平坦な `gate_model_task_result` に `measurements` を直接持つ。
PennyLane-Braket プラグイン（1.35）は前者を測定として解釈できず、**例外を出さずに
全ビット0を返した**。`BraketSolver._validate_samples` がこれを検知するようになったが、
既に課金済みのタスクの結果は S3 に残っているので、捨てずに回収する。

課金済みタスク1本が $8.30 である以上、「解析コードのバグで買ったデータを捨てる」のは
選択肢ではない。このスクリプトは ARN を受け取り、どちらのスキーマでも測定結果を
取り出して、実験CSVと同じ列の1行に変換する。

実行例:
    python -m experiments.paper01_qubo_baseline.recover_task \
        --task-arn arn:aws:braket:us-east-1:000000000000:quantum-task/5f0c0d93-... \
        --n 8 --backend braket_ionq --results-tag ionq
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np

from experiments.paper01_qubo_baseline.run_experiment import (
    DATA_END_DATE,
    RESULTS_DIR,
    SEED,
    build_problem,
    write_csv,
)
from src.finance.data_loader import FinanceDataLoader
from src.solvers.braket_results import shot_counts_from_task
from src.finance.metrics import PortfolioMetrics
from src.qubo.portfolio import PortfolioQUBO
from src.solvers.classical_solver import ClassicalSolver

RISK_FREE_RATE = 0.001


def summarise(counts: Counter, problem, data, k: int, metrics) -> dict:
    """Score measured bitstrings against the exact optimum.

    Args:
        counts: Bitstring -> shot count.
        problem: The QUBO the circuit encoded.
        data: Its price data, for the portfolio metrics.
        k: Required cardinality.
        metrics: ``PortfolioMetrics`` instance.

    Returns:
        A row describing the best feasible shot and the shot distribution.
    """
    n = problem.n_variables
    optimum = ClassicalSolver(method="exact").solve(problem, seed=SEED)
    total = sum(counts.values())

    scored = []
    for bits, count in counts.items():
        x = np.array([float(c) for c in bits])
        scored.append((float(x @ problem.Q @ x), int(x.sum()), bits, count))
    scored.sort()

    feasible = [s for s in scored if s[1] == k]
    feasible_shots = sum(s[3] for s in feasible)
    if not feasible:
        raise RuntimeError(
            f"no feasible shot among {total}: hardware noise left nothing with "
            f"cardinality {k}"
        )

    energy, _, bits, _ = feasible[0]
    selected = [i for i, c in enumerate(bits) if c == "1"]
    stats = metrics.evaluate(selected, data.returns, data.covariance, target_k=k)
    return {
        "n_assets": n,
        "n_select": k,
        "energy_quantum": energy,
        "energy_classical": optimum.energy_no_offset,
        "gap_pct": abs(energy - optimum.energy_no_offset)
        / abs(optimum.energy_no_offset) * 100.0,
        "selected_assets": " ".join(str(i) for i in selected),
        "classical_assets": " ".join(str(i) for i in optimum.selected_assets),
        "sharpe_ratio": stats.sharpe_ratio,
        "sharpe_classical": metrics.evaluate(
            optimum.selected_assets, data.returns, data.covariance,
            target_k=k).sharpe_ratio,
        "shots": total,
        "distinct_bitstrings": len(counts),
        "feasible_shots": feasible_shots,
        "feasible_fraction": feasible_shots / total,
        # Best of all shots regardless of feasibility: shows whether the
        # constraint or the objective is what noise destroyed.
        "best_energy_any_cardinality": scored[0][0],
        "data_start": data.start_date,
        "data_end": data.end_date,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recover a completed Braket task's measurements into a result row")
    p.add_argument("--task-arn", required=True)
    p.add_argument("--n", type=int, required=True, help="Assets in the circuit")
    p.add_argument("--backend", required=True, help="Recorded in the output row")
    p.add_argument("--qaoa-seed", type=int, default=SEED)
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--end-date", default=DATA_END_DATE)
    p.add_argument("--results-tag", required=True,
                   help="Output goes to paper01_hardware_<tag>.csv")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    loader, formulator = FinanceDataLoader(), PortfolioQUBO()
    metrics = PortfolioMetrics(risk_free_rate=RISK_FREE_RATE)

    problem, data, k = build_problem(args.n, loader, formulator, args.end_date)
    counts = shot_counts_from_task(args.task_arn, args.region)
    row = summarise(counts, problem, data, k, metrics)
    row = {"backend": args.backend, "qaoa_seed": args.qaoa_seed,
           "task_arn": args.task_arn, **row}

    out = RESULTS_DIR / f"paper01_hardware_{args.results_tag}.csv"
    existing = []
    if out.exists():
        import csv as _csv
        with open(out, newline="") as f:
            existing = [r for r in _csv.DictReader(f)
                        if r["task_arn"] != args.task_arn]
    write_csv(out, existing + [row])

    print(f"N={row['n_assets']} K={k} on {args.backend}")
    print(f"  shots {row['shots']}, {row['distinct_bitstrings']} distinct, "
          f"{row['feasible_fraction']*100:.0f}% feasible")
    print(f"  best feasible E={row['energy_quantum']:.6f} "
          f"(classical {row['energy_classical']:.6f}, gap {row['gap_pct']:.4f}%)")
    print(f"  assets {row['selected_assets']} vs classical {row['classical_assets']}")
    print(f"  Sharpe {row['sharpe_ratio']:.4f} vs classical "
          f"{row['sharpe_classical']:.4f}")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
