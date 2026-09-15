# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Paper ①  補助実験: シード分散スイープ / QAOA深さ(p)感度スイープ

論文①ドラフト §VIII (Limitations) が自ら挙げている2つの穴を埋めるための実験:

    1. "Results are single-seed per size; a multi-seed distribution would
       quantify QAOA variance."
       → --sweep seeds: 問題インスタンスを固定したまま QAOA 初期角のシードを
         振り、最適性ギャップと実行時間の分布を得る。

    2. "QAOA depth is fixed at p = 2; deeper circuits may narrow the optimality
       gap at higher runtime cost."
       → --sweep depth: p を振り、ギャップと実行時間のトレードオフを測る。

    3. v0.2 の depth スイープが自ら残した交絡:
       「p は 2p 個のパラメータを持つのに ADAM ステップ数は50固定なので、
         p=4 の悪化が ansatz のせいか最適化不足かを分離できない」(§VIII)
       → --sweep grid: p × ステップ数の2次元グリッドを回す。各 p について
         ステップ数を増やしたときギャップが下がり止まる点（=最適化が飽和した点）を
         見れば、「p を増やしても効かない」が最適化不足の産物かどうか判別できる。
         判定の読み方:
           - 各 p で十分ステップを与えれば p=4 が p=2 に追いつく/上回る
             → v0.2 の p=4 悪化は**最適化不足**（ステップ数が交絡していた）。
           - 飽和後も p=4 が p=2 に劣る
             → 悪化は**深さ自体**に起因（50ステップは無罪）。

インスタンスが固定される理由:
    FinanceDataLoader.load_nikkei_subset の random_seed は実装上は未使用で
    (docstring: "Unused here; reserved for reproducibility documentation")、
    NIKKEI225_TICKERS の先頭 N 銘柄を決定的に取る。したがって N を固定すれば
    問題インスタンスは完全に同一であり、シードを振って動くのは
    BraketSolver.solve(seed=...) が決める QAOA の初期角のみ。
    これは「固定インスタンス上での QAOA 最適化の分散」という、まさに論文で
    報告したい量になる。

古典参照解について:
    ClassicalSolver は N<=20 で brute force (厳密解)、N>20 で simulated
    annealing (発見的) に自動で切り替わる。N<=20 に絞ったスイープなら
    ギャップは「QAOA vs 真の最適解」であり、§VIII-2 の但し書き
    (「N>20 のギャップは QAOA vs SA」) を回避できる。CSV の
    classical_method 列に実際に使われた手法を必ず記録する。

出力 (いずれも1行ごとに flush するので、途中でOOM/中断しても既存行は残る):
    - results/paper01_seed_sweep.csv
    - results/paper01_depth_sweep.csv
    - results/paper01_sweep_convergence.csv        (seeds / depth の収束曲線)
    - results/paper01_steps_depth_grid.csv         (--sweep grid)
    - results/paper01_grid_convergence.csv         (grid の収束曲線)

grid の収束曲線を別ファイルにしている理由:
    grid は同じ (N, seed, p) をステップ数だけ変えて複数回走らせるので、
    収束曲線の行を一意に識別するには steps 列が必要になる。既存の
    paper01_sweep_convergence.csv は steps 列を持たないヘッダで確定しており、
    追記で列を増やすとヘッダと行がずれて既存データを壊す。列構成が違うものは
    別ファイルに分ける。

実行例:
    # まず計画と所要時間見積もりだけ見る (GPU不要・無料)
    python -m experiments.paper01_qubo_baseline.run_sweep --sweep both --dry-run

    # GPUマシンで本番 (両スイープで約12分)
    python -m experiments.paper01_qubo_baseline.run_sweep \
        --sweep both --backend lightning_gpu --precision single

    # steps × p グリッド (48run・A6000で約1.2時間)
    python -m experiments.paper01_qubo_baseline.run_sweep \
        --sweep grid --backend lightning_gpu --precision single

    # 中断後の再開 (既に完了した組み合わせをスキップ)
    python -m experiments.paper01_qubo_baseline.run_sweep \
        --sweep grid --backend lightning_gpu --precision single --resume

再現性のセルフチェック:
    grid の steps=50 列は depth スイープと (N, seed=42, p, steps=50) が完全に一致する。
    同一設定・同一シードなので energy_quantum は depth スイープの値と一致すべきで、
    一致しなければ実行環境か乱数の扱いが変わったことを意味する。--sweep grid は
    この照合のためあえて再実行する (12run・約9分ぶんの重複は検証コストとして許容)。
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
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from src.finance.data_loader import FinanceDataLoader
from src.finance.metrics import PortfolioMetrics
from src.qubo.portfolio import PortfolioQUBO
from src.solvers.braket_solver import (
    LOCAL_BACKENDS,
    QPU_BACKENDS,
    BraketSolver,
)
from src.solvers.classical_solver import ClassicalSolver

from experiments.paper01_qubo_baseline.run_experiment import (
    _TAG_CHARS,
    DATA_END_DATE,
    RESULTS_DIR,
    SEED,
    build_problem,
)

#: Sweep 1 — QAOA initial-angle seeds on a fixed instance.
DEFAULT_SEED_SWEEP_N = [8, 12, 16, 20]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]

#: Sweep 2 — QAOA depth. p=2 is the value used in the main experiment.
DEFAULT_DEPTH_SWEEP_N = [12, 16, 20]
DEFAULT_P_VALUES = [1, 2, 3, 4]

#: Sweep 3 — the steps x p grid that separates depth from optimizer budget.
#: Steps rise geometrically because we are looking for the point where the gap
#: stops improving; a linear ladder would spend most of its runs past it.
#:
#: The 200 rung was dropped after the first single-seed pass: the expectation
#: value it optimizes is already flat between 50 and 100 steps at every depth, so
#: 200 measured nothing that 100 had not, while costing 4x a 50-step run. 400 is
#: kept as the witness that an 8x budget still does not move the objective. The
#: freed time goes into seeds instead, which is where the actual variance is.
DEFAULT_GRID_N = [12, 16, 20]
DEFAULT_GRID_P = [1, 2, 3, 4]
DEFAULT_GRID_STEPS = [50, 100, 400]
#: The grid runs every cell at several initial-angle seeds. One seed per cell
#: cannot separate depth from initialization: the first pass had N=16, p=3
#: converge to <H> = -2.91 where p=2 reached -5.07, which looks like "depth 3 is
#: worse" but is one unlucky starting angle.
DEFAULT_GRID_SEEDS = [42, 43, 44, 45, 46]

#: Measured QAOA wall-clock on lightning.gpu (RTX A6000, single precision,
#: p=2, 50 ADAM steps), from the first full GPU run (2026-08). Used only
#: to print an up-front time estimate so a sweep is never started blind.
MEASURED_RUNTIME_S = {8: 5.5, 12: 10.0, 16: 18.0, 20: 30.0,
                      24: 150.0, 28: 2520.0, 30: 11304.0}

#: SV1 bills $0.075 per minute of simulation with a three-second floor per
#: task, and adjoint differentiation makes one optimizer step exactly one task
#: (measured in §VI-C: 51 tasks for a 50-step run plus one sampling task). The
#: floor dominates at these sizes, so tasks x 3 s is the right estimate and it
#: is an under-estimate only for N >= 20, where a task took 14.3 s.
SV1_USD_PER_MINUTE = 0.075
SV1_MIN_TASK_SECONDS = 3.0

#: Measured SV1 simulation seconds per analytic task, from the §VI-C run. The
#: three-second floor covers everything up to N=16, but N=20 took 14.3 s and
#: pays for its real duration -- so a floor-only estimate under-states a sweep
#: that includes N=20 by about 3x, which is exactly the size of error a cost
#: guard must not make.
SV1_TASK_SECONDS = {8: 0.7, 12: 0.9, 16: 1.6, 20: 14.3}

#: IonQ Forte-1, per Braket's published pricing (2026-08). Hardware bills per
#: task *and* per shot, so the shot count -- irrelevant to an SV1 bill -- is the
#: dominant term here: 1000 shots is $80.30 for a single sampling task.
IONQ_USD_PER_TASK = 0.30
IONQ_USD_PER_SHOT = 0.08

#: Hard ceiling on shots when a QPU does the sampling. The project default is
#: 1000, which costs $80.30 per run and blows a $50 budget on the first task.
#: 100 shots is $8.30, the figure the IonQ campaign was planned around.
MAX_QPU_SHOTS = 200

#: Refuse a billed sweep larger than this without an explicit override. The
#: grid alone is 180 runs -- about 9,180 SV1 tasks and well over $100 -- and
#: the whole point of a guard is that the expensive plan is the one you have to
#: ask for twice.
MAX_CLOUD_RUNS = 20

SEED_SWEEP_CSV = "paper01_seed_sweep.csv"
DEPTH_SWEEP_CSV = "paper01_depth_sweep.csv"
CONVERGENCE_CSV = "paper01_sweep_convergence.csv"
GRID_SWEEP_CSV = "paper01_steps_depth_grid.csv"
GRID_CONVERGENCE_CSV = "paper01_grid_convergence.csv"

#: Columns that identify one sweep run, for --resume de-duplication. "steps"
#: belongs here because the grid sweep varies it: without it, the same (N, seed,
#: p) at 50 and at 400 steps would collide and --resume would drop the second.
#: The seed/depth CSVs already written by v0.2 carry a steps column, so adding
#: it here does not invalidate them for --resume.
RUN_KEY_FIELDS = ("sweep", "n_assets", "qaoa_seed", "p_layers", "steps")


@dataclass(frozen=True)
class SweepRun:
    """One planned QAOA run within a sweep.

    Attributes:
        sweep: Which sweep this run belongs to ("seeds", "depth" or "grid").
        n_assets: Problem size N.
        qaoa_seed: Seed for the QAOA initial angles.
        p_layers: QAOA depth.
        steps: ADAM optimizer steps. Fixed across the seed and depth sweeps;
            varied by the grid sweep, which is the whole point of it.
    """

    sweep: str
    n_assets: int
    qaoa_seed: int
    p_layers: int
    steps: int

    def key(self) -> tuple[str, ...]:
        """Return the identity tuple used to skip already-completed runs."""
        return (self.sweep, str(self.n_assets), str(self.qaoa_seed),
                str(self.p_layers), str(self.steps))


class IncrementalCSV:
    """CSV writer that flushes after every row.

    A sweep can run for hours on a GPU and may die on an out-of-memory error at
    the largest N. Flushing per row means every completed run survives.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = None
        self._writer: csv.DictWriter | None = None

    def append(self, row: dict) -> None:
        """Write one row, creating the file and header on first call.

        Args:
            row: Column name to value. Its keys define the header on the first
                append to a new file.

        Raises:
            ValueError: If the file already exists with a different column set.
                Appending a changed schema would write rows that no longer line
                up with the header, silently corrupting the runs already in the
                file — refusing is the only safe response.
        """
        if self._writer is None:
            new_file = not self.path.exists() or self.path.stat().st_size == 0
            if not new_file:
                with open(self.path, newline="") as f:
                    existing = next(csv.reader(f), [])
                if existing != list(row.keys()):
                    raise ValueError(
                        f"{self.path.name} has columns {existing}, but this run "
                        f"would append {list(row.keys())}. Refusing to corrupt "
                        f"the existing rows; move the file aside or write to a "
                        f"new one.")
            self._fh = open(self.path, "a", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=list(row.keys()))
            if new_file:
                self._writer.writeheader()
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        """Close the underlying file handle if it was opened."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None


def estimate_runtime_s(n_assets: int, p_layers: int, steps: int) -> float:
    """Estimate QAOA wall-clock from the measured p=2 / 50-step GPU baseline.

    Cost is dominated by statevector work, which is linear in circuit depth
    (hence in ``p_layers``) and in the number of optimizer iterations. Sizes
    between measured points are interpolated on a log scale; sizes outside the
    measured range fall back to a 2^N extrapolation from the nearest point.

    Args:
        n_assets: Problem size N.
        p_layers: QAOA depth.
        steps: Number of ADAM steps.

    Returns:
        Estimated wall-clock seconds. Indicative only — never reported as data.
    """
    import math

    known = sorted(MEASURED_RUNTIME_S)
    if n_assets in MEASURED_RUNTIME_S:
        base = MEASURED_RUNTIME_S[n_assets]
    elif n_assets < known[0] or n_assets > known[-1]:
        anchor = known[0] if n_assets < known[0] else known[-1]
        base = MEASURED_RUNTIME_S[anchor] * 2.0 ** (n_assets - anchor)
    else:
        lo = max(k for k in known if k < n_assets)
        hi = min(k for k in known if k > n_assets)
        frac = (n_assets - lo) / (hi - lo)
        base = math.exp(
            math.log(MEASURED_RUNTIME_S[lo]) * (1 - frac)
            + math.log(MEASURED_RUNTIME_S[hi]) * frac
        )
    return base * (p_layers / 2.0) * (steps / 50.0)


def format_duration(seconds: float) -> str:
    """Render a duration as a compact human-readable string."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


def build_plan(args: argparse.Namespace) -> list[SweepRun]:
    """Expand the CLI arguments into the ordered list of runs to execute.

    "both" keeps its v0.2 meaning of seeds + depth so existing commands and
    their --resume state behave unchanged; "all" is the superset including the
    grid.
    """
    plan: list[SweepRun] = []
    if args.sweep in ("seeds", "both", "all"):
        for n in sorted(args.seed_n):
            for seed in args.seeds:
                plan.append(SweepRun("seeds", n, seed, args.p_layers, args.steps))
    if args.sweep in ("depth", "both", "all"):
        for n in sorted(args.depth_n):
            for p in sorted(args.p_values):
                plan.append(SweepRun("depth", n, args.seed, p, args.steps))
    if args.sweep in ("grid", "all"):
        # Ordered cheapest-first within each size: if the grid is cut short, the
        # rows that survive still cover every p at the lower step counts rather
        # than one p exhaustively. Seeds are the innermost loop so that a cell is
        # either fully populated or not started -- a half-seeded cell would give
        # a median computed over a different n than its neighbours.
        for n in sorted(args.grid_n):
            for steps in sorted(args.grid_steps):
                for p in sorted(args.grid_p):
                    for seed in args.grid_seeds:
                        plan.append(SweepRun("grid", n, seed, p, steps))
    return plan


def load_completed_keys(paths: list[Path]) -> set[tuple[str, ...]]:
    """Read existing sweep CSVs and return the identity tuples already present."""
    done: set[tuple[str, ...]] = set()
    for path in paths:
        if not path.exists():
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if all(field in row for field in RUN_KEY_FIELDS):
                    done.add(tuple(row[field] for field in RUN_KEY_FIELDS))
    return done


def optimality_gap_pct(quantum_energy: float, classical_energy: float) -> float | None:
    """Percentage by which QAOA's energy exceeds the classical reference.

    Both energies are minimisation objectives, so a positive gap means QAOA is
    worse. Returns None when the reference energy is zero.
    """
    if classical_energy == 0:
        return None
    return (quantum_energy - classical_energy) / abs(classical_energy) * 100.0


def iter_sizes(plan: list[SweepRun]) -> Iterator[int]:
    """Yield the distinct problem sizes in the plan, in ascending order."""
    yield from sorted({run.n_assets for run in plan})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paper ① seed-variance and QAOA-depth sweeps")
    p.add_argument("--sweep", default="both",
                   choices=["seeds", "depth", "grid", "both", "all"],
                   help="'both' = seeds + depth (v0.2 behaviour); "
                        "'all' adds the steps x p grid")
    p.add_argument("--backend", default="lightning_cpu",
                   choices=["lightning_cpu", "lightning_gpu", "braket_local",
                            "braket_sv1"],
                   help="braket_sv1 is billed and needs --results-tag; the plan "
                        "is costed and refused above --max-cloud-runs. Real "
                        "hardware is never valid here -- a sweep is hundreds of "
                        "runs and BraketSolver rejects QPUs outright")
    p.add_argument("--results-tag", default=None,
                   help="Suffix the sweep CSVs (paper01_seed_sweep_<tag>.csv). "
                        "Required for billed backends so a cloud run cannot "
                        "overwrite the local sweeps behind Figures 5-7")
    p.add_argument("--sample-backend", default=None,
                   choices=["lightning_cpu", "lightning_gpu", "braket_local",
                            "braket_sv1", "braket_ionq"],
                   help="Run only the final sampling circuit here, optimizing on "
                        "--backend. The only way to reach real hardware: a QPU "
                        "cannot host the optimization loop")
    p.add_argument("--s3-bucket", default=None,
                   help="Required for braket_sv1 / braket_ionq")
    p.add_argument("--max-cloud-runs", type=int, default=MAX_CLOUD_RUNS,
                   help="Refuse a billed plan larger than this many runs")
    p.add_argument("--yes-bill-me", action="store_true",
                   help="Acknowledge the printed cost estimate and proceed")
    p.add_argument("--precision", default="double", choices=["single", "double"])
    p.add_argument("--seed-n", type=int, nargs="+", default=DEFAULT_SEED_SWEEP_N,
                   help="Problem sizes for the seed sweep (<=20 keeps the "
                        "classical reference exact)")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                   help="QAOA initial-angle seeds")
    p.add_argument("--depth-n", type=int, nargs="+", default=DEFAULT_DEPTH_SWEEP_N,
                   help="Problem sizes for the depth sweep")
    p.add_argument("--p-values", type=int, nargs="+", default=DEFAULT_P_VALUES,
                   help="QAOA depths to compare")
    p.add_argument("--grid-n", type=int, nargs="+", default=DEFAULT_GRID_N,
                   help="Problem sizes for the steps x p grid")
    p.add_argument("--grid-p", type=int, nargs="+", default=DEFAULT_GRID_P,
                   help="QAOA depths in the grid")
    p.add_argument("--grid-steps", type=int, nargs="+",
                   default=DEFAULT_GRID_STEPS,
                   help="ADAM step counts in the grid; --steps is ignored for "
                        "the grid sweep because the grid varies it")
    p.add_argument("--grid-seeds", type=int, nargs="+",
                   default=DEFAULT_GRID_SEEDS,
                   help="Initial-angle seeds run at every grid cell; --seed is "
                        "ignored for the grid sweep because the grid varies it")
    p.add_argument("--p-layers", type=int, default=2,
                   help="Fixed depth used during the seed sweep")
    p.add_argument("--shot-seed", type=int, default=None,
                   help="Re-seed the RNG just before the final sampling circuit, "
                        "leaving the optimized angles alone. Running the same "
                        "sweep twice with different values isolates shot-sampling "
                        "randomness from everything else that can redraw the "
                        "returned portfolio")
    p.add_argument("--seed", type=int, default=SEED,
                   help="Fixed QAOA seed used during the depth sweep")
    p.add_argument("--steps", type=int, default=50, help="ADAM optimizer steps")
    p.add_argument("--shots", type=int, default=1000,
                   help="Shots for final sampling")
    p.add_argument("--end-date", default=DATA_END_DATE,
                   help="Last day of the price window, YYYY-MM-DD. Must match "
                        "the main experiment's window or the gaps are not "
                        "comparable with Table I")
    p.add_argument("--resume", action="store_true",
                   help="Skip runs already present in the output CSVs")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and time estimate, then exit")
    return p.parse_args(argv)


#: Every file a sweep writes. The tag has to reach all five: an SV1 run whose
#: summary went to a tagged file still appended its convergence traces to the
#: local one, and since the convergence rows carry no backend column the two
#: runs were indistinguishable once mixed. Tagging by filename is the fix that
#: matches the rest of the project; adding a column would break the header of
#: the files already written.
SWEEP_CSV_NAMES = (SEED_SWEEP_CSV, DEPTH_SWEEP_CSV, GRID_SWEEP_CSV,
                   CONVERGENCE_CSV, GRID_CONVERGENCE_CSV)


def sweep_csv_names(tag: str | None) -> tuple[str, str, str, str, str]:
    """Sweep CSV filenames, suffixed when the run names itself.

    An untagged run writes the files behind Figures 5-7. A billed run must not,
    for the same reason ``run_experiment`` requires a tag: the local sweeps are
    the reference a cloud run is being compared against.

    Returns:
        ``(seed, depth, grid, convergence, grid convergence)`` filenames.
    """
    if tag is None:
        return SWEEP_CSV_NAMES
    if not tag or not set(tag) <= _TAG_CHARS:
        raise ValueError(
            f"--results-tag must be non-empty [A-Za-z0-9-_], got {tag!r}")
    return tuple(name.replace(".csv", f"_{tag}.csv") for name in SWEEP_CSV_NAMES)


def estimate_hardware_cost(plan: list["SweepRun"], shots: int) -> tuple[int, float]:
    """Estimate QPU tasks and dollars when hardware does only the sampling.

    One task per run -- the optimizer never touches hardware -- but each task
    carries its shots, and shots are what cost money on a QPU. This is the
    inverse of SV1, where the gradient dominates and shots are nearly free.

    Args:
        plan: The runs about to be executed.
        shots: Measurement shots in the single sampling circuit.

    Returns:
        ``(task count, estimated USD)``.
    """
    tasks = len(plan)
    return tasks, tasks * (IONQ_USD_PER_TASK + shots * IONQ_USD_PER_SHOT)


def estimate_cloud_cost(plan: list["SweepRun"]) -> tuple[int, float]:
    """Estimate SV1 tasks and dollars for a plan.

    One task per optimizer step plus one for the final sampling circuit, each
    charged for the greater of its simulation time and the three-second floor.
    Per-task durations come from the measured §VI-C run; sizes above the largest
    measured one are extrapolated by doubling per qubit, since an SV1 task is a
    statevector simulation.

    Args:
        plan: The runs about to be executed.

    Returns:
        ``(task count, estimated USD)``.
    """
    largest = max(SV1_TASK_SECONDS)
    seconds = 0.0
    tasks = 0
    for run in plan:
        per_task = SV1_TASK_SECONDS.get(
            run.n_assets,
            SV1_TASK_SECONDS[largest] * 2 ** (run.n_assets - largest),
        )
        count = run.steps + 1
        tasks += count
        seconds += count * max(SV1_MIN_TASK_SECONDS, per_task)
    return tasks, seconds / 60.0 * SV1_USD_PER_MINUTE


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    slots = {args.backend, args.sample_backend} - {None}
    billed = bool(slots - LOCAL_BACKENDS)
    on_hardware = bool(slots & QPU_BACKENDS)
    if on_hardware and args.shots > MAX_QPU_SHOTS:
        print(f"--shots {args.shots} on {sorted(slots & QPU_BACKENDS)[0]} would "
              f"cost ${IONQ_USD_PER_TASK + args.shots * IONQ_USD_PER_SHOT:,.2f} "
              f"per run (hardware bills per shot). The ceiling is "
              f"{MAX_QPU_SHOTS}; 100 shots is $8.30.")
        return 2
    if billed and args.results_tag is None:
        print(f"--results-tag is required for backend {args.backend}: an "
              f"untagged run overwrites the sweep CSVs behind Figures 5-7, "
              f"which are the local reference this run is compared against. "
              f"Try --results-tag sv1.")
        return 2
    if billed and args.sweep in ("grid", "all"):
        print(f"--sweep {args.sweep} includes the steps x p x seed grid: "
              f"{len(build_plan(args))} runs. That is a four-figure task count "
              f"on a metered backend and is never the right thing to buy. Run "
              f"--sweep seeds or --sweep depth on the cloud and keep the grid "
              f"local.")
        return 2

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        (seed_name, depth_name, grid_name,
         convergence_name, grid_convergence_name) = sweep_csv_names(args.results_tag)
    except ValueError as exc:
        print(exc)
        return 2
    seed_csv = RESULTS_DIR / seed_name
    depth_csv = RESULTS_DIR / depth_name
    grid_csv = RESULTS_DIR / grid_name

    plan = build_plan(args)
    if args.resume:
        done = load_completed_keys([seed_csv, depth_csv, grid_csv])
        skipped = [r for r in plan if r.key() in done]
        plan = [r for r in plan if r.key() not in done]
        print(f"--resume: skipping {len(skipped)} already-completed run(s)")

    if not plan:
        print("Nothing to run.")
        return 0

    total_estimate = sum(
        estimate_runtime_s(r.n_assets, r.p_layers, r.steps) for r in plan)

    step_counts = sorted({r.steps for r in plan})
    steps_label = (str(step_counts[0]) if len(step_counts) == 1
                   else ",".join(str(s) for s in step_counts))
    print(f"Backend: {args.backend} ({args.precision} precision), "
          f"steps={steps_label}, shots={args.shots}")
    print(f"Price window ends: {args.end_date} (must match Table I)")
    print(f"Planned QAOA runs: {len(plan)}   "
          f"estimated total: {format_duration(total_estimate)}")
    print(f"{'sweep':>6} {'N':>4} {'seed':>5} {'p':>2} {'steps':>6} {'est':>9}")
    print("-" * 39)
    for run in plan:
        est = estimate_runtime_s(run.n_assets, run.p_layers, run.steps)
        print(f"{run.sweep:>6} {run.n_assets:>4} {run.qaoa_seed:>5} "
              f"{run.p_layers:>2} {run.steps:>6} {format_duration(est):>9}")

    if billed:
        if on_hardware:
            tasks, usd = estimate_hardware_cost(plan, args.shots)
            detail = (f"1 sampling task per run at {args.shots} shots; "
                      f"${IONQ_USD_PER_TASK}/task + ${IONQ_USD_PER_SHOT}/shot")
        else:
            tasks, usd = estimate_cloud_cost(plan)
            detail = (f"at ${SV1_USD_PER_MINUTE}/min, "
                      f"{SV1_MIN_TASK_SECONDS:.0f}s floor, "
                      f"measured per-task durations from §VI-C")
        print(f"\nBILLED BACKEND {sorted(slots - LOCAL_BACKENDS)[0]}: "
              f"{len(plan)} runs "
              f"-> ~{tasks:,} tasks, ~${usd:,.2f} ({detail})")
        if len(plan) > args.max_cloud_runs:
            print(f"Refusing: {len(plan)} runs exceeds --max-cloud-runs "
                  f"{args.max_cloud_runs}. Narrow the plan, or raise the ceiling "
                  f"deliberately.")
            return 2
        if not args.yes_bill_me and not args.dry_run:
            print("Refusing: pass --yes-bill-me to acknowledge the estimate "
                  "above and proceed.")
            return 2

    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return 0

    loader = FinanceDataLoader()
    formulator = PortfolioQUBO()
    metrics = PortfolioMetrics(risk_free_rate=0.001)
    classical_solver = ClassicalSolver(method="auto")

    # One classical reference per size, reused across every run at that size:
    # the instance is identical, so re-solving would only burn time.
    print("\nComputing classical reference per size...")
    problems: dict[int, tuple] = {}
    reference: dict[int, tuple[float, str]] = {}
    for n in iter_sizes(plan):
        problem, data, k = build_problem(n, loader, formulator, args.end_date)
        problems[n] = (problem, data, k)
        res = classical_solver.solve(problem, seed=SEED)
        method = res.metadata.get("method", "")
        reference[n] = (res.energy_no_offset, method)
        # Ask the solver whether it proved optimality rather than matching on a
        # method name: the exhaustive set grew from {brute_force} to
        # {brute_force, exact} and a name check silently mislabelled the new
        # one as a heuristic, which is the wrong way round for a gap's meaning.
        proven = res.metadata.get("exhaustive", False)
        print(f"  N={n:>3} K={k}  E={res.energy_no_offset:>13.6f}  "
              f"({method}, {'exact' if proven else 'heuristic'})")

    writers = {
        "seeds": IncrementalCSV(seed_csv),
        "depth": IncrementalCSV(depth_csv),
        "grid": IncrementalCSV(grid_csv),
    }
    # The grid's convergence rows need a steps column to be identifiable, which
    # the v0.2 convergence file's header does not have; see the module docstring.
    # seeds and depth share one writer because they share that header.
    sweep_convergence = IncrementalCSV(RESULTS_DIR / convergence_name)
    grid_convergence = IncrementalCSV(RESULTS_DIR / grid_convergence_name)
    convergence_writers = {
        "seeds": sweep_convergence,
        "depth": sweep_convergence,
        "grid": grid_convergence,
    }
    failures: list[tuple[SweepRun, str]] = []

    try:
        for idx, run in enumerate(plan, start=1):
            problem, data, k = problems[run.n_assets]
            classical_energy, classical_method = reference[run.n_assets]
            est = estimate_runtime_s(run.n_assets, run.p_layers, run.steps)
            print(f"\n[{idx}/{len(plan)}] sweep={run.sweep} N={run.n_assets} "
                  f"seed={run.qaoa_seed} p={run.p_layers} steps={run.steps} "
                  f"(est {format_duration(est)})")

            solver = BraketSolver(
                backend=args.backend,
                p_layers=run.p_layers,
                n_shots=args.shots,
                n_optimizer_steps=run.steps,
                precision=args.precision,
                s3_bucket=args.s3_bucket,
                sample_backend=args.sample_backend,
            )
            try:
                res = solver.solve(problem, seed=run.qaoa_seed,
                                   shot_seed=args.shot_seed)
            except Exception as exc:
                traceback.print_exc()
                failures.append((run, str(exc)))
                continue

            stats = metrics.evaluate(
                res.selected_assets, data.returns, data.covariance, target_k=k)
            gap = optimality_gap_pct(res.energy_no_offset, classical_energy)

            writers[run.sweep].append({
                "sweep": run.sweep,
                "n_assets": run.n_assets,
                "n_select": k,
                "qaoa_seed": run.qaoa_seed,
                "p_layers": run.p_layers,
                "energy_quantum": res.energy_no_offset,
                "energy_classical": classical_energy,
                "gap_pct": gap,
                "runtime_s": res.runtime_seconds,
                "feasible": stats.feasible,
                "sharpe_ratio": stats.sharpe_ratio,
                "classical_method": classical_method,
                "backend": res.backend,
                "precision": args.precision,
                "steps": run.steps,
                "shots": args.shots,
                "data_start": data.start_date,
                "data_end": data.end_date,
            })
            for step, energy in enumerate(res.metadata.get("convergence", [])):
                row = {
                    "sweep": run.sweep,
                    "n_assets": run.n_assets,
                    "qaoa_seed": run.qaoa_seed,
                    "p_layers": run.p_layers,
                    **({"steps": run.steps} if run.sweep == "grid" else {}),
                    "step": step,
                    "expectation": energy,
                }
                convergence_writers[run.sweep].append(row)

            gap_str = "n/a" if gap is None else f"{gap:.4f}%"
            print(f"    E={res.energy_no_offset:.6f}  gap={gap_str}  "
                  f"t={res.runtime_seconds:.1f}s  feasible={stats.feasible}")
    finally:
        for w in writers.values():
            w.close()
        sweep_convergence.close()
        grid_convergence.close()

    written = sorted({
        writers[run.sweep].path for run in plan
    } | {
        convergence_writers[run.sweep].path for run in plan
    })
    print("\nWrote → " + "\n       ".join(str(p) for p in written))

    if failures:
        print(f"\n--- {len(failures)} failure(s) ---")
        for run, msg in failures:
            print(f"  sweep={run.sweep} N={run.n_assets} seed={run.qaoa_seed} "
                  f"p={run.p_layers} steps={run.steps}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
