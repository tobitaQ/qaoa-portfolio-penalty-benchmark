# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Paper ①  実験スクリプト: QUBO Baseline

実験条件:
    - 選択銘柄数 K: max(2, N // 5)  (約20%)
    - リスク回避パラメータ λ: 0.5
    - 乱数シード: 42 (固定)
    - データ: 日経225構成銘柄 (yfinance, 過去3年)

銘柄数 N の扱い:
    古典ソルバーは N=50 まで問題なく解ける。一方 QAOA は 1銘柄=1量子ビット
    のため状態ベクトルが 2^N に比例し、シミュレータ側に上限がある:

        lightning_cpu  ... N=20 で約3分、N=24 で約78分
        lightning_gpu  ... 48GB VRAM で N=30 まで (precision="single" 必須)
        SV1 (クラウド)  ... 34量子ビット
        IonQ Forte     ... 実機の量子ビット数に依存

    したがって古典は --classical-n 全域、量子は --quantum-n のみを走らせ、
    到達できなかった N は「測定された限界」として論文に報告する。

出力:
    - results/paper01_results.csv      (定量指標)
    - results/paper01_convergence.csv  (QAOA収束曲線)

    --results-tag NAME を付けると results/paper01_results_NAME.csv 側に書く。
    タグ無しのファイルはコミット済みの基準値（補足 Table S2 の出典）なので、
    クラウドバックエンド（braket_sv1 / braket_ionq）はタグを必須にしてあり、
    ローカルでもタグ無しの実行は既存ファイルがあれば --write-canonical を
    付けない限り拒否する。

実行例:
    # ローカルCPU (無料・数分)
    python -m experiments.paper01_qubo_baseline.run_experiment

    # GPUマシン (N=30まで)
    python -m experiments.paper01_qubo_baseline.run_experiment \
        --backend lightning_gpu --precision single --quantum-n 8 12 16 20 24 28 30
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
from pathlib import Path

from src.finance.data_loader import FinanceDataLoader
from src.finance.metrics import PortfolioMetrics
from src.qubo.portfolio import PortfolioQUBO
from src.solvers.braket_solver import (
    LOCAL_BACKENDS,
    QPU_BACKENDS,
    BraketSolver,
)
from src.solvers.classical_solver import ClassicalSolver

SEED = 42
RISK_AVERSION = 0.5
LOOKBACK_YEARS = 3
RESULTS_DIR = Path(__file__).parent / "results"

#: Pinned end of the price window. FinanceDataLoader defaults end_date to
#: date.today(), which would slide the dataset forward on every run: the same N
#: and seed would yield a different QUBO — and a different optimal energy — from
#: one day to the next, making published tables unreproducible. This constant
#: freezes the window to the one used for the paper's Table I (the Step 2 GPU
#: run of 2026-08-05); the effective window is recorded in every results row.
DATA_END_DATE = "2026-08-05"

# Classical solver handles the full range; QAOA is capped by simulator memory.
DEFAULT_CLASSICAL_N = [8, 12, 16, 20, 30, 50]
DEFAULT_QUANTUM_N = [8, 12, 16, 20]

#: Characters allowed in --results-tag. The tag goes straight into a filename.
_TAG_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")


def results_paths(tag: str | None) -> tuple[Path, Path]:
    """Return the (results, convergence) CSV paths for a run.

    An untagged run writes the canonical pair that Table I and Figures 1-4 are
    generated from. Every other run must name itself, so a cross-backend check
    cannot silently overwrite the published numbers.

    Args:
        tag: Short run label, or None for the canonical pair.

    Returns:
        Tuple of (results CSV path, convergence CSV path).

    Raises:
        ValueError: If the tag contains anything but [A-Za-z0-9-_].
    """
    if tag is None:
        suffix = ""
    else:
        if not tag or not set(tag) <= _TAG_CHARS:
            raise ValueError(
                f"--results-tag must be non-empty [A-Za-z0-9-_], got {tag!r}"
            )
        suffix = f"_{tag}"
    return (
        RESULTS_DIR / f"paper01_results{suffix}.csv",
        RESULTS_DIR / f"paper01_convergence{suffix}.csv",
    )


def canonical_overwrite_refusal(paths, tag: str | None, write_canonical: bool,
                                resume: bool = False) -> str | None:
    """Explain why an untagged run may not proceed, or return None.

    The untagged file names are the committed baseline the paper reads. A clone
    therefore already holds them, and a reader who runs a driver "to see it
    work" would replace them with a shorter, different run before comparing
    anything. So an untagged run that finds its canonical output in place is
    refused unless it says ``--write-canonical``; a tagged run writes beside
    the baseline and needs no permission; a ``--resume`` run only appends the
    rows it finds missing and is the drivers' own restart path.

    Args:
        paths: The canonical output paths the run would write.
        tag: The ``--results-tag`` value, or None for the canonical names.
        write_canonical: Whether ``--write-canonical`` was passed.
        resume: Whether the run appends to, rather than replaces, its outputs.

    Returns:
        The refusal message to print, or None when the run may go ahead.
    """
    if tag is not None or write_canonical or resume:
        return None
    present = [p for p in paths if Path(p).exists()]
    if not present:
        return None
    names = ", ".join(Path(p).name for p in present)
    return (f"refusing to overwrite the committed baseline file(s) {names}: "
            f"pass --results-tag <name> to write beside them (e.g. "
            f"--results-tag reproduction), or --write-canonical to replace "
            f"them deliberately")


def build_problem(
    n_assets: int,
    loader: FinanceDataLoader,
    formulator: PortfolioQUBO,
    end_date: str | None = DATA_END_DATE,
):
    """Load data for n_assets and formulate the QUBO. Returns (problem, data, K).

    Args:
        n_assets: Number of candidate assets N.
        loader: Price-data source.
        formulator: QUBO formulator (Layer 1).
        end_date: Last day of the price window, "YYYY-MM-DD". Defaults to the
            pinned DATA_END_DATE so results stay reproducible; pass None to
            follow today's date instead (this changes the problem instance).
    """
    data = loader.load_nikkei_subset(
        n_assets=n_assets,
        end_date=end_date,
        lookback_years=LOOKBACK_YEARS,
        random_seed=SEED,
    )
    actual_k = min(max(2, n_assets // 5), data.n_assets)
    problem = formulator.formulate(
        returns=data.returns,
        covariance=data.covariance,
        num_select=actual_k,
        risk_aversion=RISK_AVERSION,
    )
    return problem, data, actual_k


def row_from_result(result, data, problem, metrics, k, method_detail: str) -> dict:
    """Build one results-CSV row from a SolverResult."""
    stats = metrics.evaluate(
        result.selected_assets, data.returns, data.covariance, target_k=k
    )
    return {
        "n_assets": data.n_assets,
        "n_select": k,
        "solver": result.backend,
        "energy_no_offset": result.energy_no_offset,
        "runtime_s": result.runtime_seconds,
        "feasible": stats.feasible,
        "expected_return": stats.expected_return,
        "volatility": stats.volatility,
        "sharpe_ratio": stats.sharpe_ratio,
        "method_detail": method_detail,
        # Provenance: the price window fully determines the QUBO instance, so a
        # results row is only reproducible if it travels with its own window.
        "data_start": data.start_date,
        "data_end": data.end_date,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows to CSV, with every key any row carries in the header.

    Taking the header from the first row alone drops columns silently. The E4
    state comparison has them only on the rows it could compare -- the
    reference arm has none -- so running it without the run artifacts present
    rewrote the file with twelve of its nineteen columns gone, and nothing said
    so. Keys keep first-seen order, and a row missing one writes an empty cell.
    """
    if not rows:
        return
    fields: dict[str, None] = {}
    for row in rows:
        fields.update(dict.fromkeys(row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), restval="")
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------------
# Merge path (--merge)
# ----------------------------------------------------------------------
#
# Rerunning only part of the sweep used to destroy the rest of the table:
# ``--skip-quantum`` writes a file containing classical rows only, and the
# quantum rows -- N=30 alone is 3.14 h on an A6000 -- were gone. The QUBO is
# fixed by the price window, so a classical-side change (a better exact solver,
# say) leaves every quantum row still valid. ``--merge`` replaces just the rows
# this run recomputed and keeps the others verbatim.

#: A results row is identified by which solver ran at which size.
RESULT_KEY = ("n_assets", "solver")

#: A convergence point is identified by size, backend, depth and step.
CONVERGENCE_KEY = ("n_assets", "backend", "p_layers", "step")


def read_csv(path: Path) -> list[dict]:
    """Read a CSV into a list of dicts, or [] if it does not exist."""
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _row_key(row: dict, key_fields: tuple[str, ...]) -> tuple[str, ...]:
    """Key a row by its identifying fields, as text.

    Rows read back from CSV are strings while freshly computed rows hold ints,
    so both sides are normalised to ``str`` before comparison.
    """
    return tuple(str(row[f]) for f in key_fields)


def _result_sort_key(row: dict) -> tuple:
    """Order rows by size, classical baseline first, then solver name."""
    solver = str(row["solver"])
    return (int(row["n_assets"]), 0 if solver.startswith("classical") else 1, solver)


def _convergence_sort_key(row: dict) -> tuple:
    return (
        int(row["n_assets"]),
        str(row["backend"]),
        int(row["p_layers"]),
        int(row["step"]),
    )


def check_mergeable(existing: list[dict], fresh: list[dict]) -> None:
    """Raise if merging ``fresh`` into ``existing`` would mix incompatible runs.

    Two ways a merge silently corrupts the table, both checked here:

    * a changed set of columns, which would drop or blank a field; and
    * a changed price window, which means the retained rows and the new rows
      describe *different QUBO instances* even at the same N. Provenance is
      recorded per row precisely so this is detectable.

    Args:
        existing: Rows already on disk.
        fresh: Rows produced by this run.

    Raises:
        ValueError: If the two sets cannot be merged.
    """
    if not existing or not fresh:
        return

    old_fields, new_fields = set(existing[0]), set(fresh[0])
    if old_fields != new_fields:
        missing = sorted(old_fields - new_fields)
        added = sorted(new_fields - old_fields)
        raise ValueError(
            f"cannot merge: column mismatch (missing {missing}, new {added}). "
            f"Rerun without --merge to regenerate the file."
        )

    if "data_start" not in old_fields:
        return
    windows: dict[str, tuple[str, str]] = {
        str(r["n_assets"]): (str(r["data_start"]), str(r["data_end"]))
        for r in existing
    }
    for r in fresh:
        n = str(r["n_assets"])
        new_window = (str(r["data_start"]), str(r["data_end"]))
        if n in windows and windows[n] != new_window:
            raise ValueError(
                f"cannot merge: N={n} was computed over "
                f"{windows[n][0]}..{windows[n][1]} but this run used "
                f"{new_window[0]}..{new_window[1]}. Different price windows are "
                f"different problem instances; rerun without --merge."
            )


def merge_rows(
    existing: list[dict],
    fresh: list[dict],
    key_fields: tuple[str, ...],
    sort_key,
) -> list[dict]:
    """Overlay ``fresh`` onto ``existing``, keyed by ``key_fields``.

    Rows of ``existing`` that this run did not recompute are carried through
    unchanged -- as the strings read from the file, so their digits round-trip
    exactly rather than through a float conversion.

    Args:
        existing: Rows already on disk.
        fresh: Rows produced by this run; these win on a key collision.
        key_fields: Fields that identify a row.
        sort_key: Ordering applied to the merged result.

    Returns:
        The merged rows in a deterministic order.
    """
    check_mergeable(existing, fresh)
    merged = {_row_key(r, key_fields): r for r in existing}
    for r in fresh:
        merged[_row_key(r, key_fields)] = r
    return sorted(merged.values(), key=sort_key)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper ① QUBO baseline experiment")
    p.add_argument("--backend", default="lightning_cpu",
                   choices=["lightning_cpu", "lightning_gpu", "braket_local",
                            "braket_sv1", "braket_ionq"],
                   help="Quantum backend (default: lightning_cpu, free/local)")
    p.add_argument("--precision", default="double", choices=["single", "double"],
                   help="Statevector precision; 'single' is required for N=30 on 48GB")
    p.add_argument("--classical-n", type=int, nargs="+", default=DEFAULT_CLASSICAL_N)
    p.add_argument("--quantum-n", type=int, nargs="+", default=DEFAULT_QUANTUM_N)
    p.add_argument("--p-layers", type=int, default=2)
    p.add_argument("--shots", type=int, default=1000,
                   help="Shots for final sampling. QPU backends bill per shot — keep small")
    p.add_argument("--steps", type=int, default=50, help="ADAM optimizer steps")
    p.add_argument("--s3-bucket", default=None, help="Required for braket_sv1/braket_ionq")
    p.add_argument("--sample-backend", default=None,
                   choices=["lightning_cpu", "lightning_gpu", "braket_local",
                            "braket_sv1", "braket_ionq"],
                   help="Run only the final sampling circuit here, optimizing on "
                        "--backend. This is the only way to reach a QPU: real "
                        "hardware cannot host the optimization loop (one billable "
                        "task per parameter-shift circuit, ~$73,000 at N=8)")
    p.add_argument("--results-tag", default=None,
                   help="Suffix the output CSVs (paper01_results_<tag>.csv). "
                        "Untagged writes the canonical files behind the "
                        "single-instance tables (Supplementary Table S2); "
                        "cloud backends must pass a tag")
    p.add_argument("--write-canonical", action="store_true",
                   help="Allow an untagged run to replace the committed "
                        "canonical CSVs when they already exist. Without it "
                        "such a run is refused, so a clone's baseline files "
                        "survive a trial invocation")
    p.add_argument("--skip-quantum", action="store_true",
                   help="Classical baseline only (fast, always free)")
    p.add_argument("--merge", action="store_true",
                   help="Overlay this run's rows onto the existing CSVs instead "
                        "of replacing them. Rows this run did not recompute are "
                        "kept verbatim — use it to refresh the classical "
                        "baseline without rerunning QAOA (N=30 is 3.14 h)")
    p.add_argument("--end-date", default=DATA_END_DATE,
                   help="Last day of the price window, YYYY-MM-DD. Defaults to "
                        "the pinned paper window; changing it changes the "
                        "problem instance and invalidates comparison with "
                        "previously published tables")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # A cloud run costs money and is a *check* of the published numbers, never
    # their source. Writing it to the canonical filenames would destroy the
    # baseline it is being compared against (the GPU sweep behind Table I took
    # nearly four hours), so require the run to name itself.
    # Refuse hardware in the optimizer slot before anything is constructed. The
    # solver enforces this too; doing it here as well means the CLI never even
    # builds an object that would have to be talked out of a $73,000 run.
    if args.backend in QPU_BACKENDS:
        print(f"--backend {args.backend} would run the 50-step optimization loop "
              f"on real hardware. QPUs cannot use adjoint differentiation, so "
              f"every step becomes one billable task per parameter-shift circuit "
              f"(~$73,000 at N=8). Optimize locally and pass "
              f"--sample-backend {args.backend} instead.")
        return 2

    cloud = {args.backend, args.sample_backend} - LOCAL_BACKENDS - {None}
    if cloud and args.results_tag is None:
        print(f"--results-tag is required for backend {sorted(cloud)[0]}: an "
              f"untagged run overwrites paper01_results.csv, the source of "
              f"Table I and Figures 1-4. Try --results-tag sv1.")
        return 2
    try:
        results_csv, convergence_csv = results_paths(args.results_tag)
    except ValueError as exc:
        print(exc)
        return 2
    refusal = canonical_overwrite_refusal(
        (results_csv, convergence_csv), args.results_tag, args.write_canonical)
    if refusal:
        print(refusal)
        return 2

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    loader = FinanceDataLoader()
    formulator = PortfolioQUBO()
    metrics = PortfolioMetrics(risk_free_rate=0.001)
    classical_solver = ClassicalSolver(method="auto")
    quantum_solver = BraketSolver(
        backend=args.backend,
        p_layers=args.p_layers,
        n_shots=args.shots,
        n_optimizer_steps=args.steps,
        s3_bucket=args.s3_bucket,
        precision=args.precision,
        sample_backend=args.sample_backend,
    )

    quantum_n = set() if args.skip_quantum else set(args.quantum_n)
    all_n = sorted(set(args.classical_n) | quantum_n)

    print(f"Backend: {args.backend} ({args.precision} precision), "
          f"p={args.p_layers}, shots={args.shots}, steps={args.steps}")
    print(f"Price window ends: {args.end_date} "
          f"({LOOKBACK_YEARS}y lookback, seed {SEED})")
    print(f"Classical N: {sorted(args.classical_n)}")
    print(f"Quantum   N: {sorted(quantum_n) or '(skipped)'}")

    # Snapshot the files once. The per-N flush below rewrites the output, so
    # merging has to overlay onto this snapshot rather than onto whatever the
    # previous flush left behind.
    prior_rows = read_csv(results_csv) if args.merge else []
    prior_convergence = read_csv(convergence_csv) if args.merge else []
    if args.merge:
        print(f"Merging into {len(prior_rows)} existing row(s) "
              f"and {len(prior_convergence)} convergence point(s)")

    def flush(rows: list[dict], convergence_rows: list[dict]) -> str | None:
        """Write the two CSVs, overlaying onto the snapshot when merging.

        Returns:
            An error message if the merge is unsafe (nothing is written in that
            case), otherwise None. The first flush happens after the first N, so
            an incompatible merge is reported in seconds rather than at the end.
        """
        try:
            merged_rows = (
                merge_rows(prior_rows, rows, RESULT_KEY, _result_sort_key)
                if args.merge else rows
            )
            # A classical-only merge run has no convergence rows; leave that
            # file alone rather than rewriting it from an empty overlay.
            merged_convergence = None
            if convergence_rows or not args.merge:
                merged_convergence = (
                    merge_rows(
                        prior_convergence, convergence_rows,
                        CONVERGENCE_KEY, _convergence_sort_key,
                    )
                    if args.merge else convergence_rows
                )
        except ValueError as exc:
            return str(exc)

        write_csv(results_csv, merged_rows)
        if merged_convergence is not None:
            write_csv(convergence_csv, merged_convergence)
        return None

    rows: list[dict] = []
    convergence_rows: list[dict] = []
    failures: list[tuple[int, str, str]] = []

    for n in all_n:
        print(f"\n{'='*60}\nN={n}\n{'='*60}")
        try:
            problem, data, k = build_problem(n, loader, formulator, args.end_date)
        except Exception as exc:
            print(f"  DATA ERROR for N={n}: {exc}")
            failures.append((n, "data", str(exc)))
            continue

        print(f"  QUBO {data.n_assets}x{data.n_assets}, K={k}, "
              f"penalty A={problem.metadata['penalty_strength']:.4f}")

        if n in set(args.classical_n):
            try:
                res = classical_solver.solve(problem, seed=SEED)
                rows.append(row_from_result(
                    res, data, problem, metrics, k, res.metadata.get("method", "")))
                print(f"    classical: E={res.energy_no_offset:.6f} "
                      f"t={res.runtime_seconds:.3f}s")
            except Exception as exc:
                traceback.print_exc()
                failures.append((n, "classical", str(exc)))

        if n in quantum_n:
            try:
                res = quantum_solver.solve(problem, seed=SEED)
                rows.append(row_from_result(
                    res, data, problem, metrics, k, f"p={args.p_layers}"))
                print(f"    quantum:   E={res.energy_no_offset:.6f} "
                      f"t={res.runtime_seconds:.3f}s")
                for step, energy in enumerate(res.metadata.get("convergence", [])):
                    convergence_rows.append({
                        "n_assets": data.n_assets,
                        "backend": res.backend,
                        "p_layers": args.p_layers,
                        "step": step,
                        "expectation": energy,
                    })
            except Exception as exc:
                traceback.print_exc()
                failures.append((n, "quantum", str(exc)))

        # Flush after every N. The sizes run last and longest -- N=30 alone is
        # over three hours on an A6000 -- so a crash or an OOM near the end of
        # the sweep would otherwise discard the whole run. Rewriting the two
        # CSVs each time is idempotent and costs nothing at these row counts;
        # the completed file is identical to a single write at the end.
        if rows:
            err = flush(rows, convergence_rows)
            if err:
                print(f"\n{err}")
                return 2

    if not rows:
        print("\nNo results produced.")
        return 1

    err = flush(rows, convergence_rows)
    if err:
        print(f"\n{err}")
        return 2
    print(f"\nSaved {len(rows)} rows → {results_csv}")
    if convergence_rows:
        print(f"Saved {len(convergence_rows)} convergence points → "
              f"{convergence_csv}")

    print("\n--- Summary ---")
    print(f"{'N':>5} {'K':>3} {'Solver':>16} {'Energy':>13} {'Runtime':>10} "
          f"{'Feas':>6} {'Sharpe':>8}")
    print("-" * 68)
    for r in rows:
        print(f"{r['n_assets']:>5} {r['n_select']:>3} {r['solver']:>16} "
              f"{r['energy_no_offset']:>13.4f} {r['runtime_s']:>10.3f} "
              f"{str(r['feasible']):>6} {r['sharpe_ratio']:>8.4f}")

    if failures:
        print(f"\n--- {len(failures)} failure(s) ---")
        for n, stage, msg in failures:
            print(f"  N={n} [{stage}]: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
