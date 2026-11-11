# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
E-S1: does the record catch a mix-up? Fault injection against the E1 record.

Principle P1 of the system paper says that a run record bound to both its
inputs (the pinned statistics, the penalty constant, the initial angles) and
its outputs (final angles, exact-state quantities, shots, batch summary) makes
a mix-up show up as an *inconsistency between records* that a machine can
check. This harness measures that claim: it takes the committed E1 record
(2,565 runs: ``paper01_e1_cross.csv``, ``paper01_e1_reference.csv``, the
per-run ``.npz`` artifacts and the derived statistics in ``data/derived/``),
injects one fault at a time into an in-memory copy, runs every check, and
records which check fails first, at which stage, and whether the headline
numbers of the study would have moved without anyone noticing.

Checks (``CHECKS``; each is one verification relation between records)
-----------------------------------------------------------------------
  schema           required columns present and filled on every ``ok`` row
  manifest         each (N, instance) block holds exactly the (penalty, seed)
                   cells its reference row prescribes; run ids unique; one
                   window and one spec version per file
  penalty          the penalty value is the one its label names (A_heur,
                   A_margin, m·A_crit) and ``a_over_crit`` is their ratio
  reference-row    the instance quantities copied onto each run row (f*, Δ_F,
                   |F|, A_crit, ...) equal the reference row's
  input-digest     the ``input_digest`` on the reference and on every run row
                   (a content hash of μ and Σ, added 2026-09-16 after the first
                   pass of this harness found the instance digest covers only
                   the ticker set) equals the digest of the pinned statistics
  reference-input  the reference row is what the pinned statistics give when
                   the feasible set is enumerated again (a clone's situation:
                   derived statistics only, no price snapshot)
  scorer           the solver's best shot, its energy and the scorer's f/V
                   decomposition agree; g_F and g_off follow from f, f*, Δ_F, A
  row-npz          the row and its artifact describe the same run: angles,
                   penalty, P_F and P_opt from the stored probabilities, shot
                   totals, feasible shots from the stored shot indices
  npz-internal     the artifact is self-consistent: trajectory starts at the
                   initial angles, convergence ends at the final expectation,
                   shots sum to the batch size, feasible probabilities sum ≤ 1
  identities       columns that are functions of other recorded values are
                   recomputed: Q_τ(S) from P_τ, P_τ and P_opt from the stored
                   probabilities and f, the rank of the best feasible shot,
                   D and |E*|, the best-expectation columns from the stored
                   convergence, the reference row's uniform baselines, and the
                   per-instance constancy of the optimum's Sharpe ratio
  state-batch      the batch is a draw from the *final* state: per run the
                   feasible-shot and optimum-shot counts have a one-sided
                   binomial tail ≥ 1e-9 under S × final P_F / P_opt, and
                   per (N, penalty) block the feasible fraction tracks final
                   P_F more closely than initial P_F (the E3 catch)
  seed-distinct    within an instance and penalty, different seeds have
                   different initial angles and different shot batches (the
                   ``shot_seed`` catch)
  state-recompute  (expensive, sampled: the reference instance's A_heur seed-42
                   run and four spaced runs per N) the state re-simulated at the
                   recorded final angles reproduces the stored probabilities

Faults (``FAULTS``; each a mix-up that has happened here or could)
---------------------------------------------------------------------
  price_perturbation   one asset's mean return in the derived statistics moved
                       by 1e-6 relative (a revised snapshot; cf. the 4.2e-7
                       yfinance revision of 2026-08)
  window_mixin         one run row of the prior-window replication (E5) pasted
                       into the main file
  optimizer_mixin      one run row of the E3 L-BFGS-B arm pasted into the main
                       file
  label_swap           an A_margin row relabelled A_heur (a setting mix-up)
  angle_swap           initial and final angles exchanged in row and artifact
  batch_from_initial   every batch summary replaced by a draw from the initial
                       state (the E3 L-BFGS-B wrapper defect)
  evaluator_from_Q     f of the best feasible shot replaced by its penalised
                       energy (an evaluator fed the QUBO matrix)
  column_drop          the ``final_P_F`` column missing (the E4 write_csv
                       defect)
  shot_seed_noop       seed 43's shot batch replaced by seed 42's (the
                       ``shot_seed`` defect: the same shots re-drawn)
  reference_stale      the reference file's f* for one instance from a
                       different penalty scale (a stale regeneration)

Controls
--------
``naive`` is a pipeline that reads the CSV and computes the headline numbers
with no check but the schema; ``relations`` is the full set. For every fault
the harness reports whether the headline numbers moved (so the fault would
have changed a conclusion) and which configuration caught it.

Nothing is written to the committed files: faults act on copies in memory or
in a temporary directory, and the harness's own first pass is the clean
record, which must pass every check (a failing clean check is a false positive
and is reported as such).

Run:
    python -m experiments.system_paper.es1_fault_injection                  # ~3 min
    python -m experiments.system_paper.es1_fault_injection --sizes 12 --no-recompute
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from experiments.paper01_qubo_baseline import run_e1_cross as e1
from experiments.paper01_qubo_baseline.run_experiment import RISK_AVERSION, write_csv
from experiments.paper01_qubo_baseline.run_instances import input_digest, instance_tickers
from src.finance.data_loader import FinanceDataLoader
from src.qubo import scoring
from src.qubo.portfolio import PortfolioQUBO
from src import record_checks

RESULTS_DIR = Path(__file__).parent / "results"
E1_RESULTS = e1.RESULTS_DIR
MATRIX_CSV = RESULTS_DIR / "paper02_es1_detection_matrix.csv"
FINDINGS_CSV = RESULTS_DIR / "paper02_es1_findings.csv"
HEADLINE_CSV = RESULTS_DIR / "paper02_es1_headline_shift.csv"
COUNTS_CSV = RESULTS_DIR / "paper02_es1_finding_counts.csv"
CLEAN_CSV = RESULTS_DIR / "paper02_es1_clean_records.csv"

SIZES = (12, 16, 20)
#: The record-only relations, their threshold and their required columns are
#: shared with the aggregation, which runs them before it reads a number.
TAIL_P = record_checks.TAIL_P
REQUIRED = record_checks.REQUIRED
NPZ_KEYS = ("initial_angles", "final_angles", "angles_trajectory", "convergence",
            "feasible_indices", "feasible_f", "final_probs_feasible",
            "initial_probs_feasible", "shot_indices", "shot_counts", "penalty",
            "hamiltonian_scale")


# ----------------------------------------------------------------------
# The record
# ----------------------------------------------------------------------

@dataclass
class Record:
    """The E1 record as the checks see it: rows, reference, artifacts, inputs."""
    rows: pd.DataFrame
    reference: pd.DataFrame
    artifacts: dict[str, dict[str, np.ndarray]]
    derived_dir: Path
    label: str = "clean"
    end_date: str = e1.DATA_END_DATE
    runs_expected: dict[int, int] | None = None

    def copy(self, label: str) -> "Record":
        return Record(self.rows.copy(deep=True), self.reference.copy(deep=True),
                      copy.deepcopy(self.artifacts), self.derived_dir, label,
                      self.end_date, self.runs_expected)


def load_record(sizes=SIZES, results_dir: Path = E1_RESULTS, suffix: str = "",
                label: str = "clean") -> Record:
    """The main E1 record, or one of its sibling arms (``suffix`` = ``_e5`` etc.).

    The window comes from the record itself, so the prior-window replication is
    checked against its own pinned statistics; the planned run count comes from
    the manifest the run wrote, so the manifest relation is checked against the
    arm's own specification rather than against E1's.
    """
    rows = pd.read_csv(results_dir / f"paper01_e1_cross{suffix}.csv", float_precision="round_trip")
    rows = rows[rows.n_assets.isin(sizes)].reset_index(drop=True)
    ref = pd.read_csv(results_dir / f"paper01_e1_reference{suffix}.csv", float_precision="round_trip")
    ref = ref[ref.n_assets.isin(sizes)].reset_index(drop=True)
    artifacts = {}
    for run_id in rows.run_id:
        with np.load(results_dir / f"e1_runs{suffix}" / f"{run_id}.npz") as z:
            artifacts[run_id] = {k: np.array(z[k]) for k in NPZ_KEYS}
    end_date = str(rows.data_end.iloc[0]) if "data_end" in rows and len(rows) else e1.DATA_END_DATE
    planned = None
    man = results_dir / f"paper01_e1_manifest{suffix}.csv"
    if man.exists():
        m = pd.read_csv(man)
        planned = {int(r.n_assets): int(r.runs_expected) for _, r in m.iterrows()
                   if int(r.n_assets) in sizes}
    return Record(rows, ref, artifacts, FinanceDataLoader.DERIVED_DIR, label, end_date, planned)


def _angles(text: str) -> np.ndarray:
    return np.array([float(v) for v in str(text).split()])


def _fmt_angles(a: np.ndarray) -> str:
    return " ".join(f"{v:.17g}" for v in a)


# ----------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------

def check_schema(rec: Record) -> list[str]:
    """One expression per fact, and it lives in ``src/record_checks.py``."""
    return record_checks.check_schema(rec.rows, rec.reference)


def check_manifest(rec: Record) -> list[str]:
    out = []
    rows = rec.rows
    if rows.run_id.duplicated().any():
        out.append(f"duplicate run_id: {rows.run_id[rows.run_id.duplicated()].tolist()[:3]}")
    keys = rows[["n_assets", "instance", "penalty_label", "qaoa_seed"]]
    if keys.duplicated().any():
        out.append(f"duplicate (N, instance, penalty, seed) cells: {int(keys.duplicated().sum())}")
    for c in ("data_start", "data_end", "spec_version", "optimizer", "backend", "precision",
              "p_layers", "steps", "shots", "stepsize"):
        if c in rows and rows[c].nunique() > 1:
            out.append(f"{c} not uniform: {rows[c].unique().tolist()}")
    if "status" in rows:
        bad = rows[~(rows.status.eq("ok") | rows.status.astype(str).str.startswith("error:"))]
        for _, r in bad.iterrows():
            out.append(f"{r.run_id}: status {r.status!r} is neither ok nor error:<type>")
        if "error" in rows:
            bad = rows[rows.status.eq("ok") & rows.error.notna()]
            for _, r in bad.iterrows():
                out.append(f"{r.run_id}: status ok but error recorded")
    seeds = sorted(rows.qaoa_seed.unique())
    ran = set(rows.penalty_label.unique())
    for _, r in rec.reference.iterrows():
        block = rows[(rows.n_assets == r.n_assets) & (rows.instance == r.instance)]
        if r.a_crit > 0:
            labels = [lab for _, lab in e1.MULTIPLES] + ["Aheur", "Amargin"]
        else:
            labels = ["A0", "Aheur", "Amargin"]
        # An arm may run a subset of the penalty grid (E5 and the E3 arms run
        # two of the six); the relation is that every block holds the same grid,
        # and the manifest's own run count says the file is whole.
        labels = [lab for lab in labels if lab in ran]
        expected = {(lab, s) for lab in labels for s in seeds}
        got = set(zip(block.penalty_label, block.qaoa_seed))
        if got != expected:
            out.append(f"N={r.n_assets} inst {r.instance}: cells {sorted(got ^ expected)[:4]} "
                       f"{'extra' if got - expected else 'missing'}")
    for n, want in (rec.runs_expected or {}).items():
        got_n = int((rows.n_assets == n).sum())
        if got_n != want:
            out.append(f"N={n}: {got_n} rows against the manifest's {want} planned")
    return out


def check_penalty(rec: Record) -> list[str]:
    """One expression per fact, and it lives in ``src/record_checks.py``."""
    return record_checks.check_penalty(rec.rows, rec.reference)


def check_reference_row(rec: Record) -> list[str]:
    """One expression per fact, and it lives in ``src/record_checks.py``."""
    return record_checks.check_reference_row(rec.rows, rec.reference)


def _reference_from_inputs(n: int, instance: int, derived_dir: Path,
                           end_date: str = e1.DATA_END_DATE) -> dict:
    """Rebuild the reference row from the derived statistics alone."""
    with _loader_dirs(cache_dir=None, derived_dir=derived_dir):
        loader = FinanceDataLoader()
        ref = e1.InstanceReference(n, instance, loader, PortfolioQUBO(), end_date)
        return ref.row()


@contextlib.contextmanager
def _loader_dirs(cache_dir: Path | None, derived_dir: Path) -> Iterator[None]:
    """Point the loader at a price cache (None = an empty one) and derived dir."""
    old = FinanceDataLoader.CACHE_DIR, FinanceDataLoader.DERIVED_DIR
    tmp = None
    try:
        if cache_dir is None:
            tmp = Path(tempfile.mkdtemp(prefix="es1_nocache_"))
            cache_dir = tmp
        FinanceDataLoader.CACHE_DIR = cache_dir
        FinanceDataLoader.DERIVED_DIR = derived_dir
        yield
    finally:
        FinanceDataLoader.CACHE_DIR, FinanceDataLoader.DERIVED_DIR = old
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


def check_reference_input(rec: Record) -> list[str]:
    out = []
    for _, q in rec.reference.iterrows():
        fresh = _reference_from_inputs(int(q.n_assets), int(q.instance), rec.derived_dir,
                                       rec.end_date)
        for col in ("f_star", "delta_F", "feasible_set_size", "a_crit", "a_heur",
                    "a_margin", "instance_digest", "input_digest"):
            a, b = q[col], fresh[col]
            same = (a == b) if isinstance(a, str) else np.isclose(a, b, rtol=0, atol=0)
            if not same:
                out.append(f"N={q.n_assets} inst {q.instance}: reference {col} {a!r} "
                           f"!= recomputed {b!r}")
    return out


def check_input_digest(rec: Record) -> list[str]:
    """One expression per fact, and it lives in ``src/record_checks.py``."""
    return record_checks.check_input_digest(rec.rows, rec.reference, rec.derived_dir, rec.end_date)


def check_scorer(rec: Record) -> list[str]:
    """One expression per fact, and it lives in ``src/record_checks.py``."""
    return record_checks.check_scorer(rec.rows, rec.reference)


def check_row_npz(rec: Record) -> list[str]:
    out = []
    for _, r in rec.rows.iterrows():
        z = rec.artifacts.get(r.run_id)
        if z is None:
            out.append(f"{r.run_id}: no artifact")
            continue
        if not np.array_equal(_angles(r.initial_angles), z["initial_angles"]):
            out.append(f"{r.run_id}: initial_angles differ between row and npz")
        if not np.array_equal(_angles(r.final_angles), z["final_angles"]):
            out.append(f"{r.run_id}: final_angles differ between row and npz")
        if float(z["penalty"]) != r.penalty:
            out.append(f"{r.run_id}: penalty differs between row and npz")
        for prefix in ("final", "initial"):
            value = r.get(f"{prefix}_P_F")
            if value is None:
                out.append(f"{r.run_id}: {prefix}_P_F absent from the row; the stored "
                           f"probabilities give {z[f'{prefix}_probs_feasible'].sum()!r}")
            elif not np.isclose(z[f"{prefix}_probs_feasible"].sum(), value, rtol=1e-12, atol=1e-15):
                out.append(f"{r.run_id}: {prefix}_P_F {value!r} != sum of stored probabilities")
        if int(z["shot_counts"].sum()) != int(r.shots):
            out.append(f"{r.run_id}: shots {r.shots} != stored counts {int(z['shot_counts'].sum())}")
        feasible = np.isin(z["shot_indices"], z["feasible_indices"])
        if int(z["shot_counts"][feasible].sum()) != int(r.feasible_shots):
            out.append(f"{r.run_id}: feasible_shots {r.feasible_shots} != stored "
                       f"{int(z['shot_counts'][feasible].sum())}")
        if int(r.best_all_index) not in set(z["shot_indices"].tolist()):
            out.append(f"{r.run_id}: best_all_index not among the stored shots")
        if pd.notna(r.get("f_best_feasible", np.nan)):
            pos = np.flatnonzero(z["feasible_indices"] == int(r.best_feasible_index))
            if pos.size != 1 or not np.isclose(z["feasible_f"][pos[0]], r.f_best_feasible, rtol=0, atol=0):
                out.append(f"{r.run_id}: f_best_feasible is not the stored f of its index")
    return out


def check_npz_internal(rec: Record) -> list[str]:
    out = []
    for _, r in rec.rows.iterrows():
        z = rec.artifacts.get(r.run_id)
        if z is None:
            continue
        if not np.array_equal(z["angles_trajectory"][0], z["initial_angles"]):
            out.append(f"{r.run_id}: trajectory does not start at the initial angles")
        if np.array_equal(z["angles_trajectory"][0], z["final_angles"]):
            out.append(f"{r.run_id}: final angles equal the trajectory's first point")
        if not np.isclose(z["convergence"][-1], r.final_expectation, rtol=1e-12, atol=0):
            out.append(f"{r.run_id}: convergence does not end at final_expectation")
        if z["final_probs_feasible"].sum() > 1 + 1e-9 or (z["final_probs_feasible"] < 0).any():
            out.append(f"{r.run_id}: stored probabilities are not a sub-distribution")
        if float(z["hamiltonian_scale"]) != r.hamiltonian_scale:
            out.append(f"{r.run_id}: hamiltonian_scale differs between row and npz")
    return out


def check_identities(rec: Record) -> list[str]:
    """Columns that are functions of other recorded values are recomputed.

    Added after the column-coverage scan of ``es1_random_faults.py`` (2026-09-16)
    listed the run-row and reference columns no relation bound: the Q_τ(S)
    columns (a function of P_τ and S), P_opt / P_τ (sums of the stored
    probabilities over the feasible set's f), the rank of the best feasible
    shot, the deflation factor and |E*|, the best-expectation columns (from
    the stored convergence), the feasible-set size, the uniform-sampler
    baselines of the reference row, and the per-instance constancy of the
    Sharpe ratio of the optimum. Each is an identity; a tolerance of 1e-9
    relative covers the libm ulp differences the numerical paper reports.
    """
    out = []
    rtol = 1e-9
    def close(a, b):
        return np.isclose(float(a), float(b), rtol=rtol, atol=1e-15)
    ref = rec.reference
    N = ref.n_assets.astype(int)
    checks = {
        "P_F_uniform_all": ref.feasible_set_size / 2.0 ** N,
        "P_tau_main_uniform_all": ref.count_within_tau_main / 2.0 ** N,
        "P_tau_main_uniform_F": ref.count_within_tau_main / ref.feasible_set_size,
        "P_opt_uniform_F": ref.n_optimal_ties / ref.feasible_set_size,
        "P_opt_uniform_all": ref.n_optimal_ties / 2.0 ** N,
        "delta_F": ref.f_max_feasible - ref.f_star,
    }
    for col, want in checks.items():
        bad = ref.index[~np.isclose(ref[col].astype(float), want.astype(float), rtol=rtol, atol=1e-15)]
        for i in bad:
            out.append(f"reference N={ref.at[i, 'n_assets']} inst {ref.at[i, 'instance']}: {col} "
                       f"{ref.at[i, col]!r} != identity {want[i]!r}")
    pos = ref[ref.a_crit > 0]
    bad = pos.index[~np.isclose(pos.a_heur_over_crit.astype(float), pos.a_heur / pos.a_crit, rtol=rtol)]
    for i in bad:
        out.append(f"reference N={ref.at[i, 'n_assets']} inst {ref.at[i, 'instance']}: a_heur_over_crit inconsistent")
    # The reference row against itself and against one artifact of its instance.
    from experiments.paper01_qubo_baseline.run_instances import instance_digest, instance_tickers
    by_inst = {k: v for k, v in rec.rows.groupby(["n_assets", "instance"])}
    if "eps_A" in ref and ref.eps_A.nunique() > 1:
        out.append(f"reference: eps_A not uniform: {ref.eps_A.unique().tolist()}")
    for i, q in ref.iterrows():
        if bool(q.a_crit_is_zero) != (float(q.a_crit) == 0.0):
            out.append(f"reference N={q.n_assets} inst {q.instance}: a_crit_is_zero disagrees with a_crit")
        if isinstance(q.cardinality_minima, str) and q.cardinality_minima.strip():
            a_m = np.array([float(v) for v in q.cardinality_minima.split()])
            a_crit, binding = scoring.critical_penalty(a_m, int(q.n_select))
            a_margin, target = scoring.margin_penalty(a_m, int(q.n_select), float(q.f_bar_feasible), float(q.eps_A))
            if not close(q.a_crit, a_crit) or int(q.a_crit_binding_m) != int(binding):
                out.append(f"reference N={q.n_assets} inst {q.instance}: a_crit/binding m disagree with cardinality_minima")
            if not close(q.a_margin, a_margin) or not close(q.a_margin_target, target):
                out.append(f"reference N={q.n_assets} inst {q.instance}: a_margin/target disagree with cardinality_minima")
        tickers = str(q.tickers).split()
        if instance_digest(tickers) != q.instance_digest:
            out.append(f"reference N={q.n_assets} inst {q.instance}: tickers do not hash to instance_digest")
        if tickers != instance_tickers(int(q.n_assets), int(q.instance)):
            out.append(f"reference N={q.n_assets} inst {q.instance}: tickers are not instance {q.instance}'s")
        block = by_inst.get((q.n_assets, q.instance))
        if block is not None:
            for col in ("data_start", "data_end"):
                if col in block and (block[col] != q[col]).any():
                    out.append(f"reference N={q.n_assets} inst {q.instance}: {col} differs from its runs")
            z = rec.artifacts.get(block.run_id.iloc[0])
            if z is not None and z["feasible_f"].size:
                f = z["feasible_f"]
                if not close(q.f_star, f.min()) or not close(q.f_max_feasible, f.max()) \
                        or not close(q.f_bar_feasible, f.mean()):
                    out.append(f"reference N={q.n_assets} inst {q.instance}: f*/f_max/f_bar disagree with the stored f")
                if int(q.optimum_index) != int(z["feasible_indices"][int(np.argmin(f))]):
                    out.append(f"reference N={q.n_assets} inst {q.instance}: optimum_index is not the argmin of the stored f")
                if q.delta_F > 0:
                    g = (f - q.f_star) / q.delta_F
                    if int(q.count_within_tau_main) != int((g <= e1.TAU_MAIN).sum()) \
                            or int(q.count_within_tau_alt) != int((g <= e1.TAU_ALT).sum()):
                        out.append(f"reference N={q.n_assets} inst {q.instance}: count_within_tau disagrees with the stored f")

    rows = rec.rows
    for prefix in ("final", "initial"):
        for s_ in (10, 100, 1000, 10000):
            for base, q in ((f"{prefix}_P_tau_main", f"{prefix}_Q_tau_main_{s_}"),
                            (f"{prefix}_P_opt", f"{prefix}_Q_opt_{s_}")):
                if base not in rows:
                    continue
                if q not in rows:
                    out.append(f"column {q} missing: it is a function of {base} and S = {s_}")
                    continue
                want = rows[base].map(lambda p_: scoring.success_probability(p_, s_))
                # Absolute 1e-12: the committed E1 columns were written before
                # success_probability moved to expm1/log1p, and the naive
                # 1-(1-P)^S loses up to 2.8e-13 absolute at small P (the
                # numerical paper's Q_tau note); a mutation of 1e-9 relative
                # on a P of order 1e-3 or more is still far outside it.
                bad = rows.index[~np.isclose(rows[q].astype(float), want, rtol=rtol, atol=1e-12)]
                for i in bad:
                    out.append(f"{rows.at[i, 'run_id']}: {q} {rows.at[i, q]!r} != 1-(1-{base})^S = {want[i]!r}")
    for i, r in rows.iterrows():
        k = int(r.n_select)
        if not close(r.deflation_D, scoring.deflation_factor(r.f_star, r.penalty, k, r.delta_F)):
            out.append(f"{r.run_id}: deflation_D is not |f*-Ak^2|/Delta_F")
        if not close(r.abs_E_star, abs(r.f_star - r.penalty * k * k)):
            out.append(f"{r.run_id}: abs_E_star is not |f*-Ak^2|")
        z = rec.artifacts.get(r.run_id)
        if z is None:
            continue
        conv = z["convergence"]
        if not close(r.best_expectation, conv.min()) or int(r.best_expectation_step) != int(conv.argmin()):
            out.append(f"{r.run_id}: best_expectation/step disagree with the stored convergence")
        # Empty on 2,555 of 2,565 E1 rows: the column arrived with the E3
        # revision of the driver and only the ten N = 12 rows re-run after it
        # carry it (same spec version, newer code -- a schema drift the
        # manifest check does not see; noted in the system paper).
        # A record written before the column existed (the pilot) simply does not
        # carry it; absence is the schema relation's business, not this one's.
        if "best_expectation_by_50" in r.index and pd.notna(r.best_expectation_by_50) \
                and not close(r.best_expectation_by_50, conv[:50].min()):
            out.append(f"{r.run_id}: best_expectation_by_50 disagrees with the stored convergence")
        if int(r.feasible_set_size) != int(z["feasible_indices"].size):
            out.append(f"{r.run_id}: feasible_set_size != stored feasible set")
        if int(r.distinct_shots) != int(z["shot_indices"].size):
            out.append(f"{r.run_id}: distinct_shots != stored shot indices")
        if not close(r.feasible_shot_fraction, r.feasible_shots / r.shots):
            out.append(f"{r.run_id}: feasible_shot_fraction != feasible_shots/shots")
        if bool(r.best_all_feasible) != bool(int(r.best_all_cardinality) == k):
            out.append(f"{r.run_id}: best_all_feasible disagrees with best_all_cardinality")
        for prefix in ("final", "initial"):
            # Single-precision state vectors: the recorded sums sit within
            # 7e-6 of 1 across the cross; 1e-4 flags a moved cell, not the arithmetic.
            if abs(float(r[f"{prefix}_probs_sum"]) - 1.0) > 1e-4:
                out.append(f"{r.run_id}: {prefix}_probs_sum {r[f'{prefix}_probs_sum']!r} is not 1")
        # Shot-level successes from the stored batch and the stored f.
        f = z["feasible_f"]
        idx, cnt = z["shot_indices"], z["shot_counts"]
        pos = {int(v): j for j, v in enumerate(z["feasible_indices"])}
        feas_j = np.array([pos.get(int(v), -1) for v in idx])
        m = feas_j >= 0
        if r.delta_F > 0:
            f_sh = f[feas_j[m]]
            want_opt = int(cnt[m][(f_sh - r.f_star) <= e1.OPT_TOL * r.delta_F].sum())
            want_tau = int(cnt[m][(f_sh - r.f_star) / r.delta_F <= e1.TAU_MAIN].sum())
            if int(r.shots_optimal) != want_opt or int(r.shots_within_tau_main) != want_tau:
                out.append(f"{r.run_id}: shots_optimal/shots_within_tau_main "
                           f"{int(r.shots_optimal)}/{int(r.shots_within_tau_main)} != stored batch "
                           f"{want_opt}/{want_tau}")
        f = z["feasible_f"]
        if r.delta_F > 0:
            g = (f - r.f_star) / r.delta_F
            for prefix in ("final", "initial"):
                pf = z[f"{prefix}_probs_feasible"]
                want = {f"{prefix}_P_tau_main": pf[g <= e1.TAU_MAIN].sum(),
                        f"{prefix}_P_tau_alt": pf[g <= e1.TAU_ALT].sum(),
                        f"{prefix}_P_opt": pf[(f - r.f_star) <= e1.OPT_TOL * r.delta_F].sum()}
                for col, w in want.items():
                    if not close(r[col], w):
                        out.append(f"{r.run_id}: {col} {r[col]!r} != sum of stored probabilities {w!r}")
        if pd.notna(r.get("f_best_feasible", np.nan)):
            f_sorted = np.sort(f)
            lo = int(np.searchsorted(f_sorted, r.f_best_feasible, side="left"))
            hi = int(np.searchsorted(f_sorted, r.f_best_feasible, side="right"))
            if float(r.rank_best_feasible) != lo + 1 or float(r.rank_ties) != hi - lo:
                out.append(f"{r.run_id}: rank {r.rank_best_feasible}/{r.rank_ties} != {lo + 1}/{hi - lo} from stored f")
            if bool(r.is_optimal) != bool(r.f_best_feasible - r.f_star <= e1.OPT_TOL * r.delta_F):
                out.append(f"{r.run_id}: is_optimal disagrees with f_best_feasible")
            if bool(r.within_tau_main) != bool(r.g_F <= e1.TAU_MAIN) or bool(r.within_tau_alt) != bool(r.g_F <= e1.TAU_ALT):
                out.append(f"{r.run_id}: within_tau disagrees with g_F")
    for (n, inst), block in rows.groupby(["n_assets", "instance"]):
        if block.sharpe_optimum.nunique() > 1:
            out.append(f"N={n} inst {inst}: sharpe_optimum is not constant across the instance's runs")
    # Each finding names what its clause read, because the check is one name
    # over three kinds of record: a column recomputed from other columns of the
    # same table, a column compared with the stored artifact, or a portfolio
    # metric recomputed from the pinned statistics. The tier count of
    # es1_tiers.py places a finding by that prefix and not by the check's name.
    return ([_clause_source(m) for m in out]
            + ["[input] " + m for m in _portfolio_metric_identities(rec)])


#: Substrings that mark an identities finding as a comparison with the artifact.
_ARTIFACT_WORDS = ("stored f", "stored convergence", "stored feasible set", "stored shot indices",
                   "stored batch", "stored probabilities")


def _clause_source(msg: str) -> str:
    return ("[artifact] " if any(w in msg for w in _ARTIFACT_WORDS) else "[table] ") + msg


def _portfolio_metric_identities(rec: Record) -> list[str]:
    """Sharpe, return and volatility of the best feasible shot, and the
    Sharpe of the optimum, recomputed from the pinned statistics."""
    from experiments.paper01_qubo_baseline.run_degeneracy import RISK_FREE_RATE
    from src.finance.metrics import PortfolioMetrics
    out = []
    metrics = PortfolioMetrics(risk_free_rate=RISK_FREE_RATE)
    with _loader_dirs(cache_dir=None, derived_dir=rec.derived_dir):
        loader = FinanceDataLoader()
        for (n, inst), block in rec.rows.groupby(["n_assets", "instance"]):
            data = loader.load(instance_tickers(int(n), int(inst)), end_date=rec.end_date,
                               lookback_years=e1.LOOKBACK_YEARS)
            k = int(block.n_select.iloc[0])
            cache: dict[int, tuple[float, float, float]] = {}
            def stats(index: int):
                if index not in cache:
                    x = scoring.bitstrings_from_indices([index], int(n))[0]
                    st = metrics.evaluate([i for i in range(int(n)) if x[i] > 0.5],
                                          data.returns, data.covariance, target_k=k)
                    cache[index] = (st.sharpe_ratio, st.expected_return, st.volatility)
                return cache[index]
            z0 = rec.artifacts.get(block.run_id.iloc[0])
            if z0 is not None:
                opt_index = int(z0["feasible_indices"][int(np.argmin(z0["feasible_f"]))])
                sh_opt = stats(opt_index)[0]
                bad = block[~np.isclose(block.sharpe_optimum.astype(float), sh_opt, rtol=1e-9, atol=1e-12)]
                for _, r in bad.iterrows():
                    out.append(f"{r.run_id}: sharpe_optimum {r.sharpe_optimum!r} != recomputed {sh_opt!r}")
            for _, r in block[block.best_feasible_index.notna()].iterrows():
                sh, ret, vol = stats(int(r.best_feasible_index))
                for col, want in (("sharpe_best_feasible", sh), ("return_best_feasible", ret),
                                  ("volatility_best_feasible", vol)):
                    if not np.isclose(float(r[col]), want, rtol=1e-9, atol=1e-12):
                        out.append(f"{r.run_id}: {col} {r[col]!r} != recomputed {want!r}")
    return out


def check_state_batch(rec: Record) -> list[str]:
    """One expression per fact, and it lives in ``src/record_checks.py``."""
    return record_checks.check_state_batch(rec.rows, rec.reference)


def check_seed_distinct(rec: Record) -> list[str]:
    out = []
    for (n, inst, lab), block in rec.rows.groupby(["n_assets", "instance", "penalty_label"]):
        if len(block) < 2:
            continue
        angles = block.initial_angles.tolist()
        if len(set(angles)) < len(angles):
            out.append(f"N={n} inst {inst} {lab}: two seeds share initial angles")
        batches = []
        for run_id in block.run_id:
            z = rec.artifacts.get(run_id)
            if z is not None:
                batches.append((z["shot_indices"].tobytes(), z["shot_counts"].tobytes()))
        if len(set(batches)) < len(batches):
            out.append(f"N={n} inst {inst} {lab}: two seeds share an identical shot batch")
    return out


def check_state_recompute(rec: Record, per_size: int = 4) -> list[str]:
    """Re-simulate a sample of runs at their recorded final angles."""
    from src.solvers.braket_solver import BraketSolver
    out = []
    formulator = PortfolioQUBO()
    with _loader_dirs(cache_dir=None, derived_dir=rec.derived_dir):
        loader = FinanceDataLoader()
        for n, block in rec.rows.groupby("n_assets"):
            # The reference instance's A_heur seed-42 run always (the run the
            # paper quotes), then a spaced sample of the block.
            anchor = block[(block.instance == 0) & (block.penalty_label == "Aheur")
                           & (block.qaoa_seed == 42)]
            spaced = block.sort_values("run_id").iloc[:: max(1, len(block) // per_size)][:per_size]
            sample = pd.concat([anchor, spaced]).drop_duplicates("run_id")
            for _, r in sample.iterrows():
                z = rec.artifacts[r.run_id]
                data = loader.load(instance_tickers(int(n), int(r.instance)),
                                   end_date=rec.end_date, lookback_years=e1.LOOKBACK_YEARS)
                problem = formulator.formulate(data.returns, data.covariance,
                                               num_select=int(r.n_select),
                                               risk_aversion=RISK_AVERSION,
                                               penalty_strength=float(r.penalty))
                solver = BraketSolver(backend="lightning_cpu", p_layers=int(r.p_layers),
                                      n_shots=1, precision="double")
                res = solver.evaluate_fixed_angles(problem, z["final_angles"], batches=0)
                pf = res["probabilities"][z["feasible_indices"]]
                # E1 ran in single precision on the GPU; double on the CPU
                # agrees to ~1e-6 in probability. A swapped state is O(1) off.
                gap = float(np.abs(pf - z["final_probs_feasible"]).max())
                if gap > 1e-5:
                    out.append(f"{r.run_id}: state at the recorded final angles is {gap:.2e} "
                               f"from the stored probabilities")
    return out


@dataclass
class Check:
    name: str
    stage: str
    fn: Callable[[Record], list[str]]
    expensive: bool = False


CHECKS = [
    Check("schema", "file", check_schema),
    Check("manifest", "file", check_manifest),
    Check("penalty", "row", check_penalty),
    Check("reference-row", "row", check_reference_row),
    Check("input-digest", "input", check_input_digest),
    Check("reference-input", "input", check_reference_input, expensive=True),
    Check("scorer", "row", check_scorer),
    Check("row-npz", "artifact", check_row_npz),
    Check("npz-internal", "artifact", check_npz_internal),
    Check("identities", "row", check_identities),
    Check("state-batch", "statistical", check_state_batch),
    Check("seed-distinct", "block", check_seed_distinct),
    Check("state-recompute", "recompute", check_state_recompute, expensive=True),
]
NAIVE = {"schema"}

# ----------------------------------------------------------------------
# The declarative control
# ----------------------------------------------------------------------
#
# A reader who validates the record with a data-quality library (Deequ,
# Great Expectations, pandera) writes constraints of a fixed vocabulary:
# presence and type, completeness, the expected number of rows, a set of
# admitted values, uniqueness of a key or of a tuple of columns, a column that
# must hold one value, a numeric range, an inequality between two columns of the
# same row, a conditional equality within a row (Deequ's ``satisfies``: where
# this column says X, that column must equal this other one), and — after a join
# — the equality of a copied column with the table it was copied from.
# What it does not include is an expectation on an aggregate of the data (a mean
# within a band), because in this record the aggregates are the results being
# checked. None of these recomputes a
# quantity from a stored artifact, hashes an external input, or compares a row
# with a statistical model of how it was produced.
#
# ``check_declarative`` is that vocabulary, applied generously (the join with
# the reference table is included), so that the thirteen relations are measured
# against current practice rather than against a pipeline with no checks at all.

PROB_COLS = ["final_P_F", "final_P_opt", "final_P_tau_main", "final_P_tau_alt",
             "initial_P_F", "initial_P_opt", "initial_P_tau_main", "initial_P_tau_alt",
             "feasible_shot_fraction", "final_Q_tau_main_1000", "final_Q_opt_1000",
             "initial_Q_tau_main_1000", "initial_Q_opt_1000"]
COUNT_COLS = ["feasible_shots", "distinct_shots", "shots_within_tau_main", "shots_optimal"]
ONE_VALUE_COLS = ["data_start", "data_end", "spec_version", "optimizer", "backend",
                  "precision", "p_layers", "steps", "shots", "stepsize"]
LABELS = {"A0", "Aheur", "Amargin"} | {lab for _, lab in e1.MULTIPLES}
DECLARATIVE_FAMILIES = ("schema", "size", "dtype", "set", "unique", "single", "range",
                        "pair", "band", "conditional", "fk")


def check_declarative(rec: Record) -> list[str]:
    """What a data-quality library can express, in eight constraint families."""
    out: list[str] = []
    rows, ref = rec.rows, rec.reference
    ok = rows[rows.status == "ok"] if "status" in rows else rows

    def add(family: str, msg: str) -> None:
        out.append(f"{family}: {msg}")

    # the expected number of rows (Deequ's hasSize; a declarative spec states it as
    # a constant, and the run's own manifest is where that constant comes from here)
    if rec.runs_expected:
        total = sum(rec.runs_expected.values())
        if len(rows) != total:
            add("size", f"{len(rows)} rows against the {total} the specification plans")
        for n, want in rec.runs_expected.items():
            got = int((rows.n_assets == n).sum())
            if got != want:
                add("size", f"N={n}: {got} rows against {want} planned")

    # presence and completeness
    for c in REQUIRED:
        if c not in rows.columns:
            add("schema", f"column {c} missing")
        elif c in ok.columns and ok[c].isna().any():
            add("schema", f"{c}: {int(ok[c].isna().sum())} empty cells on ok rows")

    # declared type
    for c in ["n_assets", "n_select", "instance", "penalty", "shots", "feasible_shots"]:
        if c in rows and not pd.api.types.is_numeric_dtype(rows[c]):
            add("dtype", f"{c} is not numeric")
    for c in ["best_shot_scorer_agrees", "is_optimal", "within_tau_main"]:
        if c in ok and not set(ok[c].dropna().astype(str)) <= {"True", "False"}:
            add("dtype", f"{c} is not boolean")
    for c in ["initial_angles", "final_angles"]:
        if c in ok.columns:
            want = 2 * int(rows.p_layers.iloc[0]) if "p_layers" in rows and len(rows) else 4
            bad = [v for v in ok[c].dropna().astype(str) if len(v.split()) != want]
            if bad:
                add("dtype", f"{c}: {len(bad)} values are not {want} numbers")

    # admitted values
    if "penalty_label" in rows:
        extra = set(rows.penalty_label.unique()) - LABELS
        if extra:
            add("set", f"penalty_label outside the vocabulary: {sorted(extra)}")
    if "status" in rows:
        bad = rows[~(rows.status.eq("ok") | rows.status.astype(str).str.startswith("error:"))]
        if len(bad):
            add("set", f"{len(bad)} rows whose status is neither ok nor error:<type>")
    if "precision" in rows and not set(rows.precision.unique()) <= {"single", "double"}:
        add("set", f"precision outside {{single, double}}: {rows.precision.unique().tolist()}")

    # keys
    if rows.run_id.duplicated().any():
        add("unique", f"{int(rows.run_id.duplicated().sum())} duplicate run_id")
    key = rows[["n_assets", "instance", "penalty_label", "qaoa_seed"]]
    if key.duplicated().any():
        add("unique", f"{int(key.duplicated().sum())} duplicate (N, instance, penalty, seed) cells")

    # one value per file
    for c in ONE_VALUE_COLS:
        if c in rows and rows[c].nunique() > 1:
            add("single", f"{c} takes {rows[c].nunique()} values: {rows[c].dropna().unique().tolist()[:4]}")

    # ranges
    for c in PROB_COLS:
        if c in ok.columns:
            v = pd.to_numeric(ok[c], errors="coerce").dropna()
            if len(v) and (v.lt(-1e-12).any() or v.gt(1 + 1e-12).any()):
                add("range", f"{c} outside [0, 1]: min {v.min():.3g}, max {v.max():.3g}")
    for c in COUNT_COLS + ["shots", "feasible_set_size", "delta_F", "penalty", "runtime_s"]:
        if c in ok.columns:
            v = pd.to_numeric(ok[c], errors="coerce").dropna()
            if len(v) and v.lt(0).any():
                add("range", f"{c} negative on {int(v.lt(0).sum())} rows")
    for c in ["g_F", "g_off"]:
        if c in ok.columns:
            v = pd.to_numeric(ok[c], errors="coerce").dropna()
            if len(v) and v.lt(-1e-12).any():
                add("range", f"{c} negative on {int(v.lt(-1e-12).sum())} rows")
    for c in ["final_probs_sum", "initial_probs_sum"]:
        if c in ok.columns:
            v = pd.to_numeric(ok[c], errors="coerce").dropna()
            if len(v) and (v - 1).abs().gt(1e-4).any():
                add("range", f"{c} is not 1 ± 1e-4 on {int((v - 1).abs().gt(1e-4).sum())} rows")
    if "rank_best_feasible" in ok.columns:
        v = pd.to_numeric(ok.rank_best_feasible, errors="coerce").dropna()
        if len(v) and v.lt(1).any():
            add("range", f"rank_best_feasible below 1 on {int(v.lt(1).sum())} rows")

    # inequalities between two columns of the same row
    def pair_le(a: str, b: str, tol: float = 1e-9) -> None:
        if a in ok.columns and b in ok.columns:
            x = pd.to_numeric(ok[a], errors="coerce")
            y = pd.to_numeric(ok[b], errors="coerce")
            m = x.notna() & y.notna() & (x > y + tol)
            if m.any():
                add("pair", f"{a} > {b} on {int(m.sum())} rows (max excess {float((x - y)[m].max()):.3g})")

    pair_le("feasible_shots", "shots")
    pair_le("distinct_shots", "shots")
    pair_le("shots_within_tau_main", "feasible_shots")
    pair_le("shots_optimal", "shots_within_tau_main")
    pair_le("final_P_opt", "final_P_tau_main")
    pair_le("final_P_tau_main", "final_P_F")
    pair_le("initial_P_opt", "initial_P_F")
    pair_le("rank_best_feasible", "feasible_set_size")
    pair_le("f_star", "f_best_feasible")          # the optimum is the feasible minimum

    # a tolerance band between two columns of a row, the width computed from the
    # row itself (Deequ's satisfies takes the arithmetic): a shot fraction should
    # sit within five standard deviations of the probability it was drawn from.
    if {"feasible_shot_fraction", "final_P_F", "shots"} <= set(ok.columns):
        pF = pd.to_numeric(ok.final_P_F, errors="coerce")
        obs = pd.to_numeric(ok.feasible_shot_fraction, errors="coerce")
        S = pd.to_numeric(ok.shots, errors="coerce")
        sd = np.sqrt(np.clip(pF * (1 - pF) / S, 0, None))
        bad = int(((obs - pF).abs() > 5 * sd + 1e-12).sum())
        if bad:
            add("band", f"the feasible-shot fraction is over five sigma from final_P_F on {bad} rows")

    # a conditional equality within the row: where the label says one thing, the
    # value must be the column the label names (Deequ's satisfies, pandera's
    # dataframe-level check). This is a predicate over columns of one row, so it
    # belongs to the vocabulary even though it encodes what a label means.
    if {"penalty_label", "penalty", "a_heur", "a_margin"} <= set(ok.columns):
        for label, col in (("Aheur", "a_heur"), ("Amargin", "a_margin")):
            g = ok[ok.penalty_label == label]
            if len(g):
                bad = int((~np.isclose(pd.to_numeric(g.penalty, errors="coerce"),
                                       pd.to_numeric(g[col], errors="coerce"),
                                       rtol=1e-12, atol=1e-15)).sum())
                if bad:
                    add("conditional", f"penalty is not {col} on {bad} rows labelled {label}")
        g0 = ok[ok.penalty_label == "A0"]
        if len(g0) and (pd.to_numeric(g0.penalty, errors="coerce").abs() > 1e-15).any():
            add("conditional", "penalty is not 0 on a row labelled A0")

    # uniqueness of a tuple rather than of a key: two runs of one instance that
    # differ only in their seed should not carry the same shot summary
    shot_cols = [c for c in ("feasible_shots", "distinct_shots", "shots_within_tau_main",
                             "shots_optimal", "best_all_index")
                 if c in ok.columns]
    if len(shot_cols) >= 4 and {"n_assets", "instance", "penalty_label"} <= set(ok.columns):
        dup = ok.duplicated(subset=["n_assets", "instance", "penalty_label"] + shot_cols, keep=False)
        if dup.any():
            add("unique", f"{int(dup.sum())} rows of one (N, instance, penalty) share a shot summary")

    # the copied instance columns against the table they were copied from
    if len(ref):
        r_ix = ref.set_index(["n_assets", "instance"])
        missing = [f"{int(a)}/{int(b)}" for a, b in
                   set(zip(rows.n_assets, rows.instance)) - set(r_ix.index)]
        if missing:
            add("fk", f"{len(missing)} (N, instance) without a reference row: {sorted(missing)[:3]}")
        cols = [c for c in ("f_star", "delta_F", "feasible_set_size", "a_crit", "a_heur",
                            "a_margin", "instance_digest", "input_digest", "n_select")
                if c in rows.columns and c in ref.columns]
        joined = rows.merge(ref[["n_assets", "instance"] + cols], on=["n_assets", "instance"],
                            how="inner", suffixes=("", "_ref"))
        for c in cols:
            a, b = joined[c], joined[f"{c}_ref"]
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                bad = int((~np.isclose(a.astype(float), b.astype(float), rtol=0, atol=0)).sum())
            else:
                bad = int((a.astype(str) != b.astype(str)).sum())
            if bad:
                add("fk", f"{c} differs from the reference row on {bad} rows")
    return out


DECLARATIVE = Check("declarative", "table", check_declarative)


# ----------------------------------------------------------------------
# Faults
# ----------------------------------------------------------------------

def _pick(rec: Record, n: int = 12, label: str = "Aheur", seed: int = 42, instance: int = 0):
    m = ((rec.rows.n_assets == n) & (rec.rows.penalty_label == label)
         & (rec.rows.qaoa_seed == seed) & (rec.rows.instance == instance))
    idx = rec.rows.index[m]
    if len(idx) != 1:
        raise RuntimeError(f"expected one row for N={n} {label} s{seed} i{instance}, got {len(idx)}")
    return idx[0]


def fault_price_perturbation(rec: Record, tmp: Path) -> Record:
    out = rec.copy("price_perturbation")
    derived = tmp / "derived"
    shutil.copytree(rec.derived_dir, derived)
    for path in derived.glob(f"stats_*_{rec.end_date}_*.csv"):
        frame = pd.read_csv(path, index_col=0, float_precision="round_trip")
        tick = frame.index[0]
        frame.loc[tick, "mu"] *= 1 + 1e-6
        frame.to_csv(path, float_format="%.17g")
    out.derived_dir = derived
    return out


def _mixin(rec: Record, source_csv: Path, source_npz_dir: Path, label: str) -> Record:
    out = rec.copy(label)
    src = pd.read_csv(source_csv, float_precision="round_trip")
    src = src[(src.n_assets == 12) & (src.penalty_label == "Aheur") & (src.qaoa_seed == 42)
              & (src.instance == 0)]
    row = src.iloc[0].copy()
    # Pasted over the same cell of the main file, run id included: the kind of
    # copy that survives a merge.
    target = _pick(rec)
    row["run_id"] = out.rows.loc[target, "run_id"]
    for c in out.rows.columns:
        if c in row.index:
            out.rows.loc[target, c] = row[c]
    with np.load(source_npz_dir / f"{src.iloc[0].run_id}.npz") as z:
        out.artifacts[row["run_id"]] = {k: np.array(z[k]) for k in NPZ_KEYS}
    return out


def _other_arm(rec: Record, candidates: tuple[str, ...]) -> str:
    """The first sibling arm the record is not, by window and optimizer.

    Pasting a record's own run back into itself is not a fault, so the source
    of a mix-in has to differ from the record under test in the attribute the
    fault is about: the window for ``window_mixin``, the optimizer for
    ``optimizer_mixin``. On the main record this picks E5 and E3 L-BFGS-B, as
    before; on E5 the window mix-in comes from the main record, and on the
    L-BFGS-B arm the optimizer mix-in does.
    """
    for suffix in candidates:
        src = pd.read_csv(E1_RESULTS / f"paper01_e1_cross{suffix}.csv", nrows=1)
        if (str(src.data_end.iloc[0]) != rec.end_date
                or str(src.optimizer.iloc[0]) != str(rec.rows.optimizer.iloc[0])):
            return suffix
    raise RuntimeError("no sibling arm differs from the record")


def fault_window_mixin(rec: Record, tmp: Path) -> Record:
    suffix = _other_arm(rec, ("_e5", ""))
    return _mixin(rec, E1_RESULTS / f"paper01_e1_cross{suffix}.csv",
                  E1_RESULTS / f"e1_runs{suffix}", "window_mixin")


def fault_optimizer_mixin(rec: Record, tmp: Path) -> Record:
    suffix = _other_arm(rec, ("_e3_lbfgs", ""))
    return _mixin(rec, E1_RESULTS / f"paper01_e1_cross{suffix}.csv",
                  E1_RESULTS / f"e1_runs{suffix}", "optimizer_mixin")


def fault_label_swap(rec: Record, tmp: Path) -> Record:
    out = rec.copy("label_swap")
    i = _pick(rec, label="Amargin")
    j = _pick(rec, label="Aheur")
    out.rows.loc[i, "penalty_label"] = "Aheur"
    out.rows.loc[j, "penalty_label"] = "Amargin"
    return out


def fault_angle_swap(rec: Record, tmp: Path) -> Record:
    out = rec.copy("angle_swap")
    i = _pick(rec)
    a, b = out.rows.loc[i, "initial_angles"], out.rows.loc[i, "final_angles"]
    out.rows.loc[i, "initial_angles"], out.rows.loc[i, "final_angles"] = b, a
    z = out.artifacts[out.rows.loc[i, "run_id"]]
    z["initial_angles"], z["final_angles"] = z["final_angles"].copy(), z["initial_angles"].copy()
    return out


def fault_batch_from_initial(rec: Record, tmp: Path) -> Record:
    """The L-BFGS-B wrapper defect: the batch is drawn from the *initial* state.

    The three counts are nested (optimum ⊆ within τ ⊆ feasible) and the defect
    drew them from one state, so they stayed mutually consistent; only their
    relation to the *final* state was wrong. The draw here is therefore one
    multinomial over the initial state's probabilities rather than three
    independent binomials — an earlier version drew them independently, which
    broke the ordering and let the declarative control (Section 4.1) catch the
    replay for a reason the defect never gave anyone.
    """
    out = rec.copy("batch_from_initial")
    rng = np.random.default_rng(0)
    rows = out.rows
    S = rows.shots.astype(int).to_numpy()
    pF = rows.initial_P_F.clip(0, 1).to_numpy()
    pT = rows.initial_P_tau_main.clip(0, 1).to_numpy() if "initial_P_tau_main" in rows else pF
    pO = rows.initial_P_opt.clip(0, 1).to_numpy()
    pT = np.clip(pT, pO, pF)
    n_opt = rng.binomial(S, pO)
    n_tau = n_opt + rng.binomial(S - n_opt, np.where(pO < 1, (pT - pO) / np.maximum(1 - pO, 1e-12), 0.0).clip(0, 1))
    feas = n_tau + rng.binomial(S - n_tau, np.where(pT < 1, (pF - pT) / np.maximum(1 - pT, 1e-12), 0.0).clip(0, 1))
    rows["feasible_shots"] = feas
    rows["feasible_shot_fraction"] = feas / S
    rows["shots_optimal"] = n_opt
    if "shots_within_tau_main" in rows:
        rows["shots_within_tau_main"] = n_tau
    # The artifacts' shot arrays are left as they were: the defect changed
    # what was sampled, and the summary columns are what the tables read.
    return out


def fault_evaluator_from_Q(rec: Record, tmp: Path) -> Record:
    out = rec.copy("evaluator_from_Q")
    rows = out.rows
    has = rows.f_best_feasible.notna()
    k = rows.n_select
    E = rows.f_best_feasible - rows.penalty * k * k   # feasible: (m-k)^2 = 0
    rows.loc[has, "f_best_feasible"] = E[has]
    rows.loc[has, "g_F"] = ((E - rows.f_star) / rows.delta_F)[has]
    return out


def fault_column_drop(rec: Record, tmp: Path) -> Record:
    out = rec.copy("column_drop")
    out.rows = out.rows.drop(columns=["final_P_F"])
    return out


def fault_shot_seed_noop(rec: Record, tmp: Path) -> Record:
    out = rec.copy("shot_seed_noop")
    for inst in range(3):
        i = _pick(rec, seed=42, instance=inst)
        j = _pick(rec, seed=43, instance=inst)
        zi = out.artifacts[out.rows.loc[i, "run_id"]]
        zj = out.artifacts[out.rows.loc[j, "run_id"]]
        zj["shot_indices"] = zi["shot_indices"].copy()
        zj["shot_counts"] = zi["shot_counts"].copy()
        for c in ("shots", "feasible_shots", "feasible_shot_fraction", "distinct_shots",
                  "shots_optimal", "shots_within_tau_main", "best_all_index",
                  "best_all_cardinality", "best_all_feasible", "best_all_E", "best_all_f",
                  "energy_no_offset_solver"):
            out.rows.loc[j, c] = out.rows.loc[i, c]
    return out


def fault_reference_stale(rec: Record, tmp: Path) -> Record:
    out = rec.copy("reference_stale")
    m = (out.reference.n_assets == 12) & (out.reference.instance == 0)
    out.reference.loc[m, "f_star"] = out.reference.loc[m, "f_star"] * (1 + 1e-9)
    return out


@dataclass
class Fault:
    name: str
    apply: Callable[[Record, Path], Record]
    description: str


FAULTS = [
    Fault("price_perturbation", fault_price_perturbation,
          "one mean return in the derived statistics moved by 1e-6 relative"),
    Fault("window_mixin", fault_window_mixin, "one run from the other market window pasted into the file"),
    Fault("optimizer_mixin", fault_optimizer_mixin, "one run from another optimizer arm pasted into the file"),
    Fault("label_swap", fault_label_swap, "A_margin and A_heur rows of one cell relabelled"),
    Fault("angle_swap", fault_angle_swap, "initial and final angles exchanged in row and artifact"),
    Fault("batch_from_initial", fault_batch_from_initial,
          "every batch summary drawn from the initial state (E3 L-BFGS-B defect)"),
    Fault("evaluator_from_Q", fault_evaluator_from_Q,
          "f of the best feasible shot replaced by its penalised energy"),
    Fault("column_drop", fault_column_drop, "final_P_F column missing (E4 write_csv defect)"),
    Fault("shot_seed_noop", fault_shot_seed_noop, "seed 43's batch replaced by seed 42's (3 instances)"),
    Fault("reference_stale", fault_reference_stale, "reference f* of one instance off by 1e-9 relative"),
]


# ----------------------------------------------------------------------
# Headline numbers (what a naive pipeline reports)
# ----------------------------------------------------------------------

def headline(rec: Record) -> dict[str, float]:
    """The study's main contrasts, computed the naive way from the CSV alone."""
    out = {}
    rows = rec.rows
    def col(b, name):
        # A naive pipeline would crash on a dropped column; here the headline
        # number becomes NaN, which counts as "moved" against the clean value.
        return b[name] if name in b else pd.Series(np.nan, index=b.index)
    for n in sorted(rows.n_assets.unique()):
        for lab in ("Aheur", "Amargin"):
            b = rows[(rows.n_assets == n) & (rows.penalty_label == lab)]
            out[f"N{n}_{lab}_median_g_off"] = float(col(b, "g_off").median())
            out[f"N{n}_{lab}_median_g_F"] = float(col(b, "g_F").median())
            out[f"N{n}_{lab}_mean_feasible_fraction"] = float(col(b, "feasible_shot_fraction").mean())
            out[f"N{n}_{lab}_median_final_P_F"] = float(col(b, "final_P_F").median())
            out[f"N{n}_{lab}_optimum_hits"] = int((col(b, "shots_optimal") > 0).sum())
    return out


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def run_check(check: Check, rec: Record) -> list[str]:
    """A check that cannot run on the record (a column it needs is gone) is a finding."""
    try:
        return check.fn(rec)
    except Exception as exc:
        return [f"check raised {type(exc).__name__}: {exc}"]


#: Seconds each check took, by (record label, check name), filled by ``evaluate``.
#: The paper argues that detection happens at the cheapest stage that can see a
#: fault, so what each stage costs belongs in the record and not only in a log.
SECONDS: dict[tuple[str, str], float] = {}


def evaluate(rec: Record, checks: list[Check], log=print) -> dict[str, list[str]]:
    findings = {}
    for c in checks:
        t0 = time.perf_counter()
        findings[c.name] = run_check(c, rec)
        secs = time.perf_counter() - t0
        SECONDS[(rec.label, c.name)] = round(secs, 3)
        log(f"    {rec.label:<20} {c.name:<17} {len(findings[c.name]):>4} findings ({secs:.1f} s)")
    return findings


def run(sizes=SIZES, recompute: bool = True, log=print, record: str = ""):
    """The detection matrix on the main record, or on a sibling arm (``record``)."""
    checks = [c for c in CHECKS if recompute or not c.expensive]
    log(f"loading the E1{record} record for N in {list(sizes)} ...")
    clean = load_record(sizes, suffix=record)
    log(f"  {len(clean.rows)} rows, {len(clean.artifacts)} artifacts")
    base_head = headline(clean)

    matrix, findings_rows, head_rows, count_rows = [], [], [], []
    tmp_root = Path(tempfile.mkdtemp(prefix="es1_"))
    try:
        for fault in [None] + FAULTS:
            if fault is None:
                rec = clean
            else:
                tmp = tmp_root / fault.name
                tmp.mkdir()
                rec = fault.apply(clean, tmp)
            found = evaluate(rec, checks, log)
            declarative = run_check(DECLARATIVE, rec)
            log(f"    {rec.label:<20} {'declarative':<17} {len(declarative):>4} findings")
            failing = [c.name for c in checks if found[c.name]]
            first = next((c for c in checks if found[c.name]), None)
            head = headline(rec)
            moved = {k: (head.get(k), base_head.get(k)) for k in base_head
                     if not (pd.isna(head.get(k)) and pd.isna(base_head.get(k)))
                     and head.get(k) != base_head.get(k)}
            matrix.append({
                "fault": rec.label,
                "description": fault.description if fault else "the committed record",
                **{c.name: ("FAIL" if found[c.name] else "pass") for c in checks},
                "detected_by_naive": "yes" if any(n in NAIVE for n in failing) else "no",
                "detected_by_declarative": "yes" if declarative else "no",
                "declarative_families": ";".join(
                    f for f in DECLARATIVE_FAMILIES
                    if any(m.startswith(f + ":") for m in declarative)),
                "detected_by_relations": "yes" if failing else "no",
                "first_failing_check": first.name if first else "",
                "stage": first.stage if first else "",
                "n_findings": sum(len(v) for v in found.values()),
                "headline_numbers_moved": len(moved),
                "headline_total": len(base_head),
            })
            found["declarative (control)"] = declarative
            for name, msgs in found.items():
                # A subject can fail one check twice -- state-batch tests a
                # row's feasible count and its optimum count separately -- so
                # the number of findings is not the number of things at fault,
                # and a paper that says "rows" has to be able to say which. Every
                # message names its subject before the first colon: a run id, a
                # reference, an (N, penalty) block, or the file. Every pair is
                # written, including those with no finding, because what a check
                # that finds nothing costs is what "the cheapest stage" is about.
                subjects = {m.split(":", 1)[0].strip() for m in msgs}
                count_rows.append({"fault": rec.label, "check": name,
                                   "n_findings": len(msgs), "n_subjects": len(subjects),
                                   "seconds": SECONDS.get((rec.label, name), "")})
                for m in msgs[:20]:
                    findings_rows.append({"fault": rec.label, "check": name, "finding": m})
            for k, (a, b) in moved.items():
                head_rows.append({"fault": rec.label, "quantity": k, "clean": b, "faulted": a})
            log(f"  {rec.label}: {'DETECTED by ' + first.name if first else 'not detected'}; "
                f"{len(moved)}/{len(base_head)} headline numbers moved")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return matrix, findings_rows, head_rows, count_rows


#: The committed records that the same thirteen relations should pass: the main
#: cross, the prior-window replication, the two optimizer arms and the pilot.
#: Each was produced by the same driver and carries its own reference, manifest,
#: artifacts and window, so a false positive here is a false positive of the
#: relation and not of the record.
CLEAN_ARMS = [
    ("", "E1 cross (main, ADAM 0.1 x 50, 2023-08-06 to 2026-08-05)"),
    ("_e5", "E5 prior-window replication (same design, 2020-08-05 to 2023-08-05)"),
    ("_e3_adam003", "E3 arm, ADAM 0.03 x 100"),
    ("_e3_lbfgs", "E3 arm, L-BFGS-B, 100 gradients"),
    ("_pilot", "the pilot block (one seed)"),
]


def clean_records(sizes=SIZES, recompute: bool = True, log=print) -> list[dict]:
    """Run every relation on each committed clean record (no faults injected).

    "Zero false positives" on one record is one observation; this makes it five,
    over two market windows and three optimizer settings.
    """
    checks = [c for c in CHECKS if recompute or not c.expensive]
    out = []
    for suffix, description in CLEAN_ARMS:
        label = f"e1{suffix}" if suffix else "e1"
        log(f"loading {label} ...")
        rec = load_record(sizes, suffix=suffix, label=label)
        log(f"  {len(rec.rows)} rows, {len(rec.artifacts)} artifacts, window ends {rec.end_date}")
        found = evaluate(rec, checks, log)
        found["declarative (control)"] = run_check(DECLARATIVE, rec)
        failing = {k: v for k, v in found.items() if v}
        out.append({
            "record": label,
            "description": description,
            "window_end": rec.end_date,
            "rows": len(rec.rows),
            "references": len(rec.reference),
            "artifacts": len(rec.artifacts),
            "checks_run": len(checks),
            "checks_failing": len(set(failing) - {"declarative (control)"}),
            "false_positives": "; ".join(f"{k} ({len(v)}): {v[0]}" for k, v in failing.items()) or "none",
        })
        log(f"  {label}: {out[-1]['checks_failing']} of {len(checks)} relations report a finding")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    p.add_argument("--no-recompute", action="store_true",
                   help="skip the two expensive checks (reference-input, state-recompute)")
    p.add_argument("--clean-records", action="store_true",
                   help="run every relation on the five committed clean records and stop "
                        "(the false-positive measurement; injects nothing)")
    p.add_argument("--record", default="",
                   help="inject into a sibling arm instead of the main record (its file "
                        "suffix, e.g. _e5); the outputs carry the same suffix")
    args = p.parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.clean_records:
        rows = clean_records(tuple(args.sizes), not args.no_recompute)
        write_csv(CLEAN_CSV, rows)
        print(f"wrote {CLEAN_CSV.name} ({len(rows)})")
        for r in rows:
            print(f"  {r['record']:<14} {r['rows']:>5} rows  "
                  f"{r['checks_failing']}/{r['checks_run']} relations flag  {r['false_positives'][:90]}")
        return 0
    matrix, findings, heads, counts = run(tuple(args.sizes), not args.no_recompute,
                                          record=args.record)
    tagged = lambda p: p.with_name(p.stem + args.record + p.suffix)
    write_csv(tagged(MATRIX_CSV), matrix)
    write_csv(tagged(FINDINGS_CSV), findings)
    write_csv(tagged(HEADLINE_CSV), heads)
    write_csv(tagged(COUNTS_CSV), counts)
    print(f"wrote {tagged(MATRIX_CSV).name} ({len(matrix)}), "
          f"{tagged(FINDINGS_CSV).name} ({len(findings)}), "
          f"{tagged(COUNTS_CSV).name} ({len(counts)}), "
          f"{tagged(HEADLINE_CSV).name} ({len(heads)})")
    cols = [c for c in matrix[0] if c not in ("description", "n_findings", "headline_total")]
    print("\t".join(cols))
    for row in matrix:
        print("\t".join(str(row[c]) for c in cols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
