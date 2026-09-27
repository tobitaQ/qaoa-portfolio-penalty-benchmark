# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
E-S1, second part: random mutations and column coverage of the verification relations.

The ten faults of ``es1_fault_injection.py`` were chosen by the authors --
four replays of real defects, six judgements of plausible mistakes -- which
leaves the objection that the checks were written for the faults. This
harness answers it two ways, with the same record and the same checks.

**Column coverage.** For every column of the run row, the reference row and
the per-run artifact, one random cell is perturbed (a numeric value by a
relative 10⁻³, or by ±1 for a count; a string by a different valid value)
and the cheap checks are run. A column is *covered* if some check fails.
The uncovered columns are listed by name: they are the parts of the record
that no relation binds, and the honest limit of P1 as implemented.

**Random mutations.** ``--faults`` mutations are drawn from a fixed set of
operators with a seeded generator (Table below), applied one at a time to a
copy of the record, and the cheap checks run; for each we record whether it
was detected, by which check first, at what magnitude, and whether the
headline numbers moved. Operators:

  cell_perturb      one numeric cell of a run row × (1 + δ), δ ∈ {1e-9, 1e-6, 1e-3, 1e-1}, sign random
  cell_from_other   one cell of a run row replaced by the same column's value from another row of the same N
  row_metrics_copy  every non-key column of one run row copied onto another row of the same N
  row_delete        one run row removed
  label_shuffle     the penalty labels of one (N, instance, seed) block permuted
  reference_perturb one numeric cell of a reference row × (1 + δ)
  npz_perturb       one element of one artifact array × (1 + δ) (or ±1 for counts)
  npz_swap          the artifacts of two runs of the same N exchanged
  input_perturb     one μ or Σ entry of the derived statistics × (1 + δ)
  column_drop       one non-key column removed from the run rows

Only the cheap checks run (everything but reference-input and
state-recompute), so a mutation that only those two could see counts as
undetected here; the ten-fault harness showed both to be redundant for every
fault a cheaper check catches, and input-digest now covers the input.

Outputs: ``paper02_es1_column_coverage.csv`` (one row per column),
``paper02_es1_random_faults.csv`` (one row per mutation) and
``paper02_es1_random_summary.csv`` (per operator and per magnitude).

Run:
    python -m experiments.system_paper.es1_random_faults --faults 300      # ~1.5 h on N = 12, 16, 20
    python -m experiments.system_paper.es1_random_faults --sizes 12 --faults 60 --no-coverage
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.paper01_qubo_baseline.run_experiment import write_csv
from experiments.system_paper import es1_fault_injection as es1

RESULTS_DIR = es1.RESULTS_DIR
COVERAGE_CSV = RESULTS_DIR / "paper02_es1_column_coverage.csv"
FAULTS_CSV = RESULTS_DIR / "paper02_es1_random_faults.csv"
SUMMARY_CSV = RESULTS_DIR / "paper02_es1_random_summary.csv"

CHEAP = [c for c in es1.CHECKS if not c.expensive]


def light_copy(rec, label: str, touch: tuple[str, ...] = ()):
    """A copy that shares the artifacts except for the runs about to be mutated.

    ``Record.copy`` deep-copies 2,565 artifacts (~100 MB) per fault, which made
    a coverage pass a minute per column; a mutation touches at most two runs.
    """
    arts = dict(rec.artifacts)
    for rid in touch:
        arts[rid] = {k: v.copy() for k, v in rec.artifacts[rid].items()}
    return es1.Record(rec.rows.copy(deep=True), rec.reference.copy(deep=True), arts,
                      rec.derived_dir, label, rec.end_date, rec.runs_expected)
KEYS = {"run_id", "n_assets", "instance", "penalty_label", "qaoa_seed", "spec_version"}
DELTAS = (1e-9, 1e-6, 1e-3, 1e-1)
NPZ_ARRAYS = ("initial_angles", "final_angles", "angles_trajectory", "convergence",
              "final_probs_feasible", "initial_probs_feasible", "shot_counts", "feasible_f")


def _is_count(series: pd.Series) -> bool:
    return series.dtype.kind in "iu" or (series.dtype.kind == "f" and np.all(np.mod(series.dropna(), 1) == 0)
                                          and series.dropna().abs().max() >= 1)


def perturb_value(v, delta: float, rng: np.random.Generator, count: bool):
    if isinstance(v, (bool, np.bool_)):
        return not v
    if isinstance(v, str):
        return v + "x"
    if count:
        return v + (1 if rng.random() < 0.5 else -1) if v != 0 else v + 1
    if v == 0:
        return delta
    return v * (1 + delta * (1 if rng.random() < 0.5 else -1))


# ----------------------------------------------------------------------
# Mutation operators: each returns (record, description) or None if no-op
# ----------------------------------------------------------------------

def op_cell_perturb(rec, rng, tmp):
    cols = [c for c in rec.rows.columns if c not in KEYS and rec.rows[c].dtype.kind in "fiub"]
    col = rng.choice(cols)
    i = rng.integers(len(rec.rows))
    delta = float(rng.choice(DELTAS))
    v = rec.rows.at[i, col]
    if pd.isna(v):
        return None
    out = light_copy(rec, "cell_perturb")
    out.rows.at[i, col] = perturb_value(v, delta, rng, _is_count(rec.rows[col]))
    if out.rows.at[i, col] == v:
        return None
    return out, {"target": col, "magnitude": delta, "row": rec.rows.at[i, "run_id"]}


def op_cell_from_other(rec, rng, tmp):
    cols = [c for c in rec.rows.columns if c not in KEYS]
    col = rng.choice(cols)
    i = rng.integers(len(rec.rows))
    same_n = rec.rows.index[rec.rows.n_assets == rec.rows.at[i, "n_assets"]]
    j = rng.choice(same_n)
    if j == i or (pd.notna(rec.rows.at[i, col]) and rec.rows.at[i, col] == rec.rows.at[j, col]):
        return None
    out = light_copy(rec, "cell_from_other")
    out.rows.at[i, col] = rec.rows.at[j, col]
    return out, {"target": col, "magnitude": "", "row": rec.rows.at[i, "run_id"]}


def op_row_metrics_copy(rec, rng, tmp):
    i = rng.integers(len(rec.rows))
    same_n = rec.rows.index[rec.rows.n_assets == rec.rows.at[i, "n_assets"]]
    j = rng.choice(same_n)
    if j == i:
        return None
    out = light_copy(rec, "row_metrics_copy")
    cols = [c for c in rec.rows.columns if c not in KEYS]
    out.rows.loc[i, cols] = rec.rows.loc[j, cols].values
    return out, {"target": "all non-key columns", "magnitude": "", "row": rec.rows.at[i, "run_id"]}


def op_row_delete(rec, rng, tmp):
    i = rng.integers(len(rec.rows))
    out = light_copy(rec, "row_delete")
    rid = rec.rows.at[i, "run_id"]
    out.rows = out.rows.drop(index=i).reset_index(drop=True)
    return out, {"target": "row", "magnitude": "", "row": rid}


def op_label_shuffle(rec, rng, tmp):
    groups = rec.rows.groupby(["n_assets", "instance", "qaoa_seed"]).indices
    key = list(groups)[rng.integers(len(groups))]
    idx = groups[key]
    if len(idx) < 2:
        return None
    labels = rec.rows.loc[idx, "penalty_label"].tolist()
    perm = rng.permutation(len(labels))
    if all(labels[k] == labels[perm[k]] for k in range(len(labels))):
        return None
    out = light_copy(rec, "label_shuffle")
    out.rows.loc[idx, "penalty_label"] = [labels[k] for k in perm]
    return out, {"target": "penalty_label", "magnitude": "", "row": f"N{key[0]} i{key[1]} s{key[2]}"}


def op_reference_perturb(rec, rng, tmp):
    cols = [c for c in rec.reference.columns if c not in ("n_assets", "instance", "spec_version")
            and rec.reference[c].dtype.kind in "fiub"]
    col = rng.choice(cols)
    i = rng.integers(len(rec.reference))
    delta = float(rng.choice(DELTAS))
    v = rec.reference.at[i, col]
    if pd.isna(v):
        return None
    out = light_copy(rec, "reference_perturb")
    out.reference.at[i, col] = perturb_value(v, delta, rng, _is_count(rec.reference[col]))
    if out.reference.at[i, col] == v:
        return None
    return out, {"target": f"reference.{col}", "magnitude": delta,
                 "row": f"N{rec.reference.at[i, 'n_assets']} i{rec.reference.at[i, 'instance']}"}


def op_npz_perturb(rec, rng, tmp):
    rid = rec.rows.at[rng.integers(len(rec.rows)), "run_id"]
    name = rng.choice(NPZ_ARRAYS)
    arr = rec.artifacts[rid][name]
    flat_index = rng.integers(arr.size)
    delta = float(rng.choice(DELTAS))
    out = light_copy(rec, "npz_perturb", touch=(rid,))
    a = out.artifacts[rid][name]
    v = a.flat[flat_index]
    if a.dtype.kind in "iu":
        a.flat[flat_index] = v + 1
    else:
        a.flat[flat_index] = perturb_value(float(v), delta, rng, False)
    return out, {"target": f"npz.{name}", "magnitude": delta if a.dtype.kind not in "iu" else 1, "row": rid}


def op_npz_swap(rec, rng, tmp):
    n = rng.choice(sorted(rec.rows.n_assets.unique()))
    ids = rec.rows.run_id[rec.rows.n_assets == n].tolist()
    a, b = rng.choice(ids, size=2, replace=False)
    out = light_copy(rec, "npz_swap")
    out.artifacts[a], out.artifacts[b] = rec.artifacts[b], rec.artifacts[a]
    return out, {"target": "npz", "magnitude": "", "row": f"{a}<->{b}"}


def op_input_perturb(rec, rng, tmp):
    derived = tmp / f"derived_{rng.integers(1 << 30)}"
    shutil.copytree(rec.derived_dir, derived)
    delta = float(rng.choice(DELTAS))
    for path in derived.glob("stats_*_2026-08-05_*.csv"):
        frame = pd.read_csv(path, index_col=0, float_precision="round_trip")
        r = rng.integers(len(frame))
        c = rng.integers(len(frame.columns))
        v = frame.iat[r, c]
        frame.iat[r, c] = perturb_value(float(v), delta, rng, False)
        frame.to_csv(path, float_format="%.17g")
        target = f"derived.{frame.columns[c]}[{frame.index[r]}]"
    out = light_copy(rec, "input_perturb")
    out.derived_dir = derived
    return out, {"target": target, "magnitude": delta, "row": ""}


def op_column_drop(rec, rng, tmp):
    cols = [c for c in rec.rows.columns if c not in KEYS]
    col = rng.choice(cols)
    out = light_copy(rec, "column_drop")
    out.rows = out.rows.drop(columns=[col])
    return out, {"target": col, "magnitude": "", "row": ""}


OPERATORS = {
    "cell_perturb": op_cell_perturb, "cell_from_other": op_cell_from_other,
    "row_metrics_copy": op_row_metrics_copy, "row_delete": op_row_delete,
    "label_shuffle": op_label_shuffle, "reference_perturb": op_reference_perturb,
    "npz_perturb": op_npz_perturb, "npz_swap": op_npz_swap,
    "input_perturb": op_input_perturb, "column_drop": op_column_drop,
}
WEIGHTS = {"cell_perturb": 0.30, "cell_from_other": 0.12, "row_metrics_copy": 0.06,
           "row_delete": 0.04, "label_shuffle": 0.06, "reference_perturb": 0.10,
           "npz_perturb": 0.14, "npz_swap": 0.04, "input_perturb": 0.08, "column_drop": 0.06}


def run_checks(rec) -> dict[str, list[str]]:
    return {c.name: es1.run_check(c, rec) for c in CHEAP}


def first_failing(found: dict[str, list[str]]) -> str:
    return next((c.name for c in CHEAP if found[c.name]), "")


# ----------------------------------------------------------------------
# Column coverage
# ----------------------------------------------------------------------

def coverage(clean, rng, log) -> list[dict]:
    base = es1.headline(clean)
    out = []
    targets = [("row", c) for c in clean.rows.columns if c not in KEYS]
    targets += [("reference", c) for c in clean.reference.columns
                if c not in ("n_assets", "instance", "spec_version")]
    targets += [("npz", c) for c in NPZ_ARRAYS]
    for kind, col in targets:
        t0 = time.perf_counter()
        if kind == "npz":
            rid_touch = clean.rows.at[rng.integers(len(clean.rows)), "run_id"]
            rec = light_copy(clean, f"coverage:{kind}.{col}", touch=(rid_touch,))
        else:
            rec = light_copy(clean, f"coverage:{kind}.{col}")
        if kind == "row":
            filled = rec.rows.index[rec.rows[col].notna()]
            if len(filled) == 0:
                # ``error`` is empty on every ok row: nothing to perturb, and
                # a column that is never filled cannot be bound by a relation.
                out.append({"kind": kind, "column": col, "covered": "n/a (column empty)",
                            "first_failing_check": "", "failing_checks": "",
                            "headline_numbers_moved": 0, "where": "", "seconds": 0.0})
                log(f"  [coverage] {kind}.{col:<28} empty column")
                continue
            i = rng.choice(filled)
            v = rec.rows.at[i, col]
            rec.rows.at[i, col] = perturb_value(v, 1e-3, rng, _is_count(clean.rows[col]))
            where = rec.rows.at[i, "run_id"]
        elif kind == "reference":
            filled = rec.reference.index[rec.reference[col].notna()]
            i = rng.choice(filled) if len(filled) else 0
            v = rec.reference.at[i, col]
            rec.reference.at[i, col] = perturb_value(v, 1e-3, rng, _is_count(clean.reference[col]))
            where = f"N{rec.reference.at[i, 'n_assets']} i{rec.reference.at[i, 'instance']}"
        else:
            rid = rid_touch
            a = rec.artifacts[rid][col]
            k = rng.integers(a.size)
            a.flat[k] = a.flat[k] + 1 if a.dtype.kind in "iu" else perturb_value(float(a.flat[k]), 1e-3, rng, False)
            where = rid
        found = run_checks(rec)
        moved = sum(1 for k, v in es1.headline(rec).items() if v != base.get(k) and not (pd.isna(v) and pd.isna(base.get(k))))
        first = first_failing(found)
        out.append({"kind": kind, "column": col, "covered": "yes" if first else "no",
                    "first_failing_check": first,
                    "failing_checks": ";".join(n for n, f in found.items() if f),
                    "headline_numbers_moved": moved, "where": where,
                    "seconds": round(time.perf_counter() - t0, 1)})
        log(f"  [coverage] {kind}.{col:<28} {'covered by ' + first if first else 'NOT COVERED'}")
    return out


# ----------------------------------------------------------------------
# Random mutations
# ----------------------------------------------------------------------

def random_faults(clean, n_faults: int, rng, log, tmp: Path) -> list[dict]:
    base = es1.headline(clean)
    names = list(OPERATORS)
    probs = np.array([WEIGHTS[n] for n in names])
    probs /= probs.sum()
    rows = []
    k = 0
    while len(rows) < n_faults:
        name = str(rng.choice(names, p=probs))
        result = OPERATORS[name](clean, rng, tmp)
        if result is None:
            continue
        rec, info = result
        k += 1
        t0 = time.perf_counter()
        found = run_checks(rec)
        first = first_failing(found)
        declarative = es1.run_check(es1.DECLARATIVE, rec)
        head = es1.headline(rec)
        moved = sum(1 for kk, v in head.items() if v != base.get(kk) and not (pd.isna(v) and pd.isna(base.get(kk))))
        rows.append({"fault_no": k, "operator": name, **info,
                     "detected": "yes" if first else "no", "first_failing_check": first,
                     "detected_by_declarative": "yes" if declarative else "no",
                     "declarative_families": ";".join(
                         f for f in es1.DECLARATIVE_FAMILIES
                         if any(msg.startswith(f + ":") for msg in declarative)),
                     "n_failing_checks": sum(1 for f in found.values() if f),
                     "failing_checks": ";".join(n for n, f in found.items() if f),
                     "headline_numbers_moved": moved, "seconds": round(time.perf_counter() - t0, 1)})
        log(f"  [{k:>3}/{n_faults}] {name:<18} {str(info['target']):<34} {str(info['magnitude']):<6} "
            f"{'detected by ' + first if first else 'NOT DETECTED'}  moved={moved}")
        if k % 10 == 0:
            write_csv(FAULTS_CSV.with_name(FAULTS_CSV.stem + "_partial.csv"), rows)
        if rec.derived_dir != clean.derived_dir:
            shutil.rmtree(rec.derived_dir, ignore_errors=True)
    return rows


def _decl(g: pd.DataFrame) -> dict:
    """The control's numbers for the same group, so the increment is in one row."""
    if "detected_by_declarative" not in g:
        return {}
    return {"detected_declarative": int((g.detected_by_declarative == "yes").sum()),
            "rate_declarative": round((g.detected_by_declarative == "yes").mean(), 3),
            "moved_and_undetected_declarative": int(((g.headline_numbers_moved > 0)
                                                     & (g.detected_by_declarative == "no")).sum())}


def summarise(faults: list[dict]) -> list[dict]:
    df = pd.DataFrame(faults)
    out = []
    for name, g in df.groupby("operator"):
        out.append({"group": "operator", "value": name, "n": len(g),
                    "detected": int((g.detected == "yes").sum()),
                    "rate": round((g.detected == "yes").mean(), 3),
                    "moved_headline": int((g.headline_numbers_moved > 0).sum()),
                    "moved_and_undetected": int(((g.headline_numbers_moved > 0) & (g.detected == "no")).sum()),
                    **_decl(g)})
    mag = df[df.magnitude.astype(str) != ""]
    for m, g in mag.groupby(mag.magnitude.astype(str)):
        out.append({"group": "magnitude", "value": m, "n": len(g),
                    "detected": int((g.detected == "yes").sum()),
                    "rate": round((g.detected == "yes").mean(), 3),
                    "moved_headline": int((g.headline_numbers_moved > 0).sum()),
                    "moved_and_undetected": int(((g.headline_numbers_moved > 0) & (g.detected == "no")).sum()),
                    **_decl(g)})
    out.append({"group": "all", "value": "", "n": len(df), "detected": int((df.detected == "yes").sum()),
                "rate": round((df.detected == "yes").mean(), 3),
                "moved_headline": int((df.headline_numbers_moved > 0).sum()),
                "moved_and_undetected": int(((df.headline_numbers_moved > 0) & (df.detected == "no")).sum()),
                **_decl(df)})
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sizes", type=int, nargs="+", default=list(es1.SIZES))
    p.add_argument("--faults", type=int, default=300)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--no-coverage", action="store_true")
    p.add_argument("--without", nargs="*", default=[],
                   help="check names to leave out (e.g. identities, for the before/after comparison)")
    p.add_argument("--tag", default="", help="suffix for the output files")
    args = p.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    global CHEAP
    CHEAP = [c for c in CHEAP if c.name not in set(args.without)]
    suffix = f"_{args.tag}" if args.tag else ""
    cov_csv = COVERAGE_CSV.with_name(COVERAGE_CSV.stem + suffix + ".csv")
    faults_csv = FAULTS_CSV.with_name(FAULTS_CSV.stem + suffix + ".csv")
    summary_csv = SUMMARY_CSV.with_name(SUMMARY_CSV.stem + suffix + ".csv")
    print(f"loading the record for N in {args.sizes} ...", flush=True)
    clean = es1.load_record(tuple(args.sizes))
    clean_found = run_checks(clean)
    assert not any(clean_found.values()), {k: v[:2] for k, v in clean_found.items() if v}
    print(f"  {len(clean.rows)} rows; clean record passes {len(CHEAP)} cheap checks", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.no_coverage:
        cov = coverage(clean, rng, print)
        write_csv(cov_csv, cov)
        n_cov = sum(1 for c in cov if c["covered"] == "yes")
        print(f"coverage: {n_cov}/{len(cov)} columns covered; uncovered: "
              f"{[c['kind'] + '.' + c['column'] for c in cov if c['covered'] == 'no']}", flush=True)
    if args.faults <= 0:
        return 0
    tmp = Path(tempfile.mkdtemp(prefix="es1rand_"))
    try:
        faults = random_faults(clean, args.faults, rng, print, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    write_csv(faults_csv, faults)
    summary = summarise(faults)
    write_csv(summary_csv, summary)
    for s in summary:
        print(f"  {s['group']:<9} {str(s['value']):<18} n={s['n']:>3} detected={s['detected']:>3} "
              f"rate={s['rate']:.3f} moved={s['moved_headline']:>3} moved&undetected={s['moved_and_undetected']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
