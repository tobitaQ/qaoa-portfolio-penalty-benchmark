# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""The declarative control of E-S1, run as an actual pandera schema.

``es1_fault_injection.check_declarative`` states the declarative vocabulary in
hand-written pandas. A reviewer may reasonably ask whether that re-implementation
is a fair stand-in for the library a reader would actually reach for, or whether
its boundary was drawn where the comparison flatters the relations. This module
answers by building the same expectations as a ``pandera`` schema and validating
the record with it, so the control is a run of the library rather than a
description of one.

What is expressed here is pandera's own constraint vocabulary: column presence
and nullability, declared dtype, an admitted set (``isin``), a numeric range
(``in_range``, ``ge``), field and multi-field uniqueness, and dataframe-level
checks for the predicates that read two columns of one row. The reference join
is included, as it is in the hand-written control, because both tables are part
of the record a reader receives.

What is *not* expressed is anything that leaves the tables: opening a per-run
``.npz``, hashing the pinned statistics, recomputing a quantity from a stored
state. pandera can run such a function -- any callable can be registered -- but
then the relation is this platform's and the library supplies the runner and the
report, which is the distinction Section 4.1 draws.

pandera is not in ``requirements-repro.txt``: the reproduction of the paper's
numbers must not depend on it. It is in ``requirements-control.txt``, and
installing it changes neither the pinned pandas nor the pinned numpy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import pandera.pandas as pa
from pandera.errors import SchemaErrors

from . import es1_fault_injection as es1

#: Which constraint family a pandera failure belongs to. The built-in checks
#: report themselves by name, so the family is read from the name rather than
#: assigned by hand.
_FAMILY_BY_CHECK = (
    ("column_in_dataframe", "schema"),
    ("not_nullable", "schema"),
    ("dtype", "dtype"),
    ("isin", "set"),
    ("str_matches", "set"),
    ("in_range", "range"),
    ("greater_than_or_equal_to", "range"),
    ("less_than_or_equal_to", "range"),
    ("field_uniqueness", "unique"),
    ("multiple_fields_uniqueness", "unique"),
)

#: Inequalities between two columns of one row, as in the hand-written control.
_PAIRS = (("feasible_shots", "shots"),
          ("distinct_shots", "shots"),
          ("shots_within_tau_main", "feasible_shots"),
          ("shots_optimal", "shots_within_tau_main"),
          ("final_P_opt", "final_P_tau_main"),
          ("final_P_tau_main", "final_P_F"),
          ("initial_P_opt", "initial_P_F"),
          ("rank_best_feasible", "feasible_set_size"),
          ("f_star", "f_best_feasible"))


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame[col], errors="coerce")


def _family(check_name: str) -> str:
    for needle, family in _FAMILY_BY_CHECK:
        if needle in check_name:
            return family
    head = check_name.split(":", 1)[0]
    return head if head in es1.DECLARATIVE_FAMILIES or head == "size" else "check"


def _pair_check(a: str, b: str, tol: float = 1e-9) -> pa.Check:
    def holds(df: pd.DataFrame) -> bool:
        if a not in df.columns or b not in df.columns:
            return True
        x, y = _num(df, a), _num(df, b)
        m = x.notna() & y.notna() & (x > y + tol)
        return not bool(m.any())
    return pa.Check(holds, name=f"pair: {a} <= {b}", error=f"pair: {a} > {b}")


def _band_check() -> pa.Check:
    """The shot fraction within five standard deviations of its probability."""
    def holds(df: pd.DataFrame) -> bool:
        if not {"feasible_shot_fraction", "final_P_F", "shots"} <= set(df.columns):
            return True
        pF, obs, S = _num(df, "final_P_F"), _num(df, "feasible_shot_fraction"), _num(df, "shots")
        sd = np.sqrt(np.clip(pF * (1 - pF) / S, 0, None))
        return not bool(((obs - pF).abs() > 5 * sd + 1e-12).any())
    return pa.Check(holds, name="band: feasible-shot fraction within five sigma of final_P_F",
                    error="band: a shot fraction is over five sigma from final_P_F")


def _conditional_check() -> pa.Check:
    """Where the label names a column, the penalty must equal that column."""
    def holds(df: pd.DataFrame) -> bool:
        if not {"penalty_label", "penalty", "a_heur", "a_margin"} <= set(df.columns):
            return True
        for label, col in (("Aheur", "a_heur"), ("Amargin", "a_margin")):
            g = df[df.penalty_label == label]
            if len(g) and not np.allclose(_num(g, "penalty"), _num(g, col),
                                          rtol=1e-12, atol=1e-15):
                return False
        g0 = df[df.penalty_label == "A0"]
        return not (len(g0) and bool((_num(g0, "penalty").abs() > 1e-15).any()))
    return pa.Check(holds, name="conditional: penalty is the column its label names",
                    error="conditional: penalty is not the column its label names")


def _shot_summary_check() -> pa.Check:
    """Two runs of one instance that differ only in seed share no shot summary."""
    cols = ("feasible_shots", "distinct_shots", "shots_within_tau_main",
            "shots_optimal", "best_all_index")
    def holds(df: pd.DataFrame) -> bool:
        have = [c for c in cols if c in df.columns]
        key = ["n_assets", "instance", "penalty_label"]
        if len(have) < 4 or not set(key) <= set(df.columns):
            return True
        return not bool(df.duplicated(subset=key + have, keep=False).any())
    return pa.Check(holds, name="unique: one (N, instance, penalty) shares no shot summary",
                    error="unique: rows of one (N, instance, penalty) share a shot summary")


def _reference_check(ref: pd.DataFrame) -> pa.Check:
    """The copied instance columns against the table they were copied from."""
    cols = ("f_star", "delta_F", "feasible_set_size", "a_crit", "a_heur",
            "a_margin", "instance_digest", "input_digest", "n_select")
    def holds(df: pd.DataFrame) -> bool:
        if not len(ref) or not {"n_assets", "instance"} <= set(df.columns):
            return True
        r_ix = ref.set_index(["n_assets", "instance"])
        if set(zip(df.n_assets, df.instance)) - set(r_ix.index):
            return False
        use = [c for c in cols if c in df.columns and c in ref.columns]
        joined = df.merge(ref[["n_assets", "instance"] + use],
                          on=["n_assets", "instance"], how="inner", suffixes=("", "_ref"))
        for c in use:
            a, b = joined[c], joined[f"{c}_ref"]
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                if not np.array_equal(a.astype(float), b.astype(float)):
                    return False
            elif not (a.astype(str) == b.astype(str)).all():
                return False
        return True
    return pa.Check(holds, name="fk: instance columns equal their reference row",
                    error="fk: an instance column differs from its reference row")


def _size_check(expected: dict[int, int] | None) -> pa.Check | None:
    if not expected:
        return None
    total = sum(expected.values())
    def holds(df: pd.DataFrame) -> bool:
        if len(df) != total:
            return False
        if "n_assets" not in df.columns:
            return True
        return all(int((df.n_assets == n).sum()) == want for n, want in expected.items())
    return pa.Check(holds, name="size: the row count the specification plans",
                    error=f"size: the table is not the {total} rows planned")


def _sum_to_one_check(col: str) -> pa.Check:
    def holds(df: pd.DataFrame) -> bool:
        if col not in df.columns:
            return True
        v = _num(df, col).dropna()
        return not (len(v) and bool((v - 1).abs().gt(1e-4).any()))
    return pa.Check(holds, name=f"range: {col} is 1 +/- 1e-4",
                    error=f"range: {col} is not 1 +/- 1e-4")


def _angle_width_check(col: str, want: int) -> pa.Check:
    def holds(df: pd.DataFrame) -> bool:
        if col not in df.columns:
            return True
        vals = df[col].dropna().astype(str)
        return all(len(v.split()) == want for v in vals)
    return pa.Check(holds, name=f"dtype: {col} holds {want} numbers",
                    error=f"dtype: {col} does not hold {want} numbers")


def build_schema(rec: es1.Record) -> pa.DataFrameSchema:
    """The record's expectations as a pandera schema."""
    rows, ref = rec.rows, rec.reference
    width = 2 * int(rows.p_layers.iloc[0]) if "p_layers" in rows and len(rows) else 4

    # A column can carry several expectations -- final_P_F is required *and* a
    # probability -- so the specs are accumulated and the Column built once.
    # Assigning Column objects in sequence silently dropped required=True for
    # the fifteen columns that appear in two lists.
    spec: dict[str, dict] = {}

    def want(col: str, *, dtype=None, check=None, required=None, nullable=None) -> None:
        e = spec.setdefault(col, {"dtype": None, "checks": [], "required": False,
                                  "nullable": True})
        if dtype is not None:
            e["dtype"] = dtype
        if check is not None:
            e["checks"].append(check)
        if required:
            e["required"] = True
        if nullable is False:
            e["nullable"] = False

    for c in es1.REQUIRED:
        want(c, required=True, nullable=False)
    for c in es1.PROB_COLS:
        want(c, check=pa.Check.in_range(-1e-12, 1 + 1e-12))
    for c in es1.COUNT_COLS + ["shots", "feasible_set_size", "delta_F", "penalty", "runtime_s"]:
        want(c, check=pa.Check.ge(0))
    for c in ("g_F", "g_off"):
        want(c, check=pa.Check.ge(-1e-12))
    want("rank_best_feasible", check=pa.Check.ge(1))
    for c in ("n_assets", "n_select", "instance"):
        want(c, check=pa.Check(lambda s: pd.api.types.is_numeric_dtype(s),
                               name=f"dtype: {c} is numeric",
                               error=f"dtype: {c} is not numeric"))
    want("penalty_label", check=pa.Check.isin(sorted(es1.LABELS)))
    want("precision", check=pa.Check.isin(["single", "double"]))
    want("status", check=pa.Check(
        lambda s: s.eq("ok") | s.astype(str).str.startswith("error:"),
        name="set: status is ok or error:<type>",
        error="set: a status is neither ok nor error:<type>"))
    for c in ("best_shot_scorer_agrees", "is_optimal", "within_tau_main"):
        want(c, check=pa.Check(lambda s: set(s.dropna().astype(str)) <= {"True", "False"},
                               name=f"dtype: {c} is boolean",
                               error=f"dtype: {c} is not boolean"))
    for c in es1.ONE_VALUE_COLS:
        want(c, check=pa.Check(lambda s: s.nunique(dropna=True) <= 1,
                               name=f"single: {c} takes one value",
                               error=f"single: {c} takes more than one value"))

    columns = {c: pa.Column(e["dtype"], checks=e["checks"] or None,
                            nullable=e["nullable"], required=e["required"],
                            unique=(c == "run_id"), coerce=False)
               for c, e in spec.items()}
    if "run_id" not in columns:
        columns["run_id"] = pa.Column(nullable=True, required=False, unique=True)

    checks = [_pair_check(a, b) for a, b in _PAIRS]
    checks += [_band_check(), _conditional_check(), _shot_summary_check(),
               _reference_check(ref), _sum_to_one_check("final_probs_sum"),
               _sum_to_one_check("initial_probs_sum"),
               _angle_width_check("initial_angles", width),
               _angle_width_check("final_angles", width)]
    size = _size_check(rec.runs_expected)
    if size is not None:
        checks.append(size)

    return pa.DataFrameSchema(
        columns=columns,
        checks=checks,
        unique=["n_assets", "instance", "penalty_label", "qaoa_seed"],
        strict=False, coerce=False, name="E1 run table")


def check_pandera(rec: es1.Record) -> list[str]:
    """Validate the record with pandera; one message per failed expectation."""
    rows = rec.rows
    ok = rows[rows.status == "ok"] if "status" in rows else rows
    try:
        build_schema(rec).validate(ok, lazy=True)
    except SchemaErrors as err:
        cases = err.failure_cases
        seen: list[str] = []
        for _, row in cases.iterrows():
            name = str(row.get("check", ""))
            col = row.get("column")
            fam = _family(name)
            msg = f"{fam}: {name}" + (f" [{col}]" if isinstance(col, str) and col else "")
            if msg not in seen:
                seen.append(msg)
        return seen
    return []


PANDERA = es1.Check("pandera", "table", check_pandera)
