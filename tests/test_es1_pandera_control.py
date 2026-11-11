# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Regression tests for the declarative control stated as a pandera schema.

Two defects in this schema were caught by running it and not by reading it, and
both would have been reported as detections rather than as errors, because the
harness treats a check that cannot run as a finding:

  * ``pa.Column``'s first positional argument is a dtype, so a Check passed
    there raised a TypeError and every record came back "detected", including
    the committed one;
  * assigning a Column per list meant the last list won, which dropped
    ``required=True`` for the fifteen columns that appear in two of them, and
    with it the detection of a missing column.

The tests below pin the committed record as clean, pin each family on a
one-column edit, and pin the merge that the second defect broke.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

pytest.importorskip("pandera", reason="the control needs requirements-control.txt")

from experiments.system_paper import es1_fault_injection as es1  # noqa: E402
from experiments.system_paper import es1_pandera_control as pc  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "experiments" \
    / "paper01_qubo_baseline" / "results"


@pytest.fixture(scope="module")
def rec() -> es1.Record:
    """A record of real rows and a real reference, without the artifacts.

    ``check_pandera`` reads only the two tables, so the 99 MB of per-run
    artifacts the relations need are not loaded here.
    """
    rows = pd.read_csv(RESULTS / "paper01_e1_cross.csv")
    ref = pd.read_csv(RESULTS / "paper01_e1_reference.csv")
    rows = rows[rows.n_assets == 12].reset_index(drop=True)
    return es1.Record(rows=rows, reference=ref, artifacts={},
                      derived_dir=RESULTS, label="test")


def test_the_committed_record_raises_nothing(rec) -> None:
    assert pc.check_pandera(rec) == []


def test_a_probability_outside_its_range_is_a_range_finding(rec) -> None:
    bad = rec.copy("probability")
    bad.rows.loc[0, "final_P_F"] = 1.5
    out = pc.check_pandera(bad)
    assert out and any(m.startswith("range") for m in out), out


def test_a_missing_required_column_is_a_schema_finding(rec) -> None:
    """The case the clobbered ``required`` flag lost: final_P_F is in two lists."""
    bad = rec.copy("column")
    bad.rows = bad.rows.drop(columns=["final_P_F"])
    out = pc.check_pandera(bad)
    assert out and any(m.startswith("schema") for m in out), out


def test_a_broken_inequality_between_two_columns_is_a_pair_finding(rec) -> None:
    bad = rec.copy("pair")
    bad.rows.loc[0, "feasible_shots"] = int(bad.rows.loc[0, "shots"]) + 1
    out = pc.check_pandera(bad)
    assert out and any(m.startswith("pair") for m in out), out


def test_a_column_that_must_hold_one_value_is_a_single_finding(rec) -> None:
    bad = rec.copy("single")
    bad.rows.loc[0, "optimizer"] = "nobody's optimizer"
    out = pc.check_pandera(bad)
    assert out and any(m.startswith("single") for m in out), out


def test_a_column_in_two_lists_keeps_every_expectation(rec) -> None:
    """final_P_F is required *and* a probability; the schema must carry both."""
    col = pc.build_schema(rec).columns["final_P_F"]
    assert col.required is True
    assert col.nullable is False
    assert col.checks, "the range check was dropped"


def test_an_unparseable_schema_would_not_look_like_a_detection(rec) -> None:
    """``run_check`` turns an exception into a finding, so the schema must build."""
    out = es1.run_check(pc.PANDERA, rec)
    assert not any("raised" in m for m in out), out
