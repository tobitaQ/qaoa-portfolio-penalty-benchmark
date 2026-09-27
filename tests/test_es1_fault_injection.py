# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT
"""E-S1 harness: the clean record passes, each injected fault is caught where claimed.

The detection matrix is evidence for the system paper, so the two properties
it rests on are pinned here: zero findings on the committed N = 12 record
(no false positives), and for every fault the check the paper names as the
one that catches it does so on the N = 12 block. Runs on the committed CSVs
and artifacts; skips if they are not present (a clone of the public package
without the ``e1_runs`` artifacts).
"""

from __future__ import annotations

import pytest

pytest.importorskip("pennylane")

from experiments.system_paper import es1_fault_injection as es1  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (es1.E1_RESULTS / "e1_runs" / "N12_i00_Aheur_s42.npz").exists(),
    reason="E1 artifacts not present")

CHEAP = [c for c in es1.CHECKS if not c.expensive]


@pytest.fixture(scope="module")
def clean():
    return es1.load_record((12,))


@pytest.fixture(scope="module")
def clean_findings(clean):
    return {c.name: es1.run_check(c, clean) for c in CHEAP}


def test_the_committed_record_passes_every_cheap_check(clean_findings):
    assert {k: v for k, v in clean_findings.items() if v} == {}


def test_reference_rebuilt_from_derived_statistics_alone_matches(clean):
    """A clone has only ``data/derived``; the reference must come back bit for bit."""
    small = es1.Record(clean.rows, clean.reference[clean.reference.instance < 3].reset_index(drop=True),
                       clean.artifacts, clean.derived_dir)
    assert es1.check_reference_input(small) == []


@pytest.mark.parametrize("fault_name, expected_check", [
    ("window_mixin", "manifest"),
    ("optimizer_mixin", "manifest"),
    ("label_swap", "penalty"),
    ("angle_swap", "npz-internal"),
    ("batch_from_initial", "row-npz"),
    ("evaluator_from_Q", "scorer"),
    ("column_drop", "schema"),
    ("shot_seed_noop", "state-batch"),
    ("reference_stale", "reference-row"),
    ("price_perturbation", "input-digest"),
])
def test_each_fault_is_caught_by_the_named_check(clean, tmp_path, fault_name, expected_check):
    fault = next(f for f in es1.FAULTS if f.name == fault_name)
    rec = fault.apply(clean, tmp_path)
    found = {c.name: es1.run_check(c, rec) for c in CHEAP}
    first = next(c.name for c in CHEAP if found[c.name])
    assert first == expected_check, found


def test_faults_do_not_touch_the_clean_record(clean, tmp_path):
    before = clean.rows.copy(deep=True)
    a0 = {k: v["final_angles"].copy() for k, v in clean.artifacts.items()}
    for fault in es1.FAULTS:
        if fault.name == "price_perturbation":
            continue
        d = tmp_path / fault.name
        d.mkdir()
        fault.apply(clean, d)
    assert clean.rows.equals(before)
    assert all((clean.artifacts[k]["final_angles"] == v).all() for k, v in a0.items())


def test_price_perturbation_is_caught_by_the_input_digest(clean, tmp_path):
    """The instance digest covers tickers, not values: a moved statistic passes
    every check that reads only the record. The first pass of this harness
    found that (it was caught only by re-enumerating the reference); the
    input_digest column added afterwards catches it at the input stage."""
    fault = next(f for f in es1.FAULTS if f.name == "price_perturbation")
    rec = fault.apply(clean, tmp_path)
    # Checks that read only the record cannot see a moved statistic; the two
    # that read the pinned statistics (input-digest, and the portfolio-metric
    # identities added later) both do.
    record_only = [c for c in CHEAP if c.name not in ("input-digest", "identities")]
    assert all(es1.run_check(c, rec) == [] for c in record_only)
    findings = es1.check_input_digest(rec)
    assert findings and all("pinned statistics" in f for f in findings)
    assert any("sharpe" in f for f in es1.check_identities(rec))
    small = es1.Record(rec.rows, rec.reference, rec.artifacts, rec.derived_dir)
    assert es1.check_reference_input(small)


def test_angle_swap_is_also_caught_by_recomputing_the_anchor_run(clean, tmp_path):
    fault = next(f for f in es1.FAULTS if f.name == "angle_swap")
    rec = fault.apply(clean, tmp_path)
    findings = es1.check_state_recompute(rec, per_size=1)
    assert any("N12_i00_Aheur_s42" in f for f in findings)


def test_headline_moves_silently_for_the_setting_mixup(clean, tmp_path):
    """label_swap changes the A_heur-vs-A_margin contrast and no naive check sees it."""
    fault = next(f for f in es1.FAULTS if f.name == "label_swap")
    rec = fault.apply(clean, tmp_path)
    assert es1.headline(rec) != es1.headline(clean)
    assert es1.check_schema(rec) == []


def test_identities_hold_on_the_clean_record_and_see_a_moved_q_column(clean, tmp_path):
    """The identity check (added after the coverage scan) passes the committed
    N = 12 block and flags a Q_τ column that no longer equals 1-(1-P_τ)^S."""
    assert es1.check_identities(clean) == []
    rec = clean.copy("q_moved")
    i = rec.rows.index[rec.rows.final_P_tau_main > 1e-3][0]
    rec.rows.at[i, "final_Q_tau_main_1000"] *= 1 + 1e-3
    found = es1.check_identities(rec)
    assert found and "final_Q_tau_main_1000" in found[0]


def test_identities_see_a_moved_rank_and_a_moved_uniform_baseline(clean):
    rec = clean.copy("rank_moved")
    i = rec.rows.index[rec.rows.rank_best_feasible.notna()][0]
    rec.rows.at[i, "rank_best_feasible"] += 1
    assert any("rank" in f for f in es1.check_identities(rec))
    rec = clean.copy("baseline_moved")
    rec.reference.at[0, "P_F_uniform_all"] *= 1.001
    assert any("P_F_uniform_all" in f for f in es1.check_identities(rec))


# ----------------------------------------------------------------------
# The declarative control, and the sibling records
# ----------------------------------------------------------------------

def test_the_declarative_control_reports_nothing_on_the_committed_record(clean):
    """A control that fires on a clean record measures nothing."""
    assert es1.check_declarative(clean) == []


def test_every_declarative_finding_names_one_of_the_documented_families(clean, tmp_path):
    rec = es1.fault_column_drop(clean, tmp_path)
    found = es1.check_declarative(rec)
    assert found, "a dropped column is within the control's vocabulary"
    assert all(m.split(":")[0] in es1.DECLARATIVE_FAMILIES for m in found)


def test_the_control_does_not_see_the_two_faults_that_read_another_file(clean, tmp_path):
    """The increment the thirteen relations buy, pinned as the paper states it.

    A constraint language over one table reaches the other eight. These two are
    checked against something that table does not hold: the statistics behind an
    input's name, and the run's stored artifact.
    """
    for fault in (es1.fault_price_perturbation, es1.fault_angle_swap):
        where = tmp_path / fault.__name__
        where.mkdir(exist_ok=True)
        rec = fault(clean, where)
        assert es1.check_declarative(rec) == [], fault.__name__


def test_the_band_reaches_the_batch_drawn_from_the_initial_state(clean, tmp_path):
    """A row predicate with a tolerance computed from the row catches the replay."""
    where = tmp_path / "band"
    where.mkdir(exist_ok=True)
    found = es1.check_declarative(es1.fault_batch_from_initial(clean, where))
    assert any(m.startswith("band:") for m in found), found


def test_the_control_sees_the_two_families_it_was_given(clean, tmp_path):
    """A conditional equality and a tuple's uniqueness are in its vocabulary."""
    for fault, family in ((es1.fault_label_swap, "conditional"),
                          (es1.fault_shot_seed_noop, "unique")):
        where = tmp_path / fault.__name__
        where.mkdir(exist_ok=True)
        found = es1.check_declarative(fault(clean, where))
        assert any(m.startswith(family + ":") for m in found), (fault.__name__, found)


def test_a_sibling_arm_is_checked_against_its_own_window_and_manifest():
    """The prior-window replication has different statistics, references and digests."""
    if not (es1.E1_RESULTS / "e1_runs_e5" / "N12_i00_Aheur_s42.npz").exists():
        pytest.skip("E5 artifacts not present")
    rec = es1.load_record((12,), suffix="_e5", label="e1_e5")
    assert rec.end_date == "2023-08-05" != es1.e1.DATA_END_DATE
    assert rec.runs_expected and sum(rec.runs_expected.values()) == len(rec.rows)
    for name in ("manifest", "penalty", "reference-row", "input-digest", "scorer",
                 "row-npz", "npz-internal", "identities", "state-batch", "seed-distinct"):
        check = next(c for c in es1.CHECKS if c.name == name)
        assert es1.run_check(check, rec) == [], name


def test_only_the_batch_clause_of_seed_distinct_fires_on_the_shot_seed_replay(clean, tmp_path):
    """``shot_seed_noop`` copies a batch and leaves the angles, so the angle clause is silent."""
    found = es1.check_seed_distinct(es1.fault_shot_seed_noop(clean, tmp_path))
    assert found, "the batch clause should fire"
    assert all("shot batch" in m for m in found), found
    assert not any("initial angles" in m for m in found), found
