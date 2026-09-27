# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT
"""Random-mutation harness: operators mutate a copy, never the clean record."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pennylane")

from experiments.system_paper import es1_fault_injection as es1  # noqa: E402
from experiments.system_paper import es1_random_faults as rf  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (es1.E1_RESULTS / "e1_runs" / "N12_i00_Aheur_s42.npz").exists(),
    reason="E1 artifacts not present")


@pytest.fixture(scope="module")
def clean():
    return es1.load_record((12,))


def test_every_operator_leaves_the_clean_record_untouched(clean, tmp_path):
    rng = np.random.default_rng(1)
    rows_before = clean.rows.copy(deep=True)
    ref_before = clean.reference.copy(deep=True)
    arts_before = {k: {a: v.copy() for a, v in d.items()} for k, d in clean.artifacts.items()}
    for name, op in rf.OPERATORS.items():
        produced = 0
        for _ in range(20):
            out = op(clean, rng, tmp_path)
            if out is not None:
                produced += 1
                break
        assert produced, f"{name} produced no mutation in 20 draws"
    assert clean.rows.equals(rows_before) and clean.reference.equals(ref_before)
    assert all((clean.artifacts[k][a] == v).all() for k, d in arts_before.items() for a, v in d.items())


def test_light_copy_isolates_only_the_touched_run(clean):
    rid = clean.rows.at[0, "run_id"]
    other = clean.rows.at[1, "run_id"]
    c = rf.light_copy(clean, "x", touch=(rid,))
    assert c.artifacts[rid] is not clean.artifacts[rid]
    assert c.artifacts[other] is clean.artifacts[other]
    c.artifacts[rid]["final_angles"][0] += 1.0
    assert c.artifacts[rid]["final_angles"][0] != clean.artifacts[rid]["final_angles"][0]


def test_perturb_value_changes_every_kind():
    rng = np.random.default_rng(0)
    assert rf.perturb_value(True, 1e-3, rng, False) is False
    assert rf.perturb_value("Aheur", 1e-3, rng, False) != "Aheur"
    assert rf.perturb_value(5, 1e-3, rng, True) in (4, 6)
    assert rf.perturb_value(0.0, 1e-3, rng, False) == 1e-3
    assert rf.perturb_value(2.0, 1e-3, rng, False) != 2.0


def test_summary_counts(tmp_path):
    faults = [{"operator": "a", "magnitude": 1e-3, "detected": "yes", "headline_numbers_moved": 0},
              {"operator": "a", "magnitude": 1e-3, "detected": "no", "headline_numbers_moved": 2},
              {"operator": "b", "magnitude": "", "detected": "yes", "headline_numbers_moved": 1}]
    s = {(r["group"], str(r["value"])): r for r in rf.summarise(faults)}
    assert s[("operator", "a")]["rate"] == 0.5 and s[("operator", "a")]["moved_and_undetected"] == 1
    assert s[("all", "")]["detected"] == 2 and s[("magnitude", "0.001")]["n"] == 2
