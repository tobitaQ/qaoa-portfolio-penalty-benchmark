# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Regression tests for the injection gates of the input-rerun fault.

Four attempts at this experiment failed before it worked, and every one failed
in the same way: the injection was not the shape it was meant to be, and the
numbers that came back were read as a result anyway.

  1. The re-run used a different backend and precision from the committed
     record, so the five headline numbers that moved had moved because of the
     backend.
  2. The perturbed ticker was in neither of the two instances being re-run, so
     the re-run read the committed inputs and nothing happened.
  3. Only the derived-statistics directory was redirected, leaving the price
     cache to be read first, so again nothing happened.
  4. Only two of the six penalty settings were re-run, so the instance's other
     rows kept the old input digest beside the new reference and the
     declarative control fired on that rather than on the fault.

(1) and (4) are now gates that abort. (2) is prevented by drawing the ticker
from the instances' shared set. (3) is prevented by going through the harness's
own loader context. These tests hold all four.
"""

from __future__ import annotations

import pandas as pd
import pytest

from experiments.system_paper import es1_fault_injection as es1
from experiments.system_paper import es1_input_rerun_fault as rr


def _rec(rows: pd.DataFrame) -> es1.Record:
    return es1.Record(rows=rows, reference=pd.DataFrame(), artifacts={},
                      derived_dir=rr.RESULTS, label="t")


def _rows(n_rows: int, **over) -> pd.DataFrame:
    base = {"run_id": [f"r{i}" for i in range(n_rows)], "n_assets": 12, "instance": 0,
            "backend": "lightning_gpu", "precision": "single", "steps": 50,
            "p_layers": 2, "shots": 1000, "optimizer": "adam",
            "input_digest": "aaaa", "final_P_F": 0.5, "g_off": 0.1}
    base.update(over)
    return pd.DataFrame(base)


def test_a_changed_backend_aborts() -> None:
    clean = _rec(_rows(4))
    arm = _rec(_rows(4, backend="lightning_cpu"))
    with pytest.raises(SystemExit) as e:
        rr.check_injection(clean, arm)
    assert "backend" in str(e.value)


def test_a_changed_precision_aborts() -> None:
    clean = _rec(_rows(4))
    arm = _rec(_rows(4, precision="double"))
    with pytest.raises(SystemExit) as e:
        rr.check_injection(clean, arm)
    assert "precision" in str(e.value)


def test_a_block_covering_an_instance_in_part_aborts() -> None:
    """The failure that made the declarative control fire on the splice."""
    clean = _rec(_rows(6))
    arm = _rec(_rows(2, run_id=["r0", "r1"], input_digest="bbbb"))
    with pytest.raises(SystemExit) as e:
        rr.check_injection(clean, arm)
    assert "in part" in str(e.value)


def test_two_digests_in_one_instance_abort() -> None:
    clean = _rec(_rows(4))
    arm = _rec(_rows(4, input_digest=["bbbb", "bbbb", "cccc", "cccc"]))
    with pytest.raises(SystemExit) as e:
        rr.check_injection(clean, arm)
    assert "digest" in str(e.value)


def test_a_faithful_injection_passes_and_reports_what_changed() -> None:
    clean = _rec(_rows(4))
    arm = _rec(_rows(4, input_digest="bbbb", final_P_F=[0.5, 0.5, 0.6, 0.7]))
    gate = rr.check_injection(clean, arm)
    assert gate["inputs_changed"] is True
    assert gate["rows_rerun"] == 4
    assert gate["rows_with_changed_state"] == 2


def test_an_ineffective_perturbation_is_reported_not_aborted() -> None:
    """A delta too small to change the inputs is a finding, not a crash."""
    clean = _rec(_rows(4))
    gate = rr.check_injection(clean, _rec(_rows(4)))
    assert gate["inputs_changed"] is False
    assert gate["rows_with_changed_state"] == 0


def test_the_ticker_is_drawn_from_the_instances_that_hold_it(tmp_path) -> None:
    """The failure that made the perturbation miss both instances."""
    src = tmp_path / "src"
    src.mkdir()
    frame = pd.DataFrame({"mu": [0.1, 0.2, 0.3]}, index=["A.T", "B.T", "C.T"])
    frame.index.name = "ticker"
    frame.to_csv(src / f"stats_2023-08-06_{rr.WINDOW_END}_3_deadbeef.csv")
    for seed in range(12):
        info = rr.perturb_statistics(src, tmp_path / f"d{seed}", 1e-6, seed, ["B.T"])
        assert info["cell"] == "mu[B.T]", info
