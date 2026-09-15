# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""The two facts behind "177 tasks to sample one bitstring", each on its own evidence."""

from datetime import datetime, timedelta, timezone
from math import comb

import pytest

from experiments.paper01_qubo_baseline import repro_parameter_shift_fallback as repro
from experiments.paper01_qubo_baseline import sv1_task_ledger as ledger


def test_param_shift_returns_two_tapes_per_gate_parameter():
    qml = pytest.importorskip("pennylane")
    n_params, n_tapes, census = repro.param_shift_tape_count(8, 2, qml)
    assert n_params == 2 * (comb(8, 2) + 2 * 8) == 88
    assert n_tapes == 2 * n_params == 176
    assert census == {"Hadamard": 8, "PauliRot": 88}


def test_ledger_groups_tasks_into_runs_by_submission_gap():
    t0 = datetime(2026, 8, 9, 8, 47, tzinfo=timezone.utc)
    tasks = [(t0 + timedelta(seconds=5 * k), 0, f"arn:aws:braket:us-east-1:123456789012:quantum-task/a{k}")
             for k in range(50)]
    tasks += [(t0 + timedelta(seconds=250 + 3 * k), 1000, f"arn:x:{k}") for k in range(177)]
    tasks += [(t0 + timedelta(hours=1), 0, "arn:y"), (t0 + timedelta(hours=1, seconds=4), 1000, "arn:z")]
    rows = ledger.ledger(sorted(tasks), gap=120.0)
    assert [(r["tasks"], r["analytic_tasks"], r["sampling_tasks"]) for r in rows] == [(227, 50, 177), (2, 1, 1)]
    assert rows[0]["shots_total"] == 177_000
    # The account id never reaches a committed file.
    assert ":000000000000:" in rows[0]["first_task_id"] and "123456789012" not in rows[0]["first_task_id"]


def test_committed_ledger_records_the_pilot():
    import csv
    from experiments.paper01_qubo_baseline.run_e1_cross import RESULTS_DIR
    path = RESULTS_DIR / "paper01_sv1_task_ledger.csv"
    if not path.exists():
        pytest.skip("ledger not present")
    rows = list(csv.DictReader(open(path)))
    first = rows[0]
    assert (int(first["tasks"]), int(first["analytic_tasks"]), int(first["sampling_tasks"])) == (227, 50, 177)
    assert sum(int(r["tasks"]) for r in rows) == 1299
