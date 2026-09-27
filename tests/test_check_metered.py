# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT
"""R14: acknowledged task count versus the provider's ledger; the estimate is persisted."""

import json
from types import SimpleNamespace

from experiments.paper01_qubo_baseline import check_metered, run_experiment


def test_r14_flags_the_historical_pilot_and_passes_the_corrected_run():
    assert check_metered.check(51, 227)["status"] == "fail"
    assert check_metered.check(204, 204)["status"] == "pass"


def test_estimate_is_written_with_the_expected_task_count(tmp_path, monkeypatch):
    monkeypatch.setattr(run_experiment, "RESULTS_DIR", tmp_path)
    args = SimpleNamespace(backend="braket_sv1", sample_backend=None, steps=50, shots=1000)
    est = run_experiment.cloud_estimate([8], 50, 1000, "braket_sv1", None)
    path = run_experiment.write_metered_estimate("pilot", est, [8], args)
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["expected_tasks"] == 51 and d["results_tag"] == "pilot" and d["usd_floor"] > 0
