# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Tests for recovering a billed Braket task's measurements from S3.

The recovery path exists because a paid hardware run was nearly lost: IonQ
returns the nested ``program_set_task_result`` schema, the PennyLane plugin did
not read it, and the solver reported an all-zero bitstring instead of raising.
The shots were in S3 the whole time. These tests pin the schema handling, since
getting it wrong silently is exactly the failure being recovered from.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from experiments.paper01_qubo_baseline.recover_task import summarise
from src.solvers.braket_results import samples_from_counts, shot_counts_from_task
from src.finance.metrics import PortfolioMetrics
from src.qubo.portfolio import PortfolioQUBO


class _FakeBraket:
    def __init__(self, status="COMPLETED", shots=100):
        self._status, self._shots = status, shots

    def get_quantum_task(self, quantumTaskArn):
        return {"status": self._status, "shots": self._shots,
                "outputS3Bucket": "bucket", "outputS3Directory": "prefix"}


def _install(monkeypatch, objects, status="COMPLETED", shots=100):
    """Point the recovery at an in-memory S3 and Braket."""
    import src.solvers.braket_results as mod

    class _FakeSession:
        def __init__(self, region_name=None):
            pass

        def client(self, name):
            assert name == "braket"
            return _FakeBraket(status, shots)

    monkeypatch.setitem(__import__("sys").modules, "boto3",
                        type("m", (), {"Session": _FakeSession}))
    monkeypatch.setattr(mod, "_read_s3_json",
                        lambda session, bucket, key: objects[key])


class TestSchemaHandling:
    def test_the_flat_gate_model_schema_gives_one_row_per_shot(self, monkeypatch):
        objects = {"prefix/results.json": {
            "braketSchemaHeader": {"name": "braket.task_result.gate_model_task_result"},
            "measurements": [[0, 1], [0, 1], [1, 0]],
        }}
        _install(monkeypatch, objects, shots=3)
        assert shot_counts_from_task("arn") == Counter({"01": 2, "10": 1})

    def test_the_nested_program_set_schema_is_followed_to_its_leaf(self, monkeypatch):
        """IonQ's shape: results -> programs/0/results -> executables/0."""
        objects = {
            "prefix/results.json": {
                "braketSchemaHeader": {
                    "name": "braket.task_result.program_set_task_result"},
                "programResults": ["programs/0/results.json"],
            },
            "prefix/programs/0/results.json": {
                "executableResults": ["executables/0.json"]},
            "prefix/programs/0/executables/0.json": {
                "measuredQubits": [0, 1, 2],
                "measurementProbabilities": {"000": 0.25, "011": 0.75},
            },
        }
        _install(monkeypatch, objects, shots=100)
        assert shot_counts_from_task("arn") == Counter({"000": 25, "011": 75})

    def test_probabilities_round_back_to_the_requested_shot_total(self, monkeypatch):
        objects = {
            "prefix/results.json": {
                "braketSchemaHeader": {
                    "name": "braket.task_result.program_set_task_result"},
                "programResults": ["programs/0/results.json"]},
            "prefix/programs/0/results.json": {
                "executableResults": ["executables/0.json"]},
            "prefix/programs/0/executables/0.json": {
                "measurementProbabilities": {"00": 0.14, "01": 0.08, "11": 0.78}},
        }
        _install(monkeypatch, objects, shots=100)
        assert sum(shot_counts_from_task("arn").values()) == 100

    def test_an_incomplete_task_is_refused(self, monkeypatch):
        _install(monkeypatch, {}, status="FAILED")
        with pytest.raises(RuntimeError, match="not COMPLETED"):
            shot_counts_from_task("arn")

    def test_a_result_without_measurements_is_refused(self, monkeypatch):
        """Better to fail than to return an empty Counter that scores as zeros."""
        objects = {"prefix/results.json": {
            "braketSchemaHeader": {"name": "braket.task_result.gate_model_task_result"},
        }}
        _install(monkeypatch, objects)
        with pytest.raises(RuntimeError, match="no measurements"):
            shot_counts_from_task("arn")


class _Data:
    def __init__(self, returns, covariance):
        self.returns, self.covariance = returns, covariance
        self.start_date, self.end_date = "2023-08-06", "2026-08-05"


class TestSummarise:
    @staticmethod
    def _instance():
        rng = np.random.default_rng(9)
        n, k = 6, 2
        returns = rng.uniform(0.05, 0.30, n)
        cov = np.eye(n) * 0.04
        problem = PortfolioQUBO().formulate(returns, cov, num_select=k)
        return problem, _Data(returns, cov), k

    def test_the_best_feasible_shot_wins_not_the_best_shot(self):
        """A lower-energy infeasible string must not become the answer."""
        problem, data, k = self._instance()
        n = problem.n_variables
        best = np.zeros(n)
        best[[0, 1, 2]] = 1.0          # K+1 assets: lower energy, infeasible
        feasible = np.zeros(n)
        feasible[[0, 1]] = 1.0
        counts = Counter({"".join(str(int(b)) for b in best): 60,
                          "".join(str(int(b)) for b in feasible): 40})

        row = summarise(counts, problem, data, k, PortfolioMetrics(0.001))

        assert row["selected_assets"] == "0 1"
        assert row["feasible_shots"] == 40
        assert row["feasible_fraction"] == pytest.approx(0.4)
        assert row["best_energy_any_cardinality"] <= row["energy_quantum"]

    def test_no_feasible_shot_raises_rather_than_inventing_one(self):
        problem, data, k = self._instance()
        counts = Counter({"111111": 100})
        with pytest.raises(RuntimeError, match="no feasible shot"):
            summarise(counts, problem, data, k, PortfolioMetrics(0.001))

    def test_the_gap_is_measured_against_the_exact_optimum(self):
        problem, data, k = self._instance()
        n = problem.n_variables
        x = np.zeros(n)
        x[[0, 1]] = 1.0
        counts = Counter({"".join(str(int(b)) for b in x): 100})

        row = summarise(counts, problem, data, k, PortfolioMetrics(0.001))

        assert row["gap_pct"] == pytest.approx(
            abs(row["energy_quantum"] - row["energy_classical"])
            / abs(row["energy_classical"]) * 100)
        assert row["gap_pct"] >= 0.0

    def test_shot_statistics_describe_the_whole_batch(self):
        problem, data, k = self._instance()
        counts = Counter({"110000": 30, "101000": 20, "111000": 50})
        row = summarise(counts, problem, data, k, PortfolioMetrics(0.001))
        assert row["shots"] == 100
        assert row["distinct_bitstrings"] == 3
        assert row["feasible_shots"] == 50


class TestSamplesFromCounts:
    """Counts -> one row per shot, which is what the solver scores."""

    def test_counts_expand_to_one_row_per_shot(self):
        rows = samples_from_counts(Counter({"01": 2, "10": 1}), n=2)
        assert rows.shape == (3, 2)
        assert sorted(map(tuple, rows)) == [(0.0, 1.0), (0.0, 1.0), (1.0, 0.0)]

    def test_bit_j_is_qubit_j(self):
        """Ordering decides which asset was selected; getting it silently wrong
        was the failure mode this whole path exists to avoid."""
        rows = samples_from_counts(Counter({"1000": 1}), n=4)
        assert list(rows[0]) == [1.0, 0.0, 0.0, 0.0]

    def test_a_wrong_width_is_refused(self):
        with pytest.raises(RuntimeError, match="expected 4"):
            samples_from_counts(Counter({"101": 1}), n=4)

    def test_no_shots_is_refused(self):
        with pytest.raises(RuntimeError, match="no shots"):
            samples_from_counts(Counter(), n=4)
