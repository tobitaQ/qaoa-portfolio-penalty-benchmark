# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
The hardware evidence has to outlive the bucket it was written to.

The IonQ shots of Sec. V-D (Supplementary Sec. S-VIII) were written to a Braket results bucket that is torn
down once the experiments are finished, and they cost $24.90 to produce. They
are archived under ``data/braket_results/`` and read back through
``shot_counts_from_dir``. These tests check that the archive still reproduces
the committed table, so a teardown does not quietly turn the hardware section
into numbers nobody can re-derive.
"""

from __future__ import annotations

import csv
import json
import pathlib
from pathlib import Path

import pytest

from src.solvers.braket_results import shot_counts_from_dir

ARCHIVE = Path(__file__).resolve().parents[1] / "data" / "braket_results"
HARDWARE_CSV = (Path(__file__).resolve().parents[1] / "experiments"
                / "paper01_qubo_baseline" / "results" / "paper01_hardware_ionq.csv")

def test_the_archive_is_present() -> None:
    """The archive is committed, so its absence is a failure and not a skip.

    This started as a skipif. That is why the loss of 2026-08-14 was invisible:
    an unanchored ``braket_results/`` in .gitignore meant ``git add -A`` never
    committed the archive, the suite went green by skipping, and the only copy
    died with the working tree. The tasks survived only because the S3 bucket
    was still up. A missing artifact must be loud.
    """
    assert ARCHIVE.exists(), f"{ARCHIVE} is missing; it is committed, so this is a loss"
    assert HARDWARE_CSV.exists(), f"{HARDWARE_CSV} is missing"
    tasks = sorted((ARCHIVE / "ionq").glob("*/results.json"))
    assert len(tasks) == 4, f"expected 4 archived IonQ tasks, found {len(tasks)}"


def _rows() -> list[dict]:
    with open(HARDWARE_CSV) as handle:
        return list(csv.DictReader(handle))


def _task_dir(row: dict) -> Path:
    return ARCHIVE / "ionq" / row["task_arn"].rsplit("/", 1)[1]


@pytest.mark.parametrize("row", _rows(), ids=lambda r: f"N{r['n_assets']}")
def test_archive_reproduces_the_reported_shots(row: dict) -> None:
    """Every counted quantity in the hardware table comes back out of the archive."""
    counts = shot_counts_from_dir(_task_dir(row))
    k = int(row["n_select"])

    assert sum(counts.values()) == int(row["shots"])
    assert len(counts) == int(row["distinct_bitstrings"])
    assert sum(v for bits, v in counts.items()
               if bits.count("1") == k) == int(row["feasible_shots"])


def test_archived_tasks_are_the_nested_program_set_schema() -> None:
    """The archive keeps the schema the plugin could not read, not a flattened copy.

    Flattening at archive time would erase the defect Supplementary Sec. S-VIII reports, and the
    reader under test would no longer exercise the branch that matters.
    """
    for row in _rows():
        result = json.loads((_task_dir(row) / "results.json").read_text())
        schema = result["braketSchemaHeader"]["name"]
        assert "program_set" in schema, schema
        assert (_task_dir(row) / "programs" / "0" / "results.json").exists()


def test_reader_rejects_a_directory_without_measurements(tmp_path: Path) -> None:
    """A truncated archive fails loudly rather than reporting zero shots."""
    (tmp_path / "results.json").write_text(json.dumps({
        "braketSchemaHeader": {"name": "braket.task_result.gate_model_task_result"},
        "taskMetadata": {"shots": 100},
    }))
    with pytest.raises(RuntimeError, match="no measurements"):
        shot_counts_from_dir(tmp_path)


class TestHardwareShotProvenance:
    """What the four IonQ tasks actually returned (review M9-a).

    Table XVI prints integers like "89 of 100 shots infeasible". A reviewer
    cannot tell from the paper whether those are measured shot records, counts
    the SDK reconstructed from returned probabilities, or something an
    error-mitigation setting produced. The archived task records answer it, so
    the answer is pinned here rather than left to a reading of the prose.
    """

    ROOT = pathlib.Path(__file__).resolve().parents[1] / "data" / "braket_results" / "ionq"

    def _tasks(self):
        return sorted(d for d in self.ROOT.iterdir() if d.is_dir())

    def test_there_are_four_tasks(self):
        assert len(self._tasks()) == 4

    def test_the_device_returned_probabilities_and_no_shot_sequence(self):
        for task in self._tasks():
            executable = json.loads(
                (task / "programs" / "0" / "executables" / "0.json").read_text())
            assert "measurementProbabilities" in executable
            assert "measurements" not in executable, (
                f"{task.name} has a shot sequence after all; the paper's "
                "provenance note needs rewriting")

    def test_the_probabilities_are_exact_hundredths_of_100_shots(self):
        """Which is what makes the reconstructed counts exact rather than estimates."""
        for task in self._tasks():
            meta = json.loads((task / "metadata.json").read_text())
            assert meta["requestedShots"] == 100
            assert meta["successfulShots"] == 100
            probs = json.loads(
                (task / "programs" / "0" / "executables" / "0.json").read_text()
            )["measurementProbabilities"]
            counts = [p * 100 for p in probs.values()]
            for c in counts:
                assert abs(c - round(c)) < 1e-9, f"{task.name}: {c} is not a whole shot"
            assert round(sum(counts)) == 100

    def test_no_error_mitigation_was_requested(self):
        """A debiasing setting would make the returned numbers post-processed."""
        for task in self._tasks():
            meta = json.loads((task / "metadata.json").read_text())
            assert meta.get("deviceParameters") in (None, {}, "None")
