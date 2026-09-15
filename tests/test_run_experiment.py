# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Tests for the paper ① experiment runner's output routing.

The runner's two CSVs are the source of Table I and Figures 1-4. A cloud run
(Step 4-B compares SV1 against the GPU baseline) must land somewhere else, or
the comparison destroys its own reference.
"""

from __future__ import annotations

import csv

import pytest

from experiments.paper01_qubo_baseline.run_experiment import (
    RESULT_KEY,
    RESULTS_DIR,
    _result_sort_key,
    check_mergeable,
    main,
    merge_rows,
    parse_args,
    results_paths,
)


class TestResultsPaths:
    def test_untagged_run_writes_the_canonical_pair(self):
        results, convergence = results_paths(None)
        assert results == RESULTS_DIR / "paper01_results.csv"
        assert convergence == RESULTS_DIR / "paper01_convergence.csv"

    def test_tag_suffixes_both_files(self):
        results, convergence = results_paths("sv1")
        assert results == RESULTS_DIR / "paper01_results_sv1.csv"
        assert convergence == RESULTS_DIR / "paper01_convergence_sv1.csv"

    @pytest.mark.parametrize("tag", ["", "../paper01", "a b", "sv1/x"])
    def test_tag_cannot_escape_the_results_directory(self, tag):
        with pytest.raises(ValueError):
            results_paths(tag)


class TestCloudGuard:
    def test_cloud_backend_without_a_tag_refuses_to_run(self, capsys):
        """Never let a paid run overwrite the published CSVs."""
        code = main([
            "--backend", "braket_sv1",
            "--s3-bucket", "some-bucket",
            "--quantum-n", "8",
            "--classical-n", "8",
        ])
        assert code == 2
        assert "--results-tag" in capsys.readouterr().out
        # It must bail out before touching AWS or the canonical files.
        assert (RESULTS_DIR / "paper01_results.csv").exists()

    def test_local_backend_still_defaults_to_the_canonical_pair(self):
        args = parse_args(["--backend", "lightning_gpu"])
        assert args.results_tag is None
        assert results_paths(args.results_tag)[0].name == "paper01_results.csv"


class TestMergeRows:
    """--merge must refresh recomputed rows and leave the rest untouched.

    The reason it exists: N=30 QAOA is 3.14 h on an A6000, so a classical-side
    rerun that dropped the quantum rows would cost a day to undo.
    """

    @staticmethod
    def _row(n, solver, energy, *, window=("2023-08-06", "2026-08-05")):
        return {
            "n_assets": n,
            "n_select": 2,
            "solver": solver,
            "energy_no_offset": energy,
            "runtime_s": 1.0,
            "feasible": True,
            "expected_return": 0.1,
            "volatility": 0.2,
            "sharpe_ratio": 0.5,
            "method_detail": "x",
            "data_start": window[0],
            "data_end": window[1],
        }

    def test_quantum_rows_survive_a_classical_rerun(self):
        existing = [
            self._row(8, "classical_auto", -1.0),
            self._row(8, "lightning_gpu", -0.9),
            self._row(30, "classical_auto", -3.0),
            self._row(30, "lightning_gpu", -2.9),
        ]
        fresh = [self._row(8, "classical_auto", -1.5),
                 self._row(30, "classical_auto", -3.5)]

        merged = merge_rows(existing, fresh, RESULT_KEY, _result_sort_key)

        assert len(merged) == 4
        by_key = {(r["n_assets"], r["solver"]): r for r in merged}
        assert by_key[(8, "classical_auto")]["energy_no_offset"] == -1.5
        assert by_key[(30, "classical_auto")]["energy_no_offset"] == -3.5
        # Untouched rows keep their exact values.
        assert by_key[(8, "lightning_gpu")]["energy_no_offset"] == -0.9
        assert by_key[(30, "lightning_gpu")]["energy_no_offset"] == -2.9

    def test_merge_is_ordered_by_size_with_classical_first(self):
        existing = [self._row(30, "lightning_gpu", -2.9), self._row(8, "classical_auto", -1.0)]
        fresh = [self._row(30, "classical_auto", -3.0), self._row(8, "lightning_gpu", -0.9)]

        merged = merge_rows(existing, fresh, RESULT_KEY, _result_sort_key)

        assert [(r["n_assets"], r["solver"]) for r in merged] == [
            (8, "classical_auto"), (8, "lightning_gpu"),
            (30, "classical_auto"), (30, "lightning_gpu"),
        ]

    def test_string_rows_read_from_csv_match_freshly_typed_rows(self):
        """Rows come back from disk as strings; keys must still collide."""
        existing = [{k: str(v) for k, v in self._row(8, "classical_auto", -1.0).items()}]
        fresh = [self._row(8, "classical_auto", -1.5)]

        merged = merge_rows(existing, fresh, RESULT_KEY, _result_sort_key)

        assert len(merged) == 1
        assert merged[0]["energy_no_offset"] == -1.5

    def test_a_different_price_window_is_refused(self):
        """Same N over a different window is a different QUBO, not an update."""
        existing = [self._row(8, "lightning_gpu", -0.9)]
        fresh = [self._row(8, "classical_auto", -1.0, window=("2023-01-01", "2026-01-01"))]

        with pytest.raises(ValueError, match="different problem instances"):
            check_mergeable(existing, fresh)

    def test_a_changed_column_set_is_refused(self):
        existing = [self._row(8, "lightning_gpu", -0.9)]
        fresh = [dict(self._row(8, "classical_auto", -1.0), extra_column=1)]

        with pytest.raises(ValueError, match="column mismatch"):
            check_mergeable(existing, fresh)

    def test_merging_into_nothing_just_writes_the_fresh_rows(self):
        fresh = [self._row(8, "classical_auto", -1.0)]
        assert merge_rows([], fresh, RESULT_KEY, _result_sort_key) == fresh


class TestMergeEndToEnd:
    def test_classical_only_merge_keeps_the_quantum_row_verbatim(self, tmp_path, monkeypatch):
        """Full runner path: --skip-quantum --merge must not drop QAOA rows."""
        import experiments.paper01_qubo_baseline.run_experiment as mod

        monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path)
        results_csv = tmp_path / "paper01_results.csv"

        # Seed a table that already holds an expensive quantum row.
        assert main(["--classical-n", "8", "--quantum-n", "8",
                     "--backend", "lightning_cpu", "--steps", "2",
                     "--shots", "50"]) == 0
        before = list(csv.DictReader(results_csv.open()))
        quantum_before = [r for r in before if not r["solver"].startswith("classical")]
        assert quantum_before, "fixture needs a quantum row to protect"

        # Recompute the classical side only.
        assert main(["--classical-n", "8", "--skip-quantum", "--merge",
                     "--write-canonical"]) == 0

        after = list(csv.DictReader(results_csv.open()))
        quantum_after = [r for r in after if not r["solver"].startswith("classical")]
        assert quantum_after == quantum_before, "quantum rows changed under --merge"

    def test_without_merge_a_classical_rerun_still_replaces_the_file(self, tmp_path, monkeypatch):
        """Replacing is still possible — --merge is opt-in, --write-canonical is the permission."""
        import experiments.paper01_qubo_baseline.run_experiment as mod

        monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path)
        results_csv = tmp_path / "paper01_results.csv"

        assert main(["--classical-n", "8", "--quantum-n", "8",
                     "--backend", "lightning_cpu", "--steps", "2",
                     "--shots", "50"]) == 0
        assert main(["--classical-n", "8", "--skip-quantum", "--write-canonical"]) == 0

        after = list(csv.DictReader(results_csv.open()))
        assert all(r["solver"].startswith("classical") for r in after)


class TestCanonicalGuard:
    """An untagged run must not replace the committed baseline by accident.

    A clone holds the canonical CSVs, so the first thing a reader's trial run
    would do — before the guard — was overwrite the numbers they came to check.
    """

    def test_an_untagged_run_over_existing_files_is_refused(self, tmp_path, monkeypatch, capsys):
        import experiments.paper01_qubo_baseline.run_experiment as mod

        monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path)
        assert main(["--classical-n", "8", "--skip-quantum"]) == 0
        results_csv = tmp_path / "paper01_results.csv"
        before = results_csv.read_bytes()

        assert main(["--classical-n", "8", "--skip-quantum"]) == 2
        out = capsys.readouterr().out
        assert "--write-canonical" in out and "--results-tag" in out
        assert results_csv.read_bytes() == before

    def test_merge_is_not_a_loophole(self, tmp_path, monkeypatch):
        """--merge rewrites the file too, so it needs the same permission."""
        import experiments.paper01_qubo_baseline.run_experiment as mod

        monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path)
        assert main(["--classical-n", "8", "--skip-quantum"]) == 0
        assert main(["--classical-n", "8", "--skip-quantum", "--merge"]) == 2

    def test_a_tagged_run_writes_beside_the_baseline(self, tmp_path, monkeypatch):
        import experiments.paper01_qubo_baseline.run_experiment as mod

        monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path)
        assert main(["--classical-n", "8", "--skip-quantum"]) == 0
        before = (tmp_path / "paper01_results.csv").read_bytes()
        assert main(["--classical-n", "8", "--skip-quantum",
                     "--results-tag", "reproduction"]) == 0
        assert (tmp_path / "paper01_results_reproduction.csv").exists()
        assert (tmp_path / "paper01_results.csv").read_bytes() == before

    def test_the_committed_baseline_is_guarded(self):
        """The real results directory: the untagged names exist, so a bare run is refused."""
        from experiments.paper01_qubo_baseline.run_experiment import canonical_overwrite_refusal
        paths = results_paths(None)
        assert all(p.exists() for p in paths)
        assert canonical_overwrite_refusal(paths, None, False) is not None
        assert canonical_overwrite_refusal(paths, "reproduction", False) is None
        assert canonical_overwrite_refusal(paths, None, True) is None


class TestHardwareGuardCLI:
    """The runner refuses a QPU in the optimizer slot before building anything."""

    def test_qpu_as_backend_is_refused(self, capsys):
        code = main(["--backend", "braket_ionq", "--quantum-n", "8",
                     "--results-tag", "ionq", "--s3-bucket", "b"])
        assert code == 2
        out = capsys.readouterr().out
        assert "--sample-backend braket_ionq" in out

    def test_a_qpu_sampling_backend_still_needs_a_results_tag(self, capsys):
        """The cloud guard must look at both slots, not just --backend."""
        code = main(["--sample-backend", "braket_ionq", "--quantum-n", "8"])
        assert code == 2
        assert "--results-tag is required" in capsys.readouterr().out

    def test_a_local_sampling_backend_needs_no_tag(self):
        args = parse_args(["--sample-backend", "braket_local"])
        assert args.sample_backend == "braket_local"
        assert args.results_tag is None
