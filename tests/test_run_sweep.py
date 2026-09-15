# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Tests for the paper ① sweep driver's pure helpers.

These cover plan construction, resume de-duplication, the runtime estimator and
the gap formula. None of them need a quantum backend, so they run on any machine.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from src.qubo.portfolio import PortfolioQUBO
from src.solvers.classical_solver import ClassicalSolver

from experiments.paper01_qubo_baseline.run_sweep import (
    IncrementalCSV,
    MEASURED_RUNTIME_S,
    RUN_KEY_FIELDS,
    SweepRun,
    build_plan,
    estimate_cloud_cost,
    estimate_hardware_cost,
    estimate_runtime_s,
    format_duration,
    load_completed_keys,
    main,
    optimality_gap_pct,
    parse_args,
    sweep_csv_names,
)


class TestBuildPlan:
    def test_seeds_sweep_is_sizes_times_seeds_at_fixed_depth(self):
        args = parse_args(["--sweep", "seeds", "--seed-n", "8", "12",
                           "--seeds", "42", "43", "44", "--p-layers", "2"])
        plan = build_plan(args)

        assert len(plan) == 6
        assert {r.sweep for r in plan} == {"seeds"}
        assert {r.p_layers for r in plan} == {2}
        assert {(r.n_assets, r.qaoa_seed) for r in plan} == {
            (8, 42), (8, 43), (8, 44), (12, 42), (12, 43), (12, 44)}

    def test_depth_sweep_is_sizes_times_p_at_fixed_seed(self):
        args = parse_args(["--sweep", "depth", "--depth-n", "16",
                           "--p-values", "1", "2", "3", "--seed", "7"])
        plan = build_plan(args)

        assert [r.p_layers for r in plan] == [1, 2, 3]
        assert {r.qaoa_seed for r in plan} == {7}
        assert {r.sweep for r in plan} == {"depth"}

    def test_both_concatenates_the_two_sweeps(self):
        args = parse_args(["--sweep", "both", "--seed-n", "8",
                           "--seeds", "42", "--depth-n", "8",
                           "--p-values", "1", "2"])
        plan = build_plan(args)

        assert [r.sweep for r in plan] == ["seeds", "depth", "depth"]

    def test_sizes_are_ordered_ascending(self):
        args = parse_args(["--sweep", "seeds", "--seed-n", "20", "8", "16",
                           "--seeds", "42"])
        assert [r.n_assets for r in build_plan(args)] == [8, 16, 20]


class TestResume:
    def test_completed_keys_round_trip_through_csv(self, tmp_path):
        path = tmp_path / "sweep.csv"
        writer = IncrementalCSV(path)
        writer.append({"sweep": "seeds", "n_assets": 8, "n_select": 2,
                       "qaoa_seed": 42, "p_layers": 2, "steps": 50,
                       "gap_pct": 0.0})
        writer.close()

        assert load_completed_keys([path]) == {("seeds", "8", "42", "2", "50")}

    def test_missing_file_contributes_nothing(self, tmp_path):
        assert load_completed_keys([tmp_path / "absent.csv"]) == set()

    def test_run_key_matches_the_csv_columns(self, tmp_path):
        run = SweepRun("depth", 16, 42, 3, 50)
        path = tmp_path / "sweep.csv"
        writer = IncrementalCSV(path)
        writer.append({field: value for field, value in
                       zip(RUN_KEY_FIELDS, run.key())})
        writer.close()

        assert run.key() in load_completed_keys([path])

    def test_resume_filters_the_plan(self, tmp_path):
        path = tmp_path / "sweep.csv"
        writer = IncrementalCSV(path)
        writer.append({"sweep": "seeds", "n_assets": 8, "qaoa_seed": 42,
                       "p_layers": 2, "steps": 50})
        writer.close()
        done = load_completed_keys([path])

        args = parse_args(["--sweep", "seeds", "--seed-n", "8",
                           "--seeds", "42", "43"])
        remaining = [r for r in build_plan(args) if r.key() not in done]

        assert [r.qaoa_seed for r in remaining] == [43]

    def test_the_v02_sweep_csvs_still_resume(self):
        """The committed seed/depth CSVs must not all look unfinished.

        RUN_KEY_FIELDS gained "steps" for the grid sweep. Those files were
        written before that change, so if they lacked the column every completed
        run would silently re-execute — hours of GPU time at the larger sizes.
        """
        from experiments.paper01_qubo_baseline.run_sweep import (
            DEPTH_SWEEP_CSV, RESULTS_DIR, SEED_SWEEP_CSV)

        paths = [RESULTS_DIR / SEED_SWEEP_CSV, RESULTS_DIR / DEPTH_SWEEP_CSV]
        if not all(p.exists() for p in paths):
            pytest.skip("v0.2 sweep CSVs not present")

        done = load_completed_keys(paths)
        assert len(done) == 32, "expected the 20 seed + 12 depth runs"
        assert ("depth", "20", "42", "4", "50") in done

        args = parse_args(["--sweep", "both"])
        assert [r for r in build_plan(args) if r.key() not in done] == []


class TestGridSweep:
    def test_grid_is_sizes_times_steps_times_depths(self):
        args = parse_args(["--sweep", "grid", "--grid-n", "12",
                           "--grid-p", "1", "2", "--grid-steps", "50", "100"])
        plan = build_plan(args)

        assert {r.sweep for r in plan} == {"grid"}
        assert {(r.p_layers, r.steps) for r in plan} == {
            (1, 50), (2, 50), (1, 100), (2, 100)}

    def test_cheapest_step_counts_come_first(self):
        """A grid cut short should still cover every p at the low step counts."""
        args = parse_args(["--sweep", "grid", "--grid-n", "12",
                           "--grid-p", "1", "2", "--grid-steps", "400", "50",
                           "--grid-seeds", "42"])
        assert [r.steps for r in build_plan(args)] == [50, 50, 400, 400]

    def test_grid_run_uses_its_own_step_count_not_the_global_one(self):
        args = parse_args(["--sweep", "grid", "--grid-n", "12", "--grid-p", "2",
                           "--grid-steps", "200", "--steps", "50",
                           "--grid-seeds", "42"])
        assert [r.steps for r in build_plan(args)] == [200]

    def test_same_depth_at_different_steps_are_distinct_runs(self):
        """Without steps in the key, --resume would drop all but one."""
        args = parse_args(["--sweep", "grid", "--grid-n", "12", "--grid-p", "2",
                           "--grid-steps", "50", "100", "200",
                           "--grid-seeds", "42"])
        plan = build_plan(args)
        assert len({r.key() for r in plan}) == len(plan) == 3

    def test_grid_runs_every_cell_at_every_seed(self):
        """One seed per cell cannot separate depth from initialization."""
        args = parse_args(["--sweep", "grid", "--grid-n", "12",
                           "--grid-p", "1", "2", "--grid-steps", "50",
                           "--grid-seeds", "42", "43", "44"])
        plan = build_plan(args)
        assert len(plan) == 6
        assert {(r.p_layers, r.qaoa_seed) for r in plan} == {
            (p, s) for p in (1, 2) for s in (42, 43, 44)}

    def test_grid_seeds_are_distinct_runs_for_resume(self):
        """Without the seed in the key, --resume would keep only the first."""
        args = parse_args(["--sweep", "grid", "--grid-n", "12", "--grid-p", "2",
                           "--grid-steps", "50",
                           "--grid-seeds", "42", "43", "44"])
        plan = build_plan(args)
        assert len({r.key() for r in plan}) == len(plan) == 3

    def test_a_cell_is_fully_seeded_before_the_next_one_starts(self):
        """A half-seeded cell would give a median over a different n than its
        neighbours, so seeds must be the innermost loop."""
        args = parse_args(["--sweep", "grid", "--grid-n", "12",
                           "--grid-p", "1", "2", "--grid-steps", "50",
                           "--grid-seeds", "42", "43"])
        assert [(r.p_layers, r.qaoa_seed) for r in build_plan(args)] == [
            (1, 42), (1, 43), (2, 42), (2, 43)]

    def test_grid_ignores_the_global_seed(self):
        """--seed drives the depth sweep; the grid varies the seed itself."""
        args = parse_args(["--sweep", "grid", "--grid-n", "12", "--grid-p", "2",
                           "--grid-steps", "50", "--seed", "99",
                           "--grid-seeds", "42", "43"])
        assert {r.qaoa_seed for r in build_plan(args)} == {42, 43}

    def test_both_excludes_the_grid_but_all_includes_it(self):
        common = ["--seed-n", "8", "--seeds", "42", "--depth-n", "8",
                  "--p-values", "2", "--grid-n", "8", "--grid-p", "2",
                  "--grid-steps", "50", "--grid-seeds", "42"]
        both = build_plan(parse_args(["--sweep", "both", *common]))
        every = build_plan(parse_args(["--sweep", "all", *common]))

        assert "grid" not in {r.sweep for r in both}
        assert [r.sweep for r in every] == ["seeds", "depth", "grid"]


class TestIncrementalCSV:
    def test_header_written_once_across_reopen(self, tmp_path):
        path = tmp_path / "out.csv"
        first = IncrementalCSV(path)
        first.append({"a": 1, "b": 2})
        first.close()

        second = IncrementalCSV(path)
        second.append({"a": 3, "b": 4})
        second.close()

        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]

    def test_row_is_readable_before_close(self, tmp_path):
        """A GPU sweep that dies mid-run must not lose completed rows."""
        path = tmp_path / "out.csv"
        writer = IncrementalCSV(path)
        writer.append({"a": 1})

        with open(path, newline="") as f:
            assert list(csv.DictReader(f)) == [{"a": "1"}]
        writer.close()

    def test_appending_a_changed_schema_is_refused(self, tmp_path):
        """Silently appending mismatched columns would corrupt earlier runs."""
        path = tmp_path / "out.csv"
        first = IncrementalCSV(path)
        first.append({"a": 1, "b": 2})
        first.close()

        second = IncrementalCSV(path)
        with pytest.raises(ValueError, match="Refusing to corrupt"):
            second.append({"a": 3, "b": 4, "steps": 50})
        second.close()

        with open(path, newline="") as f:
            assert list(csv.DictReader(f)) == [{"a": "1", "b": "2"}]


class TestRuntimeEstimate:
    def test_measured_sizes_reproduce_the_baseline(self):
        for n, seconds in MEASURED_RUNTIME_S.items():
            assert estimate_runtime_s(n, p_layers=2, steps=50) == pytest.approx(seconds)

    def test_scales_linearly_in_depth_and_steps(self):
        base = estimate_runtime_s(16, p_layers=2, steps=50)
        assert estimate_runtime_s(16, 4, 50) == pytest.approx(2 * base)
        assert estimate_runtime_s(16, 2, 100) == pytest.approx(2 * base)

    def test_interpolated_size_lies_between_its_neighbours(self):
        assert (MEASURED_RUNTIME_S[20]
                < estimate_runtime_s(22, 2, 50)
                < MEASURED_RUNTIME_S[24])

    def test_monotonic_in_problem_size(self):
        times = [estimate_runtime_s(n, 2, 50) for n in range(8, 31)]
        assert times == sorted(times)

    def test_extrapolates_beyond_the_measured_range(self):
        assert estimate_runtime_s(32, 2, 50) > MEASURED_RUNTIME_S[30]
        assert estimate_runtime_s(4, 2, 50) < MEASURED_RUNTIME_S[8]


class TestOptimalityGap:
    def test_matching_energy_is_zero_gap(self):
        assert optimality_gap_pct(-153.27, -153.27) == pytest.approx(0.0)

    def test_reproduces_the_reported_n20_gap(self):
        """Paper Table I reports 0.053 % for N=20."""
        assert optimality_gap_pct(-335.97, -336.15) == pytest.approx(0.053, abs=1e-3)

    def test_worse_energy_gives_a_positive_gap(self):
        assert optimality_gap_pct(-99.0, -100.0) > 0

    def test_zero_reference_is_undefined(self):
        assert optimality_gap_pct(-1.0, 0.0) is None


class TestFormatDuration:
    @pytest.mark.parametrize("seconds,expected", [
        (5.5, "6s"), (30.0, "30s"), (150.0, "2.5min"), (11304.0, "3.14h"),
    ])
    def test_units_switch_at_the_expected_thresholds(self, seconds, expected):
        assert format_duration(seconds) == expected


class TestBilledBackendGuard:
    """A metered sweep must be named, costed, and acknowledged before it runs.

    run_sweep used to refuse cloud backends outright. Opening it for the SV1
    seed sweep means the refusal has to be replaced by something, not removed:
    the grid alone is 212 runs and would be a four-figure task count.
    """

    def test_a_billed_backend_needs_a_results_tag(self, capsys):
        code = main(["--backend", "braket_sv1", "--sweep", "seeds", "--dry-run"])
        assert code == 2
        assert "--results-tag is required" in capsys.readouterr().out

    def test_the_grid_is_never_allowed_on_a_billed_backend(self, capsys):
        code = main(["--backend", "braket_sv1", "--sweep", "all",
                     "--results-tag", "sv1", "--dry-run"])
        assert code == 2
        assert "keep the grid local" in capsys.readouterr().out

    def test_a_billed_run_refuses_without_acknowledgement(self, capsys):
        code = main(["--backend", "braket_sv1", "--sweep", "seeds",
                     "--results-tag", "sv1", "--seed-n", "16"])
        assert code == 2
        out = capsys.readouterr().out
        assert "BILLED BACKEND" in out
        assert "--yes-bill-me" in out

    def test_a_plan_above_the_ceiling_is_refused(self, capsys):
        code = main(["--backend", "braket_sv1", "--sweep", "seeds",
                     "--results-tag", "sv1", "--seed-n", "8", "12", "16", "20",
                     "--max-cloud-runs", "5", "--yes-bill-me", "--dry-run"])
        assert code == 2
        assert "exceeds --max-cloud-runs" in capsys.readouterr().out

    def test_local_backends_are_unaffected(self, capsys):
        code = main(["--sweep", "seeds", "--seed-n", "8", "--dry-run"])
        assert code == 0
        assert "BILLED BACKEND" not in capsys.readouterr().out

    def test_the_tag_routes_every_sweep_csv(self):
        """All five, including both convergence files.

        A tag that reached only the summary CSVs still let an SV1 run append
        its traces to the local convergence file, which has no backend column
        to tell them apart afterwards.
        """
        assert sweep_csv_names(None) == (
            "paper01_seed_sweep.csv", "paper01_depth_sweep.csv",
            "paper01_steps_depth_grid.csv", "paper01_sweep_convergence.csv",
            "paper01_grid_convergence.csv")
        assert sweep_csv_names("sv1") == (
            "paper01_seed_sweep_sv1.csv", "paper01_depth_sweep_sv1.csv",
            "paper01_steps_depth_grid_sv1.csv",
            "paper01_sweep_convergence_sv1.csv",
            "paper01_grid_convergence_sv1.csv")

    def test_no_tagged_name_collides_with_an_untagged_one(self):
        """The guarantee that matters: a tagged run cannot touch a local file."""
        assert not set(sweep_csv_names("sv1")) & set(sweep_csv_names(None))

    @pytest.mark.parametrize("tag", ["", "../x", "a b"])
    def test_a_bad_tag_is_refused(self, tag):
        with pytest.raises(ValueError):
            sweep_csv_names(tag)


class TestCloudCostEstimate:
    def test_n20_is_priced_above_the_three_second_floor(self):
        """The floor alone understates an N=20 sweep threefold."""
        plan = [SweepRun("seeds", 20, seed, 2, 50) for seed in range(5)]
        tasks, usd = estimate_cloud_cost(plan)
        assert tasks == 5 * 51
        floor_only = tasks * 3.0 / 60.0 * 0.075
        assert usd > 2.5 * floor_only

    def test_small_sizes_pay_the_floor(self):
        plan = [SweepRun("seeds", 8, 42, 2, 50)]
        tasks, usd = estimate_cloud_cost(plan)
        assert usd == pytest.approx(tasks * 3.0 / 60.0 * 0.075)

    def test_an_unmeasured_size_extrapolates_upward(self):
        """A statevector task doubles per qubit; the estimate must not flatten."""
        _, at20 = estimate_cloud_cost([SweepRun("seeds", 20, 42, 2, 50)])
        _, at24 = estimate_cloud_cost([SweepRun("seeds", 24, 42, 2, 50)])
        assert at24 > 10 * at20


class TestClassicalReferenceLabel:
    def test_the_exhaustive_flag_decides_the_label(self):
        """The sweep's "exact/heuristic" note must follow the solver's claim.

        It used to be a name check against "brute_force". When the exhaustive
        set grew to include "exact", that check called the new solver a
        heuristic — inverting what an optimality gap measured against it means.
        """
        rng = np.random.default_rng(2)
        n, k = 8, 3
        returns = rng.uniform(0.05, 0.30, n)
        cov = rng.uniform(0.001, 0.005, (n, n))
        cov = (cov + cov.T) / 2 + np.eye(n) * 0.04
        problem = PortfolioQUBO().formulate(returns, cov, num_select=k)

        for method, exhaustive in (("exact", True), ("brute_force", True),
                                   ("simulated_annealing", False)):
            res = ClassicalSolver(method=method).solve(problem, seed=42)
            assert res.metadata["exhaustive"] is exhaustive, method


class TestHardwareSamplingGuard:
    """A QPU sampling backend is billed per shot, so shots are the risk.

    On SV1 the gradient dominates and shots are nearly free; on hardware the
    economics invert. The project default of 1000 shots is $80.30 for a single
    task, which is why the ceiling is a refusal rather than a warning.
    """

    def test_the_default_shot_count_is_refused_on_hardware(self, capsys):
        code = main(["--sample-backend", "braket_ionq", "--sweep", "seeds",
                     "--seed-n", "8", "--seeds", "42", "--results-tag", "ionq",
                     "--dry-run"])
        assert code == 2
        out = capsys.readouterr().out
        assert "$80.30" in out and "ceiling is 200" in out

    def test_a_hundred_shots_is_allowed_and_costed_per_shot(self, capsys):
        code = main(["--sample-backend", "braket_ionq", "--sweep", "seeds",
                     "--seed-n", "8", "--seeds", "42", "43",
                     "--results-tag", "ionq", "--shots", "100", "--dry-run"])
        assert code == 0
        out = capsys.readouterr().out
        # Two runs, one sampling task each — the optimizer never reaches hardware.
        assert "2 runs -> ~2 tasks, ~$16.60" in out

    def test_a_qpu_sampling_backend_still_needs_a_tag(self, capsys):
        code = main(["--sample-backend", "braket_ionq", "--sweep", "seeds",
                     "--seed-n", "8", "--seeds", "42", "--shots", "100",
                     "--dry-run"])
        assert code == 2
        assert "--results-tag is required" in capsys.readouterr().out

    def test_hardware_is_never_the_optimizer_backend(self):
        """argparse must not even offer it in the --backend slot."""
        with pytest.raises(SystemExit):
            parse_args(["--backend", "braket_ionq"])


class TestHardwareCostEstimate:
    def test_cost_is_one_task_per_run_plus_shots(self):
        plan = [SweepRun("seeds", 8, 42, 2, 50), SweepRun("seeds", 8, 43, 2, 50)]
        tasks, usd = estimate_hardware_cost(plan, shots=100)
        assert tasks == 2
        assert usd == pytest.approx(2 * (0.30 + 100 * 0.08))

    def test_optimizer_steps_do_not_enter_the_hardware_bill(self):
        """The whole point of --sample-backend: steps stay local and free."""
        _, cheap = estimate_hardware_cost([SweepRun("seeds", 8, 42, 2, 50)], 100)
        _, dear = estimate_hardware_cost([SweepRun("seeds", 8, 42, 2, 400)], 100)
        assert cheap == dear

    def test_shots_dominate_the_hardware_bill(self):
        _, at100 = estimate_hardware_cost([SweepRun("seeds", 8, 42, 2, 50)], 100)
        _, at1000 = estimate_hardware_cost([SweepRun("seeds", 8, 42, 2, 50)], 1000)
        assert at1000 > 9 * at100
