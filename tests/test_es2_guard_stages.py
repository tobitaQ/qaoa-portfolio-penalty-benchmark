# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT
"""E-S2 harness: the counts it measures and the stages it records.

The harness is evidence for the system paper, so its own arithmetic has to be
pinned: the parameter-shift tape formula against the SV1 ledger, the
spending-limit model against the documented rejection rule, and the guard
toggling against the committed code (a guard that stops firing must move a
cell from ``cli``/``pre-submit`` to ``submitted``).
"""

from __future__ import annotations

import pytest

pytest.importorskip("pennylane")

from experiments.system_paper import es2_guard_stages as es2  # noqa: E402


class TestCountsAndPrices:
    def test_tape_formula_matches_the_sv1_ledger_at_n8(self):
        # Ledger run 1: 227 tasks = 50 analytic + 177 sampling = 1 + 2*2*(28+16).
        assert 1 + es2.tapes_per_gradient(8) == es2.LEDGER_SAMPLING_TASKS_N8
        assert es2.tapes_per_gradient(20) == 920

    def test_transform_agrees_with_the_formula(self):
        import pennylane as qml
        for n in (8, 12):
            _, tapes, _ = es2.param_shift_tape_count(n, es2.P_LAYERS, qml)
            assert tapes == es2.tapes_per_gradient(n)

    def test_list_prices(self):
        assert es2.ionq_task_usd(100) == pytest.approx(8.30)
        assert es2.ionq_task_usd(1000) == pytest.approx(80.30)
        # A 0.11 s sampling task bills the 3 s minimum.
        assert es2.sv1_task_usd(0.11) == pytest.approx(0.00375)
        assert es2.sv1_task_usd(14.3) == pytest.approx(14.3 / 60 * 0.075)

    def test_cap_model_rejects_the_first_task_over_the_limit(self):
        # $50 cap, $8.30 per task: six accepted, the seventh rejected.
        assert es2._cap_outcome(177, 8.30, 50.0, on_qpu=True) == ("service", 6, pytest.approx(49.80))
        # One $80.30 task is rejected outright.
        assert es2._cap_outcome(1, 80.30, 50.0, on_qpu=True) == ("service", 0, 0.0)
        # Simulators are outside the cap's scope.
        assert es2._cap_outcome(177, 0.00375, 50.0, on_qpu=False)[0] == "submitted"
        # No cap configured.
        assert es2._cap_outcome(3, 8.30, None, on_qpu=True) == ("submitted", 3, pytest.approx(24.90))


class TestMeasuredStages:
    @pytest.fixture(scope="class")
    def n8(self):
        """Measured once: the no-derivative loop count and the committed sampling path."""
        loop = es2.measure_loop_tapes(8, steps=1, device="mock")
        sampling = es2.measure_sampling_call(8)
        return loop, sampling

    def test_no_derivative_device_reproduces_177_tapes_per_step(self, n8):
        loop, _ = n8
        assert loop["tapes_per_step"] == 177 and loop["matches_formula"]

    def test_committed_sampling_path_freezes_and_sends_one_tape(self, n8):
        _, sampling = n8
        assert sampling["frozen_before_qnode"]
        assert sampling["sampling_tapes_committed"] == 1
        assert sampling["tapes_if_trainable"] == 177

    def test_qpu_optimizer_stops_at_the_cli_with_guards_and_bills_without(self, n8):
        import pennylane as qml
        loop, _ = n8
        dev = es2.NoDerivativeDevice.make(qml, 8)
        on = es2.cell_qpu_optimizer(8, "structural", None, 177, qml, dev)
        assert (on.stage, on.tasks_billed, on.usd_billed) == ("cli", 0, 0.0)
        off = es2.cell_qpu_optimizer(8, "none", None, 177, qml, dev)
        assert off.stage == "submitted"
        assert off.tasks_would_submit == 50 * 177 + 177
        assert off.usd_billed == pytest.approx(9027 * 8.30)
        capped = es2.cell_qpu_optimizer(8, "cap", 50.0, 177, qml, dev)
        assert (capped.stage, capped.tasks_billed, capped.usable_result) == ("service", 6, "no")

    def test_trainable_sampling_is_removed_before_submission(self, n8):
        _, sampling = n8
        on = es2.cell_trainable_sampling(8, "structural", None, "sv1", sampling)
        assert (on.stage, on.tasks_would_submit) == ("pre-submit", 1)
        off = es2.cell_trainable_sampling(8, "none", None, "sv1", sampling)
        assert (off.stage, off.tasks_would_submit) == ("submitted", 177)
        assert "ledger" in off.count_source
        # The cap does not see simulator tasks.
        assert es2.cell_trainable_sampling(8, "cap", 50.0, "sv1", sampling).stage == "submitted"
        # On the QPU the cap stops it, but not before $49.80.
        capped = es2.cell_trainable_sampling(8, "cap", 50.0, "qpu", sampling)
        assert (capped.stage, capped.usd_billed, capped.usable_result) == ("service", 49.80, "unknown")

    def test_default_shots_on_hardware_pass_the_structural_guards(self):
        import pennylane as qml
        cell = es2.cell_qpu_sampling_default_shots(
            8, "structural", None, qml, es2.NoDerivativeDevice.make(qml, 8))
        assert (cell.stage, cell.tasks_would_submit, cell.usd_billed) == ("submitted", 1, 80.30)
        both = es2.cell_qpu_sampling_default_shots(
            8, "both", 50.0, qml, es2.NoDerivativeDevice.make(qml, 8))
        assert (both.stage, both.usd_billed) == ("service", 0.0)

    def test_untagged_cloud_run_is_refused_and_the_baseline_survives(self, tmp_path):
        import pennylane as qml
        on = es2.cell_untagged_cloud_run("structural", qml, tmp_path / "on")
        assert (on.stage, on.baseline_overwritten) == ("cli", "no")
        off = es2.cell_untagged_cloud_run("none", qml, tmp_path / "off")
        assert (off.stage, off.baseline_overwritten) == ("submitted", "yes")

    def test_guards_are_restored_after_the_unguarded_cell(self):
        """Toggling must not leak: the committed guards are back afterwards."""
        import src.solvers.braket_solver as mod
        from experiments.paper01_qubo_baseline import run_experiment
        assert "braket_ionq" in mod.QPU_BACKENDS
        assert "braket_sv1" not in run_experiment.LOCAL_BACKENDS
        with pytest.raises(ValueError):
            mod.BraketSolver(backend="braket_ionq")
