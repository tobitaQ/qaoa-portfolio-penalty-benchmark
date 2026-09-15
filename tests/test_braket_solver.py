# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Tests for BraketSolver (QAOA via PennyLane).

These are regression tests for three defects found on 2026-08-05:
  1. ADAM never updated the parameters (plain numpy array instead of a
     pennylane.numpy tensor) — the convergence curve was perfectly flat.
  2. The Braket device rejects ``qml.sample(wires=...)`` — final sampling
     raised NotImplementedError.
  3. The un-normalized cost Hamiltonian made the gamma landscape oscillate
     faster than the optimizer step, so QAOA never converged.

Only the free local simulators are exercised; cloud backends are checked for
argument validation only, never executed.
"""

import numpy as np
import pytest

from src.qubo.portfolio import PortfolioQUBO
from src.solvers.braket_solver import (
    BACKEND_DEVICE_MAP,
    BRAKET_REGION,
    IONQ_ARN,
    LOCAL_BACKENDS,
    BraketSolver,
    _validate_samples,
    available_backends,
    frozen_angles,
)
from src.solvers.classical_solver import ClassicalSolver


@pytest.fixture
def toy_problem():
    """6-asset problem — small enough for a fast statevector simulation."""
    np.random.seed(1)
    n = 6
    returns = np.random.uniform(0.05, 0.20, n)
    cov = np.eye(n) * 0.04 + np.random.uniform(0.001, 0.005, (n, n))
    cov = (cov + cov.T) / 2
    return PortfolioQUBO().formulate(returns, cov, num_select=2)


@pytest.fixture
def scaling_problem():
    """10-asset problem — large enough to expose the Hamiltonian-scaling defect.

    The cardinality penalty A grows with N, so the Ising coefficients do too. At
    N=6 the largest coefficient is ~7 and even an un-normalized run converges, so
    a smaller fixture cannot catch the bug. At N=10 it reaches ~33 and the
    un-normalized optimizer visibly fails to settle (measured 2026-08-05).
    """
    rng = np.random.default_rng(1)
    n = 10
    returns = rng.uniform(0.05, 0.20, n)
    cov = np.eye(n) * 0.04 + rng.uniform(0.001, 0.005, (n, n))
    cov = (cov + cov.T) / 2
    return PortfolioQUBO().formulate(returns, cov, num_select=2)


@pytest.fixture
def solver():
    return BraketSolver(
        backend="lightning_cpu", p_layers=2, n_shots=200, n_optimizer_steps=25
    )


class TestBackendSelection:
    def test_rejects_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            BraketSolver(backend="not_a_backend")

    def test_rejects_unknown_precision(self):
        with pytest.raises(ValueError, match="precision"):
            BraketSolver(precision="quadruple")

    def test_precision_maps_to_dtype(self):
        assert BraketSolver(precision="single").c_dtype is np.complex64
        assert BraketSolver(precision="double").c_dtype is np.complex128

    def test_available_backends_covers_local_only(self):
        status = available_backends()
        assert set(status) == set(LOCAL_BACKENDS)
        assert all(isinstance(v, bool) for v in status.values())

    def test_lightning_cpu_is_available(self):
        """lightning.qubit ships with pennylane-lightning and must always work."""
        assert available_backends()["lightning_cpu"] is True

    def test_cloud_backends_require_s3_bucket(self, toy_problem):
        """SV1 in the optimizer slot, IonQ only as the sampling backend.

        braket_ionq cannot be the optimizer backend at all any more -- the
        hardware guard rejects it earlier, and TestHardwareGuard covers that --
        so it is exercised here through the route that is actually allowed.
        """
        with pytest.raises(ValueError, match="s3_bucket"):
            BraketSolver(backend="braket_sv1").solve(toy_problem, seed=42)
        with pytest.raises(ValueError, match="s3_bucket"):
            BraketSolver(
                backend="lightning_cpu", sample_backend="braket_ionq",
                p_layers=1, n_optimizer_steps=2, n_shots=20,
            ).solve(toy_problem, seed=42)

    def test_cloud_session_ignores_the_ambient_region(self, monkeypatch):
        """Braket has no ap-northeast-1 endpoint — the region must not leak in.

        Regression for 2026-08-09: the project's default region is
        ap-northeast-1, so an unpinned session resolved
        braket.ap-northeast-1.amazonaws.com and every SV1 run died on DNS
        before submitting a task.
        """
        monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-northeast-1")
        monkeypatch.setenv("AWS_REGION", "ap-northeast-1")
        assert BRAKET_REGION == "us-east-1"
        assert BraketSolver._aws_session().region == BRAKET_REGION

    def test_final_sampling_angles_are_not_trainable(self, toy_problem):
        """Trainable angles at sampling time cost 176 billable SV1 tasks.

        Regression for 2026-08-09: ``_sample_best`` passed the optimizer's
        tensors straight into a shots-based QNode. Sampling one PauliZ per wire
        rules out the adjoint method, so PennyLane fell back to parameter-shift
        and evaluated a gradient nobody reads — one Braket task per shifted
        circuit. Measured at N=8: 177 tasks for what should be a single shot
        batch.
        """
        import pennylane as qml
        from pennylane import numpy as pnp

        trainable = pnp.array([0.1, 0.2, 0.3, 0.4], requires_grad=True)
        assert qml.math.requires_grad(trainable)
        assert not qml.math.requires_grad(frozen_angles(trainable))
        np.testing.assert_array_equal(frozen_angles(trainable), np.asarray(trainable))

    def test_ionq_arn_is_not_retired_aria(self):
        """Aria-1 was retired; the ARN must point at a device that still exists."""
        assert "Aria" not in IONQ_ARN
        assert "Forte" in IONQ_ARN

    def test_gpu_backend_does_not_silently_fall_back(self, toy_problem):
        """A GPU run must fail loudly rather than report CPU numbers as GPU."""
        if available_backends()["lightning_gpu"]:
            pytest.skip("lightning.gpu is installed here; nothing to fall back from")
        with pytest.raises(RuntimeError, match="lightning.gpu is unavailable"):
            BraketSolver(backend="lightning_gpu").solve(toy_problem, seed=42)


class TestQAOAOptimization:
    def test_parameters_are_actually_optimized(self, solver, toy_problem):
        """Regression: a flat curve means the optimizer never moved the params."""
        conv = solver.solve(toy_problem, seed=42).metadata["convergence"]
        assert len(conv) == solver.n_optimizer_steps
        assert not np.allclose(conv, conv[0]), "convergence is flat — params not trainable"

    def test_expectation_improves(self, solver, scaling_problem):
        conv = solver.solve(scaling_problem, seed=42).metadata["convergence"]
        q = max(1, len(conv) // 4)
        assert np.mean(conv[-q:]) < np.mean(conv[:q]), "no downward trend"

    def test_optimizer_settles(self, solver, scaling_problem):
        """Regression: with an un-normalized cost Hamiltonian the gamma landscape
        oscillates faster than the ADAM step, so the curve swings across its whole
        range forever instead of settling.

        A mean-trend check is not enough — the old buggy run still drifted downward
        on average — so require the tail to be quiet relative to the overall
        excursion. Measured on this fixture: 0.02 normalized vs 0.14 un-normalized.
        """
        conv = np.asarray(solver.solve(scaling_problem, seed=42).metadata["convergence"])
        q = max(2, len(conv) // 4)
        spread = float(conv.max() - conv.min())
        assert float(np.std(conv[-q:])) < 0.1 * spread, (
            "expectation is still oscillating at the end of optimization"
        )

    def test_hamiltonian_scale_recorded(self, solver, toy_problem):
        """Convergence is in normalized units, so the scale must be reported."""
        meta = solver.solve(toy_problem, seed=42).metadata
        assert meta["hamiltonian_scale"] > 0

    def test_sampling_returns_binary_vector(self, solver, toy_problem):
        """Regression: the Braket device rejects computational-basis sampling."""
        result = solver.solve(toy_problem, seed=42)
        assert result.bitstring.shape == (toy_problem.n_variables,)
        assert set(np.unique(result.bitstring)) <= {0.0, 1.0}

    def test_energy_reported_in_original_units(self, solver, toy_problem):
        """Normalizing H must not leak into the reported energy."""
        result = solver.solve(toy_problem, seed=42)
        expected = toy_problem.evaluate_no_offset(result.bitstring)
        assert result.energy_no_offset == pytest.approx(expected)

    def test_reaches_classical_optimum_on_toy_problem(self, solver, toy_problem):
        """On a 6-asset problem QAOA should land on the brute-force optimum."""
        best = ClassicalSolver(method="brute_force").solve(toy_problem, seed=42)
        result = solver.solve(toy_problem, seed=42)
        assert result.energy_no_offset <= best.energy_no_offset * 0.99 + 1e-9

    def test_deterministic_with_same_seed(self, solver, toy_problem):
        r1 = solver.solve(toy_problem, seed=7)
        r2 = solver.solve(toy_problem, seed=7)
        assert np.allclose(r1.bitstring, r2.bitstring)

    def test_runtime_recorded(self, solver, toy_problem):
        assert solver.solve(toy_problem, seed=42).runtime_seconds > 0

    def test_backend_label_propagates_to_result(self, solver, toy_problem):
        assert solver.solve(toy_problem, seed=42).backend == "lightning_cpu"


class TestDeviceMap:
    def test_every_backend_has_a_device(self):
        for backend in BACKEND_DEVICE_MAP:
            assert BACKEND_DEVICE_MAP[backend]

    def test_local_backends_are_a_subset_of_the_map(self):
        assert LOCAL_BACKENDS <= set(BACKEND_DEVICE_MAP)


class TestHardwareGuard:
    """Real hardware may sample, never optimize.

    The optimization loop on a QPU is not slow-and-expensive, it is
    unaffordable: without adjoint differentiation PennyLane falls back to
    parameter-shift and submits one billable task per shifted circuit, ~177 per
    step at N=8. A Budget alert fires after the spend, so the only thing that
    can prevent it is a check before the first task.
    """

    def test_a_qpu_cannot_be_the_optimizer_backend(self):
        with pytest.raises(ValueError, match="cannot run the optimization loop"):
            BraketSolver(backend="braket_ionq")

    def test_the_guard_fires_before_any_aws_object_exists(self, monkeypatch):
        """No credentials, no session, no bucket — it must still refuse."""
        import src.solvers.braket_solver as mod

        def explode(*args, **kwargs):
            raise AssertionError("touched AWS before refusing")

        monkeypatch.setattr(mod.BraketSolver, "_aws_session", staticmethod(explode))
        with pytest.raises(ValueError, match="braket_ionq"):
            BraketSolver(backend="braket_ionq", s3_bucket="anything")

    def test_a_qpu_is_allowed_as_the_sampling_backend(self):
        solver = BraketSolver(backend="lightning_cpu", sample_backend="braket_ionq")
        assert solver.backend == "lightning_cpu"
        assert solver.sample_backend == "braket_ionq"
        # The S3 prefix follows the billed device, not the optimizer.
        assert solver.s3_prefix == "ionq"

    def test_an_unknown_sampling_backend_is_refused(self):
        with pytest.raises(ValueError, match="Unknown sample_backend"):
            BraketSolver(backend="lightning_cpu", sample_backend="ibm_brisbane")

    def test_sampling_defaults_to_the_optimizer_backend(self):
        """Existing callers must be unaffected."""
        solver = BraketSolver(backend="lightning_cpu")
        assert solver.sample_backend == "lightning_cpu"


class TestSampleBackendRouting:
    @staticmethod
    def _toy_problem():
        rng = np.random.default_rng(3)
        n = 4
        returns = rng.uniform(0.05, 0.30, n)
        cov = np.eye(n) * 0.04
        return PortfolioQUBO().formulate(returns, cov, num_select=2)

    def test_exactly_one_device_is_built_for_sampling(self, monkeypatch):
        """The billed backend must be constructed once, for the final circuit.

        Structural check on the count that decides the invoice: if the sampling
        backend were built inside the optimizer loop this would be 50+.
        """
        solver = BraketSolver(backend="lightning_cpu", sample_backend="braket_local",
                              p_layers=1, n_optimizer_steps=2, n_shots=20)
        built: list[str] = []
        original = BraketSolver._make_device

        def spy(self, n, qml, backend=None):
            built.append(backend or self.backend)
            return original(self, n, qml, backend)

        monkeypatch.setattr(BraketSolver, "_make_device", spy)
        solver.solve(self._toy_problem(), seed=42)

        assert built.count("braket_local") == 1
        assert built.count("lightning_cpu") == 1

    def test_no_extra_device_when_the_backends_match(self, monkeypatch):
        solver = BraketSolver(backend="lightning_cpu", p_layers=1,
                              n_optimizer_steps=2, n_shots=20)
        built: list[str] = []
        original = BraketSolver._make_device

        def spy(self, n, qml, backend=None):
            built.append(backend or self.backend)
            return original(self, n, qml, backend)

        monkeypatch.setattr(BraketSolver, "_make_device", spy)
        solver.solve(self._toy_problem(), seed=42)

        assert built == ["lightning_cpu"]

    def test_the_result_records_where_sampling_ran(self):
        solver = BraketSolver(backend="lightning_cpu", sample_backend="braket_local",
                              p_layers=1, n_optimizer_steps=2, n_shots=20)
        result = solver.solve(self._toy_problem(), seed=42)
        assert result.metadata["sample_backend"] == "braket_local"


class TestSampleValidation:
    """A device result that is not measurements must raise, never score.

    IonQ Forte-1 returned 100 shots over 60 distinct bitstrings on 2026-08-09;
    the nested program-set result schema reached the solver as all +1 and the
    run reported energy 0.0 at a 100 % gap. A wrong number that looks like a
    result is worse than a crash, and $8.30 of hardware time nearly entered a
    table as one.
    """

    def test_all_identical_shots_are_refused(self):
        z = np.ones((100, 8))
        with pytest.raises(RuntimeError, match="identical shots"):
            _validate_samples(z, n=8, n_shots=100, backend="braket_ionq")

    def test_a_wrong_shape_is_refused(self):
        with pytest.raises(RuntimeError, match="expected"):
            _validate_samples(np.ones((8, 100)), n=8, n_shots=100,
                              backend="braket_ionq")

    def test_values_outside_the_pauliz_spectrum_are_refused(self):
        rng = np.random.default_rng(0)
        z = rng.choice([-1.0, 1.0], size=(100, 8))
        z[3, 2] = 0.5
        with pytest.raises(RuntimeError, match="outside"):
            _validate_samples(z, n=8, n_shots=100, backend="braket_sv1")

    def test_real_looking_samples_pass(self):
        rng = np.random.default_rng(1)
        z = rng.choice([-1.0, 1.0], size=(100, 8))
        _validate_samples(z, n=8, n_shots=100, backend="lightning_cpu")

    def test_a_single_wire_may_legitimately_repeat(self):
        """One qubit can genuinely give 100 identical shots; do not block it."""
        _validate_samples(np.ones((100, 1)), n=1, n_shots=100,
                          backend="lightning_cpu")

    def test_the_local_path_still_produces_a_feasible_solution(self):
        """End-to-end: validation must not fire on a working backend."""
        rng = np.random.default_rng(4)
        n, k = 6, 2
        returns = rng.uniform(0.05, 0.30, n)
        cov = np.eye(n) * 0.04
        problem = PortfolioQUBO().formulate(returns, cov, num_select=k)
        result = BraketSolver(backend="lightning_cpu", p_layers=1,
                              n_optimizer_steps=3, n_shots=50).solve(problem, seed=42)
        assert np.isfinite(result.energy_no_offset)


class TestSamplingDeterminism:
    """Two seeds that a single `seed` argument used to conflate.

    On a near-degenerate landscape the returned portfolio is a draw from the
    near-optimal set, so a run can differ from another because it optimized to
    different angles or because it sampled the same state differently. SV1 has no
    controllable shot RNG and was found disagreeing with itself at N = 20, which
    is why the local control exists.
    """

    @staticmethod
    def _problem(n=8, k=2):
        rng = np.random.default_rng(3)
        return PortfolioQUBO().formulate(
            rng.uniform(0.05, 0.30, n), np.eye(n) * 0.04, num_select=k)

    def test_shot_seed_leaves_the_optimizer_untouched(self):
        """The convergence trace must not depend on the sampling seed."""
        solver = BraketSolver(backend="lightning_cpu", p_layers=1,
                              n_optimizer_steps=5, n_shots=100)
        a = solver.solve(self._problem(), seed=42, shot_seed=1)
        b = solver.solve(self._problem(), seed=42, shot_seed=2)
        assert a.metadata["convergence"] == b.metadata["convergence"]

    def test_the_same_shot_seed_reproduces_the_bitstring(self):
        solver = BraketSolver(backend="lightning_cpu", p_layers=1,
                              n_optimizer_steps=5, n_shots=100)
        a = solver.solve(self._problem(), seed=42, shot_seed=7)
        sa = solver.last_samples.copy()
        b = solver.solve(self._problem(), seed=42, shot_seed=7)
        assert a.energy_no_offset == b.energy_no_offset
        assert list(a.bitstring) == list(b.bitstring)
        np.testing.assert_array_equal(sa, solver.last_samples)

    def test_a_different_shot_seed_actually_draws_different_shots(self):
        """The control of 2026-08 did not: it re-seeded NumPy's global RNG,
        which the Lightning devices consult only at construction, so both
        "shot-RNG" sweeps re-drew identical shots on every row. The test that
        would have caught it compared bitstrings, which best-of-100 makes
        robust; this one compares the shots themselves.
        """
        solver = BraketSolver(backend="lightning_cpu", p_layers=1,
                              n_optimizer_steps=5, n_shots=100)
        a = solver.solve(self._problem(), seed=42, shot_seed=1)
        sa = solver.last_samples.copy()
        b = solver.solve(self._problem(), seed=42, shot_seed=2)
        assert a.metadata["shot_seed_applied"] is True
        assert not np.array_equal(sa, solver.last_samples)
        assert a.metadata["convergence"] == b.metadata["convergence"]

    def test_an_unseedable_sampler_is_reported_not_faked(self):
        solver = BraketSolver(backend="lightning_cpu", sample_backend="braket_local",
                              p_layers=1, n_optimizer_steps=2, n_shots=20)
        res = solver.solve(self._problem(n=4), seed=42, shot_seed=3)
        assert res.metadata["shot_seed"] == 3
        assert res.metadata["shot_seed_applied"] is False

    def test_the_shot_seed_is_recorded(self):
        solver = BraketSolver(backend="lightning_cpu", p_layers=1,
                              n_optimizer_steps=5, n_shots=100)
        assert solver.solve(self._problem(), seed=42,
                            shot_seed=5).metadata["shot_seed"] == 5
        assert solver.solve(self._problem(), seed=42).metadata["shot_seed"] is None
        assert solver.solve(self._problem(), seed=42).metadata["shot_seed_applied"] is None

    def test_repeated_runs_are_bit_identical(self):
        """lightning.qubit reduces in parallel and its reduction order follows
        thread scheduling, so this fails at the default OMP thread count
        (~1e-13 on the convergence trace). The experiment scripts pin
        OMP_NUM_THREADS=1 before importing a simulator; this asserts the pin is
        doing its job in whatever environment the suite runs in."""
        import os
        if os.environ.get("OMP_NUM_THREADS") != "1":
            pytest.skip("OMP_NUM_THREADS is not pinned in this environment")
        solver = BraketSolver(backend="lightning_cpu", p_layers=2,
                              n_optimizer_steps=10, n_shots=100)
        traces = [solver.solve(self._problem(10, 3), seed=42).metadata["convergence"]
                  for _ in range(3)]
        assert traces[0] == traces[1] == traces[2]


class TestEvaluateFixedAngles:
    """Plan E4-A: the same angles measured on several backends.

    The entry point must build exactly the circuit ``solve`` builds -- the
    check is that the probability vector at the optimiser's final angles is
    the one ``record_state`` returned -- and on the sampling backend it must
    construct the device once, whatever the number of batches, because on SV1
    every construction is a session and every batch a billed task.
    """

    @staticmethod
    def _toy_problem():
        rng = np.random.default_rng(3)
        n = 4
        returns = rng.uniform(0.05, 0.30, n)
        cov = np.eye(n) * 0.04
        return PortfolioQUBO().formulate(returns, cov, num_select=2)

    @pytest.mark.parametrize("optimizer", ["adam", "lbfgs"])
    def test_reproduces_the_state_solve_recorded(self, optimizer):
        # Both optimizer paths: the recorded final angles, re-evaluated from
        # scratch, must give the exact state ``solve`` recorded beside its
        # batch -- the state every P_F / P_tau / Q_tau column derives from.
        problem = self._toy_problem()
        solver = BraketSolver(backend="lightning_cpu", p_layers=1,
                              n_optimizer_steps=3, n_shots=50, record_state=True,
                              optimizer=optimizer)
        res = solver.solve(problem, seed=42)
        assert not np.array_equal(res.metadata["final_angles"], res.metadata["initial_angles"])
        out = solver.evaluate_fixed_angles(problem, res.metadata["final_angles"],
                                           batches=2, shot_seed=7)
        np.testing.assert_array_equal(out["probabilities"], res.metadata["final_probabilities"])
        assert out["hamiltonian_scale"] == res.metadata["hamiltonian_scale"]
        assert len(out["batches"]) == 2
        assert all(b.shape == (50, 4) for b in out["batches"])
        assert set(np.unique(np.concatenate(out["batches"]))) <= {0.0, 1.0}
        assert out["task_arns"] == [] and out["execution_ms"] == []

    def test_batches_are_reproducible_from_the_shot_seed_and_differ_between_batches(self):
        problem = self._toy_problem()
        solver = BraketSolver(backend="lightning_cpu", p_layers=1, n_shots=200)
        angles = np.array([0.7, 1.1])
        a = solver.evaluate_fixed_angles(problem, angles, batches=2, shot_seed=11,
                                         probabilities=False)
        b = solver.evaluate_fixed_angles(problem, angles, batches=2, shot_seed=11,
                                         probabilities=False)
        assert a["probabilities"] is None
        np.testing.assert_array_equal(a["batches"][0], b["batches"][0])
        np.testing.assert_array_equal(a["batches"][1], b["batches"][1])
        assert not np.array_equal(a["batches"][0], a["batches"][1])

    def test_exactly_one_sampling_device_for_any_number_of_batches(self, monkeypatch):
        solver = BraketSolver(backend="lightning_cpu", sample_backend="braket_local",
                              p_layers=1, n_shots=20)
        built: list[str] = []
        original = BraketSolver._make_device

        def spy(self, n, qml, backend=None):
            built.append(backend or self.backend)
            return original(self, n, qml, backend)

        monkeypatch.setattr(BraketSolver, "_make_device", spy)
        out = solver.evaluate_fixed_angles(self._toy_problem(), np.array([0.3, 0.9]),
                                           batches=3)
        assert built == ["braket_local"]
        assert len(out["batches"]) == 3 and out["probabilities"].shape == (16,)

    def test_refuses_probabilities_on_hardware_and_wrong_angle_count(self):
        solver = BraketSolver(backend="lightning_cpu", sample_backend="braket_ionq",
                              s3_bucket="b", p_layers=1, n_shots=10)
        with pytest.raises(ValueError, match="analytic"):
            solver.evaluate_fixed_angles(self._toy_problem(), np.array([0.3, 0.9]))
        local = BraketSolver(backend="lightning_cpu", p_layers=2, n_shots=10)
        with pytest.raises(ValueError, match="expected 4 angles"):
            local.evaluate_fixed_angles(self._toy_problem(), np.array([0.3, 0.9]))

    def test_cloud_amplitude_path_matches_the_qnode_probabilities(self):
        """SV1 refuses ``Probability`` at zero shots, so the cloud arm reads
        ``Amplitude`` over named states from a circuit converted with the
        device's own ``apply``. On the free Braket local simulator that path
        and the ordinary QNode path must agree on every requested state, which
        checks the decomposition, the wire map and the bit order together.
        """
        import pennylane as qml

        problem = self._toy_problem()
        solver = BraketSolver(backend="lightning_cpu", sample_backend="braket_local",
                              p_layers=2, n_shots=10)
        angles = np.array([0.3, 0.9, 1.4, 0.2])
        idx = np.array([0, 3, 5, 6, 9, 10, 12, 15])
        out = solver.evaluate_fixed_angles(problem, angles, batches=1,
                                           probabilities=True, state_indices=idx)
        full = out["probabilities"]

        dev = solver._make_device(4, qml, "braket_local")
        h, J, _ = PortfolioQUBO().to_ising(problem)
        scale = float(max(np.max(np.abs(h)), np.max(np.abs(J)), 1e-12))
        cost_h = solver._build_cost_hamiltonian(h / scale, J / scale, 4, qml)
        mixer_h = solver._build_mixer_hamiltonian(4, qml)
        with qml.tape.QuantumTape() as tape:
            for i in range(4):
                qml.Hadamard(wires=i)
            for gamma, beta in zip(angles[:2], angles[2:]):
                qml.ApproxTimeEvolution(cost_h, gamma, 1)
                qml.ApproxTimeEvolution(mixer_h, beta, 1)
            qml.expval(qml.PauliZ(0))
        probs, task_id, ms = solver._cloud_state_probabilities(dev, tape, idx, 4, qml)
        np.testing.assert_allclose(probs, full[idx], rtol=0, atol=1e-12)
        assert task_id and np.isnan(ms)
