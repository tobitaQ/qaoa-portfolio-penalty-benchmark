# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Quantum QUBO solver via Amazon Braket + PennyLane QAOA.

Backend selection at construction time:
    "braket_local"  → LocalSimulator (free, development)
    "braket_sv1"    → SV1 cloud simulator ($0.075/min)
    "braket_ionq"   → IonQ real hardware (paper finalization only)

QAOA maps the QUBO to an Ising Hamiltonian and uses p alternating layers of
cost and mixer unitaries. Parameters are optimized with classical gradient descent.
"""

from __future__ import annotations

from typing import Literal

from datetime import datetime, timedelta, timezone

import numpy as np

from src.qubo.portfolio import PortfolioQUBO, QUBOProblem
from src.solvers.braket_results import (execution_duration_ms, resolve_task_arn,
                                        samples_from_counts, shot_counts_from_task)
from src.solvers.base import AbstractSolver, SolverResult

BACKEND_DEVICE_MAP = {
    # --- Local simulators (free) ---
    "lightning_cpu": "lightning.qubit",   # C++ statevector, adjoint diff
    "lightning_gpu": "lightning.gpu",     # NVIDIA cuQuantum, needs Linux + CUDA
    "braket_local":  "braket.local.qubit",  # Braket SDK local (parameter-shift: slow)
    # --- Cloud ---
    "braket_sv1":    "braket.aws.qubit",
    "braket_ionq":   "braket.aws.qubit",
}

BACKEND_S3_PREFIX = {
    "braket_sv1":  "sv1",
    "braket_ionq": "ionq",
}

SV1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
# IonQ Aria-1 was retired; Forte-1 is the current gate-based IonQ device (us-east-1).
IONQ_ARN = "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1"

#: Region the cloud backends talk to. Amazon Braket does not exist in
#: ap-northeast-1, which is this project's usual default region, so boto3 would
#: resolve braket.ap-northeast-1.amazonaws.com and fail before a single task is
#: submitted. Pinning it here makes the backend name alone decide where the task
#: lands, independent of the caller's AWS_DEFAULT_REGION.
BRAKET_REGION = "us-east-1"

#: Backends that run entirely on the local machine at zero AWS cost.
LOCAL_BACKENDS = frozenset({"lightning_cpu", "lightning_gpu", "braket_local"})

#: Real quantum hardware. Billed per task *and* per shot, and -- decisively --
#: unable to run adjoint differentiation, so PennyLane falls back to
#: parameter-shift and submits one billable task per shifted circuit. At N = 8
#: that is 177 tasks per optimizer step; a 50-step run at IonQ's
#: $0.30/task + $0.08/shot with 100 shots is
#:
#:     177 x 50 x ($0.30 + 100 x $0.08) = ~$73,000.
#:
#: An AWS Budget does not prevent this -- Budgets alert after the fact, they do
#: not stop execution -- so the guard has to live here, before any task is
#: submitted. QPUs are therefore accepted only as ``sample_backend``: the
#: optimizer runs locally and hardware sees exactly one final sampling task.
QPU_BACKENDS = frozenset({"braket_ionq"})


def _validate_samples(z: np.ndarray, n: int, n_shots: int, backend: str) -> None:
    """Reject a sample array that cannot have come from a real measurement.

    Measured on IonQ Forte-1, 2026-08-09: the device returned 100 shots over 60
    distinct bitstrings, but the result came back under the nested
    ``program_set_task_result`` schema and reached us as all +1 -- every qubit
    read as |0> on every shot. Nothing raised. The run reported a bitstring of
    zeros, an energy of exactly 0.0 and a 100 % optimality gap, which is a
    plausible-looking row rather than an obvious failure, and $8.30 of hardware
    time would have entered a table as a result.

    The checks are deliberately weak enough never to fire on real data: any
    QAOA state over n >= 2 qubits that produced 100 identical shots would be a
    delta function, and eigenvalues outside {-1, +1} are not measurements at all.

    Args:
        z: PauliZ eigenvalues, shape (n_shots, n).
        n: Number of wires.
        n_shots: Number of shots requested.
        backend: Sampling backend, named in the error for diagnosis.

    Raises:
        RuntimeError: If the array cannot be a measurement of this circuit.
    """
    if z.shape != (n_shots, n):
        raise RuntimeError(
            f"{backend} returned samples of shape {z.shape}, expected "
            f"({n_shots}, {n}). The device result was not parsed as "
            f"measurements; do not treat the returned bitstring as data."
        )
    if not np.isin(z, (-1.0, 1.0)).all():
        raise RuntimeError(
            f"{backend} returned PauliZ values outside {{-1, +1}}; the result "
            f"was not parsed as measurements."
        )
    if n > 1 and (z == z[0]).all():
        raise RuntimeError(
            f"{backend} returned {n_shots} identical shots "
            f"({'|0>' if z[0][0] == 1.0 else 'mixed'} on every wire). A QAOA "
            f"state over {n} qubits does not do that; the result was almost "
            f"certainly not parsed as measurements. Recover the shots from the "
            f"task's S3 output rather than trusting this bitstring."
        )


def frozen_angles(params) -> np.ndarray:
    """Strip trainability from optimized QAOA angles.

    PennyLane decides whether to differentiate a QNode from the trainability of
    its inputs, and ``pennylane.numpy`` tensors are trainable by default. A
    plain NumPy array is not, so this is what a "the optimization is finished"
    hand-off looks like.

    Args:
        params: Angles from the optimizer (``pennylane.numpy`` tensor or array).

    Returns:
        The same values as a plain float NumPy array.
    """
    return np.asarray(params, dtype=float)


def available_backends() -> dict[str, bool]:
    """Report which local simulator backends can actually be instantiated here.

    Cloud backends are omitted because availability depends on AWS credentials
    rather than on the local installation.

    Returns:
        Mapping of backend name → whether importing its device succeeds.
    """
    import pennylane as qml

    status: dict[str, bool] = {}
    for name in LOCAL_BACKENDS:
        try:
            qml.device(BACKEND_DEVICE_MAP[name], wires=2)
            status[name] = True
        except Exception:
            status[name] = False
    return status


class BraketSolver(AbstractSolver):
    """
    QAOA-based solver using Amazon Braket + PennyLane.

    Args:
        backend: Device to use. One of "lightning_cpu", "lightning_gpu",
            "braket_local", "braket_sv1", "braket_ionq".
        p_layers: Number of QAOA layers (depth). Higher → better quality, slower.
        n_shots: Number of measurement shots per circuit evaluation.
            Note: on QPU backends every shot is billed, so keep this small.
        n_optimizer_steps: Steps for parameter optimization (ADAM).
        s3_bucket: S3 bucket for cloud backends (required for sv1/ionq).
        s3_prefix: S3 key prefix.
        precision: Statevector amplitude precision for the lightning backends.
            "single" (complex64) halves memory and is what makes N=30 fit in
            48 GB of VRAM; "double" (complex128) is the safer default.
        record_state: Put the final and initial states' probability vectors in
            the result metadata (plan E1/E2). Only for local simulators.
        optimizer: ``"adam"`` (the published setting) or ``"lbfgs"``
            (SciPy L-BFGS-B on the analytic expectation with adjoint
            gradients; plan E3's control B). For L-BFGS ``n_optimizer_steps``
            caps the number of *gradient evaluations*, not iterations, so the
            two optimizers are compared on the same count of the expensive
            operation; the objective/gradient evaluation counts are recorded.
        stepsize: ADAM step size. 0.1 is the published setting; plan E3's
            control A uses 0.03. Ignored by L-BFGS.
    """

    def __init__(
        self,
        backend: Literal[
            "lightning_cpu", "lightning_gpu",
            "braket_local", "braket_sv1", "braket_ionq",
        ] = "lightning_cpu",
        p_layers: int = 2,
        n_shots: int = 1000,
        n_optimizer_steps: int = 100,
        s3_bucket: str | None = None,
        s3_prefix: str | None = None,
        precision: Literal["single", "double"] = "double",
        sample_backend: str | None = None,
        record_state: bool = False,
        optimizer: Literal["adam", "lbfgs"] = "adam",
        stepsize: float = 0.1,
    ) -> None:
        if backend not in BACKEND_DEVICE_MAP:
            raise ValueError(
                f"Unknown backend {backend!r}. "
                f"Choose from {sorted(BACKEND_DEVICE_MAP)}"
            )
        if precision not in ("single", "double"):
            raise ValueError("precision must be 'single' or 'double'")
        if backend in QPU_BACKENDS:
            raise ValueError(
                f"{backend!r} is real hardware and cannot run the optimization "
                f"loop: QPUs have no adjoint differentiation, so PennyLane falls "
                f"back to parameter-shift and bills one task per shifted circuit "
                f"(~$73,000 for a 50-step run at N=8). Optimize on a simulator "
                f"and pass sample_backend={backend!r} to put only the final "
                f"sampling task on hardware."
            )
        if sample_backend is not None and sample_backend not in BACKEND_DEVICE_MAP:
            raise ValueError(
                f"Unknown sample_backend {sample_backend!r}. "
                f"Choose from {sorted(BACKEND_DEVICE_MAP)}"
            )
        if record_state and backend not in LOCAL_BACKENDS:
            raise ValueError(
                "record_state needs a local simulator backend: the exact "
                "probability vector is not available from a cloud device"
            )
        super().__init__(backend=backend)
        self.p_layers = p_layers
        self.n_shots = n_shots
        self.n_optimizer_steps = n_optimizer_steps
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix or BACKEND_S3_PREFIX.get(
            sample_backend or backend, "braket")
        self.precision = precision
        #: Where the single final sampling circuit runs. Defaults to the
        #: optimizer's backend, so existing callers are unaffected.
        self.sample_backend = sample_backend or backend
        #: Also return the exact probability vector of the final and the
        #: initial QAOA state (2^N floats each) and the expectation at the
        #: final angles. Local simulators only; evaluated *after* the sampling
        #: circuit so the default result is untouched (the analytic circuits
        #: consume no RNG, but ordering them last makes that irrelevant).
        self.record_state = record_state
        if optimizer not in ("adam", "lbfgs"):
            raise ValueError("optimizer must be 'adam' or 'lbfgs'")
        self.optimizer = optimizer
        self.stepsize = float(stepsize)

    @property
    def c_dtype(self) -> type:
        """Complex dtype implied by ``precision``."""
        return np.complex64 if self.precision == "single" else np.complex128

    def solve(
        self, problem: QUBOProblem, seed: int = 42, shot_seed: int | None = None
    ) -> SolverResult:
        """Run QAOA and return the best sampled bitstring.

        Args:
            problem: The QUBO to solve.
            seed: Seeds the QAOA initial angles, and by default the measurement
                sampling too.
            shot_seed: Re-seeds the global RNG immediately before the final
                sampling circuit, leaving the optimized angles untouched. This
                separates two things a single seed conflates. On a
                near-degenerate landscape the returned portfolio is a *draw*
                from the near-optimal set, so a run can differ from another for
                two unrelated reasons: it optimized to different angles, or it
                sampled the same state differently. SV1 has no controllable shot
                RNG and disagreed with itself at N = 20 (Supplementary S-VII), which is why
                the local control exists.

        Returns:
            The best sampled bitstring and its metrics.
        """
        try:
            import pennylane as qml
            from pennylane import numpy as pnp
        except ImportError as e:
            raise ImportError("Install pennylane and pennylane-braket: pip install pennylane pennylane-braket") from e

        np.random.seed(seed)
        n = problem.n_variables

        # Convert QUBO → Ising
        h, J, ising_offset = PortfolioQUBO().to_ising(problem)

        # Normalize the cost Hamiltonian before building the circuit.
        #
        # gamma is a phase-rotation angle applied to the cost Hamiltonian, so the
        # optimization landscape in gamma has period ~2*pi/max|coeff|. The Ising
        # coefficients grow with the cardinality penalty A (itself O(N)), which for
        # N=12 already makes that period ~0.06 — far smaller than a typical ADAM
        # step, so the optimizer jumps across many periods and never converges.
        # Scaling H by a constant leaves argmin_x unchanged, so the reported energy
        # (computed from the original Q in _make_result) is unaffected; it only puts
        # gamma on an O(1) scale and makes a single step size valid across all N.
        h_scale = float(max(np.max(np.abs(h)), np.max(np.abs(J)), 1e-12))
        cost_h = self._build_cost_hamiltonian(h / h_scale, J / h_scale, n, qml)
        mixer_h = self._build_mixer_hamiltonian(n, qml)

        # Select device (shots passed to qnode, not device, per PennyLane ≥0.38)
        dev = self._make_device(n, qml)

        # QAOA circuit — analytic (no shots) for gradient-based optimization
        @qml.qnode(dev)
        def qaoa_layer(params):
            gammas = params[:self.p_layers]
            betas  = params[self.p_layers:]
            # Uniform superposition
            for i in range(n):
                qml.Hadamard(wires=i)
            for gamma, beta in zip(gammas, betas):
                qml.ApproxTimeEvolution(cost_h, gamma, 1)
                qml.ApproxTimeEvolution(mixer_h, beta, 1)
            return qml.expval(cost_h)

        # Optimize parameters
        rng = np.random.default_rng(seed)
        # Must be a pennylane.numpy tensor with requires_grad=True, otherwise the
        # optimizer silently leaves the parameters untouched (flat convergence curve).
        params = pnp.array(
            rng.uniform(0, np.pi, size=2 * self.p_layers), requires_grad=True
        )
        t0 = self._timer()
        energies = []
        # Angles *before* each step, so trajectory[i] is where energies[i] was
        # evaluated; the angles after the last step are ``final_angles`` and
        # have no expectation in ``energies`` (step_and_cost returns the cost
        # at the pre-step point).
        trajectory = []
        evaluations = {"objective": 0, "gradient": 0}
        if self.optimizer == "adam":
            opt = qml.AdamOptimizer(stepsize=self.stepsize)
            for step in range(self.n_optimizer_steps):
                trajectory.append(frozen_angles(params).copy())
                params, cost = opt.step_and_cost(qaoa_layer, params)
                energies.append(float(cost))
            # step_and_cost evaluates the objective and its gradient once each.
            evaluations = {"objective": self.n_optimizer_steps,
                           "gradient": self.n_optimizer_steps}
            final_angles = frozen_angles(params).copy()
        else:
            final_angles = self._lbfgs(qaoa_layer, frozen_angles(params),
                                       trajectory, energies, evaluations, qml)
        t_optimised = self._timer()

        # Sample best bitstring
        # A different sampling backend means exactly one circuit runs there --
        # the whole point of --sample-backend for metered hardware.
        sample_dev = (
            dev if self.sample_backend == self.backend
            else self._make_device(n, qml, self.sample_backend)
        )
        # ``final_angles``, not ``params``: the ADAM loop above rebinds
        # ``params`` to the array each step returns, but the L-BFGS-B branch
        # receives a frozen copy and returns its result into ``final_angles``,
        # leaving ``params`` at the initial angles. Until 2026-09-13 this call
        # read ``params``, so the L-BFGS-B path sampled its batch from the
        # *initial* state while recording the optimised one (found when the
        # E3 batches' feasible fraction tracked initial_P_F, not final_P_F).
        best_x, best_energy = self._sample_best(
            sample_dev, final_angles, problem, n, qml, cost_h, mixer_h, shot_seed
        )
        runtime = self._timer() - t0

        state: dict = {}
        if self.record_state:
            t_state = self._timer()

            @qml.qnode(dev)
            def probs_circuit(angles):
                gammas = angles[:self.p_layers]
                betas = angles[self.p_layers:]
                for i in range(n):
                    qml.Hadamard(wires=i)
                for gamma, beta in zip(gammas, betas):
                    qml.ApproxTimeEvolution(cost_h, gamma, 1)
                    qml.ApproxTimeEvolution(mixer_h, beta, 1)
                return qml.probs(wires=range(n))

            best_step = int(np.argmin(energies)) if energies else None
            state = {
                "final_probabilities": np.asarray(probs_circuit(final_angles), dtype=float),
                "initial_probabilities": np.asarray(
                    probs_circuit(trajectory[0] if trajectory else final_angles), dtype=float),
                "expectation_at_final_angles": float(qaoa_layer(final_angles)),
                "best_expectation": (energies[best_step] if energies else None),
                "best_expectation_step": best_step,
                "best_expectation_angles": (trajectory[best_step] if energies else None),
                "state_seconds": self._timer() - t_state,
            }

        return self._make_result(
            problem,
            best_x,
            runtime,
            {
                "p_layers": self.p_layers,
                "n_shots": self.n_shots,
                "n_optimizer_steps": self.n_optimizer_steps,
                "precision": self.precision,
                "sample_backend": self.sample_backend,
                "feasible_shot_fraction": self._feasible_fraction(problem),
                "shot_seed": shot_seed,
                "shot_seed_applied": getattr(self, "shot_seed_applied", None),
                # Convergence values are expectations of the NORMALIZED cost
                # Hamiltonian; multiply by hamiltonian_scale for Ising units.
                "hamiltonian_scale": h_scale,
                "final_expectation": energies[-1] if energies else None,
                "convergence": energies,
                "optimizer": self.optimizer,
                "stepsize": self.stepsize if self.optimizer == "adam" else None,
                "objective_evaluations": evaluations["objective"],
                "gradient_evaluations": evaluations["gradient"],
                # Present only on the L-BFGS-B path; None says "not applicable"
                # rather than "converged", which is the distinction the E3
                # tables needed and did not have.
                "scipy_status": evaluations.get("scipy_status"),
                "scipy_message": evaluations.get("scipy_message"),
                "scipy_iterations": evaluations.get("scipy_nit"),
                "scipy_final_grad_norm": evaluations.get("scipy_grad_norm"),
                "scipy_version": evaluations.get("scipy_version"),
                "scipy_ftol": evaluations.get("scipy_ftol"),
                "scipy_gtol": evaluations.get("scipy_gtol"),
                "initial_angles": trajectory[0] if trajectory else final_angles,
                "final_angles": final_angles,
                "angles_trajectory": np.asarray(trajectory, dtype=float),
                "optimisation_seconds": t_optimised - t0,
                "sampling_seconds": runtime - (t_optimised - t0),
                "last_samples": getattr(self, "last_samples", None),
                **state,
            },
        )

    def evaluate_fixed_angles(
        self,
        problem: QUBOProblem,
        angles,
        *,
        batches: int = 1,
        shot_seed: int | None = None,
        probabilities: bool = True,
        state_indices=None,
    ) -> dict:
        """Run the QAOA circuit at *given* angles: no optimisation, no gradient.

        Plan E4-A needs the same state measured on several backends, so the
        angles come from outside (a stored E1 run) and the circuit is exactly
        the one ``solve`` builds -- same Hamiltonian normalisation, same gate
        sequence -- on the ``sample_backend`` device. Two things are measured:

        * the exact probabilities of the state -- the full 2^N vector on a
          local simulator (``qml.probs``), or on a cloud simulator the
          probabilities of the basis states in ``state_indices`` (SV1 refuses
          the ``Probability`` result type at zero shots but returns
          ``Amplitude`` for a list of states, so the feasible set's mass is
          read from one analytic task), and
        * ``batches`` independent batches of ``n_shots`` measurements, each a
          separate circuit execution and, on a cloud backend, a separate
          billable task.

        On a cloud backend the shots are read from the task's S3 output rather
        than from the plugin's parse (the discipline of ``_sample_best``), the
        task ARNs and the simulator's billed ``executionDuration`` are
        returned, and the metered device is constructed exactly once.

        Args:
            problem: The QUBO whose Ising form defines the circuit.
            angles: ``2 * p_layers`` QAOA angles (gammas then betas).
            batches: Number of independent ``n_shots`` batches to draw.
            shot_seed: Local simulators only: re-seeds the *device's* shot
                generator before batch ``b`` with ``shot_seed + b`` so the
                draws are reproducible (Lightning devices; the Braket local
                simulator cannot be seeded and ``shot_seed_applied`` says so).
                Ignored on cloud backends, whose samplers are not seedable.
            probabilities: Also return the exact probabilities. Refused on
                real hardware, which has no analytic mode.
            state_indices: Basis-state indices (wire 0 the most significant
                bit) whose probabilities are wanted; required on a cloud
                backend when ``probabilities`` is set, optional locally.

        Returns:
            A dict with ``probabilities`` (2^N floats; ``None`` on a cloud
            backend), ``state_probabilities`` (aligned with ``state_indices``,
            or ``None``), ``batches``
            (list of ``(n_shots, N)`` 0/1 arrays), ``task_arns``,
            ``execution_ms`` (per task, cloud only), ``hamiltonian_scale``,
            ``backend``, ``seconds``.
        """
        import pennylane as qml

        backend = self.sample_backend
        if probabilities and backend in QPU_BACKENDS:
            raise ValueError("a QPU has no analytic mode; pass probabilities=False")
        cloud = backend not in LOCAL_BACKENDS
        if probabilities and cloud and state_indices is None:
            raise ValueError("a cloud simulator returns amplitudes of named states only; "
                             "pass state_indices (e.g. the feasible set)")
        angles = frozen_angles(angles)
        if angles.shape != (2 * self.p_layers,):
            raise ValueError(f"expected {2 * self.p_layers} angles, got {angles.shape}")
        n = problem.n_variables
        h, J, _ = PortfolioQUBO().to_ising(problem)
        h_scale = float(max(np.max(np.abs(h)), np.max(np.abs(J)), 1e-12))
        cost_h = self._build_cost_hamiltonian(h / h_scale, J / h_scale, n, qml)
        mixer_h = self._build_mixer_hamiltonian(n, qml)
        # One device for everything below: on a metered backend this is the
        # construction a test spies on.
        dev = self._make_device(n, qml, backend)
        t0 = self._timer()

        def layers(params):
            gammas = params[:self.p_layers]
            betas = params[self.p_layers:]
            for i in range(n):
                qml.Hadamard(wires=i)
            for gamma, beta in zip(gammas, betas):
                qml.ApproxTimeEvolution(cost_h, gamma, 1)
                qml.ApproxTimeEvolution(mixer_h, beta, 1)

        out: dict = {"backend": backend, "hamiltonian_scale": h_scale,
                     "task_arns": [], "execution_ms": [], "probabilities": None,
                     "state_indices": state_indices, "state_probabilities": None,
                     "batches": [], "shot_seed": None if cloud else shot_seed,
                     "shot_seed_applied": None}

        if probabilities and not cloud:
            @qml.qnode(dev)
            def probs_circuit(params):
                layers(params)
                return qml.probs(wires=range(n))

            P = np.asarray(probs_circuit(angles), dtype=float)
            if P.shape != (2 ** n,) or abs(P.sum() - 1.0) > 1e-4:   # fp32 states sum to 1 ± ~1e-6
                raise RuntimeError(f"{backend} returned a probability vector of shape "
                                   f"{P.shape} summing to {P.sum()!r}")
            out["probabilities"] = P
            if state_indices is not None:
                out["state_probabilities"] = P[np.asarray(state_indices, dtype=np.int64)]
        elif probabilities:
            with qml.tape.QuantumTape() as tape:
                layers(angles)
                qml.expval(qml.PauliZ(0))   # placeholder; replaced by Amplitude below
            probs, arn, ms = self._cloud_state_probabilities(dev, tape, state_indices, n, qml)
            out["state_probabilities"] = probs
            out["task_arns"].append(arn)
            out["execution_ms"].append(ms)

        @qml.qnode(dev, shots=self.n_shots)
        def sample_circuit(params):
            layers(params)
            return [qml.sample(qml.PauliZ(i)) for i in range(n)]

        for b in range(batches):
            if not cloud and shot_seed is not None:
                out["shot_seed_applied"] = self._seed_sampler(dev, shot_seed + b)
            submitted_after = datetime.now(timezone.utc) - timedelta(seconds=5)
            raw = sample_circuit(angles)
            if cloud:
                arn = resolve_task_arn(dev, submitted_after, BRAKET_REGION)
                if arn in out["task_arns"]:
                    raise RuntimeError(f"task {arn} was already attributed to an earlier "
                                       f"circuit of this run; refusing to double-count it")
                samples = samples_from_counts(shot_counts_from_task(arn, BRAKET_REGION), n)
                out["task_arns"].append(arn)
                out["execution_ms"].append(execution_duration_ms(arn, BRAKET_REGION))
            else:
                z = np.atleast_2d(np.asarray(raw, dtype=float))
                if z.shape[0] == n:
                    z = z.T
                _validate_samples(z, n, self.n_shots, backend)
                samples = (1.0 - z) / 2.0
            if samples.shape != (self.n_shots, n):
                raise RuntimeError(f"batch {b} has shape {samples.shape}")
            out["batches"].append(samples)
        out["seconds"] = self._timer() - t0
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _lbfgs(self, qnode, x0: np.ndarray, trajectory: list, energies: list,
               evaluations: dict, qml) -> np.ndarray:
        """Minimise the analytic expectation with SciPy L-BFGS-B.

        Every objective evaluation is appended to ``trajectory``/``energies``
        (so the record is per evaluation, not per iteration, and the line
        search's extra evaluations are visible), and the run stops once the
        gradient has been evaluated ``n_optimizer_steps`` times, which is the
        budget ADAM gets. Angles are unbounded, as in the ADAM path.
        """
        import scipy
        from scipy.optimize import minimize

        budget = self.n_optimizer_steps
        # ftol and gtol are SciPy's defaults and are recorded rather than set,
        # because which of them stopped a run is only interpretable next to
        # their values and the precision the objective was computed at.
        evaluations["scipy_version"] = scipy.__version__
        evaluations["scipy_ftol"] = 2.220446049250313e-09
        evaluations["scipy_gtol"] = 1e-05
        evaluations["precision"] = self.precision

        class _Budget(Exception):
            pass

        best = {"x": np.asarray(x0, dtype=float).copy(), "f": np.inf}

        def objective(x):
            if evaluations["gradient"] >= budget:
                raise _Budget
            x = np.asarray(x, dtype=float)
            f = float(qnode(x))
            evaluations["objective"] += 1
            trajectory.append(x.copy())
            energies.append(f)
            if f < best["f"]:
                best["x"], best["f"] = x.copy(), f
            return f

        def gradient(x):
            # SciPy hands over a plain array, which PennyLane treats as
            # non-trainable (qml.grad then warns and returns nothing), so the
            # angles are re-wrapped as a trainable tensor for the adjoint pass.
            from pennylane import numpy as pnp

            x = pnp.array(np.asarray(x, dtype=float), requires_grad=True)
            g = np.asarray(qml.grad(qnode)(x), dtype=float)
            if g.shape != x.shape:
                raise RuntimeError("L-BFGS gradient has the wrong shape; "
                                   "the angles were not trainable")
            evaluations["gradient"] += 1
            return g

        try:
            res = minimize(objective, np.asarray(x0, dtype=float), jac=gradient,
                           method="L-BFGS-B",
                           options={"maxfun": 10 * budget, "maxiter": 10 * budget})
            final = np.asarray(res.x, dtype=float)
            # Why it stopped, not just that it did. "Converged" was inferred
            # from having used fewer gradient evaluations than the cap, which is
            # also what a failed line search looks like -- and with a
            # single-precision expectation, SciPy's default ftol of ~2e-9 sits
            # two orders of magnitude below the noise floor of the objective, so
            # a line search that cannot make progress is the likely stop. The
            # tolerances are SciPy's defaults; they are recorded so a reader
            # does not have to know that.
            evaluations["scipy_status"] = int(res.status)
            evaluations["scipy_message"] = str(res.message)
            evaluations["scipy_nit"] = int(res.nit)
            evaluations["scipy_grad_norm"] = float(np.max(np.abs(res.jac)))
        except _Budget:
            final = best["x"]
            evaluations["scipy_status"] = -1
            evaluations["scipy_message"] = "stopped at the gradient budget"
            evaluations["scipy_nit"] = -1
            evaluations["scipy_grad_norm"] = float("nan")
        return final.copy()

    def _build_cost_hamiltonian(
        self, h: np.ndarray, J: np.ndarray, n: int, qml
    ):
        coeffs, ops = [], []
        for i in range(n):
            if abs(h[i]) > 1e-12:
                coeffs.append(h[i])
                ops.append(qml.PauliZ(i))
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    coeffs.append(J[i, j])
                    ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
        if not coeffs:
            coeffs, ops = [0.0], [qml.Identity(0)]
        return qml.Hamiltonian(coeffs, ops)

    def _build_mixer_hamiltonian(self, n: int, qml):
        coeffs = [-1.0] * n
        ops = [qml.PauliX(i) for i in range(n)]
        return qml.Hamiltonian(coeffs, ops)

    def _make_device(self, n: int, qml, backend: str | None = None):
        """Create a PennyLane device WITHOUT shots (shots are set on the QNode).

        Args:
            n: Wire count.
            qml: The imported ``pennylane`` module.
            backend: Which backend to build. Defaults to the optimizer's; the
                final sampling step passes ``sample_backend`` so hardware can be
                reached without ever hosting the optimization loop.
        """
        backend = backend or self.backend
        device_str = BACKEND_DEVICE_MAP[backend]

        if backend in ("lightning_cpu", "lightning_gpu"):
            try:
                return qml.device(device_str, wires=n, c_dtype=self.c_dtype)
            except Exception as e:
                if backend == "lightning_gpu":
                    # Never fall back to CPU silently: a paper must not report
                    # CPU timings under a GPU label.
                    raise RuntimeError(
                        "lightning.gpu is unavailable on this machine. It requires "
                        "Linux + NVIDIA CUDA:\n"
                        "    pip install -r requirements-gpu.txt\n"
                        f"Use backend='lightning_cpu' instead. Original error: {e}"
                    ) from e
                raise

        if backend == "braket_local":
            return qml.device(device_str, wires=n)

        if backend in ("braket_sv1", "braket_ionq"):
            if not self.s3_bucket:
                raise ValueError(f"s3_bucket is required for {backend}")
            arn = SV1_ARN if backend == "braket_sv1" else IONQ_ARN
            return qml.device(
                device_str,
                device_arn=arn,
                wires=n,
                s3_destination_folder=(self.s3_bucket, self.s3_prefix),
                aws_session=self._aws_session(),
            )

        raise ValueError(f"Unknown backend: {backend}")

    @staticmethod
    def _cloud_state_probabilities(dev, tape, state_indices, n: int, qml):
        """Exact probabilities of named basis states from one analytic cloud task.

        SV1 rejects ``Probability`` at zero shots (``minShots: 1``) but serves
        ``Amplitude`` for a list of basis states at zero shots. The tape is
        decomposed to the plugin device's own gate set -- the same expansion
        the device applies when a QNode runs on it -- converted with the
        device's ``apply``, given an ``Amplitude`` result type over the
        requested states, and submitted through the device's underlying
        ``AwsDevice`` with ``shots=0``.

        Args:
            dev: The PennyLane-Braket AWS device.
            tape: A tape holding the circuit's operations (its measurement is
                ignored).
            state_indices: Basis-state indices, wire 0 the most significant bit.
            n: Number of wires.
            qml: The imported ``pennylane`` module.

        Returns:
            ``(probabilities aligned with state_indices, task ARN, execution ms)``.
        """
        from src.qubo.scoring import bitstrings_from_indices

        idx = np.asarray(state_indices, dtype=np.int64)
        # ``qml.device`` wraps the plugin's legacy device in a facade; the
        # gate set, ``apply`` and the underlying AwsDevice live on the target.
        plugin = getattr(dev, "target_device", dev)
        [expanded], _ = qml.transforms.decompose(tape, gate_set=set(plugin.operations))
        circuit = plugin.apply(expanded.operations, rotations=None)
        states = ["".join(str(int(b)) for b in row)
                  for row in bitstrings_from_indices(idx, n)]
        circuit.amplitude(state=states)
        if hasattr(plugin, "_s3_folder"):
            task = plugin._device.run(circuit, s3_destination_folder=plugin._s3_folder, shots=0,
                                      poll_timeout_seconds=plugin._poll_timeout_seconds,
                                      poll_interval_seconds=plugin._poll_interval_seconds)
        else:
            # The Braket local simulator (free): same conversion, no S3, no bill.
            task = plugin._device.run(circuit, shots=0)
        result = task.result()
        amplitudes = result.values[0]
        if not isinstance(amplitudes, dict) or len(amplitudes) != len(states):
            raise RuntimeError(f"task {task.id} returned {type(amplitudes).__name__} of "
                               f"length {len(amplitudes) if hasattr(amplitudes, '__len__') else '?'}, "
                               f"expected amplitudes for {len(states)} states")
        probs = np.asarray([abs(complex(amplitudes[s])) ** 2 for s in states], dtype=float)
        ms = (execution_duration_ms(task.id, BRAKET_REGION) if hasattr(plugin, "_s3_folder")
              else float("nan"))
        return probs, task.id, ms

    @staticmethod
    def _seed_sampler(dev, seed: int) -> bool:
        """Seed the *device's* shot generator, and say whether that was possible.

        Found 2026-09-12 while building the E4 fixed-angle path: the earlier
        control re-seeded NumPy's global generator before the sampling circuit,
        but the Lightning devices draw their own seed from that generator once,
        at construction (``seed="global"``), and thereafter sample from a
        private ``numpy.random.Generator``. Re-seeding the global state before
        sampling therefore changed nothing, and the two "shot-RNG control"
        sweeps of 2026-08 re-drew identical shots on every row -- 10 of 10
        bitstrings agreed because the samples were the same, not because
        best-of-shots is insensitive to sampling. The Braket local simulator
        has no seedable generator at all.

        Args:
            dev: The PennyLane device about to run a shots circuit.
            seed: The shot seed.

        Returns:
            ``True`` if the device exposes a generator that was re-seeded;
            ``False`` if the device's sampling cannot be seeded from here, in
            which case the caller must record the seed as not applied rather
            than claim reproducible shots.
        """
        if not hasattr(dev, "_rng"):
            return False
        rng = np.random.default_rng(seed)
        dev._rng = rng
        # The device hands its generator to the statevector object when that is
        # first built and reuses the object afterwards, so a generator swapped
        # on the device alone is never consulted once a circuit has run.
        sv = getattr(dev, "_statevector", None)
        if sv is not None and hasattr(sv, "_rng"):
            sv._rng = rng
        return True

    @staticmethod
    def _aws_session():
        """Build a Braket session pinned to :data:`BRAKET_REGION`.

        Returns:
            An ``AwsSession`` whose boto3 session targets a Braket-supported
            region regardless of the caller's environment.
        """
        import boto3
        from braket.aws import AwsSession

        return AwsSession(boto_session=boto3.Session(region_name=BRAKET_REGION))

    def _feasible_fraction(self, problem) -> float | None:
        """Share of the final shots that met the cardinality constraint.

        The reported energy is a minimum over shots, so it says nothing about
        how often the circuit produced a usable portfolio at all. On hardware
        that gap was 89 of 100 shots infeasible under a reported gap of zero.
        """
        k = problem.metadata.get("num_select")
        samples = getattr(self, "last_samples", None)
        if k is None or samples is None:
            return None
        return float(np.mean(np.sum(samples, axis=1) == k))

    def _sample_best(self, dev, params, problem, n, qml, cost_h, mixer_h,
                     shot_seed: int | None = None):
        """Run final circuit with shots and return the lowest-energy sample.

        The Braket PennyLane device does not support computational-basis
        ``qml.sample(wires=...)``; it only supports observable-based sampling.
        We therefore sample PauliZ on each wire and map eigenvalues to bits:
        z = +1 → x = 0,  z = -1 → x = 1.

        The angles arrive from the optimizer as trainable tensors and must be
        frozen first. Optimization is over, so no gradient is read here, but a
        QNode returning one sample per wire cannot use the adjoint method, and
        with trainable inputs PennyLane falls back to parameter-shift and
        evaluates 2x(number of parameterised gates) extra circuits. On SV1 each
        of those is a separate billable task: measured 2026-08-09, N=8 sampling
        submitted 177 tasks instead of 1, and the waste grows as O(N^2) with the
        cost Hamiltonian's term count.
        """
        params = frozen_angles(params)
        self.shot_seed_applied = (self._seed_sampler(dev, shot_seed)
                                  if shot_seed is not None else None)
        # Stamped before the run so the billed task can be found afterwards even
        # when the plugin does not record it -- see resolve_task_arn.
        submitted_after = datetime.now(timezone.utc) - timedelta(seconds=5)

        @qml.qnode(dev, shots=self.n_shots)
        def sample_circuit(params):
            gammas = params[:self.p_layers]
            betas  = params[self.p_layers:]
            for i in range(n):
                qml.Hadamard(wires=i)
            for gamma, beta in zip(gammas, betas):
                qml.ApproxTimeEvolution(cost_h, gamma, 1)
                qml.ApproxTimeEvolution(mixer_h, beta, 1)
            return [qml.sample(qml.PauliZ(i)) for i in range(n)]

        raw = sample_circuit(params)

        if self.sample_backend in QPU_BACKENDS:
            # Do not trust the plugin's parse on hardware. IonQ returns the
            # nested program-set schema, which plugin 1.35 silently turns into
            # an array that is not the measurements; all three tasks on
            # 2026-08-09 came back as a bitstring of zeros while the real shots
            # -- 60, 79 and 92 distinct strings out of 100 -- sat in S3. The
            # circuit has already run and been billed at this point, so the
            # only question is whether we read the answer or invent one.
            samples = samples_from_counts(
                shot_counts_from_task(
                    resolve_task_arn(dev, submitted_after, BRAKET_REGION),
                    BRAKET_REGION),
                n)
        else:
            # shape (n, n_shots) → (n_shots, n)
            z = np.atleast_2d(np.asarray(raw, dtype=float))
            if z.shape[0] == n:
                z = z.T
            _validate_samples(z, n, self.n_shots, self.sample_backend)
            samples = (1.0 - z) / 2.0  # {+1,-1} → {0,1}

        # Kept so the caller can ask what fraction of the shots satisfied the
        # cardinality constraint. Only the best shot decides the reported
        # energy, which is exactly the summary Sec. VI-A argues hides a failed
        # run, so the distribution behind it has to remain inspectable.
        self.last_samples = samples

        best_energy = np.inf
        best_x = samples[0]
        for x in samples:
            e = float(x @ problem.Q @ x)
            if e < best_energy:
                best_energy = e
                best_x = x
        return best_x, best_energy
