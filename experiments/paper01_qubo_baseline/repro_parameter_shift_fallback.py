# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Minimal reproduction of the two facts behind Sec. VI-B's "177 tasks to sample one bitstring".

They have separate evidence and are kept apart here:

1. **The gradient transform's tape count.** For the p-layer QAOA ansatz of
   Sec. III-A on a dense N-asset QUBO, every term of the cost Hamiltonian and
   every mixer term is one parameterised ``PauliRot``, so there are
   p·[C(N, 2) + 2N] gate parameters and ``qml.gradients.param_shift`` returns
   2·p·[C(N, 2) + 2N] shifted tapes — 176 at N = 8, 920 at N = 20. This is a
   property of the transform and is reproduced locally, with no device.

2. **What the metered service executed.** The SV1 task log of the N = 8 pilot
   (``paper01_sv1_task_ledger.csv``, written by ``sv1_task_ledger.py`` from the
   archived task records) shows 227 tasks: 50 analytic tasks for the fifty
   optimiser steps and 177 = 1 + 176 tasks for the single sampling call —
   the forward sample plus the transform's tapes. The mechanism is the
   framework's: a QNode that returns ``qml.sample`` per wire cannot use the
   adjoint method, and when its angles arrive marked trainable the
   ``braket.aws.qubit`` device's execution path evaluated a parameter-shift
   gradient that nothing read. ``frozen_angles()`` removes the trainable flag,
   after which the same call is 1 task (51 for the whole run).

The *local* devices do not reproduce fact 2: ``lightning.qubit`` and
``braket.local.qubit`` execute the sampling QNode once whether or not the
angles are trainable (this script prints their tracker counts to show it).
That is why the ledger, not a local tracker, is the evidence for the bill,
and why the defect was found only on the metered service.

Run:
    python -m experiments.paper01_qubo_baseline.repro_parameter_shift_fallback [--n 8] [--p 2]
"""

from __future__ import annotations

import argparse
import collections
import platform
import sys
from math import comb

import numpy as np

from src.solvers.braket_solver import frozen_angles

ROTATION_GATES = ("Hadamard", "RZ", "RX", "MultiRZ", "PauliRot", "IsingZZ")


def dense_hamiltonians(n: int, qml, seed: int = 0):
    """A dense random cost Hamiltonian (every pair coupled) and the X mixer."""
    rng = np.random.default_rng(seed)
    coeffs = list(rng.normal(size=n)) + [float(rng.normal()) for _ in range(comb(n, 2))]
    obs = [qml.PauliZ(i) for i in range(n)] + [
        qml.PauliZ(i) @ qml.PauliZ(j) for i in range(n) for j in range(i + 1, n)]
    cost_h = qml.Hamiltonian(coeffs, obs)
    mixer_h = qml.Hamiltonian([-1.0] * n, [qml.PauliX(i) for i in range(n)])
    return cost_h, mixer_h


def layers(params, n: int, p: int, cost_h, mixer_h, qml) -> None:
    """The ansatz exactly as ``BraketSolver`` queues it."""
    for i in range(n):
        qml.Hadamard(wires=i)
    for gamma, beta in zip(params[:p], params[p:]):
        qml.ApproxTimeEvolution(cost_h, gamma, 1)
        qml.ApproxTimeEvolution(mixer_h, beta, 1)


def param_shift_tape_count(n: int, p: int, qml) -> tuple[int, int, dict]:
    """(gate parameters, shifted tapes, gate census) for the expectation tape."""
    from pennylane import numpy as pnp
    cost_h, mixer_h = dense_hamiltonians(n, qml)
    params = pnp.array(np.linspace(0.1, 0.4, 2 * p), requires_grad=True)
    with qml.queuing.AnnotatedQueue() as q:
        layers(params, n, p, cost_h, mixer_h, qml)
        qml.expval(cost_h)
    tape = qml.tape.QuantumScript.from_queue(q)
    expanded = tape.expand(depth=3, stop_at=lambda op: op.name in ROTATION_GATES)
    n_gate_params = sum(op.num_params for op in expanded.operations)
    # Only the gates are shifted: the Hamiltonian coefficients inside the
    # measured observable are parameters of the tape too, and are not gates.
    expanded.trainable_params = list(range(n_gate_params))
    tapes, _ = qml.gradients.param_shift(expanded)
    census = dict(collections.Counter(op.name for op in expanded.operations))
    return n_gate_params, len(tapes), census


def local_execution_counts(n: int, p: int, qml, device: str, shots: int = 1000) -> dict:
    """Executions the local device records for one sampling call, trainable vs. frozen."""
    from pennylane import numpy as pnp
    cost_h, mixer_h = dense_hamiltonians(n, qml)
    dev = qml.device(device, wires=n, shots=shots)

    @qml.qnode(dev, shots=shots)
    def sample_circuit(params):
        layers(params, n, p, cost_h, mixer_h, qml)
        return [qml.sample(qml.PauliZ(i)) for i in range(n)]

    trainable = pnp.array(np.linspace(0.1, 0.4, 2 * p), requires_grad=True)
    out = {}
    for label, params in (("trainable", trainable), ("frozen", frozen_angles(trainable))):
        with qml.Tracker(dev) as tracker:
            sample_circuit(params)
        out[label] = int(tracker.totals.get("executions", 0))
    out["diff_method"] = str(sample_circuit.diff_method)
    return out


def package_versions() -> dict[str, str]:
    """The plugin and SDK versions the metered path ran under (review round 3, §6.4)."""
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for name in ("amazon-braket-pennylane-plugin", "amazon-braket-sdk", "pennylane-lightning"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not installed"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--skip-local", action="store_true", help="tape count only")
    args = ap.parse_args(argv)
    import pennylane as qml

    print(f"python {platform.python_version()}, pennylane {qml.__version__}, "
          f"{', '.join(f'{k} {v}' for k, v in package_versions().items())}, N = {args.n}, p = {args.p}")
    n_params, n_tapes, census = param_shift_tape_count(args.n, args.p, qml)
    expected = 2 * args.p * (comb(args.n, 2) + 2 * args.n)
    print(f"decomposed ansatz: {census}")
    print(f"gate parameters: {n_params}; param_shift tapes: {n_tapes}; "
          f"2·p·[C(N,2) + 2N] = {expected}; +1 forward = {n_tapes + 1}")
    if not args.skip_local:
        for device in ("lightning.qubit", "braket.local.qubit"):
            try:
                counts = local_execution_counts(args.n, args.p, qml, device)
            except Exception as exc:  # a missing plugin is reported, not hidden
                print(f"{device}: not run ({exc.__class__.__name__}: {exc})")
                continue
            print(f"{device}: sampling QNode executions — trainable angles {counts['trainable']}, "
                  f"frozen angles {counts['frozen']} (diff_method={counts['diff_method']}); "
                  f"the local device does not reproduce the metered path")
    print("metered path: see results/paper01_sv1_task_ledger.csv (227 = 50 + 1 + 176 at N = 8).")
    print("What is observed there is the task count; that it arose from a parameter-shift\n"
          "evaluation of the trainable angles is the explanation consistent with the count\n"
          "(1 + 2·p·[C(N,2) + 2N]) and with the fix (frozen angles → 1 task). The metered\n"
          "path was not re-run with instrumentation, so the causal path is inferred, not traced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
