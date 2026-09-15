# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Where 177 and 921 come from (review M1 / R6).

The paper prices a QPU optimization loop at one billable task per
parameter-shift circuit and reports 177 at N = 8 and 921 at N = 20. Those two
numbers carried the largest figure in the paper, and a reviewer could not check
them: they were measured once, on a metered backend, under a plugin version.
They are checked here instead, against the gradient transform itself, on a free
local device.

The claim is narrow and worth stating exactly. Shifting each parameterised gate
of the decomposed ansatz costs

    1 + 2 · p · [C(N, 2) + 2N]

circuits: one unshifted evaluation, and two per parameterised gate appearance --
C(N, 2) two-qubit ZZ rotations, N single-qubit Z rotations and N mixer X
rotations in each of p layers. This is *not* 1 + 2·(2p): the two variational
angles of a layer are shared by every gate in it, and a shared parameter's shift
rule is set by the generator's frequencies, not by the number of symbols the
user typed.
"""

from __future__ import annotations

import math

import pytest

pennylane = pytest.importorskip("pennylane")
qml = pennylane
from pennylane import numpy as pnp  # noqa: E402

P_LAYERS = 2


def _hamiltonians(n: int):
    """A dense cost Hamiltonian of the paper's shape, and the transverse mixer."""
    coeffs, obs = [], []
    for i in range(n):
        coeffs.append(0.5)
        obs.append(qml.PauliZ(i))
        for j in range(i + 1, n):
            coeffs.append(0.25)
            obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
    return (qml.Hamiltonian(coeffs, obs),
            qml.Hamiltonian([1.0] * n, [qml.PauliX(i) for i in range(n)]))


def _ansatz(params, n, cost, mixer):
    gammas, betas = params[:P_LAYERS], params[P_LAYERS:]
    for i in range(n):
        qml.Hadamard(wires=i)
    for gamma, beta in zip(gammas, betas):
        qml.ApproxTimeEvolution(cost, gamma, 1)
        qml.ApproxTimeEvolution(mixer, beta, 1)


def _shift_circuits(n: int) -> int:
    """Circuits a parameter-shift gradient of this ansatz costs, counted as tapes."""
    cost, mixer = _hamiltonians(n)
    params = pnp.array([0.1] * (2 * P_LAYERS), requires_grad=True)
    tape = qml.tape.make_qscript(
        lambda p: (_ansatz(p, n, cost, mixer), qml.expval(cost))[-1])(params)
    tape.trainable_params = list(range(len(tape.get_parameters())))
    tapes, _ = qml.gradients.param_shift(tape)
    return 1 + len(tapes)


def _formula(n: int) -> int:
    return 1 + 2 * P_LAYERS * (math.comb(n, 2) + 2 * n)


@pytest.mark.parametrize("n, expected", [(8, 177), (12, 361), (16, 609), (20, 921)])
def test_the_published_circuit_counts(n, expected):
    assert _formula(n) == expected
    assert _shift_circuits(n) == expected


def test_the_count_is_not_two_per_variational_angle():
    """The number the paper used to compare against, and should not have.

    "Parameter-shift needs 8 circuits per step because there are 2p = 4
    parameters" treats a layer's shared angle as one gate. It is off by a factor
    of 44 at N = 8.
    """
    assert 1 + 2 * (2 * P_LAYERS) == 9
    assert _shift_circuits(8) == 177


def test_sampling_needs_one_circuit():
    """The defect is a gradient where none is read, not an expensive gradient."""
    n = 8
    cost, mixer = _hamiltonians(n)
    dev = qml.device("lightning.qubit", wires=n, shots=100)
    frozen = pnp.array([0.1] * (2 * P_LAYERS), requires_grad=False)

    @qml.qnode(dev)
    def sample(params):
        _ansatz(params, n, cost, mixer)
        return [qml.sample(qml.PauliZ(i)) for i in range(n)]

    with qml.Tracker(dev) as tracker:
        sample(frozen)
    assert tracker.totals["executions"] == 1
