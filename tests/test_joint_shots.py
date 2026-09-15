# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""The per-wire PauliZ sampling path returns joint shots, not stitched marginals.

Supplementary Sec. S-IX (2): the Braket device rejects computational-basis ``sample(wires=...)``,
so the solver returns ``[qml.sample(qml.PauliZ(i)) for i in range(n)]`` and maps
eigenvalues to bits. A referee (round 3, §6.5) asked whether each wire's column
comes from the same shot -- a joint measurement -- or from separate executions
whose marginals are then combined, which would destroy every correlation in
the bitstring distribution. On a Bell or GHZ state the two are trivially
distinguishable: a joint measurement never returns wires that disagree.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest

from src.solvers.braket_solver import _validate_samples


def _ghz_samples(device_name: str, n: int, shots: int) -> np.ndarray:
    dev = qml.device(device_name, wires=n, shots=shots)

    @qml.qnode(dev, shots=shots)
    def circuit():
        qml.Hadamard(wires=0)
        for i in range(1, n):
            qml.CNOT(wires=[0, i])
        return [qml.sample(qml.PauliZ(i)) for i in range(n)]

    z = np.atleast_2d(np.asarray(circuit(), dtype=float))
    if z.shape[0] == n:            # (n, shots) → (shots, n), as the solver does
        z = z.T
    _validate_samples(z, n, shots, device_name)
    return z


@pytest.mark.parametrize("device_name", ["lightning.qubit", "braket.local.qubit"])
@pytest.mark.parametrize("n", [2, 3])
def test_per_wire_pauliz_samples_are_joint_measurements(device_name, n):
    z = _ghz_samples(device_name, n, shots=400)
    # Every shot lies in {|0...0>, |1...1>}: the wires agree on every row.
    assert (z == z[:, [0]]).all(), "wires disagree within a shot: marginals were stitched"
    # ...and both branches actually occur, so the test is not passing on a
    # collapsed state.
    assert 0.3 < np.mean(z[:, 0] == 1.0) < 0.7


def test_stitched_marginals_would_have_failed_that_check():
    """The check above has teeth: independent marginals disagree half the time."""
    rng = np.random.default_rng(0)
    z = rng.choice([-1.0, 1.0], size=(400, 2))
    assert not (z == z[:, [0]]).all()
