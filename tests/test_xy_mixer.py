# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""The constraint-preserving ansatz has to preserve the constraint.

Everything E6 claims rests on one property: the Dicke initial state has all of
its amplitude on bitstrings of weight K, and every mixer factor commutes with
the total Hamming weight, so the state never leaves the feasible set -- at any
angle, at any depth, under any Trotter ordering. If that fails, P_F is no
longer 1 by construction and the comparison against the penalty runs is a
comparison of two different things.

The tests therefore check the property rather than a number: random angles,
several depths, and an assertion that the mass outside the feasible set stays
at zero to machine precision.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.qubo.portfolio import PortfolioQUBO
from src.solvers.braket_solver import BraketSolver


def _problem(n: int, k: int, penalty: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.08, 0.05, n)
    a = rng.normal(0, 0.02, (n, n))
    cov = a @ a.T + np.eye(n) * 0.02
    return PortfolioQUBO().formulate(mu, cov, num_select=k, risk_aversion=0.5,
                                     penalty_strength=penalty)


def _weights(n: int) -> np.ndarray:
    return np.array([bin(i).count("1") for i in range(2 ** n)])


def test_the_dicke_state_is_uniform_on_the_feasible_set_and_zero_elsewhere():
    s = BraketSolver(backend="lightning_cpu", mixer="xy_ring", n_select=3)
    psi = s._dicke_state(6)
    w = _weights(6)
    assert np.allclose(np.abs(psi[w == 3]) ** 2, 1 / 20)      # C(6,3) = 20
    assert np.all(psi[w != 3] == 0)
    assert np.isclose(np.sum(np.abs(psi) ** 2), 1.0)


def test_a_mixer_that_cannot_prepare_its_own_weight_is_refused():
    with pytest.raises(ValueError, match="n_select"):
        BraketSolver(backend="lightning_cpu", mixer="xy_ring")
    with pytest.raises(ValueError, match="weight 7"):
        BraketSolver(backend="lightning_cpu", mixer="xy_ring", n_select=7)._dicke_state(5)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_no_amplitude_leaves_the_feasible_set_at_random_angles(p):
    import pennylane as qml

    n, k = 7, 3
    problem = _problem(n, k)
    s = BraketSolver(backend="lightning_cpu", p_layers=p, record_state=True,
                     mixer="xy_ring", n_select=k)
    h, j, _ = PortfolioQUBO().to_ising(problem)
    scale = max(np.max(np.abs(h)), np.max(np.abs(j)), 1e-12)
    cost_h = s._build_cost_hamiltonian(h / scale, j / scale, n, qml)
    mixer_h = s._build_mixer_hamiltonian(n, qml)
    dev = qml.device("lightning.qubit", wires=n)

    @qml.qnode(dev)
    def circuit(angles):
        s._prepare_initial_state(n, qml)
        for gamma, beta in zip(angles[:p], angles[p:]):
            qml.ApproxTimeEvolution(cost_h, gamma, 1)
            s._apply_mixer(beta, n, qml, mixer_h)
        return qml.probs(wires=range(n))

    rng = np.random.default_rng(7)
    w = _weights(n)
    for _ in range(5):
        probs = np.asarray(circuit(rng.uniform(-np.pi, np.pi, 2 * p)))
        assert probs[w != k].sum() < 1e-12
        assert np.isclose(probs[w == k].sum(), 1.0)


def test_the_published_ansatz_is_unchanged_by_the_new_options():
    """The transverse path must still start uniform over all 2^N states."""
    import pennylane as qml

    n = 5
    s = BraketSolver(backend="lightning_cpu")
    dev = qml.device("lightning.qubit", wires=n)

    @qml.qnode(dev)
    def circuit():
        s._prepare_initial_state(n, qml)
        return qml.probs(wires=range(n))

    assert np.allclose(np.asarray(circuit()), 1 / 2 ** n)


def test_every_shot_of_an_xy_run_satisfies_the_cardinality_constraint():
    n, k = 8, 3
    res = BraketSolver(backend="lightning_cpu", p_layers=2, n_shots=128,
                       n_optimizer_steps=3, record_state=True,
                       mixer="xy_ring", n_select=k).solve(_problem(n, k), seed=42)
    samples = res.metadata["last_samples"]
    assert samples.shape == (128, n)
    assert np.all(samples.sum(axis=1) == k)
    # and the recorded exact state agrees: all of its mass is feasible
    assert np.isclose(res.metadata["final_probabilities"][_weights(n) == k].sum(), 1.0)
    assert int(np.sum(res.bitstring)) == k


def test_the_pair_gate_is_the_xy_evolution_the_paper_writes_down():
    """Sec. IV-C states the mixer factor as exp(i (beta/2) (XX + YY)).

    That is the convention of the gate the solver calls, not a derivation: a
    reader rebuilding the ansatz from the paper needs the sign and the factor
    of two to match. One layer is the even pairs then the odd ones, N pair
    gates for N qubits, and the paper's resource sentence counts them.
    """
    import pennylane as qml
    from scipy.linalg import expm

    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    xx_yy = np.kron(X, X) + np.kron(Y, Y)
    beta = 0.37
    assert np.allclose(qml.IsingXY(2 * beta, wires=[0, 1]).matrix(),
                       expm(1j * (beta / 2) * xx_yy))

    n = 8
    recorded = []

    class _Spy:
        """Records the wires each pair gate is applied to, in order."""
        def __getattr__(self, name):
            return getattr(qml, name)

        @staticmethod
        def IsingXY(angle, wires):            # noqa: N802  (mirrors PennyLane)
            recorded.append((angle, tuple(wires)))

    BraketSolver(backend="lightning_cpu", mixer="xy_ring", n_select=3)._apply_mixer(
        beta, n, _Spy(), None)
    assert [w for _a, w in recorded] == [(0, 1), (2, 3), (4, 5), (6, 7),
                                         (1, 2), (3, 4), (5, 6), (7, 0)]
    assert all(a == 2 * beta for a, _w in recorded)
    assert len(recorded) == n
