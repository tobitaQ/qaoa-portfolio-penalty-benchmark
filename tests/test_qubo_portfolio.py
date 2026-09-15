# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Tests for Layer 1: PortfolioQUBO formulation."""

import numpy as np
import pytest

from src.qubo.portfolio import PortfolioQUBO, QUBOProblem


@pytest.fixture
def small_problem() -> tuple[np.ndarray, np.ndarray]:
    """3-asset toy problem with known structure."""
    np.random.seed(0)
    returns = np.array([0.10, 0.15, 0.08])
    cov = np.array([
        [0.04, 0.01, 0.005],
        [0.01, 0.09, 0.02],
        [0.005, 0.02, 0.03],
    ])
    return returns, cov


class TestPortfolioQUBO:
    def test_matrix_shape(self, small_problem):
        returns, cov = small_problem
        q = PortfolioQUBO().formulate(returns, cov, num_select=2)
        assert q.Q.shape == (3, 3)
        assert q.n_variables == 3

    def test_upper_triangular(self, small_problem):
        returns, cov = small_problem
        q = PortfolioQUBO().formulate(returns, cov, num_select=2)
        lower = np.tril(q.Q, k=-1)
        assert np.allclose(lower, 0), "Q must be upper-triangular"

    def test_evaluate_zero_for_zero_vector(self, small_problem):
        returns, cov = small_problem
        q = PortfolioQUBO().formulate(returns, cov, num_select=2)
        x = np.zeros(3)
        # offset = A * K^2, but xQx = 0
        assert q.evaluate_no_offset(x) == pytest.approx(0.0)

    def test_penalty_enforces_feasibility(self, small_problem):
        """Feasible solution (K=2 selected) must have lower energy than infeasible."""
        returns, cov = small_problem
        q = PortfolioQUBO().formulate(returns, cov, num_select=2)

        x_feasible   = np.array([1.0, 1.0, 0.0])  # K=2 ✓
        x_infeasible = np.array([1.0, 0.0, 0.0])  # K=1 ✗

        e_f = q.evaluate_no_offset(x_feasible)
        e_i = q.evaluate_no_offset(x_infeasible)
        assert e_f < e_i, f"Feasible ({e_f:.4f}) should beat infeasible ({e_i:.4f})"

    def test_risk_aversion_effect(self, small_problem):
        """λ=1 (pure risk) vs λ=0 (pure return) must produce different matrices."""
        returns, cov = small_problem
        q_risk   = PortfolioQUBO().formulate(returns, cov, num_select=2, risk_aversion=1.0)
        q_return = PortfolioQUBO().formulate(returns, cov, num_select=2, risk_aversion=0.0)
        assert not np.allclose(q_risk.Q, q_return.Q)

    def test_metadata_stored(self, small_problem):
        returns, cov = small_problem
        q = PortfolioQUBO().formulate(returns, cov, num_select=2, risk_aversion=0.7)
        assert q.metadata["risk_aversion"] == pytest.approx(0.7)
        assert q.metadata["num_select"] == 2

    def test_custom_asset_names(self, small_problem):
        returns, cov = small_problem
        names = ["AAPL", "GOOG", "MSFT"]
        q = PortfolioQUBO().formulate(returns, cov, num_select=2, asset_names=names)
        assert q.asset_names == names

    def test_to_ising_round_trip(self, small_problem):
        """Energy from QUBO and Ising must agree for any binary vector."""
        returns, cov = small_problem
        problem = PortfolioQUBO().formulate(returns, cov, num_select=2)
        h, J, offset = PortfolioQUBO().to_ising(problem)

        for bits in [[0, 0, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1]]:
            x = np.array(bits, dtype=float)
            z = 1.0 - 2.0 * x          # {0,1} → {+1,-1}
            qubo_e = problem.evaluate_no_offset(x)
            ising_e = float(h @ z + sum(J[i, j] * z[i] * z[j]
                                        for i in range(3) for j in range(i+1, 3)))
            assert qubo_e == pytest.approx(ising_e + offset - problem.offset, abs=1e-8), \
                f"Round-trip mismatch for x={bits}"

    def test_invalid_num_select(self, small_problem):
        returns, cov = small_problem
        with pytest.raises(ValueError):
            PortfolioQUBO().formulate(returns, cov, num_select=0)
        with pytest.raises(ValueError):
            PortfolioQUBO().formulate(returns, cov, num_select=4)

    def test_invalid_risk_aversion(self, small_problem):
        returns, cov = small_problem
        with pytest.raises(ValueError):
            PortfolioQUBO().formulate(returns, cov, num_select=2, risk_aversion=1.5)
