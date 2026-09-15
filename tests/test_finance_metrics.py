# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Tests for PortfolioMetrics."""

import numpy as np
import pytest

from src.finance.metrics import PortfolioMetrics


@pytest.fixture
def two_asset_universe():
    returns = np.array([0.10, 0.20])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    return returns, cov


class TestPortfolioMetrics:
    def test_sharpe_ratio_sign(self, two_asset_universe):
        returns, cov = two_asset_universe
        m = PortfolioMetrics(risk_free_rate=0.0)
        stats = m.evaluate([0, 1], returns, cov, target_k=2)
        assert stats.sharpe_ratio > 0

    def test_equal_weights_by_default(self, two_asset_universe):
        returns, cov = two_asset_universe
        m = PortfolioMetrics()
        stats = m.evaluate([0, 1], returns, cov, target_k=2)
        assert np.allclose(stats.weights, [0.5, 0.5])

    def test_feasibility_flag(self, two_asset_universe):
        returns, cov = two_asset_universe
        m = PortfolioMetrics()
        s_ok = m.evaluate([0, 1], returns, cov, target_k=2)
        s_bad = m.evaluate([0], returns, cov, target_k=2)
        assert s_ok.feasible
        assert not s_bad.feasible

    def test_empty_selection(self, two_asset_universe):
        returns, cov = two_asset_universe
        m = PortfolioMetrics()
        stats = m.evaluate([], returns, cov, target_k=2)
        assert stats.n_selected == 0
        assert stats.sharpe_ratio == pytest.approx(0.0)

    def test_compare_returns_all_solvers(self, two_asset_universe):
        returns, cov = two_asset_universe
        m = PortfolioMetrics()
        s1 = m.evaluate([0], returns, cov, target_k=1)
        s2 = m.evaluate([1], returns, cov, target_k=1)
        table = m.compare([("classical", s1), ("quantum", s2)])
        assert "classical" in table
        assert "quantum" in table
        assert table["quantum"]["expected_return"] > table["classical"]["expected_return"]

    def test_volatility_nonnegative(self, two_asset_universe):
        returns, cov = two_asset_universe
        m = PortfolioMetrics()
        stats = m.evaluate([0, 1], returns, cov, target_k=2)
        assert stats.volatility >= 0
