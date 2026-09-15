# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Tests for the instance-axis sweep.

Every number the paper reports comes from one instance per size. This sweep is
what turns those observations into distributions, so two properties matter more
than the arithmetic: the published instance has to still be in the sample, and
no instance may reach the network — otherwise a sweep run on a different day
samples different data and the distribution is not reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.paper01_qubo_baseline.run_instances import (
    instance_digest,
    instance_tickers,
    measure,
)
from src.finance.data_loader import NIKKEI225_TICKERS, FinanceDataLoader
from src.finance.metrics import PortfolioMetrics
from src.qubo.portfolio import PortfolioQUBO
from src.solvers.classical_solver import ClassicalSolver


class TestInstanceSelection:
    @pytest.mark.parametrize("n", [12, 16, 20, 24, 28, 30])
    def test_instance_zero_is_the_published_subset(self, n):
        """The sweep must contain the number the paper already reports."""
        assert instance_tickers(n, 0) == sorted(NIKKEI225_TICKERS[:n])

    def test_selection_is_deterministic(self):
        assert instance_tickers(20, 7) == instance_tickers(20, 7)

    def test_different_indices_give_different_subsets(self):
        drawn = {tuple(instance_tickers(20, i)) for i in range(1, 15)}
        assert len(drawn) == 14

    @pytest.mark.parametrize("n", [12, 20, 30])
    def test_every_instance_stays_inside_the_cached_universe(self, n):
        """Offline reproducibility: a subset outside the cache means a download.

        The committed snapshot holds 50 tickers. Drawing from anything wider
        would send the sweep to yfinance, whose adjusted closes are revised
        between calls -- the exact failure this project cached prices to avoid.
        """
        universe = set(NIKKEI225_TICKERS[:50])
        for instance in range(25):
            picked = instance_tickers(n, instance)
            assert len(picked) == n
            assert len(set(picked)) == n
            assert set(picked).issubset(universe)

    def test_instances_are_sorted_matching_the_loader_convention(self):
        picked = instance_tickers(16, 3)
        assert picked == sorted(picked)

    def test_a_size_beyond_the_universe_is_refused(self):
        with pytest.raises(ValueError, match="cached universe"):
            instance_tickers(60, 0)

    def test_digest_ignores_ticker_order(self):
        a = ["7203.T", "6861.T", "8306.T"]
        assert instance_digest(a) == instance_digest(list(reversed(a)))


class TestOfflineReproducibility:
    def test_a_drawn_instance_loads_without_the_network(self, monkeypatch):
        """Every instance must be served by slicing the committed snapshot.

        yfinance is replaced with a bomb, so any cache miss fails the test
        instead of silently fetching fresh data.
        """
        import sys
        import types

        def explode(*args, **kwargs):
            raise AssertionError("the sweep reached the network")

        fake = types.ModuleType("yfinance")
        fake.download = explode
        monkeypatch.setitem(sys.modules, "yfinance", fake)

        loader = FinanceDataLoader()
        for instance in (0, 1, 9):
            data = loader.load(
                instance_tickers(16, instance),
                end_date="2026-08-05",
                lookback_years=3,
            )
            assert data.n_assets == 16

    def test_the_published_instance_reproduces_the_canonical_loader(self):
        """Instance 0 must be the same data ``load_nikkei_subset`` returns."""
        loader = FinanceDataLoader()
        drawn = loader.load(instance_tickers(20, 0), end_date="2026-08-05",
                            lookback_years=3)
        canonical = loader.load_nikkei_subset(n_assets=20, end_date="2026-08-05",
                                              lookback_years=3)
        assert drawn.tickers == canonical.tickers
        assert np.array_equal(drawn.returns, canonical.returns)
        assert np.array_equal(drawn.covariance, canonical.covariance)


class TestMeasure:
    def test_the_census_optimum_is_the_solver_optimum(self):
        """The sweep aborts on disagreement; assert it has nothing to abort on."""
        loader = FinanceDataLoader()
        formulator = PortfolioQUBO()
        metrics = PortfolioMetrics(risk_free_rate=0.001)
        data = loader.load(instance_tickers(16, 4), end_date="2026-08-05",
                           lookback_years=3)
        k = 3
        problem = formulator.formulate(data.returns, data.covariance,
                                       num_select=k, risk_aversion=0.5)

        measured = measure(problem, data, k, metrics)
        solved = ClassicalSolver(method="exact").solve(problem, seed=42)
        assert measured["row"]["energy_optimum"] == solved.energy_no_offset

    def test_deflation_is_the_penalty_constant_over_the_objective_spread(self):
        loader = FinanceDataLoader()
        formulator = PortfolioQUBO()
        metrics = PortfolioMetrics(risk_free_rate=0.001)
        data = loader.load(instance_tickers(12, 2), end_date="2026-08-05",
                           lookback_years=3)
        k = 2
        problem = formulator.formulate(data.returns, data.covariance,
                                       num_select=k, risk_aversion=0.5)

        row = measure(problem, data, k, metrics)["row"]
        assert row["deflation_factor"] == pytest.approx(
            row["abs_energy_optimum"] / row["objective_range"]
        )
        # |E*| is the penalty constant to within the O(1) objective.
        assert abs(row["abs_energy_optimum"] - problem.offset) < 1.0
