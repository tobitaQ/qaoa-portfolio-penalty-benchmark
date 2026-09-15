# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""The derived statistics reproduce the price path bit for bit.

The public package ships ``data/derived/stats_*.csv`` instead of the price
snapshots. That is only sound if a loader that cannot see any price file
returns, for every instance the paper uses, the same ``returns`` and
``covariance`` arrays -- same values, same ticker order -- as the loader that
slices the snapshot. This pins that for both windows and all 420 (size,
instance) subsets, and pins that the committed files are the current output of
the generator.
"""

from __future__ import annotations

import io
import contextlib
import pathlib

import numpy as np
import pytest

from experiments.paper01_qubo_baseline.derive_input_stats import WINDOWS, main as derive_main
from experiments.paper01_qubo_baseline.run_e1_cross import LOOKBACK_YEARS
from experiments.paper01_qubo_baseline.run_instances import instance_tickers
from src.finance.data_loader import FinanceDataLoader

ROOT = pathlib.Path(__file__).resolve().parents[1]
SIZES = (8, 12, 16, 20, 24, 28, 30)


#: Captured at import, before any fixture redirects the loader's directories.
REAL_CACHE_DIR = FinanceDataLoader.CACHE_DIR


def _has_snapshots() -> bool:
    return any(REAL_CACHE_DIR.glob("close_*.csv"))


@pytest.fixture
def derived_only(monkeypatch, tmp_path):
    """A loader that sees the committed derived files and no price snapshot."""
    monkeypatch.setattr(FinanceDataLoader, "CACHE_DIR", tmp_path / "no-prices")
    return FinanceDataLoader()


class TestDerivedStatistics:
    @pytest.mark.parametrize("end", WINDOWS)
    def test_every_instance_matches_the_price_path(self, derived_only, end):
        if not _has_snapshots():
            pytest.skip("price snapshots are not distributed; run where they are")
        with_prices = FinanceDataLoader()
        with_prices.CACHE_DIR = REAL_CACHE_DIR
        for n in SIZES:
            for inst in range(30):
                tk = instance_tickers(n, inst)
                a = with_prices.load(tk, end_date=end, lookback_years=LOOKBACK_YEARS)
                b = derived_only.load(tk, end_date=end, lookback_years=LOOKBACK_YEARS)
                assert a.prices is not None and b.prices is None
                assert a.tickers == b.tickers
                assert np.array_equal(a.returns, b.returns), (n, inst)
                assert np.array_equal(a.covariance, b.covariance), (n, inst)
                assert b.returns.flags["C_CONTIGUOUS"] and b.covariance.flags["C_CONTIGUOUS"]

    @pytest.mark.parametrize("end", WINDOWS)
    def test_derived_path_serves_the_universe_and_its_slices(self, derived_only, end):
        """Runs everywhere: the committed files are self-consistent."""
        full = derived_only.load(instance_tickers(50, 0), end_date=end, lookback_years=LOOKBACK_YEARS)
        assert full.n_assets == 50 and full.prices is None
        assert full.tickers == sorted(full.tickers)
        assert np.array_equal(full.covariance, full.covariance.T)
        idx = {t: i for i, t in enumerate(full.tickers)}
        for n, inst in ((12, 0), (20, 7), (30, 29)):
            sub = derived_only.load(instance_tickers(n, inst), end_date=end, lookback_years=LOOKBACK_YEARS)
            ii = [idx[t] for t in sub.tickers]
            assert np.array_equal(sub.returns, full.returns[ii])
            assert np.array_equal(sub.covariance, full.covariance[np.ix_(ii, ii)])

    def test_committed_files_are_the_generator_output(self, monkeypatch, tmp_path):
        if not _has_snapshots():
            pytest.skip("price snapshots are not distributed; run where they are")
        monkeypatch.setattr(FinanceDataLoader, "DERIVED_DIR", tmp_path / "derived")
        with contextlib.redirect_stdout(io.StringIO()):
            derive_main()
        for fresh in sorted((tmp_path / "derived").glob("stats_*.csv")):
            committed = ROOT / "data" / "derived" / fresh.name
            assert committed.exists(), fresh.name
            assert committed.read_bytes() == fresh.read_bytes(), fresh.name

    def test_missing_window_falls_through(self, derived_only):
        """A window with no derived file must not be silently served from another."""
        assert derived_only._read_derived_stats(instance_tickers(8, 0), "2019-01-01", "2019-12-31") is None
