# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Tests for the on-disk price cache that makes the QUBO bit-reproducible.

Motivation: pinning the price window by date fixed *which* days are used but not
*what values* come back. Re-downloading the same window returns adjusted closes
that differ in their last digits, which perturbs the QUBO by ~1e-10 relative --
enough for single-precision QAOA to occasionally land in a different basin. The
cache is the fix, so these tests assert exact equality, not approximate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.finance.data_loader import FinanceDataLoader


@pytest.fixture
def loader(tmp_path, monkeypatch) -> FinanceDataLoader:
    """A loader whose data directories are redirected into a temp dir, not the
    repo -- both the price cache and the derived statistics, which would
    otherwise serve the repository's real tickers and hide the download path."""
    ld = FinanceDataLoader()
    monkeypatch.setattr(type(ld), "CACHE_DIR", tmp_path / "prices")
    monkeypatch.setattr(type(ld), "DERIVED_DIR", tmp_path / "derived")
    return ld


def _fake_close(tickers: list[str], n_days: int = 40) -> pd.DataFrame:
    """A price frame with values that are not exactly representable in few digits."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-08-07", periods=n_days, freq="B", name="Date")
    data = 1000.0 + rng.standard_normal((n_days, len(tickers))).cumsum(axis=0) * np.pi
    return pd.DataFrame(data, index=idx, columns=tickers)


def test_cache_round_trip_is_bit_exact(loader) -> None:
    """%.17g must survive the CSV round trip with no loss.

    Approximate equality would defeat the purpose: the drift being eliminated is
    itself only ~1e-10 relative.
    """
    frame = _fake_close(["7203.T", "6758.T", "9984.T"])
    path = loader._cache_path(list(frame.columns), "2023-08-06", "2026-08-05")
    loader._write_cache(path, frame)
    restored = loader._read_cache(path)

    assert restored is not None
    assert list(restored.columns) == list(frame.columns)
    # .to_numpy() comparison, exact: not assert_frame_equal's default tolerance.
    assert np.array_equal(restored.to_numpy(), frame.to_numpy())


def test_second_load_hits_cache_and_skips_download(loader, monkeypatch) -> None:
    """The download path must run once; the second load must not touch the network."""
    tickers = ["7203.T", "6758.T"]
    frame = _fake_close(tickers)
    calls = {"n": 0}

    def fake_download(*args, **kwargs):
        calls["n"] += 1
        # yfinance's multi-ticker shape: a "Close" level the loader indexes into.
        return pd.concat({"Close": frame}, axis=1)

    import yfinance
    monkeypatch.setattr(yfinance, "download", fake_download)

    first = loader.load(tickers, start_date="2023-08-06", end_date="2026-08-05")
    assert calls["n"] == 1
    second = loader.load(tickers, start_date="2023-08-06", end_date="2026-08-05")
    assert calls["n"] == 1, "second load re-downloaded instead of using the cache"

    # The whole point: identical inputs give bit-identical QUBO ingredients.
    assert np.array_equal(first.returns, second.returns)
    assert np.array_equal(first.covariance, second.covariance)


def test_use_cache_false_forces_download_and_refreshes(loader, monkeypatch) -> None:
    tickers = ["7203.T", "6758.T"]
    calls = {"n": 0}

    def fake_download(*args, **kwargs):
        calls["n"] += 1
        return pd.concat({"Close": _fake_close(tickers)}, axis=1)

    import yfinance
    monkeypatch.setattr(yfinance, "download", fake_download)

    loader.load(tickers, start_date="2023-08-06", end_date="2026-08-05")
    loader.load(tickers, start_date="2023-08-06", end_date="2026-08-05",
                use_cache=False)
    assert calls["n"] == 2


def test_corrupt_cache_falls_back_to_download(loader, monkeypatch) -> None:
    """A truncated cache file must not be fatal -- it should be re-fetched."""
    tickers = ["7203.T", "6758.T"]
    path = loader._cache_path(tickers, "2023-08-06", "2026-08-05")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("this is not a csv\x00\x00")

    calls = {"n": 0}

    def fake_download(*args, **kwargs):
        calls["n"] += 1
        return pd.concat({"Close": _fake_close(tickers)}, axis=1)

    import yfinance
    monkeypatch.setattr(yfinance, "download", fake_download)

    data = loader.load(tickers, start_date="2023-08-06", end_date="2026-08-05")
    assert calls["n"] == 1
    assert len(data.tickers) == 2


def test_cache_key_separates_windows_and_ticker_sets(loader) -> None:
    a = loader._cache_path(["7203.T", "6758.T"], "2023-08-06", "2026-08-05")
    b = loader._cache_path(["7203.T", "6758.T"], "2022-08-06", "2026-08-05")
    c = loader._cache_path(["7203.T", "9984.T"], "2023-08-06", "2026-08-05")
    assert len({a, b, c}) == 3


def test_cache_key_is_order_independent(loader) -> None:
    """Ticker order is a caller detail, not a different dataset."""
    a = loader._cache_path(["7203.T", "6758.T"], "2023-08-06", "2026-08-05")
    b = loader._cache_path(["6758.T", "7203.T"], "2023-08-06", "2026-08-05")
    assert a == b
