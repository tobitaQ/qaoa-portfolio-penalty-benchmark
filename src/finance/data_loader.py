# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Finance data loader using yfinance (free, research-purpose data source).

Downloads adjusted close prices, computes:
    - annualised expected returns  (252 trading days)
    - annualised covariance matrix (252 trading days)

PortfolioData is the standard input to PortfolioQUBO.formulate().
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Nikkei 225 top-50 liquid tickers available on yfinance (as of 2025)
NIKKEI225_TICKERS: list[str] = [
    "7203.T",  # Toyota
    "6861.T",  # Keyence
    "8306.T",  # Mitsubishi UFJ
    "9984.T",  # SoftBank Group
    "6758.T",  # Sony
    "4502.T",  # Takeda
    "6954.T",  # Fanuc
    "8035.T",  # Tokyo Electron
    "9432.T",  # NTT
    "7267.T",  # Honda
    "6902.T",  # Denso
    "4063.T",  # Shin-Etsu Chemical
    "9433.T",  # KDDI
    "8316.T",  # SMFG
    "6501.T",  # Hitachi
    "6098.T",  # Recruit
    "7974.T",  # Nintendo
    "4661.T",  # Oriental Land
    "4519.T",  # Chugai Pharma
    "9022.T",  # JR Central
    "8766.T",  # Tokio Marine
    "6367.T",  # Daikin
    "4543.T",  # Terumo
    "3382.T",  # Seven & i
    "2914.T",  # JT
    "5108.T",  # Bridgestone
    "8801.T",  # Mitsui Fudosan
    "6762.T",  # TDK
    "6301.T",  # Komatsu
    "7751.T",  # Canon
    "4568.T",  # Daiichi Sankyo
    "6503.T",  # Mitsubishi Electric
    "8031.T",  # Mitsui & Co
    "8058.T",  # Mitsubishi Corp
    "5401.T",  # Nippon Steel
    "6857.T",  # Advantest
    "9201.T",  # JAL
    "9202.T",  # ANA
    "8411.T",  # Mizuho
    "7011.T",  # Mitsubishi Heavy
    "6702.T",  # Fujitsu
    "4307.T",  # Nomura Research
    "6971.T",  # Kyocera
    "7733.T",  # Olympus
    "4901.T",  # Fujifilm
    "6645.T",  # Omron
    "7832.T",  # Bandai Namco
    "4523.T",  # Eisai
    "3407.T",  # Asahi Kasei
    "5713.T",  # Sumitomo Metal Mining
]


@dataclass
class PortfolioData:
    """Input data for QUBO formulation."""

    tickers: list[str]
    returns: np.ndarray        # Annualised expected returns, shape (N,)
    covariance: np.ndarray     # Annualised covariance matrix, shape (N, N)
    prices: Optional[pd.DataFrame]  # Raw adjusted closes; None when loaded from derived statistics
    start_date: str
    end_date: str

    @property
    def n_assets(self) -> int:
        return len(self.tickers)


class FinanceDataLoader:
    """
    Downloads stock price data with yfinance and computes portfolio statistics.

    Downloads are cached to disk, and the cache is the reproducibility boundary
    of the whole study. Pinning the price *window* by date is not enough: two
    runs on the same day over the same window get adjusted closes that differ in
    their last digits, which moves the QUBO by ~1e-10 relative. That sounds
    negligible, and for the reported energies it is, but single-precision QAOA
    amplifies it — the same (N, p, steps, seed) can land in a different basin and
    report a gap that differs by up to 0.08 percentage points, which is the size
    of the depth effects being measured. Caching the raw closes makes the whole
    pipeline bit-reproducible across processes; verified in
    ``tests/test_data_loader_cache.py``.
    """

    #: Cached price frames live here and are tracked in git (a few hundred KB) so
    #: a clone reproduces the published numbers without hitting the network.
    CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "prices"

    #: Derived statistics -- the annualised mean-return vector and covariance
    #: matrix of a snapshot -- live here and are what the public package ships.
    #: The price snapshots are third-party data that the author cannot
    #: redistribute; the statistics are the author's computation and are the
    #: only inputs the pipeline reads (``PortfolioQUBO`` and the metrics take
    #: ``returns`` and ``covariance``; nothing downstream opens ``prices``).
    #: Every subset's statistics are bit-identical to the slice of the
    #: superset's, so one file per window reproduces every instance
    #: (``tests/test_derived_stats.py``).
    DERIVED_DIR = Path(__file__).resolve().parents[2] / "data" / "derived"

    def _cache_path(self, tickers: list[str], start: str, end: str) -> Path:
        """Cache file for one (ticker set, window). Name is content-derived.

        The ticker list goes through a hash rather than into the filename: the
        subsets here run to 50 symbols, which overflows sane filename limits.
        Sorted so that ticker order does not fork the cache.
        """
        digest = hashlib.sha256(
            "\x00".join(sorted(tickers)).encode()).hexdigest()[:16]
        return self.CACHE_DIR / f"close_{start}_{end}_{len(tickers)}_{digest}.csv"

    @staticmethod
    def _read_cache(path: Path) -> "pd.DataFrame | None":
        """Load a cached frame, or None if it is unusable.

        ``float_precision="round_trip"`` is required, not cosmetic: the default C
        parser's float conversion is not correctly rounded, so without it the
        values that come back differ from the ones written in their last bits --
        exactly the drift the cache exists to remove.
        """
        try:
            frame = pd.read_csv(path, index_col=0, parse_dates=True,
                                float_precision="round_trip")
        except Exception:
            # A truncated or malformed cache must not be fatal: fall through to
            # a fresh download, which overwrites it.
            return None
        # read_csv accepts a lot of garbage without raising (a text file parses
        # as one string column), so validate the shape we actually need.
        if frame.empty or frame.shape[1] == 0:
            return None
        if not all(np.issubdtype(dt, np.floating) for dt in frame.dtypes):
            return None
        return frame

    @staticmethod
    def _write_cache(path: Path, close: "pd.DataFrame") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # %.17g is the shortest format that round-trips float64 exactly, which is
        # the entire point of the cache.
        close.to_csv(path, float_format="%.17g")

    def _read_superset_cache(
        self, tickers: list[str], start: str, end: str
    ) -> "pd.DataFrame | None":
        """Serve ``tickers`` by slicing a cached snapshot that contains them all.

        Cache files are named ``close_<start>_<end>_<count>_<digest>.csv``, so
        candidates for the same window are found by prefix and the digest never
        has to be guessed. The smallest sufficient superset is used, which keeps
        the choice deterministic when several would do.

        Columns come back in **sorted** order, which is the convention every
        cache file already follows (yfinance returns sorted columns and the
        cache is written as downloaded). Returning them in the caller's order
        instead would be a silent one-ULP change in every statistic: the
        selected assets are the same, but the covariance matrix is permuted, so
        ``x @ Q @ x`` accumulates its terms in a different order. Measured on
        N = 24 before this was pinned down: 5.7e-14 in the objective.

        Args:
            tickers: Tickers needed; order is not significant.
            start: First day of the window, "YYYY-MM-DD".
            end: Last day of the window, "YYYY-MM-DD".

        Returns:
            The sliced price frame, or None if no cached superset covers it.
        """
        wanted = set(tickers)
        best: "pd.DataFrame | None" = None
        for path in sorted(self.CACHE_DIR.glob(f"close_{start}_{end}_*.csv")):
            frame = self._read_cache(path)
            if frame is None or not wanted.issubset(frame.columns):
                continue
            if best is None or frame.shape[1] < best.shape[1]:
                best = frame
        return None if best is None else best.loc[:, sorted(wanted)]

    def _derived_path(self, tickers: list[str], start: str, end: str) -> Path:
        digest = hashlib.sha256(
            "\x00".join(sorted(tickers)).encode()).hexdigest()[:16]
        return self.DERIVED_DIR / f"stats_{start}_{end}_{len(tickers)}_{digest}.csv"

    def write_derived_stats(self, data: "PortfolioData") -> Path:
        """Write ``data``'s statistics as one frame: index = tickers, column
        ``mu`` = annualised mean return, then one column per ticker = the
        annualised covariance row. ``%.17g`` round-trips float64 exactly."""
        frame = pd.DataFrame(data.covariance, index=data.tickers, columns=data.tickers)
        frame.insert(0, "mu", data.returns)
        path = self._derived_path(data.tickers, data.start_date, data.end_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, float_format="%.17g")
        return path

    def _read_derived_stats(
        self, tickers: list[str], start: str, end: str
    ) -> "tuple[list[str], np.ndarray, np.ndarray] | None":
        """Serve ``tickers`` from a derived-statistics file covering them all.

        Same discovery and ordering rules as ``_read_superset_cache``: files
        for the window are found by prefix, the smallest sufficient one is
        used, and the tickers come back sorted. Slicing the statistics is
        exact -- a subset's mean and covariance computed from its own price
        columns are the same float64 values as the superset's sliced -- so
        this path and the price path return identical arrays.
        """
        wanted = sorted(set(tickers))
        best: "pd.DataFrame | None" = None
        for path in sorted(self.DERIVED_DIR.glob(f"stats_{start}_{end}_*.csv")):
            try:
                frame = pd.read_csv(path, index_col=0, float_precision="round_trip")
            except Exception:
                continue
            if "mu" not in frame.columns or not set(wanted).issubset(frame.index):
                continue
            if best is None or frame.shape[0] < best.shape[0]:
                best = frame
        if best is None:
            return None
        mu = np.ascontiguousarray(best.loc[wanted, "mu"].to_numpy(dtype=np.float64))
        cov = np.ascontiguousarray(best.loc[wanted, wanted].to_numpy(dtype=np.float64))
        return wanted, mu, cov

    @staticmethod
    def _canonicalise(close: "pd.DataFrame") -> "pd.DataFrame":
        """Rebuild the frame with a fixed dtype and memory layout.

        Bit-identical *values* are not sufficient for bit-identical *statistics*.
        A frame that came out of ``read_csv`` is stored one block per column,
        while one sliced out of a fresh download is a single consolidated block;
        ``DataFrame.cov`` runs over ``to_numpy()``, so the two layouts hand the
        pairwise summation its terms in a different order and the result differs
        in its last bits. On the price windows in ``data/prices`` the two agree
        exactly, but on shorter series they do not (see
        ``tests/test_data_loader_cache.py``), and a discrepancy that only shows
        up on some inputs is worse than one that always does.

        Forcing a C-contiguous float64 array before any statistic is computed
        makes the cached and freshly downloaded paths agree by construction.
        """
        return pd.DataFrame(
            np.ascontiguousarray(close.to_numpy(dtype=np.float64)),
            index=close.index,
            columns=close.columns,
        )

    def load(
        self,
        tickers: list[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_years: int = 3,
        random_seed: int = 42,
        use_cache: bool = True,
    ) -> PortfolioData:
        """
        Download price data and compute annualised returns and covariance.

        Args:
            tickers: List of ticker symbols (Yahoo Finance format).
            start_date: Start date "YYYY-MM-DD". Inferred from lookback_years if None.
            end_date: End date "YYYY-MM-DD". Defaults to today.
            lookback_years: Years of history to use when start_date is None.
            random_seed: Unused here; reserved for reproducibility documentation.
            use_cache: Read/write the on-disk price cache (see ``_cache_path``).
                Pass False to force a fresh download; the cache is then refreshed.

        Returns:
            PortfolioData with returns and covariance ready for QUBO formulation.
        """
        end = end_date or date.today().isoformat()
        start = start_date or (
            date.fromisoformat(end) - timedelta(days=365 * lookback_years)
        ).isoformat()

        close = None
        cache = self._cache_path(tickers, start, end)
        if use_cache and cache.exists():
            close = self._read_cache(cache)

        # Fall back to any cached superset before going to the network. Caching
        # each ticker set separately made the cache internally inconsistent: the
        # N=16 snapshot was fetched seven minutes after the others and yfinance
        # had already revised every one of its sixteen series by up to 4.2e-7
        # relative, so two experiment sizes disagreed about the same asset over
        # the same window. Slicing one snapshot makes the sizes mutually
        # comparable by construction, and is also what lets an instance sweep
        # draw arbitrary subsets without touching the network.
        if close is None and use_cache:
            close = self._read_superset_cache(tickers, start, end)

        # No price snapshot: the derived statistics are sufficient and are what
        # a clone of the public package has.
        if close is None and use_cache:
            derived = self._read_derived_stats(tickers, start, end)
            if derived is not None:
                available, mu, cov = derived
                if len(available) < 2:
                    raise ValueError(f"Too few assets in the derived statistics: {available}")
                return PortfolioData(tickers=available, returns=mu, covariance=cov,
                                     prices=None, start_date=start, end_date=end)

        if close is None:
            try:
                import yfinance as yf
            except ImportError as e:
                raise ImportError("Install yfinance: pip install yfinance") from e

            prices_raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )

            # Handle single vs multi-ticker download format
            if len(tickers) == 1:
                close = prices_raw[["Close"]].rename(columns={"Close": tickers[0]})
            else:
                close = prices_raw["Close"]
            self._write_cache(cache, close)

        # Drop assets with insufficient data (>5% missing)
        threshold = 0.95 * len(close)
        close = close.dropna(axis=1, thresh=int(threshold))
        close = close.ffill().dropna()
        available_tickers = list(close.columns)

        if len(available_tickers) < 2:
            raise ValueError(
                f"Too few assets with sufficient data: {available_tickers}"
            )

        close = self._canonicalise(close)

        # Annualised statistics (252 trading days)
        daily_returns = close.pct_change().dropna()
        mu = daily_returns.mean().values * 252
        cov = daily_returns.cov().values * 252

        return PortfolioData(
            tickers=available_tickers,
            returns=mu,
            covariance=cov,
            prices=close,
            start_date=start,
            end_date=end,
        )

    def load_nikkei_subset(
        self,
        n_assets: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_years: int = 3,
        random_seed: int = 42,
        use_cache: bool = True,
    ) -> PortfolioData:
        """
        Convenience method: load first n_assets tickers from NIKKEI225_TICKERS.

        Experiment sizes from Paper ①: n_assets ∈ {10, 20, 30, 50}.
        """
        if n_assets > len(NIKKEI225_TICKERS):
            raise ValueError(
                f"n_assets={n_assets} exceeds available tickers ({len(NIKKEI225_TICKERS)})"
            )
        tickers = NIKKEI225_TICKERS[:n_assets]
        return self.load(
            tickers,
            start_date=start_date,
            end_date=end_date,
            lookback_years=lookback_years,
            random_seed=random_seed,
            use_cache=use_cache,
        )
