# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Write the derived input statistics the public package ships instead of prices.

The pipeline consumes one thing from the market data: the annualised mean
return vector and covariance matrix of a price window (``PortfolioData.returns``
and ``.covariance``). The price snapshots themselves are third-party data that
cannot be redistributed, so the package carries these statistics for the
50-ticker universe of each window; every instance is a slice of them, bit for
bit (``tests/test_derived_stats.py``). This script regenerates the files from
the local snapshots and prints the snapshot digests recorded in
``data/README.md``.

Usage:
    python -m experiments.paper01_qubo_baseline.derive_input_stats
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.finance.data_loader import FinanceDataLoader  # noqa: E402
from experiments.paper01_qubo_baseline.run_instances import instance_tickers  # noqa: E402
from experiments.paper01_qubo_baseline.run_e1_cross import LOOKBACK_YEARS  # noqa: E402

#: The two windows the paper uses: E0–E4 and the E5 replication.
WINDOWS = ("2026-08-05", "2023-08-05")


def main() -> int:
    loader = FinanceDataLoader()
    tickers = instance_tickers(50, 0)
    for end in WINDOWS:
        data = loader.load(tickers, end_date=end, lookback_years=LOOKBACK_YEARS)
        if data.prices is None:
            raise SystemExit(f"{end}: no price snapshot on this host; the statistics "
                             "can only be regenerated where the snapshot is")
        out = loader.write_derived_stats(data)
        snap = loader._cache_path(data.tickers, data.start_date, data.end_date)
        digest = hashlib.sha256(snap.read_bytes()).hexdigest()[:16] if snap.exists() else "-"
        print(f"{out.name}: {data.n_assets} tickers, {len(data.prices)} trading days "
              f"{data.start_date} – {data.end_date}, from snapshot {snap.name} (sha256 {digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
