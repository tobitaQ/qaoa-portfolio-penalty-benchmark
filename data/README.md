# Data

Three kinds of files live here, and they are licensed differently. None is
covered by the MIT licence in `LICENSE`, which applies to the source code only.

## `derived/` — the input statistics the pipeline reads (author's computation)

`stats_<start>_<end>_50_<digest>.csv`, one per price window (2023-08-06 → 2026-08-05
for E0–E4, 2020-08-05 → 2023-08-05 for E5): the annualised mean return (`mu`) and the
annualised covariance matrix (one column per ticker) of the fifty-ticker universe,
written with 17 significant digits and read back with round-trip parsing. These two
arrays are the only thing any experiment consumes from the market data
(`PortfolioQUBO` and the metrics take `returns` and `covariance`; nothing reads the
price series). Every instance of the paper is a subset of the fifty tickers, and a
subset's statistics computed from its own price columns are bit-identical to the
slice of these files (`tests/test_derived_stats.py`, 420 subsets × 2 windows), so a
clone reproduces every number from them with no network access.

They are the author's computation and are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The price series cannot be
recovered from them.

## Price snapshots (third-party data — not distributed)

The statistics were computed by `src/finance/data_loader.py` from two close-price
snapshots obtained through `yfinance` from Yahoo Finance for research use (50 Nikkei 225
tickers; 731 and 735 trading days; sha256 prefixes `66a78a3a0b91c361` and
`063a70d29450bc0e`). Those snapshots are the provider's data and are not
redistributed here. What they were: the fifty tickers of `NIKKEI225_TICKERS` in
`src/finance/data_loader.py` (the sorted list and its digest are also recorded per
instance in `experiments/paper01_qubo_baseline/results/paper01_e1_reference.csv`),
daily adjusted close (`yf.download(tickers, start, end, auto_adjust=True)` at the
default daily interval, the `Close` column), over 2023-08-06 → 2026-08-05 (E0–E4) and
2020-08-05 → 2023-08-05 (E5). The acquisition code is the loader itself and the
derivation is `experiments/paper01_qubo_baseline/derive_input_stats.py`. The loader looks for them under `data/prices/` and, when they are
absent, serves the derived statistics instead; `use_cache=False` re-downloads, but
`yfinance` revises adjusted closes between calls (Sec. IV-D of the paper), so a fresh
download will not reproduce the statistics bit for bit. Anyone re-fetching the series is
responsible for the provider's terms.

## `braket_results/` — the author's own Amazon Braket task records

The four IonQ Forte-1 task records and the SV1 smoke-test record are the raw
output of tasks the author ran and paid for. They are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), the same terms as the
result CSVs (see `experiments/paper01_qubo_baseline/results/README.md`). The
account number in the task ARNs is masked; see `braket_results/README.md` for
the schema and the redaction.
