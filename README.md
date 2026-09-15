# Measuring Metric and Sampling Distortions in Penalty-Encoded QAOA Portfolio Benchmarks

Reproduction package for the paper of the same title (IEEE Transactions on Quantum
Engineering submission; the manuscript PDFs of this version are in `papers/` and in the Zenodo
deposit). Every number in the paper comes from a result file in this repository, and
`papers/check_tables.py`
checks every table cell of the main text and the Supplementary Material against
those files.

- Author: Hiroaki Tobita, Advanced Institute of Industrial Technology, Tokyo.
- Manuscript: `papers/paper01_draft.md` (main text) and `papers/paper01_supplement.md`
  (Supplementary Material); `paper01.tex` / `.pdf` and `paper01_supplement.tex` / `.pdf`
  are built from them by `papers/md_to_latex.py`.
- Nothing here needs network access to a market-data provider: the annualised
  mean returns and covariance of each price window — the only inputs the pipeline
  reads — are committed under `data/derived/`; the price snapshots themselves are
  third-party data and are not distributed (see `data/README.md`).
- Cloud and hardware backends (Amazon Braket SV1, IonQ) are opt-in: they require an
  explicit backend name and a `--results-tag`, so a default invocation cannot bill
  an AWS account.
- The committed result files are protected from a trial run: `run_experiment.py` and
  `run_e1_cross.py` write their untagged (canonical) file names only when those files
  are absent, when `--resume` appends to them, or when `--write-canonical` is passed;
  otherwise they stop and ask for a `--results-tag`. The remaining drivers (`run_sweep.py`,
  `run_instances.py`, ... — the earlier single-instance series) write their fixed file
  names under `results/`; run them with `--results-tag` where the option exists, or in a
  copy of the directory.

## Three levels of reproduction

| Level | What it does | Cost |
|---|---|---|
| 1. Tables and figures from the committed CSVs | `python -m notebooks.generate_tqe_figures`, then `cd papers && python check_tables.py` (prints the number of cells that agree) | seconds, no GPU |
| 2. One small instance end to end | `python -m experiments.paper01_qubo_baseline.run_experiment --backend lightning_cpu --classical-n 8 12 --quantum-n 8 12 --results-tag mytest` | minutes on a CPU |
| 3. The E1 cross from the manifest | `python -m experiments.paper01_qubo_baseline.run_e1_cross --backend lightning_gpu --precision single --resume --results-tag reproduction`, then compare `results/paper01_e1_cross_reproduction.csv` with the committed `paper01_e1_cross.csv` (the state-derived columns agree bit for bit on the same GPU class; the timing columns do not) | 13.9 GPU h (RTX A6000) |

Setup:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # minimum versions; add requirements-gpu.txt for lightning.gpu
python -m pytest -q                      # the test suite pins the scoring rules, the sampler and the aggregations
```

Python ≥ 3.11. `requirements.txt` states the minimum versions the code runs with;
`requirements-repro.txt` is the `pip freeze` of the environment that produced the paper's
numbers (Python 3.11.9, Ubuntu 20.04, CUDA 12.1, driver 570.133 — install it with
`pip install -r requirements-repro.txt` on a CUDA 12 Linux host to reproduce the runs and the
figure renders with the same library versions; different Matplotlib versions change figure
layout, and different simulator versions may change single-precision runs). The GPU runs of the
paper used PennyLane 0.45.1 with pennylane-lightning-gpu 0.45.0
(cuStateVec) at single precision on one RTX A6000; the CPU runs used lightning.qubit at double
precision. Single-precision runs are not bit-portable across hosts (Supplementary Sec. S-VI:
the double-precision rows of the diagnostic block reproduced bit for bit on a second host, the
single-precision rows did not).

## Layout

| Path | Contents |
|---|---|
| `src/qubo/` | The penalty-encoded QUBO (`portfolio.py`), the constraint terms, and the scoring rules (`scoring.py`: offset-normalized gap g_off, feasible-range gap g_F, P_F, P_τ, Q_τ(S), D_cond) |
| `src/solvers/` | `braket_solver.py` — QAOA through PennyLane on lightning.qubit / lightning.gpu / Braket local / SV1 / IonQ, with the billing guards (a QPU is accepted only for the final sampling task, never as the optimisation backend; see `cdk/README.md` for the account-level prerequisites); `classical_solver.py` — exact C(N, K) enumeration and simulated annealing; `braket_results.py` — reads Braket task records, including IonQ's nested schema |
| `src/finance/` | Input loading — the derived statistics, or a local price snapshot when one is present (`data_loader.py`) — and the portfolio metrics |
| `data/derived/` | The annualised mean-return vector and covariance matrix of the fifty-ticker universe for each window (2023-08-06 → 2026-08-05 for E0–E4, 2020-08-05 → 2023-08-05 for E5), written with 17 significant digits and read back with round-trip parsing; every instance is a bit-exact slice (`derive_input_stats.py`, `tests/test_derived_stats.py`) |
| `data/braket_results/` | The four IonQ Forte-1 task records in the device's own nested schema, and the SV1 smoke-test record |
| `experiments/paper01_qubo_baseline/` | Every driver and aggregation script (below) and `results/`, the CSVs the paper reads |
| `notebooks/` | `generate_tqe_figures.py` (the paper's figures; `PAPER_FIGURES=1` writes the print renders to `notebooks/figures/paper/`) and `generate_figures.py` (the single-instance figures of the Supplementary Material) |
| `papers/` | The manuscript sources, `md_to_latex.py`, `validate_latex.py`, `check_tables.py`, `check_claims.py`, and the literature census (`audit_census.csv`, tallied by `audit_tally.py`) |
| `cdk/` | The AWS CDK stacks: the Braket results bucket and execution role, and the Budgets alarms deployed before any metered task |
| `tests/` | The test suite |

## Result files by experiment

All under `experiments/paper01_qubo_baseline/results/`. "One row per run" files carry the
initial and final angles, evaluation counts, shot counts, the exact-state scores and wall-clock;
on metered backends also the task identifiers (account id masked).

| Experiment (paper section) | Driver | Files |
|---|---|---|
| E0 — re-scoring the single-instance runs under both gap definitions (Sec. V-A, Fig. 1) | `run_e0_rescore.py` | `paper01_results*.csv`, `paper01_convergence*.csv`, `paper01_seed_sweep*.csv`, `paper01_depth_sweep.csv`, `paper01_steps_depth_grid.csv`, `paper01_instances*.csv`, `paper01_degeneracy*.csv`, `paper01_penalty*.csv`, `paper01_normalization_sweep.csv`, `paper01_sa_on_qubo.csv`, `paper01_coefficient_scale*.csv` (produced by `run_experiment.py`, `run_sweep.py`, `run_instances.py`, `run_degeneracy.py`, `run_penalty_sweep.py`, `run_normalization_sweep.py`, `run_sa_on_qubo.py`, `coefficient_scale.py`) |
| E1 — the penalty-weight cross, 2,565 runs (Sec. V-B, Fig. 2, Table III) | `run_e1_cross.py`, aggregated by `analyze_e1.py` | `paper01_e1_cross.csv` (one row per run), `paper01_e1_reference.csv` (one row per instance: reference optimum, A_crit, A_margin, uniform-F baselines), `paper01_e1_manifest.csv`, `paper01_e1_summary.csv`, `paper01_e1_contrasts.csv`, `paper01_e1_decomposition.csv`, `paper01_e1_thresholds.csv`, `paper01_e1_sensitivity.csv`, `paper01_e1_feasibility_split.csv`, `paper01_e1_initial_final.csv`, `paper01_e1_optimum_hits.csv`, `paper01_e1_acquisition_factors.csv`; `*_pilot.csv` is the pilot block |
| E2 — best-of-S against the uniform-F sampler, and the conditional distance D_cond (Sec. V-B, Fig. 3, Table IV) | `analyze_e1.py` (shots), `analyze_conditional.py` (D_cond, from the per-run E1 artifacts) | `paper01_e1_shots.csv`, `paper01_e1_conditional_tv.csv`, `paper01_e1_conditional_tv_summary.csv` |
| E3 — optimizer controls and the precision diagnostic (Sec. V-C, Table VI) | `run_e1_cross.py --optimizer adam --stepsize 0.03 --steps 100 --results-tag e3_adam003`, `run_e1_cross.py --optimizer lbfgs --steps 100 --results-tag e3_lbfgs`, aggregated by `analyze_e3.py`; `run_lbfgs_termination.py` for the diagnostic | `paper01_e1_*_e3_adam003.csv`, `paper01_e1_*_e3_lbfgs.csv`, `paper01_e3_summary.csv`, `paper01_e3_contrasts.csv`, `paper01_e3_multistart.csv`, `paper01_e3_lbfgs_termination_n12_16.csv` (+ `_mac.csv`, the second-host re-run), `paper01_e3_lbfgs_termination_n20.csv`, `paper01_e3_precision_pairs.csv` |
| E4-A — fixed angles on four backends (Sec. V-D, Fig. 4, Table VII) | `run_e4_fixed_angles.py --arm gpu_double\|gpu_single\|cpu_double\|sv1`, aggregated by `analyze_e4.py` | `paper01_e4_states_e4a.csv`, `paper01_e4_batches_e4a.csv` (+ `_pilot`), `paper01_e4_states_compare.csv`, `paper01_e4_batches_summary.csv`, `paper01_e4_cross_arm.csv`, `paper01_e4_billing.csv` |
| E5 — the second market window (Sec. V-B, Table V) | `run_e1_cross.py --end-date 2023-08-05 --results-tag e5`, `analyze_e5.py` | `paper01_e1_*_e5.csv`, `paper01_e5_period_compare.csv` |
| Hardware and the earlier SV1 series (Sec. V-D, Supplementary Secs. S-VII–S-IX; the parameter-shift fallback of Sec. VI-B is reproduced by `repro_parameter_shift_fallback.py` and the SV1 bill by `sv1_task_ledger.py`) | `run_experiment.py --backend braket_sv1 ...`, `recover_task.py`, `sv1_task_ledger.py`, `repro_parameter_shift_fallback.py` | `paper01_results_sv1*.csv`, `paper01_convergence_sv1*.csv`, `paper01_seed_sweep_sv1.csv`, `paper01_sweep_convergence_sv1.csv`, `paper01_sv1_replicates.csv`, `paper01_results_gpu64.csv`, `paper01_convergence_gpu64.csv`, `paper01_results_ionq16.csv`, `paper01_hardware_ionq.csv`, `paper01_sv1_task_ledger.csv` |
| Literature census (Sec. II, Supplementary Sec. S-I) | `papers/audit_tally.py` | `papers/audit_census.csv` |

The per-run E1 artifacts (`results/e1_runs*/*.npz`: the exact probability of every feasible
bitstring under the initial and the final state, per-cardinality mass, angles, trajectory and shot
counts; ≈100 MB compressed, 2,565 files for the main block) are not in the git repository. They
are deposited in the Zenodo archive of release v1.0 (`https://doi.org/10.5281/zenodo.22743582`,
the DOI the paper's availability statement names), and are needed
only to recompute the E2 columns and D_cond, which the committed CSVs already hold. Any full
state vector is reconstructed from the recorded angles and seed.

## Checking the manuscript against the files

```bash
cd papers
python md_to_latex.py && python md_to_latex.py --supplement   # Markdown → IEEEtran LaTeX
python validate_latex.py && python validate_latex.py paper01_supplement.tex
python check_tables.py     # every table cell of both documents against results/
python check_claims.py     # unscoped claims, stated identities, abstract length, figure renders
```

## Licence and data sources

The repository holds three kinds of material under three sets of terms; `LICENSE` covers the
first only.

| Material | Where | Terms |
|---|---|---|
| Source code (QUBO, solvers, scoring, drivers, aggregations, figure and table scripts, tests, CDK) | `src/`, `experiments/`, `notebooks/`, `papers/*.py`, `tests/`, `cdk/` | MIT (`LICENSE`); each file carries an SPDX header |
| Author-generated research outputs (result CSVs, the derived input statistics, the literature census, the archived Braket task records, figure renders) | `experiments/paper01_qubo_baseline/results/`, `data/derived/`, `papers/audit_census.csv`, `data/braket_results/`, `notebooks/figures/` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see `experiments/paper01_qubo_baseline/results/README.md` |
| Market-data snapshots | not distributed | Third-party data (Yahoo Finance via `yfinance`); the statistics above were computed from them — see `data/README.md` |

The price snapshots were obtained through `yfinance` from Yahoo Finance for research use and
are not redistributed; the package ships the derived statistics computed from them, which is all
the experiments read. Users re-fetching the series are responsible for the original provider's
terms. The manuscript itself is not covered by the MIT
licence: the submitted manuscript (the PDFs in `papers/` and in the Zenodo deposit) is the author's
preprint under CC BY 4.0, and the journal version is under the publisher's terms. Third-party
libraries (PennyLane, SciPy, the Braket SDK, ...) are declared as
dependencies and not copied here.

To cite, see `CITATION.cff` (release `v1.0`, DOI `10.5281/zenodo.22743582`, the same identifiers
as the paper's availability statement).
