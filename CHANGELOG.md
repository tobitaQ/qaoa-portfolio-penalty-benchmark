# Changelog

This package is the reproduction artefact of two papers. Each release is a frozen version with its
own DOI, and each paper names the version it was evaluated against.

## v1.1 — unreleased

Adds the artefacts of the second paper (the experiment platform and its evaluation) and the two
fixes the first release's readers ran into.

- `experiments/system_paper/`: the three evaluation harnesses (E-S1 fault injection and random
  mutations, E-S2 guard stages and the two service observations, E-S3 reconstruction), the
  cost-control aggregation, and their result files (`paper02_*.csv`, and the IonQ availability log).
- `papers/system_paper/`: the manuscript, its figures, the Markdown-to-acmart converter, and
  `check_tables.py`, which binds every cell of its Tables 3–11 and 13 to a result file and
  regenerates both figures for a byte comparison.
- `cdk/cost_control_patch/`: the three changes needed to deploy the provider's cost-control sample.
- `src/run_status.py` and `scripts/run_detached.sh`: the job status record — `RUNNING` with a
  heartbeat, `EXIT 0`, `FAILED code` — that the aggregation checks before it will run.
- `experiments/a1_reuse/` and `experiments/a3_constraints/`: the two questions answered from the
  committed record alone, with their result files.
- Tests for all of the above.
- **Fix:** every text file is now opened with an explicit encoding. On a system whose locale is not
  UTF-8 — a Japanese Windows, where it is cp932 — fifteen tests failed on the first em dash of the
  manuscript. Nothing a result file holds changes.
- **Fix:** the per-run flush writes beside the file and moves it into place with `os.replace`, and
  the run's line is printed after the row is written rather than before. The previous code rewrote
  the manifest in place, so a process killed inside the rewrite could lose the runs already done.
- **Fix:** the README's setup is given per shell. The single `python -m venv .venv && .
  .venv/bin/activate` line does not work on Windows, where the environment is `Scripts\`.
- The run rows and reference rows of the E1 record gain an `input_digest` column, a content digest
  of the μ and Σ the run read. It was appended to the committed files from the pinned statistics
  without re-running anything; every column of v1.0 is unchanged.

## v1.0 — 2026-09-14 (doi:10.5281/zenodo.22743582)

The reproduction package of "Measuring Metric and Sampling Distortions in Penalty-Encoded QAOA
Portfolio Benchmarks", as submitted to IEEE Transactions on Quantum Engineering: the platform, the
drivers, the committed result files, the manuscript and its supplement, and `papers/check_tables.py`,
which checks 2,578 manuscript cells against those files.
