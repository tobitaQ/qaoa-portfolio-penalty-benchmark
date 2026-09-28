# Changelog

This package is the reproduction artefact of two papers. Each release is a frozen version with its
own DOI, and each paper names the version it was evaluated against.

## v1.1 — unreleased

Adds the artefacts of the second paper (the experiment platform and its evaluation), the two fixes
the first release's readers ran into, and three the second paper's evaluation and a review of the
drivers against its claims produced.

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
- `src/record_checks.py`: the six verification relations that need nothing but the run manifest
  and the reference table. The E-S1 harness and the aggregation both import them, so the gate and
  the checker are one implementation.
- **Fix:** `analyze_e1` runs those six before it reads a number and refuses a record that fails one,
  naming the check (`--allow-inconsistent` overrides). It previously checked the job's exit status
  and not the record, so an edited cell reached a published median without a word.
- **Fix:** `--results-tag ""` reached the canonical file names. The empty string is falsy where the
  suffix is built and not-None where the overwrite guard looks, so an empty shell variable replaced
  the committed baseline. Both drivers now pass the tag through one `checked_tag()` before a path
  exists.
- **Fix:** `--resume` compares the conditions the manifest was written under — spec version,
  backend, precision, depth, steps, optimizer, shots, window — with the ones this invocation would
  write, and refuses before skipping a cell or touching a file. It previously matched on
  (N, instance, penalty, seed) alone, so a resumed run with different steps left two experiments in
  one file and exited 0.
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
