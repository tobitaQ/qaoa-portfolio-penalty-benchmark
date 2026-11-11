# Changelog

This package is the reproduction artefact of two papers. Each deposited release is a frozen version
with its own DOI, and each paper names the version it was evaluated against. The first paper names v1.2, the version
submitted to IEEE Access; the second reconstructs v1.2 and ships v1.3.

## v1.3 — 2026-11-10 — tag; deposited on acceptance of the system paper

The release the system paper ships. The evaluation of that paper at three reconstruction levels was
made on v1.2, which this tag leaves intact; v1.3 adds what the paper's second round of measurements
produced and one change to the aggregation gate, and changes nothing else the evaluations read.

- `experiments/system_paper/results/`: the E-S1 fault injection and mutation streams repeated on
  two sibling records (`*_e5*`, `*_e3_lbfgs*`), the second instance pair of the re-run injection
  (`paper02_es1_input_rerun_pair23.csv` and the `paper01_e1_*_p23*` arms), and the clause-source
  replays (`paper02_es1_clause_sources*.csv`) that place every `identities` finding by the file its
  clause read; `paper02_es1_tiers*.csv` is now counted by clause.
- `experiments/system_paper/es1_clause_sources.py`: the replay driver; `--record` on the five E-S1
  scripts, `--ladder` on the re-run injection.
- **Fix:** the aggregation gate (`src/record_checks.py`, `analyze_e1`) runs `input-digest` whenever the
  pinned statistics for the record's window are on disk, so a record re-run from a revised input is
  refused before a number is read; the block-level correlation clause of `state-batch`, which has no
  stated false-alarm rate, is reported as a warning instead of refusing. The E-S1 harness counts both
  clauses as one check, as before. Three tests.
- `papers/system_paper/system_paper_supplement.md` and `md_to_acm.py --supplement`: the supplement.

## v1.2 — 2026-11-07 — 10.5281/zenodo.24032941

The reproduction package of the first paper as submitted to IEEE Access, after IEEE Transactions on
Quantum Engineering declined the v1.0 version without review. The paper is rebuilt around the
decomposition it already measured — acquisition probability is feasible mass times the preference
inside the feasible set — and one new experiment establishes the second factor by changing the
encoding rather than the weight. No number of the earlier blocks changed: the penalty ladder is a
re-analysis of the committed E1 record, and `papers/check_tables.py` now checks 2,641 cells.

- `experiments/paper01_qubo_baseline/run_xy_mixer.py`: the E6 driver. `--mixer xy_ring` prepares the
  Dicke state |D^N_K⟩ and mixes with a ring of XX + YY terms at A = 0, so the feasible set is
  invariant under the whole ansatz; `--mixer transverse` is the control that takes the penalty out
  of the cost and keeps the published ansatz. 450 runs each, on the same 30 instances × 5 seeds ×
  3 sizes as E1, through the same status file, manifest, reference table and evaluator.
  Results: `paper01_xy_cross.csv` (900 rows), `paper01_xy_reference.csv`, and
  `results/xy_runs/*.npz` (900 files, ≈34 MB, in the Zenodo deposit and not in git).
- `experiments/paper01_qubo_baseline/analyze_ladder.py`: the two factors of P_τ / P_τ^{uniform-F}
  per (N, penalty setting), ordered by the objective's share of the cost Hamiltonian ε_F = Δ_F / s_H
  — `paper01_e1_penalty_ladder.csv` (25 rows: the six weights of the cross at three sizes, the
  A = 0 cell the nine N = 12 instances with A_crit = 0 contribute, then the two E6 blocks). Each run is scored against its own instance's uniform-feasible floor; a test makes
  the two instances' floors differ and checks that swapping them changes the answer.
- `src/solvers/braket_solver.py`: `mixer` and `n_select` parameters, `_dicke_state()`,
  `_prepare_initial_state()` and `_apply_mixer()`. The published path is unchanged, and
  `tests/test_xy_mixer.py` pins both that and the property the block rests on — no amplitude leaves
  the feasible set at any angles, for p = 1, 2, 3 — rather than the numbers.
- `notebooks/generate_tqe_figures.py`: `fig_ladder_preference`, the preference against ε_F with the
  share of runs above the floor, the E6 block as a filled star and its control as a hollow one.
- `papers/md_to_latex.py --access` and `papers/paper01_access.tex`: the IEEE Access build (20 pages).
  The IEEEtran and TQE builds stay.
- `papers/check_style.py` and `tests/test_manuscript_layout.py`: the manuscript invariants a
  re-wrap can break without a visible diff — every table row carries its header's cell count (a
  re-flow had flattened two tables into one line each, which the cell count caught as 2,592 instead
  of 2,614), no line ends with a hyphen (`textwrap` breaks there by default, and re-joining prints
  "double- precision"), and a separator row holds nothing but separators.

- The title drops the acronym, because IEEE asks for titles without abbreviations:
  *Feasibility Without Preference: What the Quantum Approximate Optimization Algorithm
  Contributes at a Dominant Penalty Weight*. `README.md`, `CITATION.cff` and the
  supplement's own heading had kept the title the v1.0 submission carried and now follow.
- The manuscript went through a professional language edit of the abstract, introduction and
  conclusion, and the same treatment by hand for the other sections: one dialect (American),
  shorter sentences, and no elliptical possessives.

## v1.1 — not deposited; its contents are part of v1.2

Tagged 2026-09-30, while the numerical study was still under review at IEEE Transactions on
Quantum Engineering. That submission was declined without review and the study was revised for
IEEE Access, so this release was never archived on its own: v1.2 is the first deposit that carries
the material below, and it is the release the system paper ships.

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
- **Fix:** the aggregation compares the status record with the manifest beside it before it trusts
  either: the row count against the rows the record says were written, a run id that occurs twice,
  a last run the file does not hold. It previously read `done == total` out of the record alone, so
  `EXIT 0, 30/30` beside a manifest with one row deleted printed both numbers and published the
  medians of twenty-nine. The comparison is exact for a job that ended normally and allows the one
  row a kill can leave between the write and the heartbeat, and `--allow-partial` does not waive it.
- **Fix:** the job status record counts the runs that failed, and `describe()` prints the count, so
  that `EXIT 0` is read as "every planned run was attempted and recorded" rather than "every run
  succeeded". `--resume` fills the gaps the manifest does not hold and does not retry a recorded
  failure; its help says so.
- **Fix:** the public export force-adds every selected file and then refuses to commit unless what
  it tracks is what it selected. The device-availability log this project's system paper cites is
  tracked here and matched by the package's own `*.log` rule, so three exports dropped it while
  printing one count for the files selected and a smaller one for the files committed. The
  identifier scan moves onto the same list, since a file the ignore rules swallow was also a file
  nothing scanned.
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

The reproduction package of "Feasibility Without Preference: What the Quantum Approximate Optimization Algorithm Contributes at a Dominant Penalty Weight", as submitted on that date to IEEE Transactions on Quantum Engineering, which
declined it without review on 2026-10-18: the platform, the
drivers, the committed result files, the manuscript and its supplement, and `papers/check_tables.py`,
which checks 2,578 manuscript cells against those files.
