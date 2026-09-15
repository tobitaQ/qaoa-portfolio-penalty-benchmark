# Archived Amazon Braket task results

Raw device output for the billed runs the paper reports, copied out of the
Braket results bucket. The bucket is created with `auto_delete_objects=True`, so
`cdk destroy` takes the raw output with it, and the working tree is not a backup
either — see "How this was nearly lost" below.

## What is here

| Prefix | Tasks | Device | Why it is kept |
|---|---:|---|---|
| `ionq/` | 4 | IonQ Forte-1 | The hardware evidence of Sec. V-D and Supplementary Sec. S-VIII. $8.30 per task, and Forte-1's calibration moves, so these cannot be reproduced by re-running. |
| `smoketest/` | 1 | SV1 | The Bell-pair connectivity check, the one task of the "Deployment and smoke test" row of Table II and Supplementary Table S33. |

The SV1 optimization-loop tasks (1,299 objects, 159 MB) are **not** here. Every
quantity the paper takes from them is in the committed result CSVs, and the raw
payloads are too large for the repository.

## Reading them

`shot_counts_from_dir` in `src/solvers/braket_results.py` takes a task directory
and returns bitstring → shot count, handling both Braket result schemas:

```python
from src.solvers.braket_results import shot_counts_from_dir
counts = shot_counts_from_dir("data/braket_results/ionq/<task-uuid>")
```

The IonQ tasks are stored in the nested `program_set_task_result` form —
`results.json` → `programs/0/results.json` → `programs/0/executables/0.json` →
`measurementProbabilities` — deliberately, and not flattened. That nesting is
the defect Supplementary Secs. S-VIII and S-IX report: PennyLane-Braket 1.35 does not read it and does not
raise either, so flattening the archive would erase the evidence for the claim
and would stop exercising the branch that handles it.

`tests/test_braket_archive.py` checks that every counted quantity in
`results/paper01_hardware_ionq.csv` — shots, distinct bitstrings, feasible
shots — comes back out of these files.

## Redaction

The AWS account number in task ARNs and S3 locations has been replaced with
`000000000000` throughout, in three fields: `id`, `taskMetadata.id` and
`s3Location`. No measurement, timestamp or device parameter was altered. The
task UUIDs are unchanged and remain the identifiers used everywhere else.

## How this was nearly lost

`.gitignore` carried an unanchored `braket_results/` under "local experiment
logs". It was meant for the transient directory the SDK drops beside whatever it
is run from, but unanchored patterns match at any depth, so it also matched this
one. `git add -A` therefore skipped the whole archive and said nothing — the
commit that claimed to add it contains only code — and the only copy went with
the working tree when it was deleted on 2026-08-14. The tasks were recoverable
only because the S3 bucket had not been torn down.

Two things now prevent a silent repeat. The ignore rule is anchored to the
repository root with an explicit `!data/braket_results/` exception. And
`tests/test_braket_archive.py` *fails* rather than skips when the archive is
absent: skipping is what let a green suite coexist with a missing artifact,
which is the same failure mode Supplementary Sec. S-VIII reports for the plugin that returned
zeros instead of raising.
