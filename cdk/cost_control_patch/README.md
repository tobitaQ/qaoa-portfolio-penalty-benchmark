# AWS cost-control solution for Braket — as deployed for the system paper (2026-09-17)

Upstream: https://github.com/aws-samples/cost-control-for-amazon-braket (MIT-0), commit `19eaad6`
("Migrate Powertools for Lambda to v3"). Deployed once, measured, and destroyed the same day;
the measurement is `experiments/system_paper/es2_observe_cost_control.py` and its result files
`paper02_es2_cost_control_*.csv`.

`x86_64-and-numba-cache.patch` holds the three changes made to the sample:

1. Lambda images built for x86_64 instead of arm64 (`FROM public.ecr.aws/lambda/python:3.12-x86_64`,
   `Architecture.X86_64`), because the build host has no arm64 emulation. No behavioural change.
2. `NUMBA_CACHE_DIR=/tmp/numba` on the task-logger Lambda. **Without it the sample, deployed as
   documented, meters nothing**: the Braket SDK imports its default simulator, whose numba-compiled
   functions try to cache beside the source under `/var/task`, which is read-only in Lambda; the
   import fails (`RuntimeError: cannot cache function ... no locator available`), every invocation
   errors (392 of 392 in the first run), the cost alarms stay `INSUFFICIENT_DATA`, and no denial is
   ever attached — while the solution's own "Lambda Invocation Task Logger" alarm does go to ALARM
   four minutes in. 400 tasks ($1.50 at list price) passed through in that state.

3. For the five repetitions only: the monthly cost alarm's evaluation window shortened from 24 h
   (1,440 datapoints) to 5 min, so that a cleared running-cost table lets the alarm return to OK
   between runs. `datapoints_to_alarm` stays 1, so the latency from crossing to ALARM is unchanged.
   Between runs the enforcement policy was detached by hand (the sample detaches only on OK) and the
   `braket-cost-control-summary` table emptied. Runs: `paper02_es2_cost_control_*_run{1..5}.csv`,
   aggregated in `paper02_es2_cost_control_repetitions*.csv`.

`context_used.json` is the `cdk.json` context used (one region, a $0.30 monthly limit, the deny
targeted at the project's IAM user), with the account id and e-mail removed.

Deploy (needs Docker, the CDK CLI via `npx aws-cdk@2`, and `aws-cdk-lib==2.198.0 constructs==10.4.2
cdk-nag==2.35.105` in a venv): apply the patch, fill in the context, `cdk deploy --all`. Destroy in
dependency order: the control and events stacks first, the central event bus last (the bus cannot be
deleted while rules target it). Detach the enforcement policy from the controlled identities before
destroying, or the managed policy's deletion fails.
