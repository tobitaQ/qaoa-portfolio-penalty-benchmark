# CDK infrastructure (paper① reproducibility)

Minimal, region-pinned AWS footprint for the quantum-finance experiments.
**Region: `us-east-1`** (Amazon Braket is unavailable in ap-northeast-1; SV1 and
IonQ Forte-1 live in us-east-1).

## Stacks

| Stack | Resources | Bills while idle? |
|---|---|---|
| `QfrBraketStack` | S3 results bucket (encrypted, 90-day expiry) + IAM execution role scoped to Braket & that bucket | No |
| `QfrMonitoringStack` | AWS Budgets: `$50/mo` account budget (50/80/100 % actual + 100 % forecast alerts) and a Braket-only budget | No |

Cost is incurred **only when a Braket task actually runs**. The Budgets alarm is
deployed first so the ceiling is in place before any real-hardware task — a
single `n_shots=1000` IonQ run is ~$80 and would blow the paper's <$50 target.

## Prerequisites

```bash
# Python CDK library (into the project .venv is fine)
pip install -r requirements.txt
# CDK CLI comes via npx; no global install needed
npx cdk --version
```

## Validate without AWS (free, no credentials)

```bash
cd cdk
export QFR_ALERT_EMAIL=you@example.com   # required even for synth: the Budgets stack embeds it
npx cdk synth            # renders CloudFormation for all stacks
```

## Deploy (Step 4 onward — requires AWS credentials)

```bash
export CDK_DEFAULT_ACCOUNT=<your-account-id>
export QFR_ALERT_EMAIL=you@example.com     # required: where this account's Budgets alerts go
export QFR_MONTHLY_BUDGET_USD=50           # optional

npx cdk bootstrap aws://$CDK_DEFAULT_ACCOUNT/us-east-1
npx cdk deploy QfrMonitoringStack          # budget FIRST
npx cdk deploy QfrBraketStack

# One-time, account-wide, not part of any stack: Braket's service-linked role.
# Without it CreateQuantumTask returns AccessDeniedException. Allow ~1 min for
# IAM propagation before submitting the first task.
aws iam create-service-linked-role --aws-service-name braket.amazonaws.com

# The bucket name is a stack output; feed it to the experiment runner:
python -m experiments.paper01_qubo_baseline.run_experiment \
    --backend braket_sv1 --s3-bucket amazon-braket-qfr-<account>-us-east-1 \
    --quantum-n 8 12 16 20 --shots 1000
```

Tear down with `npx cdk destroy --all` (the bucket auto-empties on delete).

## Account-level prerequisites that no stack creates

Three preconditions were met only by deploying, and each blocked a task without a test or a
`cdk synth` noticing:

1. **Braket's service-linked role** is not created on first use: `CreateQuantumTask` fails with
   `AccessDeniedException: AWSServiceRoleForAmazonBraket role doesn't exist` until the command
   above has been run once per account (IAM propagation takes about a minute).
2. **The region is pinned in code**, not taken from the ambient profile: `BraketSolver` sends every
   cloud task to `us-east-1` (`BRAKET_REGION`), because a default region without a Braket endpoint
   fails in DNS before any task is created.
3. **Third-party devices (IonQ, QuEra, Rigetti) require a one-time user agreement** that can only be
   accepted in the Braket console; the API returns `AccessDeniedException: User agreement has not
   been accepted` and creates no task. SV1 needs no such step.

A fourth caution is the framework's, not the account's: a shots-based sampling circuit handed
*trainable* angles is differentiated by parameter shift, so one 1,000-shot sampling call on SV1
was billed as 177 tasks instead of one (N = 8, p = 2) until the angles were frozen
(`frozen_angles()`); `experiments/paper01_qubo_baseline/repro_parameter_shift_fallback.py`
reproduces the tape count with no device.
