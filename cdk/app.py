#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""CDK application entry point for the quantum-finance research infrastructure.

Deploys the minimal AWS footprint needed for paper① reproducibility runs:

    BraketStack      S3 results bucket + IAM role/policy for Amazon Braket
    MonitoringStack  AWS Budgets cost ceiling with e-mail alerts

Region
------
Everything is pinned to **us-east-1**. Amazon Braket is not available in
ap-northeast-1; the gate-based devices we target (SV1 simulator, IonQ Forte-1)
live in us-east-1.

Cost policy
-----------
Nothing here bills by existing (empty S3 bucket, an IAM role, a Budget are all
free). Cost is incurred only when a Braket task actually runs. The Budgets
alarm is deployed *first* so the ceiling is in place before any real-hardware
task is submitted.

Usage
-----
    cd cdk
    npx cdk synth        # render CloudFormation, no AWS credentials needed
    npx cdk deploy --all # requires credentials; run only from Step 4 onward
"""

from __future__ import annotations

import os

import aws_cdk as cdk

from stacks.braket_stack import BraketStack
from stacks.monitoring_stack import MonitoringStack

#: Braket-supported region. Do NOT change to ap-northeast-1 (unsupported).
REGION = "us-east-1"

#: Budget ceiling (USD/month). The project targets <=$50 per paper.
MONTHLY_BUDGET_USD = float(os.environ.get("QFR_MONTHLY_BUDGET_USD", "50"))

#: Where Budgets alerts are sent. There is deliberately no default: a default
#: would route another account's alerts to whoever wrote it. ``cdk synth`` and
#: ``cdk deploy`` refuse to run until the deploying party names their own address.
ALERT_EMAIL = os.environ.get("QFR_ALERT_EMAIL")
if not ALERT_EMAIL:
    raise SystemExit("QFR_ALERT_EMAIL is not set: export the e-mail address that should "
                     "receive this account's AWS Budgets alerts before running cdk.")


def main() -> None:
    app = cdk.App()

    # Account is taken from the caller's environment when available; synth works
    # without it (env-agnostic) so the templates can be rendered with no creds.
    env = cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=REGION,
    )

    braket = BraketStack(
        app,
        "QfrBraketStack",
        env=env,
        description="Amazon Braket results bucket and IAM access for paper① experiments",
    )

    MonitoringStack(
        app,
        "QfrMonitoringStack",
        monthly_budget_usd=MONTHLY_BUDGET_USD,
        alert_email=ALERT_EMAIL,
        # Tie the S3-cost view of the budget to the Braket bucket.
        results_bucket_name=braket.results_bucket.bucket_name,
        env=env,
        description="AWS Budgets cost ceiling and alerts for the research account",
    )

    cdk.Tags.of(app).add("project", "quantum-finance-research")
    cdk.Tags.of(app).add("paper", "paper01")

    app.synth()


if __name__ == "__main__":
    main()
