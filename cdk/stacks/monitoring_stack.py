# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""AWS Budgets cost ceiling and e-mail alerts.

The project targets <=$50 per paper and mandates a Budgets alarm *before* any
real-hardware (IonQ) task is submitted — a single careless ``n_shots=1000`` run
on IonQ is ~$80 and blows the budget in one shot. This stack sets:

    * an account-wide monthly cost budget with alerts at 50 / 80 / 100 % of
      actual spend and a 100 % forecast alert, and
    * a Braket-service budget so quantum spend is visible on its own.

Budgets is a billing-console resource; it is free and must be deployed in
us-east-1. No SNS topic is needed — Budgets can e-mail subscribers directly.
"""

from __future__ import annotations

from aws_cdk import Stack
from aws_cdk import aws_budgets as budgets
from constructs import Construct


def _email_notification(
    threshold: float,
    notification_type: str,
    email: str,
) -> budgets.CfnBudget.NotificationWithSubscribersProperty:
    """Build one Budgets notification that e-mails a single subscriber.

    Args:
        threshold: Percentage of the budgeted amount that triggers the alert.
        notification_type: ``"ACTUAL"`` or ``"FORECASTED"``.
        email: Destination address.

    Returns:
        A notification-with-subscribers property for ``CfnBudget``.
    """
    return budgets.CfnBudget.NotificationWithSubscribersProperty(
        notification=budgets.CfnBudget.NotificationProperty(
            comparison_operator="GREATER_THAN",
            notification_type=notification_type,
            threshold=threshold,
            threshold_type="PERCENTAGE",
        ),
        subscribers=[
            budgets.CfnBudget.SubscriberProperty(
                address=email,
                subscription_type="EMAIL",
            )
        ],
    )


class MonitoringStack(Stack):
    """Provision AWS Budgets alarms for the research account."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        monthly_budget_usd: float,
        alert_email: str,
        results_bucket_name: str | None = None,
        **kwargs,
    ) -> None:
        """Create the cost budgets.

        Args:
            monthly_budget_usd: Overall monthly ceiling in USD.
            alert_email: Address that receives the alerts.
            results_bucket_name: Unused for now; reserved for a future
                S3-scoped budget on the Braket results bucket.
        """
        super().__init__(scope, construct_id, **kwargs)

        notifications = [
            _email_notification(50.0, "ACTUAL", alert_email),
            _email_notification(80.0, "ACTUAL", alert_email),
            _email_notification(100.0, "ACTUAL", alert_email),
            # Forecast alert catches a run that is on track to overshoot before
            # the actual spend has landed.
            _email_notification(100.0, "FORECASTED", alert_email),
        ]

        # --- Overall account budget ----------------------------------------
        budgets.CfnBudget(
            self,
            "MonthlyCostBudget",
            budget=budgets.CfnBudget.BudgetDataProperty(
                budget_name="qfr-monthly-cost",
                budget_type="COST",
                time_unit="MONTHLY",
                budget_limit=budgets.CfnBudget.SpendProperty(
                    amount=monthly_budget_usd,
                    unit="USD",
                ),
            ),
            notifications_with_subscribers=notifications,
        )

        # --- Braket-only budget --------------------------------------------
        # Isolate quantum spend so simulator/hardware cost is legible on its own.
        # A tighter forecast alert (80 %) gives earlier warning on QPU tasks.
        braket_notifications = [
            _email_notification(80.0, "ACTUAL", alert_email),
            _email_notification(100.0, "ACTUAL", alert_email),
            _email_notification(80.0, "FORECASTED", alert_email),
        ]
        budgets.CfnBudget(
            self,
            "BraketServiceBudget",
            budget=budgets.CfnBudget.BudgetDataProperty(
                budget_name="qfr-braket-only",
                budget_type="COST",
                time_unit="MONTHLY",
                budget_limit=budgets.CfnBudget.SpendProperty(
                    amount=monthly_budget_usd,
                    unit="USD",
                ),
                cost_filters={"Service": ["Amazon Braket"]},
            ),
            notifications_with_subscribers=braket_notifications,
        )
