# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Amazon Braket results bucket and IAM access.

Amazon Braket writes every task's measurement results to an S3 bucket that the
caller names in ``s3_destination_folder`` (see ``BraketSolver`` for the SV1 /
IonQ path). This stack provisions:

    * a private, encrypted results bucket with lifecycle expiry, and
    * an IAM role the experiment runner (an EC2 GPU box, a Lambda, or a
      SageMaker notebook) can assume to submit Braket tasks and read/write that
      bucket — scoped to Braket + this one bucket, not account-wide S3.

Nothing here bills while idle; cost accrues only when a Braket task runs.
"""

from __future__ import annotations

from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from constructs import Construct


class BraketStack(Stack):
    """Provision the Braket results bucket and a scoped execution role.

    Attributes:
        results_bucket: S3 bucket that receives Braket task result folders.
        execution_role: IAM role granting Braket submission + bucket access.
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # --- Results bucket -------------------------------------------------
        # Braket prefixes every task with "amazon-braket-"; naming the bucket the
        # same way keeps it recognisable in the console. Name is region/account
        # qualified via CDK tokens so it stays globally unique.
        self.results_bucket = s3.Bucket(
            self,
            "BraketResultsBucket",
            bucket_name=f"amazon-braket-qfr-{self.account}-{self.region}",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=True,
            # Research outputs are reproducible by re-running; expire raw task
            # results after 90 days to avoid unbounded storage cost.
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="expire-raw-task-results",
                    expiration=Duration.days(90),
                    noncurrent_version_expiration=Duration.days(30),
                    abort_incomplete_multipart_upload_after=Duration.days(7),
                )
            ],
            # A research bucket must not block `cdk destroy`; auto-empty on delete.
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # --- Execution role -------------------------------------------------
        # Assumable by the compute that drives experiments.
        #
        # Braket additionally requires the account-level service-linked role
        # AWSServiceRoleForAmazonBraket. It is NOT created on first use: without
        # it, CreateQuantumTask fails with AccessDeniedException (observed on
        # this account, 2026-08-09). It is also account-wide rather than
        # stack-scoped, so declaring it here would make `cdk deploy` fail on any
        # account that already has it. Create it once, out of band:
        #
        #     aws iam create-service-linked-role --aws-service-name braket.amazonaws.com
        #
        # (allow ~1 min for IAM propagation before the first task).
        self.execution_role = iam.Role(
            self,
            "BraketExecutionRole",
            role_name="qfr-braket-execution-role",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("braket.amazonaws.com"),
                iam.ServicePrincipal("lambda.amazonaws.com"),
                iam.ServicePrincipal("sagemaker.amazonaws.com"),
                iam.ServicePrincipal("ec2.amazonaws.com"),
            ),
            description="Submit Amazon Braket tasks and access the results bucket",
        )

        # Braket task submission/inspection. AmazonBraketFullAccess is the AWS
        # managed policy for the service; it does not grant S3, so we add a
        # bucket-scoped grant separately below (least privilege on data).
        self.execution_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonBraketFullAccess")
        )
        self.results_bucket.grant_read_write(self.execution_role)

        # CloudWatch Logs so task/job diagnostics are retrievable.
        self.execution_role.add_to_policy(
            iam.PolicyStatement(
                sid="BraketLogs",
                effect=iam.Effect.ALLOW,
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams",
                ],
                resources=[
                    f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/braket*"
                ],
            )
        )

        # --- Outputs --------------------------------------------------------
        from aws_cdk import CfnOutput

        CfnOutput(
            self,
            "ResultsBucketName",
            value=self.results_bucket.bucket_name,
            description="Pass this as --s3-bucket to run_experiment.py for SV1/IonQ",
            export_name="QfrBraketResultsBucket",
        )
        CfnOutput(
            self,
            "ExecutionRoleArn",
            value=self.execution_role.role_arn,
            description="Role that experiment compute assumes to run Braket tasks",
            export_name="QfrBraketExecutionRoleArn",
        )
