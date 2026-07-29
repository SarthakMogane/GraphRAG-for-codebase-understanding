#!/usr/bin/env python3
import os

import aws_cdk as cdk
from graph_rag_infra.infra_stack import InfraStack


app = cdk.App()

stage = app.node.try_get_context("stage") or "dev"

# 2. Senior Pattern: Explicitly map environments based on the target stage
if stage == "prod":
    # Production must ALWAYS be locked down to a specific, immutable target
    target_account = "111122223333"  # Replace with your actual Production Account ID
    target_region = "us-east-1"      # Replace with your actual Production Region
else:
    # Dev/Local falls back to whoever is currently logged into the AWS CLI tool
    target_account = os.getenv("CDK_DEFAULT_ACCOUNT")
    target_region = os.getenv("CDK_DEFAULT_REGION")

InfraStack(app, f"GraphRAGInfraStack-{stage}",
    env=cdk.Environment(
        account=target_account,
        region=target_region
    )
)

app.synth()
