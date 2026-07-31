#!/usr/bin/env python3
import os
import aws_cdk as cdk
from graphrag_infra.infra_stack import GraphRagInfrastructureStack

app = cdk.App()

# 1. Capture the target stage (defaults to 'dev')
stage = app.node.try_get_context("stage") or "dev"

# 2. Senior Pattern: Explicitly map target destination networks
if stage == "prod":
    # Production is securely locked down to an unchangeable target
    target_account = "111122223333"  # Replace with your production AWS Account ID
    target_region = "us-east-1"      # Replace with your production AWS Region
else:
    # Local sandboxes fall back to the active AWS terminal session automatically
    target_account = os.getenv("CDK_DEFAULT_ACCOUNT")
    target_region = os.getenv("CDK_DEFAULT_REGION")
    print(f"target_account{target_account} and region {target_region}")

# 3. Instantiate the refactored modular stack layout
GraphRagInfrastructureStack(
    app, 
    f"GraphRagInfrastructureStack-{stage}",
    env=cdk.Environment(
        account=target_account,
        region=target_region
    ),
    env_modifier=stage # Injects 'dev' or 'prod' into the naming convention of constructs
)

app.synth()
