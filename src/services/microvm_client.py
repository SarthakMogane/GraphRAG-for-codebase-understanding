"""
app/services/microvm_client.py
─────────────────────────────────
Correct AWS Lambda MicroVMs client. Replaces the earlier job_consumer.py
draft's use of lambda_client.invoke_with_response_stream(TenantId=...) —
that's the regular Lambda FUNCTIONS API and does not apply here. Lambda
MicroVMs is a separate resource type with its own service client and
its own interaction model:

    run-microvm            → launch a fresh instance from a pre-built image,
                              returns {microvmId, endpoint}
    create-microvm-auth-token → mint a short-lived (<=60min) token scoped
                              to one microvmId + allowed ports
    (direct HTTPS to the endpoint, X-aws-proxy-auth: <token> header,
     SSE for progress — no invoke() call exists for this resource type)
    terminate-microvm      → explicit release, called by us immediately
                              on completion — not relied on idle-policy,
                              see module docstring in sandbox_app/app.py

boto3 service name inferred from the CLI service name (`aws lambda-microvms
...`) — verify this matches your installed botocore's service model before
first deploy; this is a ~1-month-old service and client naming should be
confirmed against `aioboto3.Session().get_available_services()` rather
than assumed with full confidence.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator, Optional
from uuid import UUID

import aioboto3
import httpx

from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class MicroVMError(Exception):
    """Raised on any unrecoverable MicroVM lifecycle or communication error."""
    pass


class MicroVMClient:
    """
    One instance shared across jobs in the orchestrator process. Holds no
    per-job state — every method takes the microvm_id / job data explicitly,
    matching the pattern already established for GitHubService and the
    other shared clients in this codebase.
    """

    def __init__(self, session: aioboto3.Session):
        self._session = session

    # ─────────────────────────────────────────────────────────────────────────
    # Launch
    # ─────────────────────────────────────────────────────────────────────────

    async def launch(
        self,
        image_identifier: str,
        run_hook_payload: dict,
        egress_connector_name: str,
        ingress_connector_name:str,
        execution_role_arn: str,
        image_version:str,
        maximum_duration_seconds: int = 900,
    ) -> tuple[str, str]:
        """
        Launch a fresh MicroVM from the given image. Returns (microvm_id, endpoint).

        maximum_duration_seconds is a hard platform-level ceiling — kept
        tight (15 min default) as a backstop distinct from and tighter
        than the application-level timeout enforced inside sandbox_app's
        own _run_job (see that module's docstring on why relying solely
        on the platform's much larger 8-hour cap is not acceptable for
        cost/operational reasons).

        idle-policy is set to effectively disable auto-resume — our model
        always explicitly terminates on completion (see terminate() below)
        rather than relying on suspend/resume semantics, since each job is
        one-shot batch work, not an interactive session worth preserving.
        """
        async with self._session.client("lambda-microvms", region_name=settings.AWS_REGION) as client:
            try:
                resp = await client.run_microvm(
                    imageIdentifier=image_identifier,
                    executionRoleArn=execution_role_arn,
                    runHookPayload=json.dumps(run_hook_payload),
                    imageVersion=image_version,
                    maximumDurationInSeconds=maximum_duration_seconds,
                    idlePolicy={
                        "maxIdleDurationSeconds": 60,
                        "suspendedDurationSeconds": 30,
                        "autoResumeEnabled": False,
                    },
                    egressNetworkConnectors=[egress_connector_name],
                    ingressNetworkConnectors=[ingress_connector_name],
                    
                )
            except Exception as e:
                raise MicroVMError(f"run-microvm failed: {e}") from e

        microvm_id = resp["microvmId"]
        endpoint = resp["endpoint"]
        start = resp["startedAt"]
        end = resp["terminatedAt"]
        state = resp["state"]
        state_reason = resp["stateReason"]
        endpoint = resp["endpoint"]
        logger.info("MicroVM launched: id=%s endpoint=%s", microvm_id, endpoint)
        return microvm_id, endpoint,start , end , state , state_reason # doo commit here for refactor(job_consumer) if we do want to save start , end in db . 

    # ─────────────────────────────────────────────────────────────────────────
    # Auth token for talking to the running instance
    # ─────────────────────────────────────────────────────────────────────────

    async def create_auth_token(self, microvm_id: str,expire:int) -> str:
        async with self._session.client("lambda-microvms", region_name=settings.AWS_REGION) as client:
            try:
                resp = await client.create_microvm_auth_token(
                    microvmIdentifier=microvm_id,
                    allowedPorts=[{"allPorts": {}}],
                    expirationInMinutes=expire
                )
            except Exception as e:
                raise MicroVMError(f"create-microvm-auth-token failed: {e}") from e
        return resp["authToken"]["X-aws-proxy-auth"]

# Terminate — errors here are logged, never raised
# ─────────────────────────────────────────────────────────────────────────
 
    async def terminate(self, microvm_id: str) -> None:
        """
        Deliberately swallows errors rather than raising — termination
        failing shouldn't fail an otherwise-successful job, and the
        idle-policy backstop (maxIdleDurationSeconds=60, set in launch())
        reclaims the instance shortly regardless.
        """
        async with self._session.client("lambda-microvms", region_name=settings.AWS_REGION) as client:
            try:
                await client.terminate_microvm(microvmIdentifier=microvm_id)
                logger.info("MicroVM terminated: id=%s", microvm_id)
            except ClientError as e:
                err = _classify(e)
                logger.error(
                    "Failed to terminate MicroVM id=%s: %s (code=%s) — "
                    "idle-policy will reclaim it within 60s regardless",
                    microvm_id, err, err.code,
                ) 