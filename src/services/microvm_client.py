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
from typing import AsyncIterator, Optional, NamedTuple
from uuid import UUID
from dataclasses import dataclass
import aioboto3
from botocore.exceptions import ClientError
import httpx
from tenacity import (
    retry,
    wait_exponential,
    retry_if_exception,
    stop_after_attempt
)
from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Error taxonomy — every code from CommonErrors.html, classified once
# ─────────────────────────────────────────────────────────────────────────────
 
# Transient — retrying the same request may succeed.
_RETRYABLE_CODES = {
    "InternalFailure",          # 500 — AWS-side issue, try again later
    "ServiceUnavailable",       # 503 — temporary, try again later
    "RequestTimeoutException",  # 408 — server didn't receive request in time
    "ThrottlingException",      # 400 — rate limited; SDK auto-retries some
    "ServiceQuotaExceededException" # of this already, but we add a layer too
                                 # since default retry config isn't guaranteed
}
 
# Permanent — retrying the identical request will fail identically.
# Each needs either a config fix (IAM policy, credentials) or a code fix
# (payload too large, malformed request) — never a blind retry.
_PERMANENT_CODES = {
    "AccessDeniedException",           # IAM policy missing the action
    "ExpiredTokenException",           # credentials need refreshing, not retry
    "IncompleteSignature",             # SDK/signing bug
    "MalformedHttpRequestException",   # our request body is wrong
    "NotAuthorized",
    "OptInRequired",                   # account not enrolled in the service
    "RequestEntityTooLargeException",  # runHookPayload exceeds the 16KB limit
    "RequestAbortedException",         # we closed the connection ourselves
    "UnknownOperationException",       # calling an action that doesn't exist
    "UnrecognizedClientException",     # bad credentials
    "ValidationError",                 # bad parameter shape/value
    "ResourceNotFoundException",       # Snapshot image identifier missing or failed build state
    "ValidationException"              # Invalid connector ARNs, bad idle policy parameters
}

@dataclass
class LaunchResult(NamedTuple):
    microvm_id: str
    endpoint: str
    start: int
    end: int
    state: str
    state_reason: str

class MicroVMError(Exception):
    """
    Raised on any MicroVM lifecycle or communication failure.
    .retryable tells the caller whether this is worth retrying at the
    job level (TransientJobFailure) or not (PermanentJobFailure) —
    mirrors the classification already established in ingestion_worker.py.
    """
    def __init__(self,message:str ,code:str = "unknown",retryable:bool = False):
        super().__init__(message)
        self.code = code 
        self.retryable = retryable

def _classify(exc:ClientError) -> MicroVMError:
    """
    Raised on any MicroVM lifecycle or communication failure.
    .retryable tells the caller whether this is worth retrying at the
    job level (TransientJobFailure) or not (PermanentJobFailure) —
    mirrors the classification already established in ingestion_worker.py.
    """
     
    code = exc.response.get("Error",{}).get("Code","Unknown")
    message = exc.response.get("Error",{}).get("Message",str(exc))
    retryable = code in _RETRYABLE_CODES

    if code not in _RETRYABLE_CODES and code not in _PERMANENT_CODES:
        logger.warning("Unrecognized MicroVM error code: %s — treating as permanent", code)

    return MicroVMError(f"{code}: {message}", code=code, retryable=retryable)

_RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}

def _classify_httpx(exc: httpx.HTTPStatusError) -> MicroVMError:
    status_code = exc.response.status_code
    retryable = status_code in _RETRYABLE_HTTP_STATUSES
    
    code = f"HTTP_{status_code}"
    message = f"Status stream HTTP error: {status_code}"

    # Extract JSON error payload from response if present
    try:
        body = exc.response.json()
        if isinstance(body, dict):
            code = body.get("code") or body.get("error") or code
            message = body.get("message") or body.get("detail") or message
    except Exception:
        pass

    return MicroVMError(message=f"[{code}] {message}", code=str(code), retryable=retryable)

def _should_retry(exc:BaseException) -> bool:
    if isinstance(exc,MicroVMError):
        return exc.retryable
    if isinstance(exc,ClientError):
        code  = exc.response.get("Error",{}).get("Code","Unknown")
        return code in _RETRYABLE_CODES
    return isinstance(exc,(httpx.TimeoutException , httpx.ConnectError))

_retry_transient = retry(
    stop=stop_after_attempt,
    wait=wait_exponential,
    retry=retry_if_exception(_should_retry),
    reraise=True
)                   

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
    @_retry_transient
    async def launch(
        self,
        image_identifier: str,
        run_hook_payload: dict,
        egress_connector_name: list[str],
        ingress_connector_name:list[str],
        execution_role_arn: str,
        image_version:str,
        maximum_duration_seconds: int = 900,
    ) -> LaunchResult:
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

        payload_json = json.dumps(run_hook_payload)
        if len(payload_json.encode()) > 16_000:
            # Fail fast with a clear, specific message rather than letting
            # AWS's RequestEntityTooLargeException surface as an opaque
            # 413 — this is a code-level bug (we're pushing too much into
            # the payload), not something a retry or a platform issue fixes.
            raise MicroVMError(
                f"run_hook_payload is {len(payload_json)} bytes, "
                f"exceeds the 16KB limit ",
                code="PayloadTooLarge",
                retryable=False,
            )
 
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
            except ClientError as e:
                raise _classify(e) from e

        microvm_id = resp["microvmId"]
        endpoint = resp["endpoint"]
        start = resp["startedAt"]
        end = resp["terminatedAt"]
        state = resp["state"]
        state_reason = resp["stateReason"]
        
        logger.info("MicroVM launched: id=%s endpoint=%s", microvm_id, endpoint)
        return LaunchResult(microvm_id, endpoint, start , end , state , state_reason)

    # ─────────────────────────────────────────────────────────────────────────
    # Auth token for talking to the running instance
    # ─────────────────────────────────────────────────────────────────────────
    @_retry_transient
    async def create_auth_token(self, microvm_id: str,expire:int) -> str:
        async with self._session.client("lambda-microvms", region_name=settings.AWS_REGION) as client:
            try:
                resp = await client.create_microvm_auth_token(
                    microvmIdentifier=microvm_id,
                    allowedPorts=[{"allPorts": {}}],
                    expirationInMinutes=expire
                )
            except ClientError as e:
                raise _classify(e) from e
        return resp["authToken"]["X-aws-proxy-auth"]

# Terminate — errors here are logged, never raised
# ─────────────────────────────────────────────────────────────────────────
    @_retry_transient
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

    # ─────────────────────────────────────────────────────────────────────────
    # Status stream 
    # ─────────────────────────────────────────────────────────────────────────
    async def stream_status(
        self,
        endpoint: str,
        auth_token: str,
        timeout_seconds: int = 600,
    ) -> AsyncIterator[dict]:
        url = f"https://{endpoint}/status"
        headers = {"X-aws-proxy-auth": auth_token}

        start_time = asyncio.get_running_loop().time()
        while True:
            elapsed = asyncio.get_running_loop().time() - start_time
            remaining_time = timeout_seconds - elapsed

            if remaining_time <= 0:
                break
            try:
                # Dynamic read timeout tied directly to remaining budget
                timeout_cfg = httpx.Timeout(
                    connect=10.0,
                    read=remaining_time,
                    write=10.0,
                    pool=10.0,
                )
                async with httpx.AsyncClient(timeout=timeout_cfg) as client:

                        async with client.stream("GET", url, headers=headers) as response:
                            response.raise_for_status()
                            async for line in response.aiter_lines():
                                # Ignore SSE comments/pings (e.g., ": ping\n\n")
                                if line.startswith(":") or not line.startswith("data:"):
                                    continue
                                payload = json.loads(line[len("data:"):].strip())
                                yield payload
                                if payload.get("phase") in ("VM_WORK_COMPLETED", "FAILED"):
                                    return

            except (httpx.RemoteProtocolError , httpx.TransportError) as e:
                # Handles unexpected drops, socket resets, and proxy idle hangouts
                elapsed = asyncio.get_running_loop().time() - start_time
                if elapsed > timeout_seconds:
                    break
                logger.warning("Status stream dropped (%s). Retrying connection...", e)
                await asyncio.sleep(1.0)
                continue
            except httpx.TimeoutException as e:
                # Retryable in principle (network blip), but at the job
                # level a hung sandbox after several minutes is treated
                # as permanent by job_consumer — retrying the exact same
                # job against a fresh MicroVM is what "Retry" on the
                # dashboard does, not an automatic SQS redrive.
                raise MicroVMError(
                    f"Status stream timed out after {timeout_seconds}s",
                    code="StatusStreamTimeout",
                    retryable=False,
                ) from e
            except httpx.HTTPStatusError as e:
                # 502, 503, 504 are retryable at the SQS JOB level (fresh VM)
                # 4xx errors are NOT retryable
                raise _classify_httpx(e) from e

        raise MicroVMError(
        f"Status stream timed out after {timeout_seconds}s across reconnects",
        code="StatusStreamTimeout",
        retryable=False,
        )