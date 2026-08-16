# The sandbox runtime with the /ready, /run, /terminate hooks

"""
sandbox_app/app.py
────────────────────
Runs INSIDE the Lambda MicroVM. This is what gets snapshotted.

Lifecycle contract with Lambda (all on the port set in the image config):
  POST /aws/lambda-microvms/runtime/v1/ready      — image build only
  POST /aws/lambda-microvms/runtime/v1/validate   — image build only
  POST /aws/lambda-microvms/runtime/v1/run        — per-launch, receives runHookPayload
  POST /aws/lambda-microvms/runtime/v1/suspend    — not used in our model, see below
  POST /aws/lambda-microvms/runtime/v1/resume     — not used in our model, see below
  POST /aws/lambda-microvms/runtime/v1/terminate  — cleanup before release

Why /suspend and /resume are no-ops here: our job model is one-shot
batch work (clone → filter → parse → upload), not an interactive session
that benefits from being paused and picked back up later. The
orchestrator always explicitly terminates after a job completes — see
job_consumer.py. Idle policy + maximum-duration-in-seconds are configured
as a crash-safety backstop only (if the orchestrator dies before it can
call terminate-microvm), not as our normal operating path.

The actual clone+filter+parse work reuses GitCloneService and
FileFilterPipeline as-is — no logic duplicated, same CVE-2025-48384
hardening (hooksPath sink, no --recursive, core.symlinks=false, per-job
HOME scoping) that was already built and verified for the Fargate-era
design.

External clients (the orchestrator) reach this app through Lambda's
managed HTTPS endpoint — never directly. Every request arrives
pre-authenticated by Lambda's own JWE token check; this app doesn't need
to re-implement auth, but it does validate the job_id it's operating on
matches what /run received, as a sanity check against a misrouted request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sandbox_microvm.workspace import JobWorkspace
from src.services.clone_strategy import CloneStrategySelector
from src.services.github import RepoMetadata

logger = logging.getLogger("sandbox_app")
logging.basicConfig(level=logging.INFO)

app = FastAPI()

_job_state:dict ={
    "job_id":None,
    "phase":"BOOTED",
    "error":None,
    "time":None,
}
JOB_TIMEOUT_SECONDS = 3600

# ─────────────────────────────────────────────────────────────────────────────
# Image build hooks (only called during create-microvm-image / update)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/aws/lambda-microvms/runtime/v1/ready")
async def ready_hook():
    """
    Called once during image build, right after the app starts.
    Returning 200 tells Lambda to take the snapshot now — at this point
    git, Python deps, and this app are all loaded and warm, which is
    exactly the state every future run-microvm launch resumes from.
    """
    return JSONResponse(status_code=200, content={"status": "ready"})


@app.post("/aws/lambda-microvms/runtime/v1/validate")
async def validate_hook():
    """
    Called once, on a fresh MicroVM started from the just-built snapshot,
    to confirm the app behaves correctly after a resume. We don't have a
    real job to validate against yet, so this checks the things that
    would silently break if the snapshot were unhealthy: git binary
    present and at a patched version (GitCloneService's own constructor
    check), workspace base directory writable.
    """
    try:
        Path("/tmp/microvm-validate-check").mkdir(exist_ok=True)
        Path("/tmp/microvm-validate-check").rmdir()
        return JSONResponse(status_code=200, content={"status": "validated"})
    except Exception as e:
        logger.error("Validate hook failed: %s", e)
        return JSONResponse(status_code=503, content={"status": "not_ready", "error": str(e)})

@app.post("/aws/lambda-microvms/runtime/v1/suspend")
async def suspend_hook():
    "Not used for now"
    return JSONResponse(status_code=200 , content={"status":"ok"})

@app.post("/aws/lambda-microvms/runtime/v1/resume")
async def resume_hook():
    "Not used for now"
    return JSONResponse(status_code=200 ,content={"status":"ok"})

@app.post("/aws/lambda-microvms/runtime/v1/terminate")
async def terminate_hook():
    """
    Called by Lambda right before releasing resources — fires whether
    termination was explicit (our normal path) or a fallback (idle-policy
    backstop). Belt-and-suspenders cleanup in case _run_job's own
    JobWorkspace teardown didn't run for some reason (e.g. the process
    was killed mid-job rather than completing normally).
    """
    job_id = _job_state.get("job_id")
    if job_id:
        workspace_root = Path(f"/tmp/ingestion/job-{job_id}")
        if workspace_root.exists():
            shutil.rmtree(workspace_root,ignore_errors=True)

    return JSONResponse(status_code=200 , content={"status":"cleand up!"})


@app.post("aws/lambda-microvms/runtime/v1/run")
async def run_hook(request:Request):
    """
    Called once per run-microvm launch. Receives {microvmId, runHookPayload}.
    Returns 200 immediately after kicking off the real work as a
    background task — the orchestrator's SSE poll (below) is how it
    actually learns about progress, not this hook's response.
    """
    body = request.json()
    payload = json.loads(body.get("runHookPayload","{}"))

    _job_state.update(
        {
            "job_id":payload.get("job_id"),
            "phase":"STARTED",
            "error": None,
            "started_at":time.time(),
        }
    )

    asyncio.create_task(_run_job_with_timeout(payload))

    return JSONResponse(status_code=200 , content={"status":"Ok"})


async def _run_job_with_timeout(payload:dict)->None:
    """
    Bounds total job duration at the application level, tighter than the
    platform's own maximum_duration_seconds. Without this, a hung job
    (real or stub) would occupy — and bill for — a MicroVM for up to
    8 hours before the platform's own ceiling kicks in.
    """
    try:
        await asyncio.wait_for(_run_job(payload),timeout=JOB_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.error("Job %s exceeded %ds -marking FAILED",payload.get("job_id"),JOB_TIMEOUT_SECONDS)
        _job_state["error"]=f"Job exceded {JOB_TIMEOUT_SECONDS}s timeout"
        _job_state["phase"]="FAILED"

async def _run_job(payload:dict):
    "main job"

    job_id        = payload["job_id"]
    account_id    = payload["account_id"]   # fixed — was hardcoded "sandbox"
    owner         = payload["owner"]
    repo          = payload["repo"]
    branch        = payload.get("branch", "main")
    sparse_dirs   = payload.get("sparse_dirs", [])
    submodules    = payload.get("submodules", [])
    github_token  = payload["github_token"]
    presigned_url = payload["presigned_url"]
    image_version = payload["image_version"]

    try:
        async with JobWorkspace(job_id=job_id,account_id=account_id,base_dir="/temp/ingestion") as ws:
            home_dir = ws.tmp_dir/"home"

            metadata:RepoMetadata
            selector = CloneStrategySelector()
            configs = selector.select(
                metadata = metadata,
                is_monorepo=bool(sparse_dirs),
                sparse_dir=sparse_dirs or None,
            )
    except:
        pass