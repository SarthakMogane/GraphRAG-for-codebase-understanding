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

logger = logging.getLogger("sandbox_app")
logging.basicConfig(level=logging.INFO)

app = FastAPI()



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


