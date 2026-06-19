"""
app/api/selection.py
──────────────────────
Phase 1 + Phase 2 API endpoints.

Endpoints:
  POST /repos/{id}/scout          — run DeepScout, return RepoScoutResult
  GET  /repos/{id}/scout          — get cached scout result (instant if cached)
  POST /repos/{id}/select         — submit user checklist selections
  GET  /repos/{id}/select         — get current saved selections
  POST /repos/{id}/select/confirm — start ingestion with saved selections

Called after repos.py's /index endpoint returns {next: "scout"}.
The frontend navigates to the scout page and calls POST /scout.
"""

import dataclasses
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from src.core.config import get_settings
from src.core.database import get_db
from src.models.database import (
    IngestionJob, JobType, Repository,
    RepoScoutResult, RepoStatus, UserSelection,
)
from src.schemas.requests import SelectionPayload
from src.schemas.responses import SelectionResponse
from src.services.scout import DeepScout, RepoScoutResult as ScoutResult
from src.services.github import GitHubService, InstallationCache
from src.workers.ingestion_task import run_ingestion
from src.utils.services_helpers import get_github_service

from  src.core.database import get_authed_read_db_dep

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()

# ─────────────────────────────────────────────────────────────────────────────
# POST /repos/{id}/scout — Phase 1
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/repos/{repo_id}/scout")
async def run_scout(
    repo_id: int,
    conn = Depends(get_authed_read_db_dep),
    gh: GitHubService = Depends(get_github_service)
):
    """
    Trigger Phase 1 Deep Scout for a repository.

    Cached by HEAD SHA — if the repo hasn't changed since last scout,
    returns the cached result instantly (0 GitHub API calls).

    Expected latency on cache miss: 2–6 seconds.
    Expected latency on cache hit:  < 50ms.
    """
    
    repo = await conn.fetchcone(
            """ SELECT   id, full_name, owner_login, repo_name, default_branch,
                         index_status, last_scout_sha, installation_id, account_id
                FROM repos 
                WHERE id = $1""",repo_id
            )
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    if repo["index_status"]== "scouting":
        raise HTTPException(status_code=409, detail="Scout already in progress")

    installation_id = repo["installation_id"]
    if not installation_id:
        raise HTTPException(
            status_code=400,
            detail="Repository verification halted: No GitHub App installation context bound.",
        )

    # ── Check cache: if HEAD SHA unchanged, serve cached result instantly ─────
    # Uses GitHubService method — no raw httpx, no _headers() call
    live_sha = await gh.get_live_head_sha(
        repo["owner_login"], repo["repo_name"],
        repo["default_branch"], installation_id,     #update : we are checking sha just so its not stale so if stale then we need to scout for thta perticular part only . maybe check is stale.? 
    )

    if live_sha and live_sha == repo["last_scout_sha"]:
        cached_json = await conn(
            """
            SELECT scout_json 
            FROM repo_scout_results
            WHERE repo_id = $1 AND head_sha = $2 
            LIMIT 1
            """,repo_id ,live_sha

        )
        if cached_json:
            logger.info(
                "Scout cache hit: %s SHA=%s", repo["full_name"], live_sha[:8]
            )
            return {
                "cached": True,
                "head_sha": live_sha,
                "scout": json.loads(cached_json),
            }

    # ── Cache miss — run the scout ─────────────────────────────────────────────
    await conn.execute("UPDATE repos SET index_status = 'scouting', updated_at = NOW() WHERE id = $1", repo_id)

    