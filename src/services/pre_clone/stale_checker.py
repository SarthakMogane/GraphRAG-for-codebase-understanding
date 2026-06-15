"""
app/services/pre_clone/stale_checker.py
─────────────────────────────────────────
Staleness detection for already-indexed repositories.

When a repo exists in our index (status=READY), we check whether new
commits have arrived since the last ingestion. If so, we route to REFRESH
instead of SERVE_CACHE.

Staleness sources:
  1. Webhook-detected (most common): GitHub tells us about new pushes/PRs
     via the webhook receiver in routes.py. The repo's status is set to
     STALE immediately when the webhook fires. By the time the user
     re-requests the repo, it's already flagged.

  2. Poll-detected (fallback): For repos whose webhooks may have been
     missed (network issues, app reinstall, etc.), we compare the stored
     SHA against the live HEAD on every validation run.

This module handles case #2 — the poll detection path.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import IngestionJob, Repository, RepoStatus
from src.services.pre_clone.types import StaleCheckResult

logger = logging.getLogger(__name__)


async def check_staleness(
    repo: Repository,
    installation_id:int
) -> StaleCheckResult:
    """
    Compare our stored HEAD SHA against the live HEAD on GitHub.

    Only called when:
      - The repo exists in DB
      - Its status is READY (not already STALE or PENDING)
      - The user submitted it again (either directly or via webhook)

    Returns a StaleCheckResult. The caller uses .is_stale to decide
    whether to route to REFRESH or SERVE_CACHE.
    """
    stored_sha = repo["index_sha"]
    last_indexed_at = repo["last_indexed_at"]

    if not stored_sha:
        # No stored SHA → treat as stale (re-ingest)
        return StaleCheckResult(
            is_stale=True,
            stored_sha=None,
            live_sha=None,
            commits_since_last_index=0,
            last_indexed_at=last_indexed_at,
        )

    # Fetch the live HEAD SHA for the default branch
    live_sha = await _get_live_head_sha(
        owner=repo["owner_login"],
        repo=repo["repo_name"],
        branch=repo["default_branch"],
        installation_id=installation_id

    )

    if live_sha is None:
        # If we can't fetch the live SHA (transient error), treat as NOT stale
        # to avoid unnecessary re-ingestions. The webhook will catch real changes.
        logger.warning(
            "Could not fetch live HEAD SHA for %s/%s — treating as fresh",
            repo["owner_login"], repo["repo_name"]
        )
        return StaleCheckResult(
            is_stale=False,
            stored_sha=stored_sha,
            live_sha=None,
            last_indexed_at=last_indexed_at,
        )

    if live_sha == stored_sha:
        logger.info(
            "%s/%s is up to date (SHA: %s)",
            repo["owner_login"], repo["repo_name"], stored_sha[:8],
        )
        return StaleCheckResult(
            is_stale=False,
            stored_sha=stored_sha,
            live_sha=live_sha,
            commits_since_last_index=0,
            last_indexed_at=last_indexed_at,
        )

    # SHAs differ — count how many commits landed since last index
    commit_count = await _count_commits_since(
        owner=repo["owner_login"],
        repo=repo["repo_name"],
        since_sha=stored_sha,
        branch=repo["default_branch"],
        installation_id=installation_id,
    )

    logger.info(
        "%s/%s is STALE: stored=%s live=%s (%d new commits)",
        repo["owner_login"], repo["repo_name"],
        stored_sha[:8], live_sha[:8], commit_count,
    )

    return StaleCheckResult(
        is_stale=True,
        stored_sha=stored_sha,
        live_sha=live_sha,
        commits_since_last_index=commit_count,
        last_indexed_at=last_indexed_at,
    )


async def check_already_processing(
    repo: dict,
    conn,
) -> Optional[str]:
    """
    Check if a repo already has an active (running/queued) ingestion job.
    Returns the job status if blocked, None if safe to proceed.

    This prevents duplicate concurrent ingestion jobs for the same repo.
    """


    active_job = await conn.fetchrow(
            """
             SELECT id,status
             FROM ingestion_jobs
             WHERE repo_id = $1
                AND status IN ('queued','runing')
            ORDER BY created_at DESC 
            LIMIT 1
            """,
            repo[id]
            
            )
    
    if active_job:
        logger.info(
            "%s/%s already has an active job (id=%d, status=%s) — blocking re-queue",
            repo["owner_login"], 
            repo["repo_name"], 
            active_job["id"], 
            active_job["status"],
        )
        return active_job["status"]

    return None


#---- helpers-----
async def _get_live_head_sha(
    gh,owner: str, repo: str, branch: str,installation_id:int
) -> Optional[str]:
    """
    Fetch the current HEAD commit SHA for a branch.
    Uses the refs API — very lightweight (no commit data, just the SHA pointer).
    """
    try:
        resp = await gh.get_live_head_sha(
            owner,repo,branch,installation_id
        )
        
        return resp
    except httpx.HTTPError:
        return None


async def _count_commits_since(
    gh,
    owner: str,
    repo: str,
    since_sha: str,
    branch: str,
    installation_id: int,
    max_count: int = 200,
) -> int:
    """
    Count commits on the default branch since our last indexed SHA.

    Uses the compare API: GET /repos/{owner}/{repo}/compare/{base}...{head}
    Base = our stored SHA, Head = current branch tip.

    Capped at max_count to avoid large API responses.
    Returns the actual count or max_count+ if more than max_count commits exist.
    """
    
    try:
        resp = await gh.get_commit_count(
            owner,
            repo,
            since_sha,branch,
            installation_id,
            max_count = max_count,
            params = {"per_page": 1}
            
            ) 
        return resp
    except httpx.HTTPError:
        return 0