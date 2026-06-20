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
import logging
from datetime import datetime, timezone
from typing import Optional
import json
from fastapi import APIRouter, Depends, HTTPException

from src.schemas.requests import SelectionPayload
from src.schemas.responses import SelectionResponse
from src.core.config import get_settings
from src.core.database import get_db
from src.models.database import (
    IngestionJob, JobType, Repository,
    RepoScoutResult, RepoStatus, UserSelection,
)
from src.services.scout import DeepScout, RepoScoutResult as ScoutResult
from src.services.github import GitHubService, InstallationCache
# from src.workers.ingestion_task import run_ingestion
from src.utils.services_helpers import get_github_service ,_serialize_scout

from  src.core.database import get_rls_conn ,get_authed_read_db_dep
from src.crud.repos_ops import _get_indexed_repos

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# POST /repos/{id}/scout — Phase 1
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/repos/{repo_id}/scout")
async def run_scout(
    repo_id: int,
    conn = Depends(get_rls_conn),
    gh: GitHubService = Depends(get_github_service)
):
    """
    Trigger Phase 1 Deep Scout for a repository.

    Cached by HEAD SHA — if the repo hasn't changed since last scout,
    returns the cached result instantly (0 GitHub API calls).

    Expected latency on cache miss: 2–6 seconds.
    Expected latency on cache hit:  < 50ms.
    """
    #update don't we need to check if ther repo is alreadyl indexed and not stale 
    async with conn.transcation():
        repo = await conn.fetchcone(
                """ SELECT   id, full_name, owner_login, repo_name, default_branch,
                                index_status, last_scout_sha, installation_id, account_id
                    FROM repos 
                    WHERE id = $1
                    FOR UPDATE NOWAIT """,repo_id
                )
        if not repo:
            raise HTTPException(status_code=404, detail="Repository not found")

        if repo["index_status"] in ("scouting", "pending", "indexing"):
            raise HTTPException(
                status_code=409, 
                detail=f"Action locked: Repository is currently in '{repo['index_status']}' status."
            )

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

    try:
        already_indexed = await _get_indexed_repos(conn)

        # Create shared InstallationCache — passed to DeepScout so it can be
        # reused by SubmoduleDecisionTree in Phase 3 (same org lookups, 0 extra calls)
        install_cache = InstallationCache(gh, installation_id)

        scout = DeepScout(
            github_service=gh,
            installation_id=installation_id,
            already_indexed_repos=already_indexed,
            install_cache=install_cache,
        )
        result: ScoutResult = await scout.run(
            owner=repo.github_owner,
            repo=repo.github_repo,
            branch=repo.default_branch,
        )

        result_dict = _serialize_scout(result)
        result_json_str = json.dumps(result_dict)

        async with conn.transaction(): 
            await conn.execute(
                """
                INSERT INTO repo_scout_results (repo_id, head_sha, scout_json, api_calls_made, duration_ms, created_at)
                VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (repo_id, head_sha) DO UPDATE SET scout_json = EXCLUDED.scout_json, created_at = NOW()
                """,
                repo_id, live_sha or "unknown", result_json_str, result.api_calls_made, result.scout_duration_ms
            )

            await conn.execute(
                "UPDATE repos SET last_scout_sha = $1, last_scout_at = NOW(), index_status = 'awaiting_ui', updated_at = NOW() WHERE id = $2",
                live_sha or "unknown", repo_id
            )
        

        logger.info(
            "Scout complete: %s — %d submodules, %d subprojects, %dms, %d API calls",
            repo.full_name,
            result.total_submodules,
            len(result.subprojects),
            result.scout_duration_ms,
            result.api_calls_made,
        )

        return {
            "cached": False,
            "head_sha": live_sha,
            "scout": result_dict,
        }
    
    
    except Exception as e:
        # Emergency fallbacks run in an isolated transaction block
        async with conn.transaction():
            await conn.execute("UPDATE repos SET index_status = 'failed', updated_at = NOW() WHERE id = $1", repo_id)
        logger.exception("Scout operation crashed out during network sync phase for repo: %s", repo_id)
        raise HTTPException(status_code=500, detail=f"Structural profiling engine network error: {str(e)}")



# ─────────────────────────────────────────────────────────────────────────────
# GET /repos/{id}/scout — fetch cached result
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/repos/{repo_id}/scout")
async def get_scout(
    repo_id: int, 
    conn = Depends(get_authed_read_db_dep) 
):
    repo = await conn.fetchrow("SELECT last_scout_sha, last_scout_at FROM repos WHERE id = $1", repo_id)
    if not repo or not repo["last_scout_sha"]:
        raise HTTPException(status_code=404, detail="No blueprint data tracked for this repository instance.")

    cached_json = await conn.fetchval(
        "SELECT scout_json FROM repo_scout_results WHERE repo_id = $1 AND head_sha = $2 LIMIT 1",
        repo_id, repo["last_scout_sha"]
    )
    if not cached_json:
        raise HTTPException(status_code=404, detail="Cached blueprint metadata missing from storage schemas.")

    return {
        "head_sha": repo["last_scout_sha"],
        "scouted_at": repo["last_scout_at"].isoformat() if repo["last_scout_at"] else None,
        "scout": json.loads(cached_json),
    }

