"""
app/services/pre_clone/pipeline.py
────────────────────────────────────
Entry gate — runs the moment a user submits a URL.

Role in the new 4-phase architecture:
  pipeline.py  = Phase 0 (entry gate, ~200ms, synchronous with HTTP request)
  deep_scout   = Phase 1 (async, 2–6s, triggered after pipeline passes)
  selection.py = Phase 2 (UI)
  ingestion    = Phase 3 (background worker)

What pipeline.py does (5 checks, in order):
  1. URL parsing          — malformed URL → 400, no DB writes
  2. Rate limit preflight — near-limit → 429 with reset time
  3. DB cache lookup      — already indexed + not stale → serve cache
  4. Already processing   — active job running → 409
  5. GitHub API validation — exists, private, not archived, not empty
     Fork check:
       is_fork=True + upstream is PUBLIC → reject (user forked OSS)
       is_fork=True + upstream is PRIVATE → proceed (user forked their own code)

What pipeline.py does NOT do anymore:
  - Fork divergence measurement (we index all private forks regardless)
  - Monorepo detection (moved to deep_scout / Phase 1)
  - Submodule detection (moved to deep_scout / Phase 1)
  - Staleness measurement (webhook handles this; pipeline just checks the flag)

Output: ValidationResult
  .should_proceed → True = hand off to deep_scout
  .routing        → NEW_INGESTION | REFRESH | SERVE_CACHE
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import Repository, RepoStatus
from src.services.github import GitHubService
from src.services.pre_clone.rate_limit_checker import check_rate_limit
from src.services.pre_clone.repo_validator import (
    RepoValidationError,
    validate_repo_exists_and_accessible,
)
from src.services.pre_clone.stale_checker import (
    check_already_processing,
    check_staleness,
)
from src.services.pre_clone.types import (
    ForkInfo,
    RoutingDecision,
    ValidationResult,
    ValidationVerdict,
)
from src.services.pre_clone.url_parser import parse_github_url

logger = logging.getLogger(__name__)


class PreClonePipeline:
    """
    Stateless entry gate. One instance per request — no shared state.

    Usage:
        pipeline = PreClonePipeline(
            github_service=gh,
            db=db,
            installation_id=user_install_id,
            force_refresh=False,
        )
        result = await pipeline.run("https://github.com/myorg/myapp")
        if not result.should_proceed:
            return error_response(result)
    """

    def __init__(
        self,
        github_service: GitHubService,
        db: AsyncSession,
        installation_id: int,
        force_refresh: bool = False,
    ):
        self.gh              = github_service
        self.db              = db
        self.installation_id = installation_id
        self.force_refresh   = force_refresh  # on retry

    async def run(self, raw_url: str) -> ValidationResult:
        """
        Run the entry gate pipeline. Returns ValidationResult.
        Fast path (cache hit): ~50ms. Full path (new repo): ~300ms.
        """
        # ── 1. URL parsing ─────────────────────────────────────────────────
        parsed = parse_github_url(raw_url)
        if not parsed.is_valid:
            return ValidationResult(
                parsed_url=parsed,
                verdict=ValidationVerdict.URL_PARSE_ERROR,
                message=parsed.error or "Invalid GitHub URL",
            )

        logger.info("Pipeline: %s", parsed.full_name)

        # ── 2. Rate limit preflight ────────────────────────────────────────
        rate_status = await check_rate_limit(
            self.gh, self.installation_id
        )
        if not rate_status.is_safe:
            from datetime import timezone
            import datetime
            secs = max(0, int(
                (rate_status.reset_at - datetime.datetime.now(timezone.utc))
                .total_seconds()
            ))
            return ValidationResult(
                parsed_url=parsed,
                verdict=ValidationVerdict.RATE_LIMIT_BLOCKED,
                rate_limit=rate_status,
                message=(
                    f"GitHub API rate limit buffer reached "
                    f"({rate_status.remaining} remaining). "
                    f"Resets in {secs // 60}m {secs % 60}s."
                ),
            )

        result = ValidationResult(
            parsed_url=parsed,
            verdict=ValidationVerdict.APPROVED,
            rate_limit=rate_status,
        )

        # ── 3. DB cache lookup ─────────────────────────────────────────────
        existing = await self._lookup_db(parsed.owner, parsed.repo)
        if existing:
            result.existing_repo_db_id = existing.id

            # 4. Already processing?
            if not self.force_refresh:
                active = await check_already_processing(existing, self.db)
                if active:
                    return ValidationResult(
                        parsed_url=parsed,
                        verdict=ValidationVerdict.ALREADY_PROCESSING,
                        existing_repo_db_id=existing.id,
                        existing_job_status=active,
                        rate_limit=rate_status,
                        message=(
                            f"Ingestion job is already {active} for "
                            f"{parsed.full_name}. Wait for it to finish."
                        ),
                    )

            # Staleness check for READY repos
            if existing.status == RepoStatus.READY and not self.force_refresh:
                stale = await check_staleness(existing, self.gh, self.installation_id, self.db)
                result.stale_check = stale
                if not stale.is_stale:
                    result.routing = RoutingDecision.SERVE_CACHE
                    result.message = (
                        f"{parsed.full_name} is up to date "
                        f"(SHA: {stale.stored_sha[:8] if stale.stored_sha else '?'}). "
                        f"Serving from cache."
                    )
                    result.default_branch   = existing.default_branch
                    result.primary_language = existing.primary_language
                    return result
                else:
                    result.routing = RoutingDecision.REFRESH

        # ── 5. GitHub API validation ───────────────────────────────────────
        try:
            repo_data = await validate_repo_exists_and_accessible(
                parsed, self.gh, self.installation_id
            )
        except RepoValidationError as e:
            return ValidationResult(
                parsed_url=parsed,
                verdict=e.verdict,
                rate_limit=rate_status,
                existing_repo_db_id=result.existing_repo_db_id,
                message=e.message,
            )

        result.github_id        = repo_data["id"]
        result.default_branch   = repo_data["default_branch"]
        result.primary_language = repo_data.get("language")
        result.size_kb          = repo_data.get("size", 0)
        result.description      = repo_data.get("description")
        result.topics           = repo_data.get("topics", [])

        # ── Fork check (simplified for private-only product) ──────────────
        if repo_data.get("fork", False):
            fork_verdict, fork_info = await self._check_fork(repo_data)
            result.fork_info = fork_info
            if fork_verdict is not None:
                result.verdict = fork_verdict
                result.message = (
                    f"{parsed.full_name} is a fork of a public repository "
                    f"({fork_info.upstream_owner}/{fork_info.upstream_repo}). "
                    f"We only index private code you wrote. Public forks are not indexed."
                )
                return result
            # Private fork of private upstream → proceed normally
            logger.info(
                "%s is a private fork — indexing as private code", parsed.full_name
            )

        # Set routing if not already set by staleness check
        if result.routing is None:
            result.routing = (
                RoutingDecision.REFRESH if existing
                else RoutingDecision.NEW_INGESTION
            )

        result.verdict = ValidationVerdict.APPROVED
        result.message = (
            f"{parsed.full_name} approved. "
            f"{'Refresh queued.' if result.routing == RoutingDecision.REFRESH else 'New ingestion queued.'}"
        )

        logger.info(
            "Pipeline approved: %s routing=%s",
            parsed.full_name, result.routing.value,
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Fork handling — private-only logic
    # ─────────────────────────────────────────────────────────────────────────

    async def _check_fork(
        self, repo_data: dict
    ) -> tuple[Optional[ValidationVerdict], ForkInfo]:
        """
        Private-only fork policy:
          - Fork of PUBLIC upstream  → reject (don't index OSS the user forked)
          - Fork of PRIVATE upstream → proceed (still the user's private code)
          - Can't determine upstream → proceed conservatively

        We do NOT measure divergence. Any private fork is worth indexing —
        the user has access to it, it may contain their customisations.
        """
        parent = repo_data.get("parent") or repo_data.get("source")

        if not parent:
            # No parent data — treat as standalone
            return None, ForkInfo(is_fork=True)

        upstream_owner   = parent["owner"]["login"]
        upstream_repo    = parent["name"]
        upstream_private = parent.get("private", False)
        upstream_id      = parent["id"]

        fork_info = ForkInfo(
            is_fork=True,
            upstream_owner=upstream_owner,
            upstream_repo=upstream_repo,
            upstream_github_id=upstream_id,
        )

        if not upstream_private:
            # Upstream is public → this is a fork of OSS → reject
            fork_info.is_diverged = False
            return ValidationVerdict.REPO_FORK_OF_PUBLIC, fork_info

        # Upstream is private → index it (user's own private infrastructure)
        return None, fork_info

    # ─────────────────────────────────────────────────────────────────────────
    # DB helpers
    # ─────────────────────────────────────────────────────────────────────────

    async def _lookup_db(self, owner: str, repo: str) -> Optional[Repository]:
        result = await self.db.execute(
            select(Repository).where(
                Repository.github_owner == owner,
                Repository.github_repo  == repo,
            )
        )
        return result.scalar_one_or_none()