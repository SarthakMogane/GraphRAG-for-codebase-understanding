"""
app/services/pre_clone/fork_detector.py
─────────────────────────────────────────
"""

import logging
from typing import Optional

import httpx

from src.core.config import get_settings
from src.services.pre_clone.types import ForkInfo, ValidationVerdict

logger = logging.getLogger(__name__)
settings = get_settings()

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

        if not repo_data.parent_info:
            # No parent data — treat as standalone
            return None, ForkInfo(is_fork=True)
        
        parent  = repo_data.parent_info

        fork_info = ForkInfo(
            is_fork=True,
            upstream_owner=parent.owner_login,
            upstream_repo=parent.name,
            upstream_github_id=parent.github_id,
        )

        if not parent.is_private:
            # Upstream is public → this is a fork of OSS → reject
            fork_info.is_diverged = False
            return ValidationVerdict.REPO_FORK_OF_PUBLIC, fork_info

        # Upstream is private → index it (user's own private infrastructure)
        return None, fork_info


"""
New Feature will be : - 

Fork detection and upstream linking.

When a user submits a fork, we have three possible outcomes:

  A) Upstream is indexed + fork hasn't diverged
     → Redirect to upstream wiki (no separate ingestion)
     → Save a ForkRecord linking fork → upstream

  B) Upstream is indexed + fork HAS diverged (unique commits/files)
     → Index fork independently, but cross-link to upstream in wiki
     → "This fork diverges from {upstream} by {N} commits"

  C) Upstream is NOT indexed
     → Index the fork on its own (treat as a standalone repo)
     → Also queue the upstream for indexing (best-effort background job)

Divergence is measured by:
  - Commits ahead of upstream (fork-unique commits)
  - File diff breadth (how many files have been modified vs upstream)

Thresholds are configurable in settings.
"""
async def detect_fork(
    repo_data: dict,
    headers: dict[str, str],
    already_indexed_repos: dict[str, int],  # "owner/repo" → db_id
) -> ForkInfo:
    """
    Analyse a repository's fork status.

    Args:
        repo_data:              Raw GitHub API response for this repo
        headers:                Authenticated request headers
        already_indexed_repos:  Map of all currently indexed repos in our DB

    Returns a ForkInfo dataclass with complete fork analysis.
    """
    if not repo_data.get("fork", False):
        return ForkInfo(is_fork=False)

    # GitHub always includes parent info on the /repos/{owner}/{repo} response
    # when fork=True. The "parent" is the direct upstream (could itself be a fork).
    # "source" is the ultimate root of the fork chain.
    parent = repo_data.get("parent")
    source = repo_data.get("source")

    if not parent:
        # Unusual — fork=True but no parent data. Treat as standalone.
        logger.warning(
            "Repository %s is marked as fork but has no parent data",
            repo_data.get("full_name")
        )
        return ForkInfo(is_fork=True)

    # We care about the DIRECT parent, not the ultimate source.
    # A fork of a fork should reference its immediate parent, not the root.
    upstream_owner = parent["owner"]["login"]
    upstream_repo  = parent["name"]
    upstream_id    = parent["id"]
    upstream_key   = f"{upstream_owner}/{upstream_repo}"

    upstream_indexed    = upstream_key in already_indexed_repos
    upstream_db_id      = already_indexed_repos.get(upstream_key)

    fork_info = ForkInfo(
        is_fork=True,
        upstream_owner=upstream_owner,
        upstream_repo=upstream_repo,
        upstream_github_id=upstream_id,
        upstream_indexed=upstream_indexed,
        upstream_repo_db_id=upstream_db_id,
    )

    logger.info(
        "Fork detected: %s → upstream %s (indexed: %s)",
        repo_data.get("full_name"), upstream_key, upstream_indexed,
    )

    # Measure divergence only if upstream is indexed (otherwise moot — we index fork standalone)
    if upstream_indexed:
        divergence = await _measure_divergence(
            fork_owner=repo_data["owner"]["login"],
            fork_repo=repo_data["name"],
            upstream_owner=upstream_owner,
            upstream_repo=upstream_repo,
            default_branch=repo_data.get("default_branch", "main"),
            headers=headers,
        )
        fork_info.commits_ahead  = divergence["ahead"]
        fork_info.commits_behind = divergence["behind"]
        fork_info.is_diverged    = _is_significantly_diverged(divergence)

        logger.info(
            "Fork divergence: %d ahead, %d behind upstream — diverged=%s",
            fork_info.commits_ahead, fork_info.commits_behind, fork_info.is_diverged,
        )

    return fork_info


async def _measure_divergence(
    fork_owner: str,
    fork_repo: str,
    upstream_owner: str,
    upstream_repo: str,
    default_branch: str,
    headers: dict[str, str],
) -> dict:
    """
    Use GitHub's Compare API to measure how far the fork has diverged.

    GitHub's compare endpoint: GET /repos/{owner}/{repo}/compare/{base}...{head}
    We compare upstream's default branch against the fork's default branch.

    Format: /repos/{fork_owner}/{fork_repo}/compare/{upstream_owner}:{branch}...{branch}
    This measures: "how does fork's branch compare to upstream's branch?"

    Returns dict with keys: ahead, behind, diff_files
    """
    comparison_url = (
        f"https://api.github.com/repos/{fork_owner}/{fork_repo}/compare/"
        f"{upstream_owner}:{default_branch}...{default_branch}"
    )

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                comparison_url,
                headers=headers,
                timeout=20.0,
            )

            if resp.status_code == 404:
                # Happens when the fork and upstream have no common commits
                # (e.g. fork was created from scratch with same name)
                # Treat as maximally diverged.
                logger.warning(
                    "No common history between fork %s/%s and upstream %s/%s",
                    fork_owner, fork_repo, upstream_owner, upstream_repo,
                )
                return {"ahead": 999, "behind": 0, "diff_files": 999}

            if resp.status_code == 422:
                # Unprocessable — usually means branches have diverged too much
                # for GitHub to compute. Treat as diverged.
                return {"ahead": 999, "behind": 0, "diff_files": 999}

            resp.raise_for_status()
            data = resp.json()

            return {
                "ahead":      data.get("ahead_by", 0),
                "behind":     data.get("behind_by", 0),
                "diff_files": len(data.get("files", [])),
                "status":     data.get("status", "unknown"),
                # status can be: "ahead", "behind", "diverged", "identical"
            }

        except httpx.TimeoutException:
            logger.warning("Divergence check timed out for %s/%s", fork_owner, fork_repo)
            # Conservative: assume not diverged → redirect to upstream
            return {"ahead": 0, "behind": 0, "diff_files": 0}

        except httpx.HTTPError as e:
            logger.warning("Divergence check failed: %s", e)
            return {"ahead": 0, "behind": 0, "diff_files": 0}


def _is_significantly_diverged(divergence: dict) -> bool:
    """
    Decide if a fork's divergence is significant enough to warrant
    independent indexing rather than redirecting to upstream.

    Criteria (any one is sufficient):
      - 10+ commits ahead of upstream
      - 20+ files differ from upstream
    """
    commits_threshold = getattr(settings, "FORK_DIVERGENCE_COMMITS_THRESHOLD", 10)
    files_threshold   = getattr(settings, "FORK_DIVERGENCE_FILES_THRESHOLD", 20)

    return (
        divergence.get("ahead", 0)      >= commits_threshold or
        divergence.get("diff_files", 0) >= files_threshold
    )


def get_fork_verdict(fork_info: ForkInfo) -> Optional[ValidationVerdict]:
    """
    Given a completed ForkInfo, return the ValidationVerdict for fork handling.
    Returns None if the fork should be indexed independently (no redirect needed).

    Called by the main validation pipeline to decide what to do with forks.
    """
    if not fork_info.is_fork:
        return None  # Not a fork — not our concern here

    if not fork_info.upstream_indexed:
        # Upstream not in our index → treat fork as standalone
        return None

    if fork_info.is_diverged:
        # Significant divergence → index independently, cross-link to upstream
        return None

    # Upstream indexed + fork not diverged → redirect to upstream wiki
    return ValidationVerdict.REPO_FORK_REDIRECTED