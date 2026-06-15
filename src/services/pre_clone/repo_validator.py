"""
app/services/pre_clone/repo_validator.py
──────────────────────────────────────────
Core repository validation checks.

Runs against the GitHub API to verify a repository is real, public,
and ingestion-worthy before we commit any compute to it.

Checks performed (in order):
  1. Repository exists (not 404)
  2. Repository is private (not public)
  3. Repository is not archived
  4. Repository is not disabled
  5. Repository is not empty (size > 0 KB, has at least one commit)
  6. Repository is not a template-only repo with no real content

Each check is independent and returns a specific ValidationVerdict on failure
so the caller knows exactly why a repo was rejected.
"""

import logging
from typing import Optional
from src.services.github import RateLimitError , RepoMetadata

import httpx

from src.services.pre_clone.types import (
    ParsedURL, ValidationVerdict,
)

logger = logging.getLogger(__name__)

# Minimum repo size in KB to be worth indexing.
# GitHub reports 0 KB for a repo with only a README commit, but 1+ KB for anything real.
_MIN_SIZE_KB = 1

# Maximum size before we warn (not block) — very large repos get a warning in the response
_WARN_SIZE_KB = 2_000_000  # 2 GB


class RepoValidationError(Exception):
    """Raised when a validation check fails. Carries the verdict."""
    def __init__(self, verdict: ValidationVerdict, message: str):
        self.verdict = verdict
        self.message = message
        super().__init__(message)


async def validate_repo_exists_and_accessible(
    parsed_url: ParsedURL,
    gh,
    installation_id
) -> RepoMetadata:
    """
    Fetch repository metadata from GitHub API and run all structural checks.

    Returns the raw GitHub API response dict on success.
    Raises RepoValidationError with specific verdict on any failure.
    """
    owner = parsed_url.owner
    repo  = parsed_url.repo

    try:
        metadata = await gh.get_repo_metadata(
            owner,
            repo,
            installation_id
        )
    
    except RateLimitError as e:
        raise RepoValidationError(
            ValidationVerdict.RATE_LIMIT_BLOCKED,
            str(e)
        )
    # We must catch HTTP status errors (401, 403, 404, 500)
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        
        if status == 404:
            raise RepoValidationError(
                ValidationVerdict.REPO_NOT_FOUND,
                f"Repository {owner}/{repo} does not exist or is not accessible."
            )
        if status == 401:
            raise RepoValidationError(
                ValidationVerdict.REPO_NOT_FOUND,
                "GitHub API authentication failed. Check App credentials."
            )
        if status == 403:
            # Check if the 403 is actually a rate limit disguised as a permissions error
            body = e.response.json() if e.response.headers.get("content-type", "").startswith("application/json") else {}
            if "rate limit" in body.get("message", "").lower():
                raise RepoValidationError(
                    ValidationVerdict.RATE_LIMIT_BLOCKED,
                    "GitHub API rate limit exceeded during validation."
                )
            raise RepoValidationError(
                ValidationVerdict.REPO_PRIVATE,
                f"Access denied to {owner}/{repo}. Ensure the app is installed on this repo."
            )
        if status >= 500:
            raise RepoValidationError(
                ValidationVerdict.REPO_NOT_FOUND,
                f"GitHub API returned server error {status}. Try again shortly."
            )
            
        # Fallback for unexpected HTTP errors (e.g., 400 Bad Request)
        raise RepoValidationError(
            ValidationVerdict.REPO_NOT_FOUND,
            f"Unexpected HTTP error {status} from GitHub."
        )
    
    except httpx.TimeoutException:
        raise RepoValidationError(
            ValidationVerdict.REPO_NOT_FOUND,
            f"GitHub API timed out fetching {owner}/{repo}. Try again shortly."
        )
    except httpx.NetworkError as e:
        raise RepoValidationError(
            ValidationVerdict.REPO_NOT_FOUND,
            f"Network error reaching GitHub API: {e}"
        )

    # ── Check 2: Visibility ───────────────────────────────────────────────────
    if not metadata.is_private:
        raise RepoValidationError(
            ValidationVerdict.REPO_PUBLIC,
            f"{owner}/{repo} is a public repository. "
            f"Only private repositories are supported in the current version. "
            f"Do your repo private !"
        )

    # ── Check 3: Archived ─────────────────────────────────────────────────────
    if metadata.is_archived:
        raise RepoValidationError(
            ValidationVerdict.REPO_ARCHIVED,
            f"{owner}/{repo} is archived. Archived repositories are read-only and "
            f"no longer actively maintained. We still index them — "
            f"but note the documentation will not auto-update since no new commits "
            f"will arrive. Pass force_index_archived=true to proceed."
            # NOTE: the actual force flag is checked by the caller, not here.
            # This validator just raises so the caller can decide.
        )

    # ── Check 4: Disabled ─────────────────────────────────────────────────────
    # GitHub marks repos as disabled when they violate ToS or are under DMCA takedown
    if metadata.is_disabled:
        raise RepoValidationError(
            ValidationVerdict.REPO_DISABLED,
            f"{owner}/{repo} has been disabled by GitHub. "
            f"This typically means a Terms of Service violation or DMCA takedown."
        )
    
    if metadata.visibility == "public":
        raise RepoValidationError(
            ValidationVerdict.REPO_PUBLIC,
            f"{owner}/{repo} has visibility '{metadata.visibility}'. Only private repositories are supported."
        )

    # ── Check 5: Empty repository ─────────────────────────────────────────────
    if metadata.is_empty:
        # Double-check: GitHub sometimes reports 0 for a newly initialised repo.
        # Verify by checking the branch list.
        is_truly_empty = await _verify_empty(gh,owner,repo,installation_id)
        if is_truly_empty:
            raise RepoValidationError(
                ValidationVerdict.REPO_EMPTY,
                f"{owner}/{repo} appears to be empty (no commits or files). "
                f"There is nothing to index yet."
            )
        # If not truly empty, proceed — GitHub size is just slow to update

    # ── Check 6: Template-only repo ───────────────────────────────────────────
    # Template repos with is_template=True and 0 non-template commits are
    # basically empty scaffolds. Still index them — but note it.
    if metadata.is_template:
        logger.info("%s/%s is a private template repository — indexing as-is", owner, repo)

    # Size warning (not a block)
    if metadata.size_kb > _WARN_SIZE_KB:
        logger.warning(
            "%s/%s is very large (%d MB). Ingestion will use sparse checkout.",
            owner, repo, metadata.size_kb // 1024,
        )

    return metadata


async def _verify_empty(gh,owner: str, repo: str, installation_id: int) -> bool:
    """
    Secondary check for empty repos.
    Fetches the branch list — if none exist, the repo is truly empty.
    This call costs 1 API request but only happens when size_kb == 0.
    """
    try:
        branches = await gh.get_repo_branches(
            owner,repo,installation_id
        )
        
        return len(branches) == 0
    except Exception:
        return False  # Conservative: don't block on any error . 