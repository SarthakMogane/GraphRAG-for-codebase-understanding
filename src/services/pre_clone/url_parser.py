"""
app/services/pre_clone/url_parser.py
──────────────────────────────────────
Secure GitHub URL parser + normalizer.

Supported:
  https://github.com/owner/repo
  https://github.com/owner/repo.git
  https://github.com/owner/repo/
  https://github.com/owner/repo/tree/main
  https://github.com/owner/repo/blob/main/file.py
  git@github.com:owner/repo.git
  github.com/owner/repo
  owner/repo
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from src.services.pre_clone.types import ParsedURL


# -----------------------------
# GitHub constraints
# -----------------------------

_MAX_OWNER_LEN = 39
_MAX_REPO_LEN = 100

_RESERVED_OWNERS = frozenset({
    "about",
    "account",
    "blog",
    "contact",
    "enterprise",
    "explore",
    "features",
    "issues",
    "login",
    "marketplace",
    "notifications",
    "pricing",
    "pulls",
    "settings",
    "site",
    "topics",
    "trending",
    "wiki",
})

# GitHub usernames/orgs:
# - alphanumeric + hyphens
# - cannot start/end with hyphen
# - max 39 chars
OWNER_RE = re.compile(
    r"^(?!.*--)[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$"
)

# GitHub repo names:
# reasonably strict + safe
REPO_RE = re.compile(
    r"^[A-Za-z0-9._-]{1,100}$"
)

SSH_RE = re.compile(
    r"^git@github\.com:(?P<owner>[^/]+)/(?P<repo>.+)$"
)


# -----------------------------
# Public API
# -----------------------------

def parse_github_url(raw: str) -> ParsedURL:
    """
    Parse and normalize a GitHub repository URL.

    Never raises.
    All errors returned inside ParsedURL.error.
    """

    if not raw or not raw.strip():
        return _error(raw, "URL is empty")

    value = raw.strip()

    try:
        owner, repo = _extract_owner_repo(value)

        if repo.endswith(".git"):
            repo = repo[:-4]

        owner = owner.strip()
        repo = repo.strip()

        err = _validate_owner_repo(owner, repo)
        if err:
            return _error(raw, err, owner, repo)

        return ParsedURL(
            raw=raw,
            owner=owner.lower(),  # normalize
            repo=repo,
            is_valid=True,
        )

    except ValueError as e:
        return _error(raw, str(e))


# -----------------------------
# Extraction
# -----------------------------

def _extract_owner_repo(value: str) -> tuple[str, str]:
    """
    Extract owner/repo from:
      - HTTPS URL
      - SSH URL
      - github.com/owner/repo
      - owner/repo
    """

    # -------------------------
    # SSH format
    # -------------------------
    ssh_match = SSH_RE.match(value)
    if ssh_match:
        return (
            ssh_match.group("owner"),
            ssh_match.group("repo"),
        )

    # -------------------------
    # Bare shorthand
    # owner/repo
    # -------------------------
    if (
        "://" not in value
        and not value.startswith("github.com/")
        and value.count("/") == 1
    ):
        owner, repo = value.split("/", 1)
        return owner, repo

    # -------------------------
    # Add scheme if missing
    # -------------------------
    if value.startswith("github.com/"):
        value = f"https://{value}"

    parsed = urlparse(value)

    # -------------------------
    # Validate host
    # -------------------------
    parsed = urlparse(value)

    # Missing host entirely
    if not parsed.netloc:
        raise ValueError(
            "Expected format: github.com/owner/repo"
        )

    # Wrong host
    if parsed.netloc.lower() != "github.com":
        raise ValueError(
            "Only github.com URLs are supported"
        )

    # -------------------------
    # Extract path parts
    # -------------------------
    parts = [p for p in parsed.path.split("/") if p]

    if len(parts) < 2:
        raise ValueError(
            "Expected format: github.com/owner/repo"
        )

    owner = parts[0]
    repo = parts[1]

    return owner, repo


# -----------------------------
# Validation
# -----------------------------

def _validate_owner_repo(owner: str, repo: str) -> str | None:
    """
    Validate owner/repo according to GitHub naming rules.
    """

    if len(owner) > _MAX_OWNER_LEN:
        return (
            f"Owner name too long "
            f"({len(owner)} > {_MAX_OWNER_LEN})"
        )

    if len(repo) > _MAX_REPO_LEN:
        return (
            f"Repo name too long "
            f"({len(repo)} > {_MAX_REPO_LEN})"
        )

    if owner.lower() in _RESERVED_OWNERS:
        return (
            f"'{owner}' is a reserved GitHub path"
        )

    if not OWNER_RE.fullmatch(owner):
        return (
            f"Invalid GitHub owner name: '{owner}'"
        )

    if not REPO_RE.fullmatch(repo):
        return (
            f"Invalid GitHub repository name: '{repo}'"
        )

    # Prevent weird edge cases
    if repo in {".", ".."}:
        return "Invalid repository name"

    return None


# -----------------------------
# Helpers
# -----------------------------

def _error(
    raw: str,
    message: str,
    owner: str = "",
    repo: str = "",
) -> ParsedURL:
    return ParsedURL(
        raw=raw,
        owner=owner,
        repo=repo,
        is_valid=False,
        error=message,
    )