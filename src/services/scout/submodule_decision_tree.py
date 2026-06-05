"""
app/services/submodule_decision_tree.py
────────────────────────────────────────
The complete submodule decision tree.

Implements every node described in the architecture:
  Node 1: URL Resolution (relative, SSH, non-GitHub, dead URLs)
  Node 2: GitHub API metadata fetch
  Node 3: Internal vs External classification
  Node 4A: External sub-tree (cross-link, library-ref, too-large, vendored)
  Node 4B: Internal sub-tree (already-indexed, size estimation, B3 gate, circular check)

Input:  A list of raw .gitmodules entries + parent repo context
Output: A list of SubmoduleDecision objects, one per entry
"""

import logging
import re
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator
from pathlib import Path
from typing import Optional
import configparser
import httpx
from datetime import datetime, timedelta, timezone
import asyncio


from src.core.config import get_settings
from src.models.database import SubmoduleOutcome
from src.services.github import GitHubService, RepoMetadata

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Input / Output Data Classes ───────────────────────────────────────────────

class GitmodulesEntry(BaseModel):
    """
    Validated entry parsed from .gitmodules file.
    Max lengths prevent memory exhaustion attacks from malicious repos.
    """
    # model_config enforces strict type checking and forbids unexpected fields
    model_config = ConfigDict(strict=True, extra="forbid")

    name: str = Field(..., max_length=255, description='[submodule "name"]')
    path: str = Field(..., max_length=1024, description="path = ...")
    # We don't use Pydantic's HttpUrl here because it might be a relative path or SSH string
    raw_url: str = Field(..., max_length=2048, description="url = ...") 
    normalized_url:Optional[str] = None
    branch: Optional[str] = Field(default=None, max_length=255) #change: submodule deafault branch 
    owner: Optional[str] = None
    repo: Optional[str] = None
    url_error: Optional[str] = None
    is_valid_url: bool = False


class SubmoduleDecision(BaseModel):
    """
    Complete decision tree result for one submodule.
    """
    model_config = ConfigDict(from_attributes=True) # Useful if mapping directly to ORM/Database models

    # Source data
    entry: GitmodulesEntry

    # Resolved
    resolved_url: Optional[str] = Field(default=None, max_length=2048)
    resolved_owner: Optional[str] = Field(default=None, max_length=255)
    resolved_repo: Optional[str] = Field(default=None, max_length=255)
    pinned_sha: Optional[str] = Field(default=None, min_length=40, max_length=40) # Git SHAs are exactly 40 chars
    is_relative_url: bool = False

    # Classification
    is_internal: Optional[bool] = None  # None = unresolvable

    # Decision
    outcome: SubmoduleOutcome = SubmoduleOutcome.INACCESSIBLE
    skip_reason: Optional[str] = Field(default=None, max_length=1024)
    should_initialize: bool = False      # True = git submodule update --init ...
    use_blob_filter: bool = False        # True for large approved submodules
    linked_repo_id: Optional[int] = Field(default=None, ge=1) # Repo IDs must be positive

    # Size estimation (from GitHub tree API)
    estimated_source_files: int = Field(default=0, ge=0)
    estimated_source_bytes: int = Field(default=0, ge=0)
    complexity_band: Optional[str] = Field(default=None, pattern="^(small|medium|large)$") 

    depth: int = Field(default=1, ge=1, le=10) # Enforce a hard maximum depth at the model level

    @field_validator("skip_reason")
    @classmethod
    def sanitize_skip_reason(cls, v: Optional[str]) -> Optional[str]:
        """Ensure error strings don't contain unescaped control characters"""
        if v is not None:
            return v.replace("\n", " ").replace("\r", "")
        return v


# ── Gitmodules Parser ─────────────────────────────────────────────────────────
from __future__ import annotations

import configparser
import logging
from pathlib import Path, PurePosixPath

from src.services.pre_clone.types import GitmodulesEntry
from src.services.pre_clone.url_parser import parse_github_url

logger = logging.getLogger(__name__)


class GitmodulesParser:
    """
    Secure parser for .gitmodules files.

    Security protections:
    - Disables config interpolation
    - Rejects malformed INI syntax
    - Enforces file size limits
    - Enforces section count limits
    - Validates submodule paths
    - Validates GitHub URLs
    - Gracefully handles malformed input
    """

    MAX_GITMODULES_SIZE = 1_048_576  # 1MB
    MAX_SUBMODULES = 100

    @classmethod
    def from_file(cls, gitmodules_path: Path) -> list[GitmodulesEntry]:
        """
        Read and parse a .gitmodules file safely.
        Never raises.
        """

        try:
            # -----------------------------------------
            # Reject symlinks
            # -----------------------------------------
            if gitmodules_path.is_symlink():
                logger.warning(
                    "Rejected symlinked .gitmodules file: %r",
                    str(gitmodules_path),
                )
                return []

            # -----------------------------------------
            # File size protection
            # -----------------------------------------
            file_size = gitmodules_path.stat().st_size

            if file_size > cls.MAX_GITMODULES_SIZE:
                logger.warning(
                    "Rejected oversized .gitmodules file "
                    "(%d bytes > %d bytes)",
                    file_size,
                    cls.MAX_GITMODULES_SIZE,
                )
                return []

            # -----------------------------------------
            # Read file safely
            # -----------------------------------------
            content = gitmodules_path.read_text(
                encoding="utf-8",
                errors="replace",
            )

            return cls().parse(content)

        except FileNotFoundError:
            return []

        except (
            PermissionError,
            OSError,
            UnicodeDecodeError,
        ) as e:
            logger.error(
                "Failed to read .gitmodules file: %s",
                e,
            )
            return []

    def parse(self, content: str) -> list[GitmodulesEntry]:
        """
        Parse .gitmodules content safely.

        Never raises.
        """

        # -----------------------------------------
        # Content size protection
        # -----------------------------------------
        if len(content.encode("utf-8")) > self.MAX_GITMODULES_SIZE:
            logger.warning(
                "Rejected oversized .gitmodules content"
            )
            return []

        # -----------------------------------------
        # Disable interpolation for security
        # -----------------------------------------
        parser = configparser.ConfigParser(
            interpolation=None,
            strict=True,
        )

        try:
            parser.read_string(content)

        except configparser.Error as e:
            logger.error(
                "Failed to parse .gitmodules INI format: %s",
                e,
            )
            return []

        sections = parser.sections()

        # -----------------------------------------
        # Section count protection
        # -----------------------------------------
        if len(sections) > self.MAX_SUBMODULES:
            logger.warning(
                "Rejected .gitmodules file with too many "
                "submodules (%d > %d)",
                len(sections),
                self.MAX_SUBMODULES,
            )
            return []

        entries: list[GitmodulesEntry] = []

        seen_paths: set[str] = set()

        for section in sections:

            # -----------------------------------------
            # Only allow:
            # submodule "name"
            # -----------------------------------------
            if not (
                section.startswith('submodule "')
                and section.endswith('"')
            ):
                logger.debug(
                    "Skipping non-submodule section: %r",
                    section,
                )
                continue

            name = section[11:-1].strip()

            # -----------------------------------------
            # Safe extraction
            # -----------------------------------------
            path = parser.get(
                section,
                "path",
                fallback=None,
            )

            url = parser.get(
                section,
                "url",
                fallback=None,
            )

            branch = parser.get(
                section,
                "branch",
                fallback=None,
            )

            # -----------------------------------------
            # Required fields
            # -----------------------------------------
            if not path or not url:
                logger.debug(
                    "Skipping submodule missing "
                    "path/url: %r",
                    section,
                )
                continue

            # -----------------------------------------
            # Normalize whitespace
            # -----------------------------------------
            path = path.strip()
            url = url.strip()

            if branch:
                branch = branch.strip()

            # -----------------------------------------
            # Validate path safety
            # -----------------------------------------
            if not self._is_safe_submodule_path(path):
                logger.warning(
                    "Rejected unsafe submodule path: %r",
                    path,
                )
                continue

            # -----------------------------------------
            # Prevent duplicate paths
            # -----------------------------------------
            if path in seen_paths:
                logger.warning(
                    "Duplicate submodule path detected: %r",
                    path,
                )
                continue

            seen_paths.add(path)

            # -----------------------------------------
            # Validate GitHub URL
            # -----------------------------------------
            parsed_url = parse_github_url(url)


            # -----------------------------------------
            # Validate schema
            # -----------------------------------------
            try:
                entry = GitmodulesEntry(
                    entry = GitmodulesEntry(
                                    name=name,
                                    path=path,

                                    raw_url=url,
                                    normalized_url=(
                                        f"https://github.com/{parsed_url.owner}/{parsed_url.repo}"
                                        if parsed_url.is_valid
                                        else None
                                    ),

                                    owner=parsed_url.owner if parsed_url.is_valid else None,
                                    repo=parsed_url.repo if parsed_url.is_valid else None,

                                    branch=branch,

                                    is_valid_url=parsed_url.is_valid,
                                    url_error=parsed_url.error,
                                )
                    )

                entries.append(entry)

            except ValueError as e:
                logger.error(
                    "Pydantic validation failed "
                    "for submodule %r: %s",
                    name,
                    e,
                )

        return entries

    @staticmethod
    def _is_safe_submodule_path(path: str) -> bool:
        """
        Validate submodule paths to prevent:
        - path traversal
        - absolute paths
        - .git directory abuse
        """

        try:
            p = PurePosixPath(path)

            # Reject absolute paths
            if p.is_absolute():
                return False

            # Reject traversal
            if ".." in p.parts:
                return False

            # Reject .git abuse
            if path.startswith(".git"):
                return False

            # Reject empty paths
            if not path.strip():
                return False

            return True

        except Exception:
            return False


# # ── Known Organization Data ───────────────────────────────────────────────────

# # Maps GitHub org login → canonical family name
# # Submodules owned by a sibling org are treated as internal
# ORG_FAMILY_MAP: dict[str, str] = {
#     "google": "google",
#     "googledeepmind": "google",
#     "googlecloudplatform": "google",
#     "googleresearch": "google",
#     "google-deepmind": "google",
#     "microsoft": "microsoft",
#     "azure": "microsoft",
#     "microsoftdocs": "microsoft",
#     "aws": "amazon",
#     "awslabs": "amazon",
#     "amazon": "amazon",
#     "meta": "meta",
#     "facebookresearch": "meta",
#     "facebookincubator": "meta",
# }

# # Well-known OSS orgs — always external regardless of name similarity
# OSS_BLOCKLIST: frozenset[str] = frozenset({
#     "abseil", "grpc", "protocolbuffers", "googleapis", "openssl",
#     "curl", "libuv", "nlohmann", "boostorg", "eigenteam",
#     "llvm", "numpy", "scipy", "pytorch", "tensorflow",
#     "kubernetes", "docker", "containerd", "opencontainers",
#     "prometheus", "grafana", "jaegertracing", "opentelemetry",
#     "redis", "mongodb", "postgres", "mysql",
#     "nodejs", "denoland", "vercel", "vitejs",
#     "django", "flask", "fastapi", "pallets",
#     "celery", "sqlalchemy", "pydantic",
# })

# # This strictly prevents path traversal characters like slashes or shell injections.
# VALID_GITHUB_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")

# # ── The Decision Tree ─────────────────────────────────────────────────────────

# class SubmoduleDecisionTree:
#     """
#     Runs the full decision tree for every entry in .gitmodules.

#     Usage:
#         tree = SubmoduleDecisionTree(
#             github_service=github_svc,
#             parent_owner="myorg",
#             parent_repo="myapp",
#             parent_default_branch="main",
#             already_indexed_repos={"myorg/auth-service": 42},  # name → repo_id
#         )
#         decisions = await tree.evaluate_all(entries, processing_stack=[])
#     """

#     def __init__(
#         self,
#         github_service: GitHubService,
#         parent_owner: str,
#         parent_repo: str,
#         parent_default_branch: str,
#         parent_org_family: Optional[str] = None,
#         already_indexed_repos: Optional[dict[str, int]] = None,
#     ):
#         self.gh = github_service
#         self.parent_owner = parent_owner
#         self.parent_repo = parent_repo
#         self.parent_branch = parent_default_branch
#         self.parent_org_family = (
#             parent_org_family
#             or ORG_FAMILY_MAP.get(parent_owner.lower())
#         )
#         # Maps "{owner}/{repo}" → repo_id for cross-linking
#         self.already_indexed: dict[str, int] = already_indexed_repos or {}

#     async def evaluate_all(
#         self,
#         entries: list[GitmodulesEntry],
#         processing_stack: list[str],   # Stack of "{owner}/{repo}" currently being processed
#         depth: int = 1,
#     ) -> list[SubmoduleDecision]:
#         """Evaluate all entries. Returns one SubmoduleDecision per entry."""
#         decisions = []
#         for entry in entries:
#             decision = await self._evaluate_one(entry, processing_stack, depth)
#             decisions.append(decision)
#             logger.info(
#                 "Submodule decision: path=%s outcome=%s reason=%s",
#                 entry.path, decision.outcome.value, decision.skip_reason or "approved"
#             )
#         return decisions

#     async def evaluate_all(
#         self,
#         entries: list[GitmodulesEntry],
#         processing_stack: list[str],
#         depth: int = 1,
#         max_concurrent: int = 5,  # Configurable concurrency limit
#     ) -> list[SubmoduleDecision]:
#         """
#         Evaluate all entries concurrently. Uses a semaphore to prevent 
#         hitting API rate limits or exhausting local file descriptors.
#         """
#         semaphore = asyncio.Semaphore(max_concurrent)
        
#         async def bounded_evaluate(entry: GitmodulesEntry) -> SubmoduleDecision:
#             async with semaphore:
#                 try:
#                     decision = await self._evaluate_one(entry, processing_stack, depth)
#                     logger.info(
#                         "Submodule decision: path=%s outcome=%s reason=%s",
#                         entry.path, 
#                         decision.outcome.value, 
#                         decision.skip_reason or "approved"
#                     )
#                     return decision
#                 except Exception as e:
#                     # Enterprise Guardrail: Catch-all at the execution boundary.
#                     # Prevents a single corrupt submodule from taking down the whole job.
#                     logger.error(
#                         "Unexpected system error evaluating submodule %s: %s", 
#                         entry.path, e, exc_info=True
#                     )
#                     error_decision = SubmoduleDecision(entry=entry, depth=depth)
#                     error_decision.outcome = SubmoduleOutcome.INACCESSIBLE
#                     error_decision.skip_reason = f"Internal evaluation error: {str(e)}"
#                     return error_decision

#         # Fire all tasks concurrently and wait for them to finish
#         tasks = [bounded_evaluate(entry) for entry in entries]
#         decisions = await asyncio.gather(*tasks)
        
#         return decisions
#     async def _evaluate_one(
#         self,
#         entry: GitmodulesEntry,
#         processing_stack: list[str],
#         depth: int,
#     ) -> SubmoduleDecision:
#         decision = SubmoduleDecision(entry=entry, depth=depth)

#         # ── Node 1: URL Resolution ────────────────────────────────────────────
#         resolved = self._resolve_url(entry.raw_url, decision)
#         if not resolved:
#             return decision   # outcome already set inside _resolve_url

#         owner, repo = resolved
#         decision.resolved_owner = owner
#         decision.resolved_repo = repo
#         decision.resolved_url = f"https://github.com/{owner}/{repo}"

#         # ── Depth Gate (before any API calls) ────────────────────────────────
#         if depth > settings.SUBMODULE_MAX_DEPTH:
#             decision.outcome = SubmoduleOutcome.DEPTH_EXCEEDED
#             decision.skip_reason = (
#                 f"Nesting depth {depth} exceeds maximum {settings.SUBMODULE_MAX_DEPTH}"
#             )
#             return decision

#         # ── Circular Reference Check (before any API calls) ───────────────────
#         repo_key = f"{owner}/{repo}"
#         if repo_key in processing_stack:
#             decision.outcome = SubmoduleOutcome.CIRCULAR_REFERENCE
#             decision.skip_reason = (
#                 f"Circular reference detected: {repo_key} "
#                 f"already in processing stack {processing_stack}"
#             )
#             return decision

#         # ── Node 2: GitHub API Metadata Fetch ────────────────────────────────
#         metadata = await self._fetch_metadata(owner, repo, decision)
#         if metadata is None:
#             return decision   # outcome set inside _fetch_metadata

#         # ── Node 3: Internal vs External Classification ───────────────────────
#         is_internal = self._classify_internal(owner, decision.is_relative_url)
#         decision.is_internal = is_internal

#         if is_internal:
#             return await self._node_4b_internal(decision, metadata, processing_stack)
#         else:
#             return await self._node_4a_external(decision, metadata)

#     # ── Node 1: URL Resolution ────────────────────────────────────────────────

#     def _resolve_url(self, raw_url: str, decision: SubmoduleDecision) -> Optional[tuple[str, str]]:
#         """
#         Normalize URL to (owner, repo). Returns None if unresolvable.
#         Sets decision.outcome and decision.skip_reason on failure.
#         """
#         url = raw_url.strip()
#         owner, repo = None, None

#         # Relative URL: ../repo.git or ./sub/repo.git
#         if url.startswith("./") or url.startswith("../"):
#             decision.is_relative_url = True
#             # Resolve against parent repo URL
#             base = f"https://github.com/{self.parent_owner}/"

#             if url.startswith("../"):
#                 resolved_path = url[3:] 
#             else:
#                 resolved_path = url[2:]

#             # ../repo.git → resolve: strip leading ../ and append to owner base
#             resolved_name = resolved_path.removesuffix(".git")
#             if "/" in resolved_path:
#                 # e.g., ../otherorg/repo
#                 parts = resolved_path.split("/", 1)
#                 owner, repo = parts[0], parts[1]
#             else:
#                 owner, repo = self.parent_owner, resolved_path

#         # SSH URL: git@github.com:owner/repo.git
#         elif url.startswith("git@github.com:"):
#             path = url[len("git@github.com:"):].removesuffix(".git")
#             parts = path.split("/", 1)
#             if len(parts) == 2:
#                 owner,repo = parts[0], parts[1]
            
#             else:
#                 decision.outcome = SubmoduleOutcome.INACCESSIBLE
#                 decision.skip_reason = f"Malformed SSH URL: {url}"
#                 return None

#         # HTTPS URL: https://github.com/owner/repo.git
      
#         elif match := re.match(r"^https://github\.com/([^/]+)/([^/]+?)(?:\.git)?$", url):
#             owner, repo = match.group(1), match.group(2)

#         # Non-GitHub host
#         elif url.startswith("https://") or url.startswith("http://"):
#             decision.outcome = SubmoduleOutcome.NON_GITHUB_HOST
#             decision.skip_reason = f"Non-GitHub host: {url}. Not currently supported."
#             return None

#         # Unrecognized format
#         else:
#             decision.outcome = SubmoduleOutcome.INACCESSIBLE
#             decision.skip_reason = f"Unrecognized URL format: {url}"
#             return None

#     # ── Enterprise Security Gate ─────────────────────────────────────────
#         # Ensure the extracted owner and repo do not contain malicious payloads
#         if owner and repo:
#             owner = owner.strip("/")
#             repo = repo.strip("/")
            
#             if not VALID_GITHUB_NAME.match(owner) or not VALID_GITHUB_NAME.match(repo):
#                 logger.warning("Security Warning: Invalid characters in submodule URL: %s", url)
#                 decision.outcome = SubmoduleOutcome.INACCESSIBLE
#                 decision.skip_reason = f"Security violation: Invalid owner/repo format parsed ({owner}/{repo})"
#                 return None
                
#             return owner, repo
            
#         return None

#     # ── Node 2: GitHub API Metadata Fetch ─────────────────────────────────────

#     async def _fetch_metadata(
#         self, owner: str, repo: str, decision: SubmoduleDecision
#     ) -> Optional[RepoMetadata]:
#         """Fetch repo metadata with strict timeout and HTTP error boundaries."""
#         try:
#             # The GitHubService beneath this should ideally be configured with 
#             # httpx.AsyncClient(timeout=10.0) to prevent infinite hangs.
#             metadata = await self.gh.get_repo_metadata(owner, repo)
            
#             if metadata.is_private:
#                 decision.outcome = SubmoduleOutcome.INACCESSIBLE
#                 decision.skip_reason = "Repository is private or requires authentication"
#                 return None
#             if metadata.visibility == "private":
#             # If your policy is to SKIP truly private repos:
#                 decision.outcome = SubmoduleOutcome.INACCESSIBLE
#                 decision.skip_reason = "Repository is strictly private"
#                 return None
#             if metadata.is_archived:
#                 logger.info("Submodule %s/%s is archived but accessible", owner, repo)
                
#             return metadata

#         except httpx.HTTPStatusError as e:
#             status = e.response.status_code
#             if status == 404:
#                 decision.outcome = SubmoduleOutcome.BROKEN_REFERENCE
#                 decision.skip_reason = f"Repository {owner}/{repo} not found (404)."
#             elif status in (403, 429):
#                 # Critical Enterprise Handling: Rate limits or Abuse mechanisms
#                 decision.outcome = SubmoduleOutcome.INACCESSIBLE
#                 decision.skip_reason = f"GitHub API Rate Limit exceeded or forbidden ({status})."
#                 logger.warning("Rate limit hit while fetching metadata for %s/%s", owner, repo)
#             else:
#                 decision.outcome = SubmoduleOutcome.INACCESSIBLE
#                 decision.skip_reason = f"GitHub API HTTP error: {status}"
#             return None
            
#         except httpx.TimeoutException:
#             # Prevents the worker from hanging forever
#             decision.outcome = SubmoduleOutcome.INACCESSIBLE
#             decision.skip_reason = "GitHub API connection timed out."
#             logger.error("Timeout fetching metadata for %s/%s", owner, repo)
#             return None
            
#         except httpx.RequestError as e:
#             # Catches DNS failures, connection resets, etc., WITHOUT catching core Python exceptions
#             decision.outcome = SubmoduleOutcome.INACCESSIBLE
#             decision.skip_reason = "Network routing/connection error."
#             logger.error("Network error fetching metadata for %s/%s: %s", owner, repo, e)
#             return None

#     # ── Node 3: Classification ────────────────────────────────────────────────

#     def _classify_internal(self, submodule_owner: str, is_relative_url: bool) -> bool:
#         """
#         Returns True if this submodule is internal to the parent organization.

#         Priority:
#         1. Relative URL → always internal
#         2. Same owner namespace → internal
#         3. Known sibling org family → internal
#         4. Known OSS blocklist → external
#         5. Different namespace, unknown org → external
#         """
#         # Rule 1: Relative URL is authoritative
#         if is_relative_url:
#             return True

#         owner_lower = submodule_owner.lower()

#         # Rule 4: OSS blocklist → external (check before same-owner to handle edge cases)
#         if owner_lower in OSS_BLOCKLIST:
#             return False

#         # Rule 2: Same owner namespace
#         if submodule_owner.lower() == self.parent_owner.lower():
#             return True

#         # Rule 3: Sibling org family
#         if self.parent_org_family:
#             submodule_family = ORG_FAMILY_MAP.get(owner_lower)
#             if submodule_family and submodule_family == self.parent_org_family:
#                 return True

#         return False

#     # ── Node 4A: External Sub-Tree ────────────────────────────────────────────

#     async def _node_4a_external(
#         self, decision: SubmoduleDecision, metadata: RepoMetadata
#     ) -> SubmoduleDecision:
#         owner = decision.resolved_owner
#         repo = decision.resolved_repo

#         # Step A1: Does this repo already have its own Code Wiki?
#         repo_key = f"{owner}/{repo}"
#         if repo_key in self.already_indexed:
#             decision.outcome = SubmoduleOutcome.EXTERNAL_CROSS_LINK
#             decision.linked_repo_id = self.already_indexed[repo_key]
#             decision.skip_reason = f"External dependency with existing wiki: {repo_key}"
#             return decision

#         # Step A2: Has package manager manifest? → library reference
#         if await self._has_package_manifest(owner, repo, metadata.default_branch):
#             decision.outcome = SubmoduleOutcome.EXTERNAL_LIBRARY_REF
#             decision.skip_reason = (
#                 f"External library with package manifest ({owner}/{repo}). "
#                 "Documented as dependency reference, not ingested."
#             )
#             return decision

#         # Step A3: Size gate — >50MB external repos are too large to vendor-ingest
#         if metadata.size_kb > settings.SUBMODULE_EXTERNAL_MAX_SIZE_KB:
#             decision.outcome = SubmoduleOutcome.EXTERNAL_TOO_LARGE
#             decision.skip_reason = (
#                 f"External repo too large to ingest: "
#                 f"{metadata.size_kb / 1024:.1f}MB > "
#                 f"{settings.SUBMODULE_EXTERNAL_MAX_SIZE_KB / 1024:.0f}MB limit"
#             )
#             return decision

#         # Step A4: Small custom external fork — ingest with reduced priority
#         decision.outcome = SubmoduleOutcome.EXTERNAL_VENDORED
#         decision.should_initialize = True
#         decision.skip_reason = None
#         logger.info(
#             "Small external fork approved for vendored ingestion: %s/%s (%dKB)",
#             owner, repo, metadata.size_kb
#         )
#         return decision

#     # ── Node 4B: Internal Sub-Tree ────────────────────────────────────────────

#     async def _node_4b_internal(
#         self,
#         decision: SubmoduleDecision,
#         metadata: RepoMetadata,
#         processing_stack: list[str],
#     ) -> SubmoduleDecision:
#         owner = decision.resolved_owner
#         repo = decision.resolved_repo
#         repo_key = f"{owner}/{repo}"

#         # Step B1: Already indexed?
#         if repo_key in self.already_indexed:
#             decision.outcome = SubmoduleOutcome.INTERNAL_CROSS_LINK
#             decision.linked_repo_id = self.already_indexed[repo_key]
#             decision.skip_reason = "Internal repo already independently indexed — cross-linking wikis"
#             return decision

#         # Step B2: Size and complexity estimation via GitHub tree API
#         await self._estimate_complexity(decision, metadata)

#         band = decision.complexity_band

#         if band in ("small", "medium"):
#             # Approved unconditionally
#             decision.outcome = SubmoduleOutcome.INTERNAL_FULL
#             decision.should_initialize = True
#             decision.use_blob_filter = (band == "medium")
#             return decision

#         # band == "large" → Step B3: Multi-criterion scoring gate
#         return await self._node_b3_large_gate(decision, metadata, processing_stack)

#     async def _node_b3_large_gate(
#         self,
#         decision: SubmoduleDecision,
#         metadata: RepoMetadata,
#         processing_stack: list[str],
#     ) -> SubmoduleDecision:
#         """
#         B3: Large internal submodule — evaluate 3 criteria before deciding.
#         Score 0–3 across: dependency centrality, infrastructure check, staleness.
#         """
#         score = 0
#         reasons = []

#         # Criterion 1: Dependency centrality
#         # Check if parent source files actively import from this submodule
#         is_central = await self._is_dependency_central(
#             decision.entry.path, decision.resolved_owner, decision.resolved_repo
#         )
#         if is_central:
#             score += 1
#             reasons.append("central dependency (actively imported by parent)")

#         # Criterion 2: Not pure infrastructure
#         is_infra = self._is_infrastructure_submodule(
#             decision.entry.name, decision.entry.path
#         )
#         if not is_infra:
#             score += 1
#             reasons.append("not pure infrastructure")
#         else:
#             reasons.append("infrastructure/tooling submodule (reduced priority)")

#         # Criterion 3: Staleness check
#         is_fresh = await self._is_recently_updated(
#             decision.resolved_owner, decision.resolved_repo, decision.pinned_sha
#         )
#         if is_fresh:
#             score += 1
#             reasons.append("recently updated (not stale)")
#         else:
#             reasons.append("stale pinned commit (not updated recently)")

#         logger.info(
#             "B3 gate score for %s/%s: %d/3 — %s",
#             decision.resolved_owner, decision.resolved_repo,
#             score, "; ".join(reasons)
#         )

#         if score >= 2:
#             # High score → queue for background ingestion
#             decision.outcome = SubmoduleOutcome.INTERNAL_QUEUED
#             decision.should_initialize = True
#             decision.use_blob_filter = True
#             decision.skip_reason = (
#                 f"Large internal submodule queued for background ingestion "
#                 f"(score {score}/3: {'; '.join(reasons)})"
#             )
#         else:
#             # Low score → README stub only
#             decision.outcome = SubmoduleOutcome.INTERNAL_STUB
#             decision.should_initialize = False
#             decision.skip_reason = (
#                 f"Large internal submodule documented as stub "
#                 f"(score {score}/3: {'; '.join(reasons)})"
#             )

#         return decision

#     # ── B3 Criterion Helpers ──────────────────────────────────────────────────

#     async def _estimate_complexity(
#         self, decision: SubmoduleDecision, metadata: RepoMetadata
#     ) -> None:
#         """Estimate source files using the Tree API without downloading content."""
#         try:
#             tree_entries = await self.gh.get_full_tree(
#                 decision.resolved_owner,
#                 decision.resolved_repo,
#                 metadata.default_branch,
#             )
#         except httpx.HTTPError as e:
#             # Only catch HTTP/Network errors. If it's a JSON decode error or MemoryError, let it crash up to evaluate_all
#             logger.warning(
#                 "Failed to fetch tree for submodule estimation %s/%s: %s. Defaulting to 'small'.", 
#                 decision.resolved_owner, decision.resolved_repo, e
#             )
#             decision.complexity_band = "small"
#             return

#         # Count only Tier 1 source files
#         from src.services.pre_clone.file_filter import LANGUAGE_MAP, FileTier
#         source_extensions = {
#             ext for ext, (_, tier) in LANGUAGE_MAP.items()
#             if tier == FileTier.TIER1_SOURCE
#         }

#         source_file_count = 0
#         source_byte_count = 0
#         for entry in tree_entries:
#             if entry.type == "blob":
#                 ext = "." + entry.path.rsplit(".", 1)[-1] if "." in entry.path else ""
#                 if ext.lower() in source_extensions:
#                     source_file_count += 1
#                     source_byte_count += entry.size or 0

#         decision.estimated_source_files = source_file_count
#         decision.estimated_source_bytes = source_byte_count

#         # Band thresholds (approximate lines of code):
#         # 500 bytes/file average → 10K files ≈ 5M bytes
#         if source_file_count < 500 or source_byte_count < 500_000:
#             band = "small"
#         elif source_file_count < 5_000 or source_byte_count < 5_000_000:
#             band = "medium"
#         else:
#             band = "large"

#         decision.complexity_band = band
#         logger.info(
#             "Complexity estimate: %s/%s → %s (%d source files, %dKB)",
#             decision.resolved_owner, decision.resolved_repo,
#             band, source_file_count, source_byte_count // 1024,
#         )

#     async def _is_dependency_central(
#         self, submodule_path: str, sub_owner: str, sub_repo: str
#     ) -> bool:
#         """
#         Check whether parent source code imports from this submodule.
#         We don't have the parent's source in memory here — this is a lightweight
#         heuristic: if the submodule path appears in any recently changed files
#         in the parent's commit history, it's being actively used.

#         A real implementation would grep the parent's materialized files,
#         but at decision-tree time we may not have them on disk yet.
#         """
#         try:
#             commits = await self.gh.get_recent_commits(
#                 self.parent_owner, self.parent_repo, self.parent_branch, count=20
#             )
#             # Check the top 5 commits' changed file lists
#             for commit in commits[:5]:
#                 files = await self.gh.get_commit_files(
#                     self.parent_owner, self.parent_repo, commit.sha
#                 )
#                 if any(submodule_path in f for f in files):
#                     return True
#             return False
        
#         except httpx.HTTPError as e:
#             logger.warning("Network error checking dependency centrality: %s", e)
#             return False  # Conservative: if we can't prove it's central, assume it isn't

#     def _is_infrastructure_submodule(self, name: str, path: str) -> bool:
#         """Check if the submodule is build tooling or CI infrastructure."""
#         infra_indicators = {
#             "build", "ci", "deploy", "infra", "toolchain", "scripts",
#             "cmake", "bazel", "tools", "devtools", "makefiles", "actions",
#         }
#         combined = f"{name.lower()} {path.lower()}"
#         return any(indicator in combined for indicator in infra_indicators)

#     async def _is_recently_updated(
#         self, owner: str, repo: str, pinned_sha: Optional[str]
#     ) -> bool:
#         """
#         Check if the pinned commit is recent (within 6 months).
#         Uses GitHub commit timestamp API.
#         """
#         if not pinned_sha:
#             return False
#         try:
            
#             commits = await self.gh.get_recent_commits(owner, repo, pinned_sha, count=1)
#             if commits:
#                 timestamp = datetime.fromisoformat(
#                     commits[0].timestamp.replace("Z", "+00:00")
#                 )
#                 cutoff = datetime.now(timezone.utc) - timedelta(days=180)
#                 return timestamp > cutoff
            
#         except (httpx.HTTPError, ValueError) as e:
#             # Catch ValueError strictly for datetime parsing failures
#             logger.warning("Failed to verify staleness for %s/%s: %s", owner, repo, e)
            
#         return False

#     # ── Package Manifest Detection ────────────────────────────────────────────

#     async def _has_package_manifest(
#         self, owner: str, repo: str, branch: str
#     ) -> bool:
#         """
#         Check if the submodule repo has a package manager manifest at its root.
#         These signal "this is a published library, not application code."
#         Fetches only the root tree — no file content.
#         """
#         PACKAGE_MANIFESTS = {
#             "package.json", "setup.py", "pyproject.toml", "Cargo.toml",
#             "pom.xml", "build.gradle", "go.mod", "composer.json",
#             "Gemfile", "*.gemspec", "pubspec.yaml", "mix.exs",
#         }
#         try:
#             root_files = await self.gh._get_root_file_list(owner, repo, branch)
#             return bool(root_files & PACKAGE_MANIFESTS)
#         except httpx.HTTPError as e:
#             logger.debug("Network error checking package manifest for %s/%s: %s", owner, repo, e)
#             return False