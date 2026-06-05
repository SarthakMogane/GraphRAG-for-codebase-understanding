"""
app/services/pre_clone/types.py
────────────────────────────────
All shared data classes and enums for the pre-clone pipeline.

Every stage in the pipeline reads and writes these objects.
Nothing in this file does I/O — pure data structures only.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class RoutingDecision(str, enum.Enum):
    """
    The three possible outcomes of the routing check.
    Decided before any GitHub API call.
    """
    SERVE_CACHE   = "serve_cache"    # Pre-built wiki exists → serve immediately
    NEW_INGESTION = "new_ingestion"  # Never seen → queue full ingestion
    REFRESH       = "refresh"        # Seen before, new commits → diff-aware refresh


class ValidationVerdict(str, enum.Enum):
    """
    Final verdict from the validation pipeline.
    APPROVED = proceed to clone. Everything else = stop with reason.
    """
    APPROVED             = "approved"
    REPO_NOT_FOUND       = "repo_not_found"
    REPO_PRIVATE         = "repo_private"
    REPO_PUBLIC          = "repo_public"
    REPO_ARCHIVED        = "repo_archived"
    REPO_DISABLED        = "repo_disabled"
    REPO_EMPTY           = "repo_empty"
    REPO_FORK_REDIRECTED = "repo_fork_redirected"   # Fork → redirected to upstream
    FORK_DIVERGED        = "fork_diverged"           # Fork diverged enough → index independently
    RATE_LIMIT_BLOCKED   = "rate_limit_blocked"
    URL_PARSE_ERROR      = "url_parse_error"
    ALREADY_PROCESSING   = "already_processing"     # Another job is actively running


class MonorepoTooling(str, enum.Enum):
    """Which monorepo tooling was detected in the repository root."""
    NONE        = "none"          # Not a monorepo (or undetectable)
    NX          = "nx"            # Nx workspace (nx.json)
    TURBOREPO   = "turborepo"     # Turborepo (turbo.json)
    RUSH        = "rush"          # Rush (rush.json)
    PNPM        = "pnpm"          # pnpm workspaces (pnpm-workspace.yaml)
    YARN        = "yarn"          # Yarn workspaces (package.json#workspaces)
    NPM         = "npm"           # npm workspaces (package.json#workspaces)
    LERNA       = "lerna"         # Lerna (lerna.json)
    GRADLE      = "gradle"        # Gradle multi-project (settings.gradle)
    BAZEL       = "bazel"         # Bazel (WORKSPACE file)
    CARGO       = "cargo"         # Cargo workspace (Cargo.toml#workspace)
    GO_MODULES  = "go_modules"    # Multiple go.mod files (inferred)
    INFERRED    = "inferred"      # No tooling config, but structure inferred

# ─────────────────────────────────────────────────────────────────────────────
# Validation Pipeline Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParsedURL:
    """Result of parsing a raw GitHub URL string."""
    raw: str
    owner: str
    repo: str
    is_valid: bool
    error: Optional[str] = None

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"


@dataclass
class RateLimitStatus:
    """
    Current state of the GitHub API rate limit.
    Fetched once at the start of each validation run.
    """
    remaining: int
    limit: int
    reset_at: datetime
    used: int

    @property
    def is_safe(self) -> bool:
        """True if we have enough calls to safely run the full validation pipeline."""
        # Validation pipeline uses at most 5 API calls per repo.
        # Buffer provides an additional safety margin.
        from src.core.config import get_settings
        return self.remaining >= get_settings().GITHUB_API_RATE_LIMIT_BUFFER

    @property
    def percent_used(self) -> float:
        return (self.used / self.limit) * 100 if self.limit else 0.0


@dataclass
class ForkInfo:
    """Details about a forked repository's relationship to its upstream."""
    is_fork: bool
    upstream_owner: Optional[str] = None
    upstream_repo: Optional[str] = None
    upstream_github_id: Optional[int] = None
    upstream_indexed: bool = False       # True if upstream is in our index
    upstream_repo_db_id: Optional[int] = None  # Our DB id for the upstream
    # Divergence metrics (populated only if is_fork=True and upstream is indexed)
    commits_ahead: int = 0               # Commits fork has that upstream doesn't
    commits_behind: int = 0              # Commits upstream has that fork doesn't
    is_diverged: bool = False            # True if fork has meaningful unique content


@dataclass
class StaleCheckResult:
    """Result of comparing our stored SHA against the live HEAD."""
    is_stale: bool
    stored_sha: Optional[str]            # What we indexed last time
    live_sha: Optional[str]              # Current HEAD on default branch
    commits_since_last_index: int = 0
    last_indexed_at: Optional[datetime] = None


@dataclass
class ValidationResult:
    """
    Complete output of the validation pipeline for one repository URL.
    This is the handoff document to the routing/scheduling layer.
    """
    # Input
    parsed_url: ParsedURL

    # Decision
    verdict: ValidationVerdict
    routing: Optional[RoutingDecision] = None

    # GitHub metadata (populated on successful validation)
    github_id: Optional[int] = None
    default_branch: Optional[str] = None
    primary_language: Optional[str] = None
    size_kb: Optional[int] = None
    description: Optional[str] = None
    topics: list[str] = field(default_factory=list)
    has_submodules: bool = False
    uses_git_lfs: bool = False

    # Check sub-results
    rate_limit: Optional[RateLimitStatus] = None
    fork_info: Optional[ForkInfo] = None
    stale_check: Optional[StaleCheckResult] = None

    # Existing DB record (if any)
    existing_repo_db_id: Optional[int] = None
    existing_job_status: Optional[str] = None   # e.g. "running" — blocks re-queue

    # Human-readable message for API response
    message: str = ""

    @property
    def should_proceed(self) -> bool:
        return self.verdict == ValidationVerdict.APPROVED

    @property
    def is_fork_redirect(self) -> bool:
        return self.verdict == ValidationVerdict.REPO_FORK_REDIRECTED


# ─────────────────────────────────────────────────────────────────────────────
# Monorepo Detection Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SubProjectScore:
    """
    Scoring breakdown for a single sub-project within a monorepo.
    Every score component is stored so decisions are auditable.
    """
    path: str                               # Relative path from repo root
    name: str                               # Package/project name

    # Raw signals
    source_file_count: int = 0
    source_byte_count: int = 0
    has_own_manifest: bool = False          # Has its own package.json / go.mod etc.
    has_own_readme: bool = False
    has_entry_point: bool = False           # Contains main.*, index.*, app.*, server.*
    is_deployable: bool = False             # Produces a deployable artifact (app/service)
    is_library: bool = False                # Produces a reusable library
    recent_commit_count: int = 0            # Commits in last 90 days touching this path
    dependent_count: int = 0               # How many other sub-projects import from this
    depth: int = 1                          # Directory nesting depth from repo root

    # Computed
    composite_score: float = 0.0
    # ── NEW: UI Recommendation Fields ───────────────────────────────────────
    recommended_action: str = "EXCLUDE"     # Expected values: "INCLUDE" | "EXCLUDE"
    recommendation_reason: Optional[str] = None


@dataclass
class MonorepoDetectionResult:
    """
    Complete result of the monorepo detection + sub-project scoring system.
    Consumed by CloneStrategySelector to build sparse checkout dir list.
    """
    is_monorepo: bool
    tooling: MonorepoTooling = MonorepoTooling.NONE

    # All discovered sub-projects with their scores
    all_subprojects: list[SubProjectScore] = field(default_factory=list)

    # Filtered lists derived from decisions
    approved_dirs: list[str] = field(default_factory=list)    # Sparse checkout set
    stub_dirs: list[str] = field(default_factory=list)         # Record but don't deep-parse
    skipped_dirs: list[str] = field(default_factory=list)      # Don't materialize

    # Package dependency graph (subproject name → list of names it imports from)
    dependency_graph: dict[str, list[str]] = field(default_factory=dict)

    # Config file that revealed the structure (for audit trail)
    detected_via: Optional[str] = None     # e.g. "nx.json", "turbo.json", "inferred"
    detection_confidence: float = 1.0      # 0.0–1.0 (inferred = lower confidence)
    warnings: list[str] = field(default_factory=list)

    @property
    def approved_count(self) -> int:
        return len(self.approved_dirs)

    @property
    def total_subproject_count(self) -> int:
        return len(self.all_subprojects)
    
