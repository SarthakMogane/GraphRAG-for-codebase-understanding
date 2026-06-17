# Structural change detection
# 2 GitHub API calls — no clone, no disk
# ─────────────────────────────────────────────────────────────────────────────
 
import hashlib
from typing import Optional
from src.schemas.responses import IndexResponse
from src.models.database import RepoStatus
from src.services.github import GitHubService
from src.core.logger import get_logger
from src.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


_STRUCTURAL_FILES = frozenset({
    # ── 1. External Source Pointer ──
    ".gitmodules",

    # ── 2. Global Monorepo Tooling Indicators ──
    "nx.json", "turbo.json", "rush.json", "lerna.json",
    "pnpm-workspace.yaml", "pnpm-workspace.yml",
    "WORKSPACE", "WORKSPACE.bazel", "go.work",

    # ── 3. Ecosystem Subproject Manifests ──
    "package.json",         # Node.js / TypeScript Workspaces
    "Cargo.toml",           # Rust Crates & Workspaces
    "go.mod",               # Go Modules
    "pyproject.toml",       # Modern Python Workspaces (uv, Poetry, Hatch)
    "setup.py",             # Legacy Python Packages
    "uv.toml",              # Explicit Astral uv layout configuration
    "pom.xml",              # Java Maven Projects
    "build.gradle",         # Java / Kotlin Gradle Groovy configuration
    "build.gradle.kts",     # Java / Kotlin Gradle Kotlin DSL configuration
    "settings.gradle",      # Multi-project JVM root mapping
    "settings.gradle.kts",  # Multi-project JVM root DSL mapping
    "composer.json",        # PHP Composer Packages
    "pubspec.yaml",         # Flutter / Dart Apps
    "mix.exs",              # Elixir Mix Applications
    "deno.json",            # Deno Runtime Environments
    "deno.jsonc"            # Deno Runtime Environments with Comments
})

_MONOREPO_DIR_SIGNALS = frozenset({
    "apps", "packages", "services", "libs", "modules", "plugins", "src", "components"
})

async def _check_structural_change(
    owner: str,
    repo: str,
    branch: str,
    current_sha: str,
    previous_sha: str,
    installation_id: int,
    _gh:GitHubService,
) -> tuple[bool, Optional[dict]]:
    """
    Compare root file structure between current HEAD and last scout SHA.
 
    Returns (is_structural: bool, diff: dict | None)
    diff shape:
      {
        "new_files":     ["nx.json"],
        "removed_files": [],
        "new_dirs":      ["packages"],
        "gitmodules_changed": True,
      }
 
    Fails safe: if API calls fail, returns (False, None) so we don't
    force a re-scout on every network blip.
    """
    try:
        current_root  = await _gh.get_full_tree(owner, repo, current_sha,  installation_id)
        previous_root = await _gh.get_full_tree(owner, repo, previous_sha, installation_id)
    except Exception as e:
        logger.warning(
            "Structural check API call failed for %s/%s: %s — treating as code-only",
            owner, repo, e,
        )
        return False, None
 
    current_structural_paths = {
        entry.path for entry in current_root 
        if entry.path.split('/')[-1] in _STRUCTURAL_FILES
    }
    previous_structural_paths = {
        entry.path for entry in previous_root 
        if entry.path.split('/')[-1] in _STRUCTURAL_FILES
    }

    new_files = sorted(current_structural_paths - previous_structural_paths)
    removed_files = sorted(previous_structural_paths - current_structural_paths)
 
    # New monorepo top-level directories
    # Isolates pure first-tier anchors (e.g., "apps", "packages")
    current_dirs = {
        entry.path for entry in current_root 
        if entry.type == "tree" and '/' not in entry.path
    } & _MONOREPO_DIR_SIGNALS

    previous_dirs = {
        entry.path for entry in previous_root 
        if entry.type == "tree" and '/' not in entry.path
    } & _MONOREPO_DIR_SIGNALS

    new_dirs = sorted(current_dirs - previous_dirs)
 
    # .gitmodules content change (if it existed in both)
    gitmodules_changed = False
    if ".gitmodules" in current_structural_paths and ".gitmodules" in previous_structural_paths:
        gitmodules_changed = await _gitmodules_content_changed(
            owner, repo, branch, previous_sha, installation_id, _gh
        )
 
    is_structural = bool(new_files or removed_files or new_dirs or gitmodules_changed)
 
    if not is_structural:
        return False, None
 
    diff = {
        "new_files":          new_files,
        "removed_files":      removed_files,
        "new_dirs":           new_dirs,
        "gitmodules_changed": gitmodules_changed,
    }
    logger.info(
        "Structural change detected in %s/%s: %s", owner, repo, diff
    )
    return True, diff



async def _gitmodules_content_changed(
    owner: str,
    repo: str,
    branch: str,
    previous_sha: str,
    installation_id: int,
    _gh:GitHubService,
) -> bool:
    """
    Check if .gitmodules content changed between previous_sha and current branch.
    Compares SHA-256 of file content. Returns True if changed.
    Fails safe: returns True (assume changed) if either fetch fails.
    """
    try:
        current_file = await _gh.get_file_content(
            owner, repo, ".gitmodules", installation_id, params={"ref": branch}
        )
        previous_file = await _gh.get_file_content(
            owner, repo, ".gitmodules", installation_id, params={"ref": previous_sha}
        )

        if current_file is None and previous_file is None:
            return False
        if current_file is None or previous_file is None:
            return True   # One side missing → structural change
        
        # Compare the native GitHub Blob SHAs. 
        # If the file contents differ by even one byte, these SHAs will not match.
        return current_file.get("sha") != previous_file.get("sha")
    
    except Exception as e:
        logger.warning(
            "Could not compare .gitmodules for %s/%s: %s — assuming changed",
            owner, repo, e,
        )
        return True   # Fail safe
    

# REFRESH handler — the complex case
# ─────────────────────────────────────────────────────────────────────────────
 
async def _handle_refresh(
    repo:dict,
    repo_id: int,
    result,
    installation_id: int,
    conn,
) -> IndexResponse:
    """
    Handle a stale repo. Decides between:
      - Silent re-ingest (code-only change, saved selection exists)
      - Structural re-scout (structure changed, show updated checklist)
    """
    stale = result.stale_check
 
    # ── Check if we have saved selections from last time ──────────────────────
    saved_selection = await conn.fetchrow(
        """
            SELECT id, selected_subprojects, selected_submodules
            FROM user_selections
            WHERE repo_id = $1 

        """,
        repo_id
    )
 
    # No saved selections → must go through scout regardless
    if not saved_selection:
        logger.info(
            "%s is stale but has no saved selections — routing to scout",
            repo["full_name"],
        )
        return IndexResponse(
            repo_id=repo_id,
            next="scout",
            message=(
                f"{repo.full_name} has new commits. "
                f"Scanning structure before indexing."
            ),
        )
 
    # ── Check for structural change ───────────────────────────────────────────
    # Cost: 2 API calls (root tree at current SHA + root tree at last scout SHA)
    # Only run if we have both SHAs to compare
    current_sha   = stale.live_sha if stale else None
    previous_sha  = repo["last_scout_sha"]
 
    structural_diff = None
    is_structural   = False
 
    if current_sha and previous_sha and current_sha != previous_sha:
        is_structural, structural_diff = await _check_structural_change(
            owner=repo["github_owner"],
            repo=repo["github_repo"],
            branch=repo["default_branch"],
            current_sha=current_sha,
            previous_sha=previous_sha,
            installation_id=installation_id,
        )
 
    # ── Structural change → re-scout with diff ────────────────────────────────
    if is_structural:
        logger.info(
            "%s has structural changes — routing to scout. diff=%s",
            repo["full_name"], ["structural_diff"],
        )
        await conn.execute(
            "UPDATE repos SET index_status = 'pending', updated_at = NOW() WHERE id = $1", 
            repo_id
        )
        
        return IndexResponse(
            repo_id=repo_id,
            next="scout",
            message=(
                f"{repo["full_name"]} has structural changes since last index. "
                f"Review the updated structure below."
            ),
            structural_diff=structural_diff,
            # Frontend uses structural_diff to show NEW/REMOVED badges
            # on the checklist. Previous selections are pre-filled.
        )
 
    # ── Code-only change → silent re-ingest with saved selections ────────────
    logger.info(
        "%s has code-only changes (%d commits) — silent re-ingest",
        repo["full_name"],
        stale.commits_since_last_index if stale else 0,
    )
 
    
    commit_count = stale.commits_since_last_index if stale else 0
    return IndexResponse(
        repo_id=repo_id,
        next="ingest",
        message=(
            f"{repo.full_name} has {commit_count} new commit(s). "
            f"Re-indexing with your saved configuration."
        ),
        job_id=job_id,
    )