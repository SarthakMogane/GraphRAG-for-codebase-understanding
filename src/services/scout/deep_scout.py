"""
app/services/deep_scout.py
───────────────────────────
PHASE 1: The Deep Scout

Runs entirely via GitHub API — zero disk, zero cloning, zero user waiting.
Produces a RepoScoutResult that the frontend renders as the Phase 2
nested checklist UI (subprojects + submodules, pre-scored and pre-selected).

What it discovers in one pass (all API, no clone):
  ┌─ Parent repo
  │    ├── Is it a monorepo?
  │    │    └── All subprojects with composite scores + auto-selection
  │    └── Has .gitmodules?
  │         └── Each submodule (Level 1) — user can toggle
  │              ├── Public?            → PUBLIC_SKIPPED  (no toggle)
  │              ├── App not installed? → INSTALL_REQUIRED (action button)
  │              ├── Already indexed?   → PRIVATE_CROSS_LINK (auto-checked)
  │              ├── Is it a monorepo?  → its subprojects with scores
  │              └── Has .gitmodules?   → Level 2 (auto-scored, no user toggle)
  └─ Dependency edge list for Phase 4 GraphRAG

How it authenticates:
  Every API call goes through GitHubService methods which use the
  installation_id to get an installation access token. No _headers()
  calls anywhere — all auth is handled by GitHubService internally.

Latency design:
  - Parent metadata + root files fetched in 2 parallel API calls
  - Parent monorepo detection runs parallel to submodule scout
  - All Level 1 submodule metadata fetched concurrently (asyncio.gather)
  - Per submodule: root files + mono detection + size all run in parallel
  - Level 2 submodule scout runs in parallel per Level 1 parent
  - InstallationCache: 1 API call per org, shared for the whole run
  Expected total latency: 2–6 seconds for most repos
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

import httpx

from src.core.config import get_settings
from src.models.database import SubmoduleOutcome
from src.services.github import GitHubService, InstallationCache
from src.services.pre_clone.types import MonorepoDetectionResult, SubProjectDecision

logger = logging.getLogger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Result data structures — map 1:1 to the frontend checklist
# ─────────────────────────────────────────────────────────────────────────────

# @dataclass
# class SubprojectNode:
#     """
#     One package/app/lib inside a monorepo.
#     Rendered as a checkbox row with a score badge in the Phase 2 UI.
#     auto_selected = True means the checkbox is pre-checked.
#     """
#     path: str
#     name: str
#     score: float
#     auto_selected: bool
#     source_file_count: int
#     has_entry_point: bool
#     dependent_count: int
#     recent_commit_count: int
#     skip_reason: Optional[str] = None


# @dataclass
# class SubmoduleNode:
#     """
#     One entry from .gitmodules — fully resolved and classified.

#     user_can_toggle:
#       depth=1 → True  → user sees a checkbox
#       depth=2 → False → system auto-decides via B3 score, no UI toggle

#     action_required:
#       True when outcome=INSTALL_REQUIRED
#       Frontend shows a "Grant Access to {org}" button instead of a checkbox
#     """
#     path: str
#     name: str
#     resolved_owner: Optional[str]
#     resolved_repo: Optional[str]
#     resolved_url: Optional[str]
#     pinned_sha: Optional[str]
#     outcome: SubmoduleOutcome
#     is_private: Optional[bool]
#     depth: int
#     auto_selected: bool
#     user_can_toggle: bool
#     action_required: bool
#     action_label: Optional[str]
#     action_url: Optional[str]
#     skip_reason: Optional[str]
#     is_monorepo: bool = False
#     monorepo_tooling: Optional[str] = None
#     subprojects: list[SubprojectNode] = field(default_factory=list)
#     nested_submodules: list[SubmoduleNode] = field(default_factory=list)
#     complexity_band: Optional[str] = None
#     estimated_source_files: int = 0
#     linked_repo_id: Optional[int] = None


# @dataclass
# class RepoScoutResult:
#     """
#     Complete Phase 1 output. Single contract between backend and frontend.

#     - Frontend renders this as the nested checklist (Phase 2 UI)
#     - Stored in repo_scout_results table (keyed by repo_id + head_sha)
#     - ingestion_task reads approved_dirs and selected_submodules from it
#     - Never re-computed if head_sha hasn't changed (cache hit → instant)
#     """
#     owner: str
#     repo: str
#     default_branch: str
#     github_id: int
#     size_kb: int
#     primary_language: Optional[str]
#     is_monorepo: bool
#     monorepo_tooling: Optional[str]
#     subprojects: list[SubprojectNode]
#     submodules: list[SubmoduleNode]
#     dependency_edges: list[dict]
#     total_submodules: int
#     private_accessible: int
#     install_required: int
#     public_skipped: int
#     auto_selected_count: int
#     scout_duration_ms: int
#     api_calls_made: int


# ─────────────────────────────────────────────────────────────────────────────
# The Deep Scout
# ─────────────────────────────────────────────────────────────────────────────

class DeepScout:
    """
    Phase 1 engine. Produces a RepoScoutResult using only GitHub API calls.

    One instance per scout run.

    The InstallationCache can be passed in from outside so that Phase 3
    (SubmoduleDecisionTree) shares the same cache — org installation
    lookups are never repeated across phases.

    Usage:
        cache  = InstallationCache(gh, installation_id)
        scout  = DeepScout(gh, installation_id, already_indexed, install_cache=cache)
        result = await scout.run("myorg", "myapp", "main")
    """

    def __init__(
        self,
        github_service: GitHubService,
        installation_id: int,
        already_indexed_repos: dict[str, int],
        install_cache: Optional[InstallationCache] = None,
    ):
        self.gh              = github_service
        self.installation_id = installation_id
        self.already_indexed = already_indexed_repos
        self.install_cache   = install_cache or InstallationCache(
            github_service, installation_id
        )
        self._api_calls = 0

    async def run(self, owner: str, repo: str, branch: str) -> RepoScoutResult:
        """
        Execute the full Phase 1 scout.
        All API calls; no disk access; no git clone.
        Returns RepoScoutResult ready for Phase 2 UI rendering.
        """
        import time
        start = time.monotonic()

        # ── Parent metadata + root files in parallel ──────────────────────────
        meta_task = asyncio.create_task(
            self.gh.get_repo_metadata(owner, repo, self.installation_id)
        )
        root_task = asyncio.create_task(
            self.gh.get_root_files(owner, repo, branch, self.installation_id)
        )
        metadata, root_files = await asyncio.gather(meta_task, root_task)
        self._api_calls += 2

#         # ── Monorepo detection + Level 1 submodule scout in parallel ──────────
#         mono_task = asyncio.create_task(
#             self._detect_monorepo(owner, repo, branch, root_files)
#         )
#         sub_task = asyncio.create_task(
#             self._scout_submodules(owner, repo, branch, root_files, depth=1)
#         )
#         mono_result, submodule_nodes = await asyncio.gather(mono_task, sub_task)

#         subproject_nodes = self._build_subproject_nodes(mono_result)
#         edges = self._build_dependency_edges(owner, repo, submodule_nodes, mono_result)
#         all_flat = self._flatten_submodules(submodule_nodes)

#         INDEXED_OUTCOMES = {
#             SubmoduleOutcome.PRIVATE_FULL,
#             SubmoduleOutcome.PRIVATE_MONOREPO,
#             SubmoduleOutcome.PRIVATE_QUEUED,
#             SubmoduleOutcome.PRIVATE_CROSS_LINK,
#         }

#         elapsed_ms = int((time.monotonic() - start) * 1000)

#         return RepoScoutResult(
#             owner=owner,
#             repo=repo,
#             default_branch=branch,
#             github_id=metadata.github_id,
#             size_kb=metadata.size_kb,
#             primary_language=metadata.primary_language,
#             is_monorepo=mono_result.is_monorepo if mono_result else False,
#             monorepo_tooling=(
#                 mono_result.tooling.value
#                 if mono_result and mono_result.is_monorepo else None
#             ),
#             subprojects=subproject_nodes,
#             submodules=submodule_nodes,
#             dependency_edges=edges,
#             total_submodules=len(all_flat),
#             private_accessible=sum(1 for s in all_flat if s.outcome in INDEXED_OUTCOMES),
#             install_required=sum(
#                 1 for s in all_flat if s.outcome == SubmoduleOutcome.INSTALL_REQUIRED
#             ),
#             public_skipped=sum(
#                 1 for s in all_flat if s.outcome == SubmoduleOutcome.PUBLIC_SKIPPED
#             ),
#             auto_selected_count=(
#                 sum(1 for s in all_flat if s.auto_selected) +
#                 sum(1 for sp in subproject_nodes if sp.auto_selected)
#             ),
#             scout_duration_ms=elapsed_ms,
#             api_calls_made=self._api_calls + self.install_cache.call_count,
#         )

#     # ─────────────────────────────────────────────────────────────────────────
#     # Submodule scouting
#     # ─────────────────────────────────────────────────────────────────────────

#     async def _scout_submodules(
#         self,
#         owner: str,
#         repo: str,
#         branch: str,
#         root_files: set[str],
#         depth: int,
#     ) -> list[SubmoduleNode]:
#         """
#         Fetch and classify all submodules for a repo via API.
#         All sibling submodules are classified concurrently.
#         Uses GitHubService.get_file_content() — no raw httpx calls.
#         """
#         if ".gitmodules" not in root_files:
#             return []

#         gitmodules_content = await self.gh.get_file_content(
#             owner, repo, ".gitmodules", branch, self.installation_id
#         )
#         self._api_calls += 1

#         if not gitmodules_content:
#             return []
          
#         # Use the production GitmodulesParser (configparser-based, size-limited)
#         from src.services.pre_clone.submodule_decision_tree import GitmodulesParser
#         if len(gitmodules_content.encode()) > GitmodulesParser.MAX_GITMODULES_SIZE:
#             return [] 
        
#         entries = GitmodulesParser().parse(gitmodules_content)

#         if not entries.is_valid_url:
#             return SubmoduleNode(
#                 path=entries.path,
#                 name=entries.name,

#                 resolved_owner=None,
#                 resolved_repo=None,
#                 resolved_url=None,
#                 pinned_sha=None,

#                 outcome=SubmoduleOutcome.INACCESSIBLE,
#                 is_private=None,
#                 depth=depth,

#                 auto_selected=False,
#                 user_can_toggle=(depth == 1),

#                 action_required=False,
#                 action_label=None,
#                 action_url=None,

#                 skip_reason=(
#                     entries.url_error
#                     or f"Invalid GitHub URL: {entries.raw_url}"
#                 ),
#             )

#         tasks = [
#             asyncio.create_task(self._classify_submodule(entry, depth))
#             for entry in entries
#         ]
#         raw = await asyncio.gather(*tasks, return_exceptions=True)

#         nodes = []
#         for i, result in enumerate(raw):
#             if isinstance(result, Exception):
#                 logger.warning("Scout error for '%s': %s", entries[i].path, result)
#                 nodes.append(SubmoduleNode(
#                     path=entries[i].path, name=entries[i].name,
#                     resolved_owner=None, resolved_repo=None,
#                     resolved_url=None, pinned_sha=None,
#                     outcome=SubmoduleOutcome.INACCESSIBLE,
#                     is_private=None, depth=depth,
#                     auto_selected=False, user_can_toggle=False,
#                     action_required=False, action_label=None, action_url=None,
#                     skip_reason=f"Scout error: {result}",
#                 ))
#             else:
#                 nodes.append(result)

#         return nodes

#     async def _classify_submodule(
#         self,
#         entry,
#         depth: int,
#     ) -> SubmoduleNode:
#         """
#         Classify one .gitmodules entry. Full pipeline:
#           resolve URL → depth gate → fetch metadata → public check →
#           installation check → already indexed → root files →
#           monorepo detection + size estimation + nested scout (parallel) →
#           determine outcome + auto_selected
#         """
#         # ── Resolve URL ───────────────────────────────────────────────────────
#         owner = entry.owner
#         repo = entry.repo
#         if not owner:
#             return SubmoduleNode(
#                 path=entry.path, name=entry.name,
#                 resolved_owner=None, resolved_repo=None,
#                 resolved_url=None, pinned_sha=None,
#                 outcome=SubmoduleOutcome.INACCESSIBLE,
#                 is_private=None, depth=depth,
#                 auto_selected=False, user_can_toggle=depth == 1,
#                 action_required=False, action_label=None, action_url=None,
#                 skip_reason=f"Cannot resolve URL: {entry.raw_url}",
#             )

#         repo_key = f"{owner}/{repo}"
#         base_url = entry.normalized_url or (
#                    f"https://github.com/{owner}/{repo}"
# )

#         # ── Depth gate ────────────────────────────────────────────────────────
#         if depth > settings.SUBMODULE_MAX_DEPTH:
#             return SubmoduleNode(
#                 path=entry.path, name=entry.name,
#                 resolved_owner=owner, resolved_repo=repo,
#                 resolved_url=base_url, pinned_sha=None,
#                 outcome=SubmoduleOutcome.DEPTH_EXCEEDED,
#                 is_private=None, depth=depth,
#                 auto_selected=False, user_can_toggle=False,
#                 action_required=False, action_label=None, action_url=None,
#                 skip_reason="Max depth reached",
#             )

#         # ── Fetch metadata via GitHubService (installation token auth) ────────
#         self._api_calls += 1
#         try:
#             meta = await self.gh.get_repo_metadata(owner, repo, self.installation_id)
#         except httpx.HTTPStatusError as e:
#             outcome = (
#                 SubmoduleOutcome.BROKEN_REFERENCE
#                 if e.response.status_code == 404
#                 else SubmoduleOutcome.INACCESSIBLE
#             )
#             return SubmoduleNode(
#                 path=entry.path, name=entry.name,
#                 resolved_owner=owner, resolved_repo=repo,
#                 resolved_url=base_url, pinned_sha=None,
#                 outcome=outcome, is_private=None, depth=depth,
#                 auto_selected=False, user_can_toggle=depth == 1,
#                 action_required=False, action_label=None, action_url=None,
#                 skip_reason=f"GitHub API {e.response.status_code}",
#             )

#         sub_branch = meta.default_branch

#         # ── Public → always skip ──────────────────────────────────────────────
#         if not meta.is_private:
#             return SubmoduleNode(
#                 path=entry.path, name=entry.name,
#                 resolved_owner=owner, resolved_repo=repo,
#                 resolved_url=base_url, pinned_sha=None,
#                 outcome=SubmoduleOutcome.PUBLIC_SKIPPED,
#                 is_private=False, depth=depth,
#                 auto_selected=False, user_can_toggle=False,
#                 action_required=False, action_label=None, action_url=None,
#                 skip_reason="Public OSS repository — not indexed by policy",
#             )

#         # ── Private → check GitHub App installation ───────────────────────────
#         # InstallationCache makes this 1 API call per org, not per submodule
#         install_id = await self.install_cache.get(owner)

#         if install_id is None:
#             app_slug = getattr(settings, "GITHUB_APP_SLUG", "your-app")
#             return SubmoduleNode(
#                 path=entry.path, name=entry.name,
#                 resolved_owner=owner, resolved_repo=repo,
#                 resolved_url=base_url, pinned_sha=None,
#                 outcome=SubmoduleOutcome.INSTALL_REQUIRED,
#                 is_private=True, depth=depth,
#                 auto_selected=False,
#                 user_can_toggle=False,
#                 action_required=True,
#                 action_label=f"Grant Access to '{owner}'",
#                 action_url=(
#                     f"https://github.com/apps/{app_slug}/installations/new"
#                     f"?suggested_target_id={owner}"
#                 ),
#                 skip_reason=f"GitHub App not installed on '{owner}'",
#             )

#         # ── Already indexed → cross-link ──────────────────────────────────────
#         if repo_key in self.already_indexed:
#             return SubmoduleNode(
#                 path=entry.path, name=entry.name,
#                 resolved_owner=owner, resolved_repo=repo,
#                 resolved_url=base_url, pinned_sha=None,
#                 outcome=SubmoduleOutcome.PRIVATE_CROSS_LINK,
#                 is_private=True, depth=depth,
#                 auto_selected=True,
#                 user_can_toggle=(depth == 1),
#                 action_required=False, action_label=None, action_url=None,
#                 skip_reason="Already indexed — will cross-link wikis",
#                 linked_repo_id=self.already_indexed[repo_key], # upDATE high security needed . 
#             )

#         # ── Fetch root files then run mono detection, size, nested scout
#         # all three in parallel ────────────────────────────────────────────────
#         self._api_calls += 1
#         sub_root_files = await self.gh.get_root_files(
#             owner, repo, sub_branch, self.installation_id
#         )

#         mono_task = asyncio.create_task(
#             self._detect_monorepo(owner, repo, sub_branch, sub_root_files)
#         )
#         size_task = asyncio.create_task(
#             self._estimate_size(owner, repo, sub_branch)
#         )
#         if depth < settings.SUBMODULE_MAX_DEPTH:
#             nested_task = asyncio.create_task(
#                 self._scout_submodules(
#                     owner, repo, sub_branch, sub_root_files, depth + 1
#                 )
#             )
#         else:
#             async def _empty():
#                 return []
#             nested_task = asyncio.create_task(_empty())  #update logic mislead if depth exced then what about teh task we created earlier , return ? 

#         mono_result, size_result, nested_nodes = await asyncio.gather(
#             mono_task, size_task, nested_task
#         )

#         file_count, byte_count, band = size_result
#         is_mono = mono_result.is_monorepo if mono_result else False

#         # ── Determine outcome and auto_selected ───────────────────────────────
#         if is_mono:
#             outcome       = SubmoduleOutcome.PRIVATE_MONOREPO  #update : need to chekc auto selecter . size ? 
#             auto_selected = True
#         elif band in ("small", "medium"):
#             outcome       = SubmoduleOutcome.PRIVATE_FULL
#             auto_selected = True
#         else:
#             b3 = await self._quick_b3_score(
#                 owner, repo, entry.path, entry.name, sub_branch
#             )
#             outcome = (
#                 SubmoduleOutcome.PRIVATE_QUEUED if b3 >= 2
#                 else SubmoduleOutcome.PRIVATE_STUB
#             )
#             auto_selected = b3 >= 2

#         return SubmoduleNode(
#             path=entry.path,
#             name=entry.name,
#             resolved_owner=owner,
#             resolved_repo=repo,
#             resolved_url=base_url,
#             pinned_sha=None,     # Populated in Phase 3 via git ls-tree after clone
#             outcome=outcome,
#             is_private=True,
#             depth=depth,
#             auto_selected=auto_selected,
#             user_can_toggle=(depth == 1),
#             action_required=False,
#             action_label=None,
#             action_url=None,
#             skip_reason=(
#                 None if auto_selected
#                 else "Large repo, B3 score low. You can still select it manually."
#             ),
#             is_monorepo=is_mono,
#             monorepo_tooling=(
#                 mono_result.tooling.value if is_mono and mono_result else None
#             ),
#             subprojects=self._build_subproject_nodes(mono_result),
#             nested_submodules=nested_nodes,
#             complexity_band=band,
#             estimated_source_files=file_count,
#         )

#     # ─────────────────────────────────────────────────────────────────────────
#     # Monorepo detection
#     # ─────────────────────────────────────────────────────────────────────────

#     async def _detect_monorepo(
#         self,
#         owner: str,
#         repo: str,
#         branch: str,
#         root_files: set[str],
#     ) -> Optional[MonorepoDetectionResult]:
#         """
#         Detect monorepo using GitHub API only. No clone, no disk.

#         The TOOLING_SIGNALS and STRUCTURAL_SIGNALS here are a FAST GATE —
#         they prevent calling MonorepoDetector at all when there is clearly
#         no monorepo structure. This saves 3+ API calls per non-monorepo submodule.

#         MonorepoDetector has the same signals internally but it's a heavier
#         class that fetches config files, scores subprojects, builds dependency
#         graphs. We only construct it after the gate passes.
#         """
#         TOOLING_SIGNALS = {
#             "nx.json", "turbo.json", "rush.json", "lerna.json",
#             "pnpm-workspace.yaml", "pnpm-workspace.yml",
#             "WORKSPACE", "WORKSPACE.bazel",
#         }
#         STRUCTURAL_SIGNALS = {"apps", "packages", "services", "libs", "modules"}

#         if not (root_files & TOOLING_SIGNALS) and not (root_files & STRUCTURAL_SIGNALS):
#             return None   # Fast exit — not a monorepo, skip MonorepoDetector entirely

#         try:
#             from src.services.pre_clone.monorepo_detector import MonorepoDetector
#             detector = MonorepoDetector(
#                 gh=self.gh,
#                 installation_id=self.installation_id,
#             )
#             result = await detector.detect(
#                 owner=owner,
#                 repo=repo,
#                 default_branch=branch,
#                 root_files=root_files,
#                 recent_commit_paths=[],
#             )
#             self._api_calls += 3
#             return result if result.is_monorepo else None
#         except Exception as e:
#             logger.warning(
#                 "Monorepo detection failed for %s/%s: %s", owner, repo, e
#             )
#             return None

#     # ─────────────────────────────────────────────────────────────────────────
#     # Size estimation
#     # ─────────────────────────────────────────────────────────────────────────

#     async def _estimate_size(
#         self, owner: str, repo: str, branch: str
#     ) -> tuple[int, int, str]:
#         """
#         Returns (source_file_count, source_byte_count, complexity_band).
#         Uses GitHubService.get_full_tree() — no content download, paths+sizes only.
#         """
#         try:
#             tree = await self.gh.get_full_tree(owner, repo, branch, self.installation_id)
#             self._api_calls += 1

#             from src.services.pre_clone.file_filter import LANGUAGE_MAP, FileTier
#             source_exts = {
#                 ext for ext, (_, tier) in LANGUAGE_MAP.items()
#                 if tier == FileTier.TIER1_SOURCE
#             }

#             count = 0
#             total_bytes = 0
#             for entry in tree:
#                 if entry.type == "blob":
#                     ext = "." + entry.path.rsplit(".", 1)[-1] if "." in entry.path else ""
#                     if ext.lower() in source_exts:
#                         count += 1
#                         total_bytes += entry.size or 0

#             if count < 500 or total_bytes < 500_000:
#                 band = "small"
#             elif count < 5_000 or total_bytes < 5_000_000:
#                 band = "medium"
#             else:
#                 band = "large"

#             return count, total_bytes, band
#         except Exception:
#             return 0, 0, "small"

#     # ─────────────────────────────────────────────────────────────────────────
#     # Quick B3 score (Phase 1 approximation)
#     # ─────────────────────────────────────────────────────────────────────────

#     async def _quick_b3_score(
#         self,
#         owner: str,
#         repo: str,
#         submodule_path: str,
#         submodule_name: str,
#         branch: str,
#     ) -> int:
#         """
#         Lightweight B3 score used only to set the auto_selected default in the UI.
#         Returns 0–3. The full B3 gate (with real centrality measurement) runs in
#         Phase 3 inside SubmoduleDecisionTree._b3_gate().

#         Uses GitHubService.get_recent_commits() — no raw httpx calls.
#         """
#         score = 0

#         # Criterion 1: Updated within 12 months
#         try:
#             from datetime import datetime, timedelta, timezone
#             commits = await self.gh.get_recent_commits(
#                 owner, repo, branch, self.installation_id, count=1
#             )
#             self._api_calls += 1
#             if commits:
#                 ts = datetime.fromisoformat(
#                     commits[0].timestamp.replace("Z", "+00:00")
#                 )
#                 if ts > datetime.now(timezone.utc) - timedelta(days=365):
#                     score += 1
#         except Exception:
#             pass

#         # Criterion 2: Not pure infrastructure
#         infra = {"build", "ci", "deploy", "infra", "toolchain", "scripts"}
#         if not any(t in f"{submodule_name} {submodule_path}".lower() for t in infra):
#             score += 1

#         # Criterion 3: Assume central (conservative — avoids hiding important repos)
#         score += 1

#         return score

#     # ─────────────────────────────────────────────────────────────────────────
#     # GraphRAG dependency edges
#     # ─────────────────────────────────────────────────────────────────────────

#     def _build_dependency_edges(
#         self,
#         owner: str,
#         repo: str,
#         submodule_nodes: list[SubmoduleNode],
#         mono_result: Optional[MonorepoDetectionResult],
#     ) -> list[dict]:
#         """
#         Build repo-level dependency edges for Phase 4 GraphRAG.

#         Two types:
#           submodule:     parent repo → submodule repo (via .gitmodules path)
#           workspace_dep: monorepo package → monorepo package (via package.json deps)

#         Symbol-level edges (function → function) are built in Phase 4
#         after AST parsing — they need actual file content which we don't have yet.

#         Example submodule edge:
#           { from: "myorg/main-app", to: "myorg/shared-lib",
#             via: "third_party/shared-lib", pinned_sha: "a3f91c...",
#             edge_type: "submodule" }

#         Example workspace_dep edge:
#           { from: "myorg/main-app/apps/api", to: "myorg/main-app/packages/ui",
#             via: "workspace_dependency", pinned_sha: None,
#             edge_type: "workspace_dep" }
#         """
#         edges      = []
#         parent_ref = f"{owner}/{repo}"

#         for node in submodule_nodes:
#             if node.resolved_owner and node.resolved_repo:
#                 edges.append({
#                     "from":       parent_ref,
#                     "to":         f"{node.resolved_owner}/{node.resolved_repo}",
#                     "via":        node.path,
#                     "pinned_sha": node.pinned_sha,
#                     "edge_type":  "submodule",
#                     "depth":      node.depth,
#                     "outcome":    node.outcome.value,
#                 })

#         if mono_result and mono_result.dependency_graph:
#             for pkg_name, deps in mono_result.dependency_graph.items():
#                 for dep in deps:
#                     edges.append({
#                         "from":       f"{parent_ref}/{pkg_name}",
#                         "to":         f"{parent_ref}/{dep}",
#                         "via":        "workspace_dependency",
#                         "pinned_sha": None,
#                         "edge_type":  "workspace_dep",
#                         "depth":      0,
#                         "outcome":    "internal",
#                     })

#         return edges

#     # ─────────────────────────────────────────────────────────────────────────
#     # Helpers
#     # ─────────────────────────────────────────────────────────────────────────


#     def _build_subproject_nodes(
#         self, mono_result: Optional[MonorepoDetectionResult]
#     ) -> list[SubprojectNode]:
#         if not mono_result or not mono_result.is_monorepo:
#             return []
#         return [
#             SubprojectNode(
#                 path=sp.path,
#                 name=sp.name,
#                 score=sp.composite_score,
#                 auto_selected=(sp.decision == SubProjectDecision.FULL_INGEST),
#                 source_file_count=sp.source_file_count,
#                 has_entry_point=sp.has_entry_point,
#                 dependent_count=sp.dependent_count,
#                 recent_commit_count=sp.recent_commit_count,
#                 skip_reason=sp.skip_reason,
#             )
#             for sp in mono_result.all_subprojects
#         ]

#     def _flatten_submodules(
#         self, nodes: list[SubmoduleNode]
#     ) -> list[SubmoduleNode]:
#         """Flatten nested submodule tree for summary count calculations."""
#         flat = []
#         for node in nodes:
#             flat.append(node)
#             flat.extend(node.nested_submodules)
#         return flat