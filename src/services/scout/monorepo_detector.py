"""
app/services/pre_clone/monorepo_detector.py
─────────────────────────────────────────────
Complete monorepo detection and sub-project scoring system.

Pipeline:
  Stage 1  — Tooling config detection
             Read nx.json / turbo.json / rush.json / pnpm-workspace.yaml /
             package.json#workspaces / Cargo.toml#workspace / settings.gradle / WORKSPACE
             → produces authoritative subproject list

  Stage 2  — Structural inference (fallback when no tooling config found)
             Scan root directory for directories containing package manifests
             → produces inferred subproject list with lower confidence

  Stage 3  — Dependency graph construction
             Parse cross-package imports from each subproject's manifest
             → builds { pkg_name → [pkg_names_it_depends_on] }

  Stage 4  — Per-subproject scoring
             Score each subproject on 6 dimensions, compute composite score

  Stage 5  — Decision threshold application
             Apply FULL_INGEST / STUB_ONLY / SKIP to each subproject
             → produces approved_dirs list for sparse checkout
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml

from src.core.config import get_settings
from src.services.pre_clone.types import (
    MonorepoDetectionResult, MonorepoTooling,
    SubProjectDecision, SubProjectScore,
)

logger = logging.getLogger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Tooling Config Detection
# ─────────────────────────────────────────────────────────────────────────────

# Tooling indicator files — checked at repo root
# Dict maps filename → tooling type
TOOLING_INDICATORS: dict[str, MonorepoTooling] = {
    "nx.json":              MonorepoTooling.NX,
    "turbo.json":           MonorepoTooling.TURBOREPO,
    "rush.json":            MonorepoTooling.RUSH,
    "lerna.json":           MonorepoTooling.LERNA,
    "pnpm-workspace.yaml":  MonorepoTooling.PNPM,
    "pnpm-workspace.yml":   MonorepoTooling.PNPM,
    "WORKSPACE":            MonorepoTooling.BAZEL,
    "WORKSPACE.bazel":      MonorepoTooling.BAZEL,
}

# Subproject manifest files — presence in a dir means it's an independent subproject
SUBPROJECT_MANIFESTS = frozenset({
    "package.json", "go.mod", "Cargo.toml", "pom.xml",
    "build.gradle", "build.gradle.kts", "pyproject.toml",
    "setup.py", "composer.json", "pubspec.yaml", "mix.exs",
})

# Entry point file stems that signal a deployable artifact
ENTRY_POINT_STEMS = frozenset({
    "main", "index", "app", "server", "start", "run", "__main__",
    "wsgi", "asgi", "manage", "cmd",
})

# Sub-project scoring weights — must sum to 1.0
SCORE_WEIGHTS = {
    "has_entry_point":        0.25,   # Deployable artifact is architecturally important
    "dependent_count":        0.25,   # Shared libs score high
    "recent_commit_count":    0.20,   # Active development is more important to document
    "source_file_count":      0.15,   # More code = more value to document
    "has_own_readme":         0.10,   # Explicit documentation intent
    "depth_penalty":          0.05,   # Penalise deep nesting
}

# Decision thresholds
FULL_INGEST_THRESHOLD = 0.35   # Score ≥ this → FULL_INGEST
STUB_THRESHOLD        = 0.15   # Score ≥ this → STUB_ONLY, else SKIP
MAX_FULL_INGEST       = 50     # Hard cap on FULL_INGEST dirs (sparse checkout limit)
MIN_SOURCE_FILES      = 3      # Sub-projects with fewer source files → SKIP always


@dataclass
class RawSubProject:
    """Discovered sub-project before scoring."""
    path: str
    name: str
    manifest_file: Optional[str] = None       # e.g. "package.json"
    declared_deps: list[str] = field(default_factory=list)   # Names of other subprojects it depends on
    declared_name: Optional[str] = None       # Name from manifest (e.g. package.json "name")


# class MonorepoDetector:
#     """
#     Detects whether a repository is a monorepo and produces the
#     complete MonorepoDetectionResult including sparse checkout dirs.

#     All file content is fetched via GitHub API tree + blob calls.
#     No git clone happens at this stage.
#     """

#     def __init__(self, headers: dict[str, str]):
#         self.headers = headers

#     async def detect(
#         self,
#         owner: str,
#         repo: str,
#         default_branch: str,
#         root_files: set[str],          # Root-level filenames (already fetched)
#         recent_commit_paths: list[str], # Files touched in recent commits (for churn)
#     ) -> MonorepoDetectionResult:
#         """
#         Main entry point. Returns MonorepoDetectionResult.

#         Args:
#             owner, repo, default_branch: Repository identity
#             root_files:    Set of filenames at repo root (from prior GitHub tree call)
#             recent_commit_paths: Flat list of file paths from recent commits
#         """
#         # ── Stage 1: Tooling detection ─────────────────────────────────────
#         tooling, detected_via, raw_subprojects = await self._stage1_tooling_detection(
#             owner, repo, default_branch, root_files
#         )

#         if not raw_subprojects:
#             # ── Stage 2: Structural inference ─────────────────────────────
#             raw_subprojects, tooling, detected_via = await self._stage2_structural_inference(
#                 owner, repo, default_branch, root_files
#             )

#         if not raw_subprojects:
#             return MonorepoDetectionResult(
#                 is_monorepo=False,
#                 tooling=MonorepoTooling.NONE,
#             )

#         # ── Stage 3: Dependency graph ──────────────────────────────────────
#         dep_graph = self._stage3_build_dependency_graph(raw_subprojects)

#         # ── Stage 4: Scoring ───────────────────────────────────────────────
#         full_tree = await self._fetch_full_tree(owner, repo, default_branch)
#         scores = await self._stage4_score_subprojects(
#             raw_subprojects=raw_subprojects,
#             dep_graph=dep_graph,
#             full_tree=full_tree,
#             recent_commit_paths=recent_commit_paths,
#         )

#         # ── Stage 5: Decision threshold ────────────────────────────────────
#         self._stage5_apply_decisions(scores)

#         approved = [s.path for s in scores if s.decision == SubProjectDecision.FULL_INGEST]
#         stubs    = [s.path for s in scores if s.decision == SubProjectDecision.STUB_ONLY]
#         skipped  = [s.path for s in scores if s.decision == SubProjectDecision.SKIP]

#         is_monorepo = len(raw_subprojects) >= 2  # 1 subproject = not really a monorepo

#         logger.info(
#             "Monorepo detection for %s/%s: tooling=%s subprojects=%d "
#             "approved=%d stubs=%d skipped=%d",
#             owner, repo, tooling.value, len(scores),
#             len(approved), len(stubs), len(skipped),
#         )

#         return MonorepoDetectionResult(
#             is_monorepo=is_monorepo,
#             tooling=tooling,
#             all_subprojects=scores,
#             approved_dirs=approved,
#             stub_dirs=stubs,
#             skipped_dirs=skipped,
#             dependency_graph=dep_graph,
#             detected_via=detected_via,
#             detection_confidence=1.0 if tooling != MonorepoTooling.INFERRED else 0.65,
#         )

#     # ─────────────────────────────────────────────────────────────────────────
#     # Stage 1: Tooling-Config Detection
#     # ─────────────────────────────────────────────────────────────────────────

#     async def _stage1_tooling_detection(
#         self, owner: str, repo: str, branch: str, root_files: set[str]
#     ) -> tuple[MonorepoTooling, Optional[str], list[RawSubProject]]:
#         """
#         Read tooling config files to get an authoritative sub-project list.
#         Returns (tooling, detected_via_filename, raw_subprojects).
#         """

#         # Check each indicator in priority order
#         for filename, tooling in TOOLING_INDICATORS.items():
#             if filename in root_files:
#                 content = await self._fetch_file(owner, repo, branch, filename)
#                 if content is None:
#                     continue

#                 subprojects = await self._parse_tooling_config(
#                     tooling, filename, content, owner, repo, branch, root_files
#                 )
#                 if subprojects:
#                     logger.info(
#                         "Monorepo tooling detected: %s via %s (%d subprojects)",
#                         tooling.value, filename, len(subprojects),
#                     )
#                     return tooling, filename, subprojects

#         # Check package.json for npm/yarn workspaces (needs content inspection)
#         if "package.json" in root_files:
#             content = await self._fetch_file(owner, repo, branch, "package.json")
#             if content:
#                 subprojects, tooling_type = await self._parse_npm_workspaces(
#                     content, owner, repo, branch
#                 )
#                 if subprojects:
#                     return tooling_type, "package.json", subprojects

#         # Check Cargo.toml for Rust workspace
#         if "Cargo.toml" in root_files:
#             content = await self._fetch_file(owner, repo, branch, "Cargo.toml")
#             if content:
#                 subprojects = await self._parse_cargo_workspace(
#                     content, owner, repo, branch
#                 )
#                 if subprojects:
#                     return MonorepoTooling.CARGO, "Cargo.toml", subprojects

#         # Check settings.gradle for Gradle multi-project
#         for gradle_file in ("settings.gradle", "settings.gradle.kts"):
#             if gradle_file in root_files:
#                 content = await self._fetch_file(owner, repo, branch, gradle_file)
#                 if content:
#                     subprojects = self._parse_gradle_settings(content)
#                     if subprojects:
#                         return MonorepoTooling.GRADLE, gradle_file, subprojects

#         return MonorepoTooling.NONE, None, []

#     async def _parse_tooling_config(
#         self,
#         tooling: MonorepoTooling,
#         filename: str,
#         content: str,
#         owner: str, repo: str, branch: str,
#         root_files: set[str],
#     ) -> list[RawSubProject]:
#         """Dispatch to the right parser for each tooling type."""
#         try:
#             if tooling == MonorepoTooling.NX:
#                 return await self._parse_nx(content, owner, repo, branch)
#             elif tooling == MonorepoTooling.TURBOREPO:
#                 return await self._parse_turborepo(content, owner, repo, branch, root_files)
#             elif tooling == MonorepoTooling.RUSH:
#                 return self._parse_rush(content)
#             elif tooling == MonorepoTooling.LERNA:
#                 return await self._parse_lerna(content, owner, repo, branch)
#             elif tooling == MonorepoTooling.PNPM:
#                 return await self._parse_pnpm_workspace(content, owner, repo, branch)
#             elif tooling == MonorepoTooling.BAZEL:
#                 return await self._parse_bazel(owner, repo, branch)
#         except Exception as e:
#             logger.warning("Failed to parse %s config: %s", filename, e)
#         return []

#     # ── Nx Parser ─────────────────────────────────────────────────────────────

#     async def _parse_nx(
#         self, nx_json_content: str, owner: str, repo: str, branch: str
#     ) -> list[RawSubProject]:
#         """
#         Nx workspace: read project.json files from each project directory.
#         nx.json defines the workspace; project.json files in each project dir
#         declare the actual project name and targets.
#         """
#         try:
#             nx_data = json.loads(nx_json_content)
#         except json.JSONDecodeError:
#             return []

#         subprojects = []

#         # Modern Nx (v16+): projects are discovered by scanning for project.json
#         # files. nx.json may contain workspaceLayout.appsDir and libsDir.
#         layout = nx_data.get("workspaceLayout", {})
#         apps_dir = layout.get("appsDir", "apps")
#         libs_dir = layout.get("libsDir", "libs")

#         # Fetch the tree to find all project.json files
#         tree = await self._fetch_full_tree(owner, repo, branch)
#         project_json_paths = [
#             e["path"] for e in tree
#             if e["path"].endswith("/project.json") or e["path"] == "project.json"
#         ]

#         for proj_path in project_json_paths:
#             # project dir is the directory containing project.json
#             proj_dir = str(Path(proj_path).parent)
#             if proj_dir == ".":
#                 proj_dir = ""

#             content = await self._fetch_file(owner, repo, branch, proj_path)
#             if not content:
#                 continue
#             try:
#                 proj_data = json.loads(content)
#                 name = proj_data.get("name", Path(proj_dir).name if proj_dir else repo)
#                 subprojects.append(RawSubProject(
#                     path=proj_dir or ".",
#                     name=name,
#                     manifest_file="project.json",
#                     declared_name=name,
#                 ))
#             except json.JSONDecodeError:
#                 continue

#         # Legacy Nx: projects listed in workspace.json
#         if not subprojects:
#             workspace_content = await self._fetch_file(owner, repo, branch, "workspace.json")
#             if workspace_content:
#                 try:
#                     ws_data = json.loads(workspace_content)
#                     for name, proj_config in ws_data.get("projects", {}).items():
#                         path = proj_config if isinstance(proj_config, str) else proj_config.get("root", name)
#                         subprojects.append(RawSubProject(
#                             path=path, name=name, manifest_file="workspace.json"
#                         ))
#                 except json.JSONDecodeError:
#                     pass

#         return subprojects

#     # ── Turborepo Parser ──────────────────────────────────────────────────────

#     async def _parse_turborepo(
#         self, turbo_json_content: str, owner: str, repo: str, branch: str,
#         root_files: set[str],
#     ) -> list[RawSubProject]:
#         """
#         Turborepo reads workspace definitions from the root package.json.
#         turbo.json defines the pipeline but not the package locations.
#         """
#         # The actual package locations are in package.json#workspaces
#         if "package.json" in root_files:
#             pkg_content = await self._fetch_file(owner, repo, branch, "package.json")
#             if pkg_content:
#                 subprojects, _ = await self._parse_npm_workspaces(
#                     pkg_content, owner, repo, branch
#                 )
#                 return subprojects
#         return []

#     # ── Rush Parser ───────────────────────────────────────────────────────────

#     def _parse_rush(self, rush_json_content: str) -> list[RawSubProject]:
#         """
#         Rush: rush.json has an explicit "projects" array with packageName and projectFolder.
#         The most declarative format — no inference needed.
#         """
#         try:
#             # Rush json may have comments (JSONC format) — strip them
#             cleaned = re.sub(r"//[^\n]*", "", rush_json_content)
#             cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
#             data = json.loads(cleaned)
#         except json.JSONDecodeError:
#             return []

#         subprojects = []
#         for project in data.get("projects", []):
#             folder = project.get("projectFolder", "")
#             name   = project.get("packageName", Path(folder).name)
#             if folder:
#                 subprojects.append(RawSubProject(
#                     path=folder,
#                     name=name,
#                     manifest_file="rush.json",
#                     declared_name=name,
#                 ))
#         return subprojects

#     # ── Lerna Parser ──────────────────────────────────────────────────────────

#     async def _parse_lerna(
#         self, lerna_json_content: str, owner: str, repo: str, branch: str
#     ) -> list[RawSubProject]:
#         """
#         Lerna: lerna.json has "packages" glob patterns.
#         We expand the globs against the actual repo tree.
#         """
#         try:
#             data = json.loads(lerna_json_content)
#         except json.JSONDecodeError:
#             return []

#         patterns = data.get("packages", ["packages/*"])
#         return await self._expand_glob_patterns(patterns, owner, repo, branch)

#     # ── pnpm Workspaces Parser ────────────────────────────────────────────────

#     async def _parse_pnpm_workspace(
#         self, yaml_content: str, owner: str, repo: str, branch: str
#     ) -> list[RawSubProject]:
#         """
#         pnpm-workspace.yaml has a "packages" list of glob patterns.
#         """
#         try:
#             data = yaml.safe_load(yaml_content)
#             patterns = data.get("packages", []) if data else []
#         except yaml.YAMLError:
#             return []

#         return await self._expand_glob_patterns(patterns, owner, repo, branch)

#     # ── npm/Yarn Workspaces Parser ────────────────────────────────────────────

#     async def _parse_npm_workspaces(
#         self, pkg_json_content: str, owner: str, repo: str, branch: str
#     ) -> tuple[list[RawSubProject], MonorepoTooling]:
#         """
#         npm/Yarn workspaces: root package.json has a "workspaces" field.
#         Can be an array of globs or an object with {packages: [...], nohoist: [...]}.
#         """
#         try:
#             data = json.loads(pkg_json_content)
#         except json.JSONDecodeError:
#             return [], MonorepoTooling.NONE

#         workspaces = data.get("workspaces")
#         if not workspaces:
#             return [], MonorepoTooling.NONE

#         # Normalise to list of glob patterns
#         if isinstance(workspaces, dict):
#             patterns = workspaces.get("packages", [])
#         elif isinstance(workspaces, list):
#             patterns = workspaces
#         else:
#             return [], MonorepoTooling.NONE

#         subprojects = await self._expand_glob_patterns(patterns, owner, repo, branch)

#         # Determine whether this is npm or yarn by checking for yarn.lock
#         tooling = MonorepoTooling.YARN  # default — most common for workspaces
#         return subprojects, tooling

#     # ── Cargo Workspace Parser ────────────────────────────────────────────────

#     async def _parse_cargo_workspace(
#         self, cargo_toml_content: str, owner: str, repo: str, branch: str
#     ) -> list[RawSubProject]:
#         """
#         Rust Cargo workspaces: root Cargo.toml has [workspace] section with members list.
#         Members can be exact paths or glob patterns.
#         """
#         try:
#             import tomllib
#         except ImportError:
#             try:
#                 pass ##need to update the code.
#                 # import tomli as tomllib
#             except ImportError:
#                 logger.warning("No TOML parser available — cannot parse Cargo.toml")
#                 return []

#         try:
#             data = tomllib.loads(cargo_toml_content)
#         except Exception:
#             return []

#         workspace = data.get("workspace", {})
#         members = workspace.get("members", [])
#         if not members:
#             return []

#         return await self._expand_glob_patterns(members, owner, repo, branch)

#     # ── Gradle Settings Parser ────────────────────────────────────────────────

#     def _parse_gradle_settings(self, settings_content: str) -> list[RawSubProject]:
#         """
#         Gradle: settings.gradle has include() / include(':project') declarations.
#         Extract project names and convert ':project:subproject' notation to paths.
#         """
#         subprojects = []
#         # Match both include('name') and include(':name') forms
#         pattern = re.compile(r"""include\s*\(\s*['":]+([^'")\s:]+)""")
#         for match in pattern.finditer(settings_content):
#             name = match.group(1).strip()
#             # Convert Gradle notation: ':subproject:module' → 'subproject/module'
#             path = name.replace(":", "/").lstrip("/")
#             subprojects.append(RawSubProject(
#                 path=path, name=name, manifest_file="settings.gradle"
#             ))
#         return subprojects

#     # ── Bazel Parser ──────────────────────────────────────────────────────────

#     async def _parse_bazel(
#         self, owner: str, repo: str, branch: str
#     ) -> list[RawSubProject]:
#         """
#         Bazel: every directory with a BUILD or BUILD.bazel file is a build target.
#         We find them by scanning the tree.
#         """
#         tree = await self._fetch_full_tree(owner, repo, branch)
#         build_dirs = set()
#         for entry in tree:
#             if entry["path"] in ("BUILD", "BUILD.bazel") or \
#                entry["path"].endswith("/BUILD") or \
#                entry["path"].endswith("/BUILD.bazel"):
#                 parent = str(Path(entry["path"]).parent)
#                 if parent != ".":
#                     build_dirs.add(parent)

#         return [
#             RawSubProject(path=d, name=Path(d).name, manifest_file="BUILD")
#             for d in sorted(build_dirs)
#         ]

#     # ─────────────────────────────────────────────────────────────────────────
#     # Stage 2: Structural Inference
#     # ─────────────────────────────────────────────────────────────────────────

#     async def _stage2_structural_inference(
#         self, owner: str, repo: str, branch: str, root_files: set[str]
#     ) -> tuple[list[RawSubProject], MonorepoTooling, Optional[str]]:
#         """
#         When no tooling config is found, infer monorepo structure from
#         directory naming conventions and presence of package manifests.
#         """
#         # Common top-level monorepo directory names
#         MONOREPO_ROOT_DIRS = frozenset({
#             "apps", "packages", "services", "libs", "modules",
#             "plugins", "internal", "components", "shared",
#         })

#         # Check if any of these directories exist at root
#         # root_files contains files AND directories at root level
#         candidate_dirs = root_files & MONOREPO_ROOT_DIRS
#         if not candidate_dirs:
#             return [], MonorepoTooling.NONE, None

#         # Fetch the tree to find manifest files inside candidate dirs
#         tree = await self._fetch_full_tree(owner, repo, branch)
#         tree_paths = {e["path"] for e in tree}

#         subprojects = []
#         for top_dir in candidate_dirs:
#             # Find all subdirs of this top_dir that have a package manifest
#             for manifest in SUBPROJECT_MANIFESTS:
#                 # Pattern: {top_dir}/{subproject}/{manifest}
#                 for entry_path in tree_paths:
#                     parts = entry_path.split("/")
#                     if (len(parts) == 3 and
#                         parts[0] == top_dir and
#                         parts[2] == manifest):
#                         subproject_path = f"{parts[0]}/{parts[1]}"
#                         subproject_name = parts[1]
#                         # Avoid duplicates
#                         if not any(s.path == subproject_path for s in subprojects):
#                             subprojects.append(RawSubProject(
#                                 path=subproject_path,
#                                 name=subproject_name,
#                                 manifest_file=manifest,
#                             ))

#         if not subprojects:
#             return [], MonorepoTooling.NONE, None

#         logger.info(
#             "Monorepo inferred from directory structure: %d subprojects in dirs %s",
#             len(subprojects), sorted(candidate_dirs),
#         )
#         return subprojects, MonorepoTooling.INFERRED, "directory_structure"

#     # ─────────────────────────────────────────────────────────────────────────
#     # Stage 3: Dependency Graph
#     # ─────────────────────────────────────────────────────────────────────────

#     def _stage3_build_dependency_graph(
#         self, raw_subprojects: list[RawSubProject]
#     ) -> dict[str, list[str]]:
#         """
#         Build a dependency graph: { project_name → [names_it_depends_on] }

#         This is used for two things:
#           1. Dependency centrality scoring: projects that many others depend on
#              get higher scores (they're core libraries)
#           2. Ensuring core libraries always make the sparse checkout set

#         Input: declared_deps from each RawSubProject's package manifest.
#         These are populated during tooling config parsing where we read
#         package.json "dependencies" fields.
#         """
#         # Build name → project map for lookup
#         name_map = {sp.declared_name or sp.name: sp for sp in raw_subprojects}

#         graph: dict[str, list[str]] = {}
#         for sp in raw_subprojects:
#             name = sp.declared_name or sp.name
#             # Filter deps to only those that are internal packages (in our subproject list)
#             internal_deps = [d for d in sp.declared_deps if d in name_map]
#             graph[name] = internal_deps

#         return graph

#     # ─────────────────────────────────────────────────────────────────────────
#     # Stage 4: Scoring
#     # ─────────────────────────────────────────────────────────────────────────

#     async def _stage4_score_subprojects(
#         self,
#         raw_subprojects: list[RawSubProject],
#         dep_graph: dict[str, list[str]],
#         full_tree: list[dict],
#         recent_commit_paths: list[str],
#     ) -> list[SubProjectScore]:
#         """
#         Score each sub-project on 6 dimensions.
#         """
#         from src.services.pre_clone.file_filter import LANGUAGE_MAP, FileTier
#         source_extensions = {
#             ext for ext, (_, tier) in LANGUAGE_MAP.items()
#             if tier == FileTier.TIER1_SOURCE
#         }

#         # Pre-compute: for each subproject path, how many files live under it
#         # This is O(subprojects × tree_size) — acceptable for typical sizes
#         tree_by_path = {}
#         for entry in full_tree:
#             if entry["type"] == "blob":
#                 top = entry["path"].split("/")[0] if "/" in entry["path"] else entry["path"]
#                 tree_by_path.setdefault(entry["path"][:entry["path"].rfind("/")], []).append(entry)

#         # Pre-compute dependent count: how many projects depend ON each project
#         reverse_dep_count: dict[str, int] = {}
#         for deps in dep_graph.values():
#             for dep in deps:
#                 reverse_dep_count[dep] = reverse_dep_count.get(dep, 0) + 1

#         # Pre-compute recent commit activity per top-level dir
#         recent_dir_counts: dict[str, int] = {}
#         for path in recent_commit_paths:
#             top_dir = path.split("/")[0]
#             recent_dir_counts[top_dir] = recent_dir_counts.get(top_dir, 0) + 1

#         scores = []
#         for sp in raw_subprojects:
#             score = SubProjectScore(
#                 path=sp.path,
#                 name=sp.declared_name or sp.name,
#             )

#             # ── Signal 1: Source file count ────────────────────────────────
#             sp_prefix = sp.path.rstrip("/") + "/"
#             source_files = [
#                 e for e in full_tree
#                 if e["path"].startswith(sp_prefix)
#                 and e["type"] == "blob"
#                 and any(e["path"].endswith(ext) for ext in source_extensions)
#             ]
#             score.source_file_count = len(source_files)
#             score.source_byte_count = sum(e.get("size", 0) for e in source_files)

#             # ── Signal 2: Entry point presence ────────────────────────────
#             entry_point_files = [
#                 e for e in full_tree
#                 if e["path"].startswith(sp_prefix)
#                 and Path(e["path"]).stem.lower() in ENTRY_POINT_STEMS
#                 and e["type"] == "blob"
#             ]
#             score.has_entry_point = len(entry_point_files) > 0
#             score.is_deployable = score.has_entry_point

#             # ── Signal 3: README presence ──────────────────────────────────
#             readme_files = [
#                 e for e in full_tree
#                 if e["path"].startswith(sp_prefix)
#                 and Path(e["path"]).name.upper().startswith("README")
#             ]
#             score.has_own_readme = len(readme_files) > 0

#             # ── Signal 4: Dependent count (reverse deps) ───────────────────
#             pkg_name = sp.declared_name or sp.name
#             score.dependent_count = reverse_dep_count.get(pkg_name, 0)

#             # ── Signal 5: Recent commit activity ──────────────────────────
#             top_component = sp.path.split("/")[0]
#             score.recent_commit_count = recent_dir_counts.get(top_component, 0)

#             # ── Signal 6: Depth penalty ────────────────────────────────────
#             score.depth = sp.path.count("/") + 1

#             # ── Composite score ────────────────────────────────────────────
#             score.composite_score = self._compute_composite_score(score, raw_subprojects)

#             scores.append(score)

#         return scores

#     def _compute_composite_score(
#         self, score: SubProjectScore, all_subprojects: list[RawSubProject]
#     ) -> float:
#         """
#         Compute a 0.0–1.0 composite score from raw signals.
#         Each dimension is normalised to [0, 1] before weighting.
#         """
#         n = len(all_subprojects)

#         # Normalise source file count (log scale — large repos shouldn't dominate)
#         import math
#         file_norm = min(1.0, math.log1p(score.source_file_count) / math.log1p(500))

#         # Entry point: binary 0 or 1
#         entry_norm = 1.0 if score.has_entry_point else 0.0

#         # Dependent count: normalised against theoretical max (n-1)
#         dep_norm = min(1.0, score.dependent_count / max(n - 1, 1))

#         # Recent commits: normalise against 50 commits as "very active"
#         commit_norm = min(1.0, score.recent_commit_count / 50)

#         # README: binary
#         readme_norm = 1.0 if score.has_own_readme else 0.0

#         # Depth penalty: shallower = better
#         # depth 1 = 1.0, depth 2 = 0.7, depth 3 = 0.4, depth 4+ = 0.1
#         depth_norm = max(0.1, 1.0 - (score.depth - 1) * 0.3)

#         composite = (
#             SCORE_WEIGHTS["has_entry_point"]     * entry_norm  +
#             SCORE_WEIGHTS["dependent_count"]      * dep_norm    +
#             SCORE_WEIGHTS["recent_commit_count"]  * commit_norm +
#             SCORE_WEIGHTS["source_file_count"]    * file_norm   +
#             SCORE_WEIGHTS["has_own_readme"]       * readme_norm +
#             SCORE_WEIGHTS["depth_penalty"]        * depth_norm
#         )

#         return round(composite, 4)

#     # ─────────────────────────────────────────────────────────────────────────
#     # Stage 5: Decision Threshold Application
#     # ─────────────────────────────────────────────────────────────────────────

#     def _stage5_apply_decisions(self, scores: list[SubProjectScore]) -> None:
#         """
#         Apply FULL_INGEST / STUB_ONLY / SKIP decisions to each scored subproject.
#         Modifies the score objects in place.
#         """
#         # Sort by composite score descending for deterministic processing
#         scores.sort(key=lambda s: s.composite_score, reverse=True)

#         full_ingest_count = 0

#         for score in scores:
#             # Hard minimum: too few source files → always skip
#             if score.source_file_count < MIN_SOURCE_FILES:
#                 score.decision = SubProjectDecision.SKIP
#                 score.skip_reason = (
#                     f"Too few source files ({score.source_file_count} < {MIN_SOURCE_FILES})"
#                 )
#                 continue

#             # Hard override: core libraries that others depend on always get full ingest
#             # even if their own score is low (e.g. a tiny but central utility lib)
#             if score.dependent_count >= 3 and full_ingest_count < MAX_FULL_INGEST:
#                 score.decision = SubProjectDecision.FULL_INGEST
#                 full_ingest_count += 1
#                 continue

#             # Score threshold application
#             if score.composite_score >= FULL_INGEST_THRESHOLD and full_ingest_count < MAX_FULL_INGEST:
#                 score.decision = SubProjectDecision.FULL_INGEST
#                 full_ingest_count += 1
#             elif score.composite_score >= STUB_THRESHOLD:
#                 score.decision = SubProjectDecision.STUB_ONLY
#                 score.skip_reason = (
#                     f"Score {score.composite_score:.2f} below full-ingest threshold "
#                     f"({FULL_INGEST_THRESHOLD})"
#                 )
#             else:
#                 score.decision = SubProjectDecision.SKIP
#                 score.skip_reason = (
#                     f"Score {score.composite_score:.2f} below stub threshold "
#                     f"({STUB_THRESHOLD})"
#                 )

#     # ─────────────────────────────────────────────────────────────────────────
#     # Helpers
#     # ─────────────────────────────────────────────────────────────────────────

#     async def _fetch_file(
#         self, owner: str, repo: str, branch: str, path: str
#     ) -> Optional[str]:
#         """Fetch a single file's content via GitHub contents API."""
#         async with httpx.AsyncClient() as client:
#             try:
#                 resp = await client.get(
#                     f"https://api.github.com/repos/{owner}/{repo}/contents/{path}",
#                     headers={**self.headers, "Accept": "application/vnd.github.raw+json"},
#                     params={"ref": branch},
#                     timeout=15.0,
#                 )
#                 if resp.status_code != 200:
#                     return None
#                 return resp.text
#             except httpx.HTTPError:
#                 return None

#     async def _fetch_full_tree(
#         self, owner: str, repo: str, branch: str
#     ) -> list[dict]:
#         """Fetch the full recursive tree. Returns raw entry dicts."""
#         async with httpx.AsyncClient() as client:
#             try:
#                 resp = await client.get(
#                     f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}",
#                     headers=self.headers,
#                     params={"recursive": "1"},
#                     timeout=30.0,
#                 )
#                 resp.raise_for_status()
#                 data = resp.json()
#                 return data.get("tree", [])
#             except httpx.HTTPError:
#                 return []

#     async def _expand_glob_patterns(
#         self, patterns: list[str], owner: str, repo: str, branch: str
#     ) -> list[RawSubProject]:
#         """
#         Expand glob patterns (like "packages/*") against the actual repo tree.
#         Returns RawSubProject for each matching directory that has a package manifest.
#         """
#         tree = await self._fetch_full_tree(owner, repo, branch)
#         tree_dirs = {
#             str(Path(e["path"]).parent)
#             for e in tree
#             if e["type"] == "blob" and Path(e["path"]).name in SUBPROJECT_MANIFESTS
#         }

#         import fnmatch
#         subprojects = []
#         for pattern in patterns:
#             # Normalize pattern: remove trailing /**
#             clean_pattern = pattern.rstrip("/**").rstrip("/*")
#             for dir_path in tree_dirs:
#                 if fnmatch.fnmatch(dir_path, clean_pattern) or \
#                    fnmatch.fnmatch(dir_path, pattern):
#                     subprojects.append(RawSubProject(
#                         path=dir_path,
#                         name=Path(dir_path).name,
#                         manifest_file="workspace_config",
#                     ))

#         # Deduplicate
#         seen = set()
#         unique = []
#         for sp in subprojects:
#             if sp.path not in seen:
#                 seen.add(sp.path)
#                 unique.append(sp)
#         return unique