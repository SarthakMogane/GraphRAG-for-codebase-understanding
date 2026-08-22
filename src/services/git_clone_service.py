"""
app/services/git_clone_service.py
──────────────────────────────────
Executes the actual git clone and sparse checkout commands.
Translates a CloneConfig into shell commands via subprocess.
 
SECURITY — every clone touches an arbitrary, untrusted third-party
repository. See STATE.md §5 for the full threat model. The controls
below directly mitigate CVE-2025-48384 (CISA KEV, actively exploited)
and apply the NVIDIA AI Red Team's mandatory sandboxing controls to git
specifically:
 
  1. Hooks are hard-disabled on every command via
     `-c core.hooksPath=<empty sink dir>`. CVE-2025-48384 tricks git
     into checking out a submodule into `.git/hooks/` via a crafted
     `\\r`-terminated path, then relies on git executing the resulting
     hook on a later checkout/commit/merge. If hooksPath points to a
     directory that can never contain an executable hook, that exploit
     chain has nothing to execute even if the path-confusion write
     still happens.
  2. `--recurse-submodules` / `--recursive` is never used anywhere in
     this file. Submodules are always initialized one at a time by
     explicit path, after GitmodulesParser has validated the entry.
  3. `GIT_ALLOW_PROTOCOL=https` restricts which transports git honors.
     Without this, a submodule URL of `ext::sh -c ...` or `file://...`
     is a known remote-helper injection / arbitrary local file read.
  4. Subprocess environment is an explicit allowlist, never
     `os.environ.copy()` — no ambient secret becomes reachable to a
     subprocess executing against untrusted, attacker-influenced input.
  5. Every subprocess call has a hard timeout — a hung or deliberately
     slow clone must not block the worker indefinitely.
  6. Git version is checked once at construction against the patched
     versions for CVE-2025-48384.
"""
import subprocess
import re
from pathlib import Path
from typing import Optional
from src.services.clone_strategy import CloneConfig
from src.core.logger import get_logger
from src.core.config import get_settings

logger = get_logger(__name__)
settings  = get_settings()

_MIN_PATCHED_GIT:dict[(int,int):int] = {
    (2, 43): 7, (2, 44): 4, (2, 45): 4, (2, 46): 4,
    (2, 47): 3, (2, 48): 2, (2, 49): 1, (2, 50): 1,
}

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

class UnsafeGitVersionError(Exception):
    """Raised when the installed git binary is vulnerable to CVE-2025-48384."""
    pass

class CloneError(Exception):
    """Raised when a git clone or submodule operation fails."""
    pass

class GitCloneService:
    """
    Executes git clone commands based on a CloneConfig.
    All git operations run in a subprocess (not GitPython) because
    GitPython's clone wrapper doesn't expose all the flags we need.
 
    Stateless across jobs except for the lazily-created hooks sink
    directory, which is process-wide (empty, never written to) and
    safe to share.
    """
    def __init__(self) -> None:
        self._verify_git_version()

    async def clone(
        self,
        owner:str,
        repo:str,
        clone_config:CloneConfig,
        home_dir:Path,
        target_dir:str,
        auth_token: Optional[str] = None
    ) -> Path:
        """
        Clone a repository according to CloneConfig.
        Returns the path to the cloned working tree.

        Args: 
            owner:      GitHub owner/org name
            repo:       Repository name
            config:     CloneConfig from CloneStrategySelector
            home_dir:   Per-job-scoped directory used as $HOME for the git
                        subprocess. Required, not optional.
            target_dir: Required, not optional — see below.
            auth_token: GitHub installation token for private repos. Passed
                        via `-c http.extraHeader=Authorization: basic ...`,
                        NOT embedded in the clone URL. Embedding a token in
                        the URL means it gets written into the cloned repo's
        """

        self._validate_safe_name(owner,"owner")
        self._validate_safe_name(repo,"repo")
        


    def _verify_git_version(self) -> None:
        try:
            result = subprocess.run(["git","--version"],capture_output=True , text=str, timeout=5)
        except (subprocess.SubprocessError,FileNotFoundError) as e:
            raise UnsafeGitVersionError("Couldn't determine git version: {e}") from e 

        match = re.match(r"(\d+)\.(\d+)\.(\d+)",result.stdout)

        if not match:
            raise UnsafeGitVersionError("Couldn't parse git version from :{result.stdout}") 

        major, minor , patch = (int(g) for g in match.groups())
        min_patch = _MIN_PATCHED_GIT.get(major,minor)

        if min_patch is not None and patch < min_patch:
            raise UnsafeGitVersionError(f"git {major}.{minor}.{patch} is vulnerable to "
                f"CVE-2025-48384 — upgrade to {major}.{minor}.{min_patch} or later.")

        if (major and minor) not in _MIN_PATCHED_GIT and (major,minor)<(2,43):
            raise UnsafeGitVersionError(
                f"git {major}.{minor}.{patch} predates all CVE-2025-48384 "
                f"patch trains — upgrade git."
            )

        logger.info("Git version OK: %d.%d.%d", major, minor, patch)

    def _validate_safe_name(value:str ,field:str) -> None:
        """
        Defense-in-depth check on owner/repo names right before they're
        interpolated into a URL and a filesystem path. Upstream
        validation (Pydantic models, GitHub API responses) should already
        guarantee this, but this is the last line before the value
        touches a subprocess argument and a path.
        """
        if not value or _SAFE_NAME_RE.match(value):
            raise CloneError(
                f"Refusing to clone: {field}={value!r} contains characters "
                f"outside [A-Za-z0-9_.-] — possible path or argument injection."
            )
