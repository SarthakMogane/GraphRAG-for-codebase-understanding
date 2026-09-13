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
import os
import signal
import re
import base64
import tempfile
import asyncio
from pathlib import Path
from typing import Optional
from src.services.clone_strategy import CloneConfig
from sandbox_microvm.workspace import WorkspacePathViolation
from src.models.database import CloneStrategy
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
    _hooks_sink: Optional[Path] = None

    def __init__(self) -> None:
        self._verify_git_version()

    async def clone(
        self,
        owner:str,
        repo:str,
        clone_config:CloneConfig,
        home_dir:Path,
        target_dir:Path,
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
        if not auth_token or not auth_token.strip():
            raise CloneError(
                f"Access Denied: Refusing unauthenticated clone for {owner}/{repo}. "
                f"This sandbox is configured for authorized tenant traffic only."
            )
        self._validate_safe_name(owner,"owner")
        self._validate_safe_name(repo,"repo")

        clone_url = f"https://github.com/{owner}/{repo}.git"
        auth_flag = self._auth_header_flag(auth_token)
        dest = Path(target_dir)
        home_dir.mkdir(parents=True ,exist_ok=True)
        dest.parent.mkdir(parents=True , exist_ok=True)

        # //strategy execution
        try:
            if clone_config.strategy == CloneStrategy.SPARSE_CHECKOUT.value:
                await self._clone_sparse(clone_url,dest,clone_config,home_dir,auth_flag)
            elif clone_config.strategy == CloneStrategy.PARTIAL_BLOB.value:
                await self._clone_partial(clone_url,dest,clone_config,home_dir,auth_flag)
            else:
                await self._clone_shallow(clone_url,dest,clone_config,home_dir,auth_flag)

            logger.info("Clone complete: %s/%s → %s", owner, repo, dest)
            return dest
        except Exception as e:
            raise CloneError(f"Clone failed for {owner}/{repo}: {e}") from e
        
    async def _clone_shallow(
        self, url:str, dest:Path, clone_config:CloneConfig, home_dir:Path,
        auth_flag:list[str] 
    )-> None:
        """
        Standard shallow clone — simplest case.
        git clone --depth 1 --single-branch <url> <dest>
        """
        cmd = self._git_cmd(
            *auth_flag,
            "clone",
            "--depth",str(clone_config.depth),
            "--single-branch",
            url,
            str(dest),
        )
        env = self._build_env(clone_config,home_dir)
        await self._run(cmd,env=env,timeout_seconds=clone_config.git_operation_timeout_seconds)

    async def _clone_partial(
            self,url:str,dest:Path,config:CloneConfig,home_dir:Path,
            auth_flags:list[str]
    ) -> None:
        """
        Partial clone with blob filter.
        git clone --depth 1 --filter=blob:none --single-branch <url> <dest>
 
        File content is NOT downloaded during clone.
        Git fetches blob content lazily when each file is first accessed.
        """
        extra =["--no-checkout"] if config.no_checkout else []
        cmd = self._git_cmd(
            *auth_flags,
            "clone",
            "--depth",str(config.depth),
            "--filter=blob:none",
            *extra,
            "--single-branch",
            url,
            str(dest),   
        )
        env = self._build_env(config,home_dir)
        await self._run(cmd=cmd,env=env,timeout_seconds=config.git_operation_timeout_seconds)

        if config.no_checkout:
            await self._run(
                self._git_cmd("checkout"),
                cwd=dest,
                env=env,
                timeout_seconds=config.git_operation_timeout_seconds,
            )

    async def _clone_sparse(
        self,url:str,dest:Path,config:CloneConfig,
        home_dir:Path,auth_flags: list[str]
    ) -> None:
        """
        Sparse checkout in cone mode — only materializes specified directories.
        Full sequence:
        1. Clone with --no-checkout and --filter=blob:none (nothing on disk)
        2. Initialize sparse-checkout in cone mode
        3. Set the approved directory list
        4. Checkout (only those directories land on disk)
 
        Cone mode is ~10x faster than non-cone mode for large repos because
        it uses simple prefix matching rather than .gitignore-style patterns.
        """

        cmd_clone = self._git_cmd(
            *auth_flags,
            "clone",
            "--depth",str(config.depth),
            "--filter=blob:none",
            "--no-checkout",
            "--single-branch",
            url,
            str(dest),
        )
        env= self._build_env(config,home_dir=home_dir)
        await self._run(cmd_clone,env=env,timeout_seconds=config.git_operation_timeout_seconds)

         #  Initialize sparse checkout in cone mode
        await self._run(
            self._git_cmd("sparse-checkout","init","--cone"),
            cwd=dest,
            env=env,
            timeout_seconds=config.git_operation_timeout_seconds,
        )
        # Set which directories to include
        dirs_to_include = config.sparse_dirs or []
        await self._run(
            self._git_cmd("sparse-checkout","set",*dirs_to_include),
            cwd=dest,
            env=env,
            timeout_seconds=config.git_operation_timeout_seconds
        )

        #  Checkout — only the specified directories land on disk
        await self._run(
            self._git_cmd("checkout"),
            cwd=dest,
            env=env,
            timeout_seconds=config.git_operation_timeout_seconds
        )

        logger.info(
            "Sparse checkout complete - materialized %d directories",len(dirs_to_include)
        )
    
    async def init_submodules(
        self,
        repo_path: Path,
        selected_submodules: list[dict], 
        auth_token: str,
        home_dir: Path,
        jobs: int = None,
    ) -> None:
        """
        Initializes submodules using the Dual-Branch Strategy with strongly-typed configs.
        """
        if not selected_submodules:
            return 

        parallel = jobs or settings.SUBMODULE_PARALLEL_JOBS
        auth_flag = self._auth_header_flag(auth_token=auth_token)

        normal_subs = []
        monorepo_subs = []

        # Route submodules using Object Attributes
        for sub in selected_submodules:
            if sub.get("is_monorepo") and sub.get("subprojects"):
                monorepo_subs.append(sub)
            else:
                normal_subs.append(sub)

        # ── BRANCH A: NORMAL SUBMODULES (Native Parallel Update) ──
        if normal_subs:
            normal_paths = [sub["path"] for sub in normal_subs]
            
            use_blob_filter = any(
            sub.get("clone_config", {}).get("strategy") == CloneStrategy.PARTIAL_BLOB.value
            or sub.get("clone_config", {}).get("filter_blob_none", False)
            for sub in normal_subs
            )
            should_skip_lfs = any(
                sub.get("clone_config", {}).get("skip_lfs", False) for sub in normal_subs
            )
            # Build batch CloneConfig for the combined git submodule command
            batch_config = CloneConfig(
                strategy=CloneStrategy.PARTIAL_BLOB if use_blob_filter else CloneStrategy.SHALLOW,
                skip_lfs=should_skip_lfs,
                filter_blob_none=use_blob_filter,
                git_operation_timeout_seconds=(
                    300 if use_blob_filter else 180
                ),
                git_retry_attempts=2,
            )

            batch_env = self._build_env(batch_config, home_dir)

            extra = ["--filter=blob:none"] if use_blob_filter else []

            
            # NOTE: For normal submodules, we do NOT manually pass `pinned_sha` here.
            # `git submodule update` automatically reads the correct pinned SHA 
            # from the parent repo's git tree.
            cmd = self._git_cmd(
                *auth_flag,
                "submodule", "update",
                "--init",
                "--depth", "1",
                f"--jobs={parallel}",
                *extra,
                "--",
                *normal_paths
            )
            
            logger.info("Initializing %d normal submodules in parallel", len(normal_paths))
            await self._run(cmd=cmd, cwd=repo_path, env=batch_env,timeout_seconds=batch_config.git_operation_timeout_seconds)

        # ── BRANCH B: MONOREPO SUBMODULES (Custom Parallel Sparse Clone) ──
        if monorepo_subs:
            logger.info("Initializing %d monorepo submodules via sparse partial clone", len(monorepo_subs))
            
            tasks = [
                self._clone_sparse_submodule(sub, repo_path, auth_flag,home_dir)
                for sub in monorepo_subs
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    failed_path = monorepo_subs[i].path
                    raise CloneError(f"Sparse clone failed for monorepo submodule {failed_path}: {res}") from res

    async def _clone_sparse_submodule(
        self, 
        submodule: dict, 
        repo_path: Path, 
        auth_flag: list[str], 
        home_dir: Path,
    ) -> None:
        """
        Manually sparse-clones a monorepo submodule. Because we bypass standard 
        submodule commands here, we MUST manually check out the pinned_sha.
        """
        sub_path = submodule["path"]
        url = submodule["url"]
        target_dir = repo_path / sub_path
        if not target_dir.is_relative_to(repo_path.resolve()):
            raise WorkspacePathViolation(f"Submodule path escape detected: {target_dir}")
        
        clone_cfg_dict = submodule.get("clone_config", {})

        strategy_str = clone_cfg_dict.get("strategy") or CloneStrategy.SPARSE_CHECKOUT.value

        sub_clone_config = CloneConfig(
            strategy=strategy_str,
            skip_lfs=clone_cfg_dict.get("skip_lfs", False),
            pinned_sha=clone_cfg_dict.get("pinned_sha"),
            sparse_dirs=clone_cfg_dict.get("sparse_dirs", []),
            git_operation_timeout_seconds=clone_cfg_dict.get(
                "git_operation_timeout_seconds",
                600,
            ),
            git_retry_attempts=clone_cfg_dict.get(
                "git_retry_attempts",
                2,
            ),
                )
        timeout_seconds=sub_clone_config.git_operation_timeout_seconds

        env = self._build_env(sub_clone_config,home_dir=home_dir)

        sparse_dirs = sub_clone_config.sparse_dirs
        if not sparse_dirs:
            subprojects = submodule.get("subprojects", [])
            sparse_dirs = [sp["path"] for sp in subprojects if isinstance(sp, dict) and "path" in sp]

        if not sparse_dirs:
            sparse_dirs = ["/"]

        # 2. Partial clone graph (REQUIRED for sparse checkout)
        # We always use blob:none here regardless of CloneConfig, because fetching 
        # blobs before sparse-checkout defeats the purpose of sparse-checkout.
        clone_cmd = self._git_cmd(
            *auth_flag,
            "clone", "--filter=blob:none", "--no-checkout",
            url, str(target_dir)
        )
        await self._run(cmd=clone_cmd, cwd=repo_path, env=env, timeout_seconds=timeout_seconds)

        # 3. Initialize sparse-checkout in cone mode
        sparse_init_cmd = self._git_cmd("sparse-checkout", "init", "--cone")
        await self._run(cmd=sparse_init_cmd, cwd=target_dir, env=env, timeout_seconds=timeout_seconds)

        # 4. Set the selected subproject directories
        sparse_set_cmd = self._git_cmd("sparse-checkout", "set", *sparse_dirs)
        await self._run(cmd=sparse_set_cmd, cwd=target_dir, env=env, timeout_seconds=timeout_seconds)

        # 5. Check out the Pinned SHA!
        # This is where pinned_sha is crucially used in the commands.
        target_ref = sub_clone_config.pinned_sha if sub_clone_config.pinned_sha else "HEAD"
        checkout_cmd = self._git_cmd("checkout", target_ref)
        await self._run(cmd=checkout_cmd, cwd=target_dir, env=env, timeout_seconds=timeout_seconds)
        
        logger.info(f"Successfully sparse-cloned monorepo submodule '{submodule["path"]}' at {target_ref}")

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
        if not value or not _SAFE_NAME_RE.match(value):
            raise CloneError(
                f"Refusing to clone: {field}={value!r} contains characters "
                f"outside [A-Za-z0-9_.-] — possible path or argument injection."
            )

    def _auth_header_flag(self,auth_token:str) -> None:
        """
        Build the -c http.extraHeader flag carrying a GitHub installation
        token, if provided.
 
        Sent via git config header, never the clone URL — see clone()'s
        docstring for why: a URL-embedded credential gets written into
        the cloned repo's own .git/config on disk (persists past the
        clone, re-exposed if that tree is ever inspected/uploaded), and
        is far more likely to leak into a log line that prints the URL.
        A config header is scoped to this one invocation only.
        """
        basic = base64.b64encode(f"x-access-token:{auth_token}".encode()).decode()

        return ["-c",f"http.extraHeader= AUTHORIZATION : basic {basic}"]    

    def _build_env(self, config:CloneConfig,home_dir:Path) -> dict[str,str]:
        """
        Explicit environment allowlist for the git subprocess.
        """
        env ={
            "PATH":os.environ.get("PATH","/usr/bin:/bin"),
            "HOME":str(home_dir),
            "GIT_TERMINAL_PROMPT":"0",# never block waiting for interactive input
            "GIT_ALLOW_PROTOCOL":"https",# only https transport is ever honored
            "GIT_CONFIG_NOSYSTEM":"1", # Ignore host-level git configs or system config
        }

        if config.skip_lfs:
            env["GIT_LFS_SKIP_SMUDGE"] = "1"
        return env

    def _hook_sink_dir(self) -> Path:
        """
        Return (creating if needed) an empty, unwritable-by-git directory
        used as core.hooksPath for every command. Because it's always
        empty, git can never find a hook script here to execute — this
        is the direct mitigation for CVE-2025-48384's exploitation step,
        independent of whatever path confusion the vulnerability causes
        during checkout.
        """
        if self._hooks_sink is None:
            sink = Path(tempfile.mkdtemp(prefix="git-hooks-sink-"))
            os.chmod(sink,0o500) # read+execute 
            self._hooks_sink = sink
            return self._hooks_sink

    def _git_cmd(self,*args)-> list[str]:
        """
        Build a git command line with mandatory safety flags always
        present, so no call site in this file can accidentally omit them.
 
        -c core.hooksPath=<empty dir>   → hooks can never execute (CVE-2025-48384)
        -c protocol.ext.allow=never     → blocks the `ext::` remote helper
                                          command-injection transport
        -c protocol.file.allow=never    → blocks `file://` local-path reads
                                          via a malicious submodule URL
        -c core.symlinks=false          → git writes a symlink's target as
                                          plain TEXT content, never creates
                                          a real OS-level symlink
        """
        return [
            "git",
            # 1. 🔒 Absolute Security Mitigations
            "-c",f"core.hooksPath={self._hook_sink_dir()}",
            "-c","protocol.ext.allow=never", #no external terminal command
            "-c","protocol.file.allow=never", #no stealing internal system files 
            "-c","core.symlinks=false",#no symbolic link file hijacking
            # 2. ⚡ MicroVM Resource Performance Optimization
            "-c", "feature.manyFiles=true",
            "-c", "checkout.workers=-1",
            "-c", "core.compression=0",
            *args 
        ]

    async def _run(
            self,
            cmd:list[str],
            cwd:Optional[Path] = None,
            env:Optional[dict] = None,
            capture_output:bool = False,
            timeout_seconds:int = 300, #update for dynamic timeout.
    )-> subprocess.CompletedProcess:
        """
        Run a git command asynchronously via asyncio subprocess.
        Raises CloneError on non-zero exit codes OR on timeout.
 
        timeout_seconds default (120s) covers a normal shallow clone of
        a large repo over a slow connection without being so generous
        that a deliberately hostile repo can tie up a worker slot for
        an unbounded amount of time.
        """
        redacted_cmd = self._redact_for_log(cmd)
        logger.debug("Running: %s (cwd=%s)", " ".join(redacted_cmd), cwd)

        process: asyncio.subprocess.Process | None = None
        
        try:
            process = await asyncio.create_subprocess_exec(    
                cmd,
                cwd=str(cwd) if cwd else None,
                env=env,
                stdout=(
                asyncio.subprocess.PIPE
                if capture_output
                else asyncio.subprocess.DEVNULL
                ),
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,

            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout= timeout_seconds
                )

            except asyncio.TimeoutError() as e:
                logger.warning(
                    "Git command timeout after %ss: %s",
                    timeout_seconds,
                    " ".join(redacted_cmd)
                )
                await self._terminate_process_tree(process)

                raise CloneError(
                    f"Git command Timed Out after {timeout_seconds}s :",
                    f"{" ".join(redacted_cmd)}"
                ) from e

            except asyncio.CancelledError:
                logger.warning(
                    "Git command cancelled: %s",
                    " ".join(redacted_cmd),
                )
                await self._terminate_process_tree(process)

                raise

            stdout_text = (
                stdout.decode(errors="replace")
                if stdout is not None
                else ""
            )

            stderr_text = (
                stderr.decode(errors="replace")
                if stderr is not None 
                else ""
            )

            if process.returncode!=0:
                redacted_stderr = self._redact_text(stderr)

                raise CloneError(
                    f"Git command failed "
                    f"(exit {process.returncode}): "
                    f"{' '.join(redacted_cmd)}\n"
                    f"stderr: {redacted_stderr}"
                )

            return (
                process.returncode,
                stdout_text,
                stderr_text,
            )

        except asyncio.CancelledError:
            # Covers cancellation that happens before communicate()
            # is entered or during process creation.
            if process is not None:
                await self._terminate_process_tree(process)

            raise

        except OSError as exc:
            raise CloneError(
                f"Failed to start Git command: "
                f"{' '.join(redacted_cmd)}: {exc}"
            ) from exc
            

    def _redact_for_log(self, cmd: list[str]) -> list[str]:
        """
        Scans the outgoing Git command array and replaces sensitive 
        HTTP authorization tokens with a safe placeholder string.
        """
        cleaned_cmd = []
        
        for argument in cmd:
            # Regex scans for any parameter starting with 'http.extraHeader=AUTHORIZATION:'
            if "http.extraHeader" in argument and "AUTHORIZATION" in argument:
                # Swap out the sensitive token payload with an inert string representation
                cleaned_cmd.append("http.extraHeader=AUTHORIZATION: basic [REDACTED]")
            else:
                # Keep standard, non-sensitive flags exactly as they are
                cleaned_cmd.append(argument)
                
        return cleaned_cmd


    def _redact_text(self, text: Optional[str], active_token: Optional[str] = None) -> str:
        """
        Sanitizes raw text (such as subprocess stderr, stdout, or error messages)
        by masking known secret patterns and any explicitly provided active token.
        """

        _PATTERNS = [
                # 1. Credentials in URLs: https://x-access-token:ghp_xxx@github.com or https://token@github.com
                (re.compile(r"(https?://)([^/\s:@]+)(?::([^/\s@]+))?@"), r"\1[REDACTED]@"),
                
                # 2. GitHub Personal Access Tokens (Classic & Fine-Grained)
                (re.compile(r"ghp_[a-zA-Z0-9]{36}"), "[REDACTED_GITHUB_TOKEN]"),
                (re.compile(r"github_pat_[a-zA-Z0-9]{22}_[a-zA-Z0-9]{59}"), "[REDACTED_GITHUB_PAT]"),
                
                # 3. Other GitHub Token types (OAuth, App, Refresh)
                (re.compile(r"gh[orsu]_[a-zA-Z0-9]{36}"), "[REDACTED_GITHUB_TOKEN]"),
                
                # 4. Bearer / Basic auth header values
                (re.compile(r"(?i)(authorization:\s*bearer\s+)[^\s]+"), r"\1[REDACTED]"),
                (re.compile(r"(?i)(authorization:\s*basic\s+)[^\s]+"), r"\1[REDACTED]"),
            ]
        
        if not text:
            return ""

        sanitized = text

        # 1. Redact exact matches of the current job's auth token if provided
        if active_token and active_token in sanitized:
            sanitized = sanitized.replace(active_token, "[REDACTED_TOKEN]")

        # 2. Run regex replacements for URL credentials and structured token types
        for pattern, replacement in self._PATTERNS:
            sanitized = pattern.sub(replacement, sanitized)

        return sanitized

    async def _terminate_process_tree(
            self,
            process:asyncio.subprocess.Process,
    )-> None:
        """
        Terminate Git and its children.

        start_new_session=True gives Git its own process group.

        This matters because Git can create child processes such as
        ssh, credential helpers, git-upload-pack, etc.
        """

        if process.returncode is not None:
            return

        try:
            os.killpg(
                process.pid,
                signal.SIGTERM
            )

        except ProcessLookupError:
            return

        try:
            asyncio.wait_for(
                process.wait(),
                timeout=5,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Git process did not terminate gracefully; killing: pid=%s",
                process.pid,
            )
            try:
                os.killpg(
                    process.pid,
                    signal.SIGTERM
                    )
            except ProcessLookupError:
                pass

            process.wait()