
from __future__ import annotations
from pathlib import Path
import shutil 
import os
import tempfile
from typing import Optional
from uuid import UUID
import logging
import stat

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKSPACE_BYTE = 2 * 1024 * 1024 * 1024  #2GB

class JobWorkspace:
    """
    single workspace for each tenant
    """

    def __init__(
        self,
        job_id:UUID | str,
        account_id:UUID | str,
        base_dir = "/temp/ingestion",
        max_byte:int = DEFAULT_MAX_WORKSPACE_BYTE,

    ):
        self.job_id = str(job_id),
        self.account_id = str(account_id),
        self._base_dir = Path(base_dir),
        self.max_byte = max_byte,
        self._root: Optional[Path] = None

    #Lifecycle

    async def __aenter__(self) -> JobWorkspace:
        self._root = self._create_root_dir
        self._create_subdir("clone")
        self._create_subdir("manifest")
        self._create_subdir("temp")

    async def __aexit__(self, exc_type, exc_val, exc_tb) ->None:
        self._distroy_root_dir
        if exc_type is not None:
            logger.warning("Workspace closed after error: Job_id:%s error=%s, detailed=%s",self.job_id,exc_val,exc_tb)
        else:
            logger.info("Workspace closed cleanly : Job_id=%s",self.job_id)

    def _create_root_dir(self):
        """
        Create a fresh, uniquely-named root directory for this job.
        Uses job_id in the name (not account_id alone) so two jobs for
        the same account never collide, and mkdtemp's randomness means
        a stale directory from a crashed prior run can never be reused
        by accident.
        """
        self._base_dir.mkdir(parent=True ,exist_ok = True)
        root_str = tempfile.mkdtemp(
            prefix=f"job-{self.job_id}-",
            dir=self._base_dir
        )
        root = Path(root_str)
        os.chmod(root,stat.S_IRWXU)

        return root

    def _distroy_root_dir(self)-> None:
        """
        Remove the entire workspace tree. Called unconditionally on exit.
        ignore_errors=True because a half-written malicious repo could in
        theory leave permission bits that make cleanup noisy — we still
        want the rest of shutdown to proceed. The S3 24h lifecycle rule
        and the fact that /tmp is itself ephemeral per Fargate task are
        the backstops if this ever fails silently.
        """
        if self._root and self._root.exists():
            shutil.rmtree(self._root,ignore_errors=True)

    def _create_subdir(self,name:str):
        sub = self._root/name
        sub.mkdir(parents=True , exist_ok=True)
        return sub

    #path containment— the actual enforcement of "no writes outside workspace"

