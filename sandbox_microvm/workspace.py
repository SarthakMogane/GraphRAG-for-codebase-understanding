
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
        self.base_dir = Path(base_dir),
        self.max_byte = max_byte,
        self._root: Optional[Path] = None

    #Lifecycle

    async def __aenter__(self) -> JobWorkspace:
        self._root = self._create_root_dir

    async def __aexit__(self, exc_type, exc_val, exc_tb) ->None:
        self._distroy_root_dir
        if exc_type is not None:
            logger.warning("Workspace closed after error: Job_id:%s error=%s, detailed=%s",self.job_id,exc_val,exc_tb)
        else:
            logger.info("Workspace closed cleanly : Job_id=%s",self.job_id)

