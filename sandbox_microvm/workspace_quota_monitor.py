from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path
from typing import Awaitable, Callable, Optional
from src.core.logger import get_logger
from sandbox_microvm.workspace import WorkspaceQuotaExceeded


logger = get_logger(__name__)


class FilesystemSpaceExceeded(Exception):
    """The filesystem is approaching its configured safety boundary."""


class WorkspaceQuotaMonitor:
    """
    Monitors a job workspace while a long-running operation is running.

    Two different measurements are intentionally used:

    1. statvfs():
       Cheap filesystem-level check performed frequently.

       This answers:
           "Is the filesystem running out of space?"

    2. os.walk():
       Workspace-specific accounting performed less frequently.

       This answers:
           "Has THIS job's workspace exceeded its logical quota?"

    The sandbox / MicroVM storage limit remains the ultimate hard boundary.
    This monitor provides earlier detection and graceful cancellation.
    """

    def __init__(
        self,
        workspace,
        *,
        filesystem_check_interval: float = 1.0,
        workspace_check_interval: float = 10.0,
        filesystem_min_free_bytes: int | None = None,
        filesystem_max_used_ratio: float | None = None,
    ):
        self.ws = workspace

        self.filesystem_check_interval = filesystem_check_interval
        self.workspace_check_interval = workspace_check_interval

        # Optional filesystem-level safety boundaries.
        #
        # Example:
        # filesystem_min_free_bytes = 512 * 1024 * 1024
        #
        # or:
        # filesystem_max_used_ratio = 0.90
        self.filesystem_min_free_bytes = filesystem_min_free_bytes
        self.filesystem_max_used_ratio = filesystem_max_used_ratio

        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._failure: Optional[BaseException] = None

    # ------------------------------------------------------------------
    # Filesystem-level monitoring
    # ------------------------------------------------------------------

    def _filesystem_usage(self) -> tuple[int, int, int]:
        """
        Return:

            total_bytes
            used_bytes
            available_bytes

        based on the filesystem containing the workspace.
        """

        root = self.ws.root

        stats = os.statvfs(root)

        total_bytes = stats.f_blocks * stats.f_frsize
        available_bytes = stats.f_bavail * stats.f_frsize
        used_bytes = total_bytes - available_bytes

        return total_bytes, used_bytes, available_bytes

    def _enforce_filesystem_limit(self) -> None:
        """
        Fast filesystem-level safety check.

        This does NOT replace workspace quota accounting because
        statvfs() measures the filesystem, not just /workspace.
        """

        # Nothing configured => nothing to enforce.
        if (
            self.filesystem_min_free_bytes is None
            and self.filesystem_max_used_ratio is None
        ):
            return

        total, used, available = self._filesystem_usage()

        if (
            self.filesystem_min_free_bytes is not None
            and available < self.filesystem_min_free_bytes
        ):
            raise FilesystemSpaceExceeded(
                f"Filesystem containing workspace is running low on space: "
                f"available={available:,} bytes, "
                f"minimum_required={self.filesystem_min_free_bytes:,} bytes"
            )

        if self.filesystem_max_used_ratio is not None:
            if total <= 0:
                raise FilesystemSpaceExceeded(
                    "Unable to determine filesystem capacity."
                )

            used_ratio = used / total

            if used_ratio >= self.filesystem_max_used_ratio:
                raise FilesystemSpaceExceeded(
                    f"Filesystem usage reached {used_ratio:.2%}; "
                    f"maximum allowed is "
                    f"{self.filesystem_max_used_ratio:.2%}"
                )

   