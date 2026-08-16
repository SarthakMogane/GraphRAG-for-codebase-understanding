from core.logger import get_logger
from dataclasses import dataclass,field
from typing import Optional
from src.models.database import CloneStrategy
from src.services.github import RepoMetadata
from sandbox_microvm.models import CloneSettings
logger = get_logger(__name__)

@dataclass
class CloneConfig:
    strategy:CloneStrategy
    depth : int = 1
    single_branch:bool = True
    filter_blob_none:bool = False
    no_checkout:bool = False
    sparse_dirs:list[str] = field(default_factory=list)
    skip_lfs:bool = False #True:GIT_LFS_SKIP_SMUDGE=1
    recurse_submodule:bool = False
    estimated_disk_mb:int = 0 

class CloneStrategySelector:
    def __init__(self,settings: CloneSettings):
        self.settings = settings
    def select(
        self,
        metadata:RepoMetadata,
        is_monorepo:bool = False,
        sparse_dir:Optional[list[str]] = None,
    ) -> CloneConfig:

        "select Strategy for cloning the repo based on args"

        size_kb = metadata.size_kb

        if is_monorepo:
            strategy = self._monorepo_strategy(metadata,sparse_dir)
        elif size_kb <= self.settings.REPO_SIZE_SMALL_KB:
            strategy = self._small_repo_strategy(metadata)
        elif size_kb <= self.settings.REPO_SIZE_MEDIUM_KB:
            strategy = self._medium_repo_strategy(metadata)
        else:
            strategy = self._large_repo_strategy(metadata)

        logger.info(
        "Clone strategy selected for %s/%s: strategy=%s size_kb=%d",
        metadata.owner, metadata.name, strategy.strategy.value, size_kb
        )

        return strategy

    def _small_repo_strategy(self, metadata:RepoMetadata) -> CloneConfig:
        """
        < 50MB: Simple shallow clone.
        Everything lands on disk; filter pipeline handles the rest.
        LFS skip only if LFS is detected (avoid downloading model weights, etc.)
        """
        return CloneConfig(
            strategy=CloneStrategy.SHALLOW,
            depth=1,
            single_branch=True,
            filter_blob_none=False,
            skip_lfs=metadata.uses_git_lfs,
            estimated_disk_mb=metadata.size_kb//1024
        )

    def _medium_repo_strategy(self, metadata:RepoMetadata) -> CloneConfig:
        """
        50MB-500MB: Partial clone with blob filtering.
        Tree structure and file metadata arrive immediately.
        File content is fetched lazily — only for files that pass the filter pipeline.
        Binary assets filtered by Phase 5 are never downloaded at all.
        """

        return CloneConfig(
            strategy=CloneStrategy.PARTIAL_BLOB,
            depth=1,
            single_branch=True,
            filter_blob_none=True,
            skip_lfs=metadata.uses_git_lfs,
            estimated_disk_mb=metadata.size_kb//2048
        )

    def _large_repo_strategy(self, metada:RepoMetadata) -> CloneConfig:
        """
         > 500MB: Partial clone + sparse checkout.
        Nothing lands on disk until specifically requested.
        sparse_dirs will be populated by MonorepoDetector separately.
        """

        return CloneConfig(
            strategy=CloneStrategy.PARTIAL_BLOB,   # Upgraded to sparse in execute phase
            depth=1,
            single_branch=True,
            filter_blob_none=True,
            no_checkout=True,          # --no-checkout: tree on disk but no files
            skip_lfs=True,
            estimated_disk_mb=50,      # Minimal until sparse checkout expands
        )

    
        