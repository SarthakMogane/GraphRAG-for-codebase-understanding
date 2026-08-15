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