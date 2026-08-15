from core.logger import get_logger
from dataclasses import dataclass,field
from src.models.database import CloneStrategy
logging = get_logger(__name__)

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

