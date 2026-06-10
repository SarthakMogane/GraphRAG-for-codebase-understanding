"""
app/core/config.py
──────────────────
All configuration loaded from environment variables via Pydantic Settings.
Every service in the system imports from here — never raw os.getenv().
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from functools import lru_cache
from typing import Optional
import os


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )
        # ── Application ──────────────────────────────────────────────────
    APP_NAME: str = "repo-ingestion"
    APP_ENV: str = Field(default="development", pattern="^(development|staging|production)$")
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"


    #fronted URL :
    # Local Development
    FRONTEND_URL:str

# When you deploy, you'll change it to:
# FRONTEND_URL="https://your-production-domain.com"
    
        # ── GitHub ────────────────────────────────────────────────────────
    # Service account GitHub App credentials for high-rate-limit API access
    GITHUB_APP_ID: int
    GITHUB_APP_PRIVATE_KEY: str          # PEM key as string (newlines as \n)
    # GITHUB_APP_INSTALLATION_ID: int
    GITHUB_WEBHOOK_SECRET: str           # For validating incoming webhook payloads

    # Per-token rate limits: 5000 req/hr for authenticated, 60 for anonymous
    GITHUB_API_RATE_LIMIT_BUFFER: int = 200   # Stop at N calls remaining

        # ── PostgreSQL ───────────────────────────────────────────────────
    DATABASE_URL: str                    # postgresql+asyncpg://user:pass@host/db
    DB_POOL_MIN_SIZE: int = 20
    DB_POOL_MIN_SIZE: int = 10

    DATABASE_READ_URL:str
    DB_READ_POOL_MIN_SIZE:int =20
    DB_READ_POOL_MAX_SIZE:int = 10

    RDS_CA_BUNDLE_PATH:str

    # crypto 
    LOCAL_ENCRYPTION_KEY:str
    KMS_KEY_ARN_TOKENS: str
    AWS_REGION:str
    APP_ENV:str

    
    # App Settings
        # ── Redis (Celery broker + result backend) ───────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_EXPIRES: int = 86400   # 24 hours in seconds

    # ── Object Storage (S3-compatible / GCS) ─────────────────────────
    STORAGE_BUCKET: str                  # Bucket for file corpus storage
    STORAGE_ENDPOINT_URL: Optional[str] = None   # None = AWS S3; set for GCS/MinIO
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None

    # ── Clone / Ingestion Limits ──────────────────────────────────────
    # These constants encode all the thresholds discussed in the architecture
    CLONE_WORK_DIR: str = "/tmp/ingestion_clones"

    # Repository size thresholds (GitHub reports in KB)
    REPO_SIZE_SMALL_KB: int = 51_200        # < 50MB  → full shallow clone
    REPO_SIZE_MEDIUM_KB: int = 512_000      # < 500MB → partial clone blob:none
    # > 500MB → sparse checkout

    # File filter thresholds
    FILE_SIZE_FULL_PARSE_BYTES: int = 512_000       # 500KB  → full parse
    FILE_SIZE_MODULE_ONLY_BYTES: int = 1_048_576    # 1MB    → module-level only
    # > 1MB text → skip, record as opaque node

    # Submodule limits
    SUBMODULE_MAX_DEPTH: int = 2             # Max nesting depth before hard skip
    SUBMODULE_EXTERNAL_MAX_SIZE_KB: int = 51_200   # 50MB external size gate
    SUBMODULE_PARALLEL_JOBS: int = 8         # git submodule update --jobs N

    # Monorepo sub-project scoring
    MONOREPO_MAX_SUBPROJECTS: int = 50       # Cap before score-based pruning kicks in
    MONOREPO_MIN_SOURCE_FILES: int = 3       # Sub-projects with fewer files are stubs

    # Commit history sampling depth
    COMMIT_HISTORY_SAMPLE_SIZE: int = 100

    # ── Celery Queue Names ────────────────────────────────────────────
    QUEUE_INGESTION: str = "ingestion"
    QUEUE_SUBMODULE: str = "submodule"
    QUEUE_WEBHOOK: str = "webhook"

    @field_validator("GITHUB_APP_PRIVATE_KEY")
    @classmethod
    def normalize_pem(cls, v: str) -> str:
        """Replace literal \n with real newlines if key was env-encoded."""
        return v.replace("\\n", "\n")
    



    PROJECT_NAME: str = "Enterprise GraphRAG API"
    VERSION: str = "1.0.0"
    
    # Security
    SESSION_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    
    # GitHub OAuth
    GITHUB_CLIENT_ID: str
    GITHUB_CLIENT_SECRET: str
    
   

@lru_cache
def get_settings() -> Settings:
    """
    Singleton settings instance.
    Use as a FastAPI dependency: settings = Depends(get_settings)
    Or import directly: from app.core.config import get_settings; s = get_settings()
    """
    return Settings()

