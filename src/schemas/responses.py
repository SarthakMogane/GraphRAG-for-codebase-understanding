from datetime import datetime
from typing import Any, Optional,List
from pydantic import BaseModel, Field, HttpUrl, field_validator
import re


class IndexResponse(BaseModel):
    repo_id:  int
    next:     str           # "scout" | "ingest" | "none"
    message:  str
    job_id:   Optional[int]    = None   # set when next="ingest"
    wiki_url: Optional[str]    = None   # set when next="none"
 
    # Set when next="scout" and it's a structural re-scout
    # Frontend uses this to show NEW/REMOVED badges on the checklist
    structural_diff: Optional[dict] = None

class RepoResponse(BaseModel):
    id: int
    github_owner: str
    github_repo: str
    status: str
    clone_strategy: Optional[str]
    is_monorepo: bool
    last_ingested_sha: Optional[str]
    last_ingested_at: Optional[datetime]
    created_at: datetime

    model_config = {"from_attributes": True}

class JobResponse(BaseModel):
    id: int
    repo_id: int
    job_type: str
    celery_task_id: Optional[str]
    status: str
    queued_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    files_total: Optional[int]
    files_accepted: Optional[int]
    files_excluded: Optional[int]
    error_message: Optional[str]

    model_config = {"from_attributes": True}


class SubmitRepoResponse(BaseModel):
    repo: RepoResponse
    job: JobResponse
    message: str

class FileManifestSummary(BaseModel):
    total_files: int
    tier1_source: int
    tier2_config: int
    tier3_docs: int
    excluded: int
    entry_points: int
    test_files: int
    generated_files: int
class SubmoduleSummary(BaseModel):
    path: str
    resolved_owner: Optional[str]
    resolved_repo: Optional[str]
    is_internal: Optional[bool]
    outcome: str
    skip_reason: Optional[str]
    complexity_band: Optional[str]

    model_config = {"from_attributes": True}

class IngestionStatusResponse(BaseModel):
    repo: RepoResponse
    latest_job: Optional[JobResponse]
    manifest_summary: Optional[FileManifestSummary]
    submodules: list[SubmoduleSummary]
class QueryResponse(BaseModel):
    """Query response schema"""
    query: str
    answer: str
    query_type: str  # 'global' or 'local'
    confidence: float
    sources: List[dict]
    verified: bool
    timing: dict
    request_id: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    components: dict


class MetricsResponse(BaseModel):
    """Metrics response"""
    total_queries: int
    avg_latency: float
    cache_hit_rate: float
    error_rate: float


class WebhookPayload(BaseModel):
    """
    Incoming GitHub webhook payload.
    We only care about push and pull_request[closed+merged] events.
    """
    ref: Optional[str] = None                # "refs/heads/main"
    before: Optional[str] = None             # SHA before push
    after: Optional[str] = None              # SHA after push
    repository: Optional[dict[str, Any]] = None
    pull_request: Optional[dict[str, Any]] = None
    action: Optional[str] = None            # "closed" for PR events