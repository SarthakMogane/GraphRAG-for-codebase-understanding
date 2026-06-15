"""
app/services/pre_clone
──────────────────────
Pre-clone pipeline — everything that runs before the first git command.

Public API:
    from app.services.pre_clone import PreClonePipeline, ValidationResult

Usage:
    pipeline = PreClonePipeline(github_service=gh, db=db)
    validation, monorepo = await pipeline.run("https://github.com/owner/repo")

    if not validation.should_proceed:
        return error_response(validation.message)

    if validation.routing == RoutingDecision.SERVE_CACHE:
        return serve_existing_wiki(validation.existing_repo_db_id)

    # Proceed to clone with:
    #   validation  → clone strategy input
    #   monorepo    → sparse checkout dirs
"""

from src.services.pre_clone.pipeline import PreClonePipeline
from src.services.pre_clone.types import (
    MonorepoDetectionResult,
    MonorepoTooling,
    ParsedURL,
    RoutingDecision,
    StaleCheckResult,
    SubProjectScore,
    ValidationResult,
    ValidationVerdict,
)
from src.services.pre_clone.url_parser import parse_github_url

__all__ = [
    "PreClonePipeline",
    "MonorepoDetectionResult",
    "MonorepoTooling",
    "ParsedURL",
    "RoutingDecision",
    "StaleCheckResult",
    "SubProjectScore",
    "ValidationResult",
    "ValidationVerdict",
    "parse_github_url",
]