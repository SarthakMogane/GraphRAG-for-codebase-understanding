"""
app/services/pre_clone/rate_limit_checker.py
──────────────────────────────────────────────
Rate limit pre-flight check.

Called once at the very start of the validation pipeline, before any
other GitHub API calls are made. If we're too close to the limit,
we stop immediately and return RATE_LIMIT_BLOCKED rather than burning
our remaining calls and then failing halfway through ingestion.

GitHub REST API limits:
  - Authenticated (GitHub App): 5,000 requests/hour per installation
  - Secondary rate limits: triggered by bursting (>100 req/min to a single endpoint)

This module fetches /rate_limit which does NOT count against the primary limit.
"""

import logging
from datetime import datetime, timezone

import httpx

from src.core.config import get_settings
from src.services.pre_clone.types import RateLimitStatus

logger = logging.getLogger(__name__)
settings = get_settings()


async def check_rate_limit(gh ,installation_id) -> RateLimitStatus:
    """
    Fetch current rate limit state from GitHub's /rate_limit endpoint.

    This endpoint is free — it does NOT decrement your remaining count.
    We call it as the very first action in the validation pipeline.

    Returns a RateLimitStatus. The caller checks .is_safe before proceeding.
    """
    try:
            resp = await gh.fetch_rate_limit(installation_id)

            remaining = resp["remaining"]
            limit     = resp["limit"]
            reset_ts  = resp["reset"]
            used      = resp["used"]

            status = RateLimitStatus(
                remaining=remaining,
                limit=limit,
                reset_at=datetime.fromtimestamp(reset_ts, tz=timezone.utc),
                used=used,
            )

            logger.info(
                "Rate limit: %d/%d remaining (%.1f%% used), resets at %s",
                remaining, limit, status.percent_used,
                status.reset_at.strftime("%H:%M:%S UTC"),
            )

            if not status.is_safe:
                seconds_to_reset = max(
                    0,
                    int((status.reset_at - datetime.now(timezone.utc)).total_seconds())
                )
                logger.warning(
                    "Rate limit buffer breached: %d remaining < %d buffer. "
                    "Reset in %ds.",
                    remaining, settings.GITHUB_API_RATE_LIMIT_BUFFER, seconds_to_reset,
                )

            return status

    except httpx.HTTPError as e:
        # If rate_limit endpoint itself fails (network issue, auth problem),
        # return a worst-case status so we fail safe — don't proceed.
        logger.error("Failed to fetch rate limit status: %s", e)
        return RateLimitStatus(
            remaining=0,
            limit=5000,
            reset_at=datetime.now(timezone.utc),
            used=5000,
        )