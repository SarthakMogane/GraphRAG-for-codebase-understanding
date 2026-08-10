import asyncio
import logging
from typing import Set

logger = logging.getLogger(__name__)

# This is your global task tracker to prevent garbage collection
_active_background_tasks: Set[asyncio.Task] = set()

def handle_task_completion(task: asyncio.Task) -> None:
    """
    The structural supervisor for background workers. 
    Triggered automatically the microsecond a pipeline concludes.
    """
    # 1. IMMEDIATE ACTION: Evict from global set to prevent RAM leaks
    _active_background_tasks.discard(task)

    try:
        # 2. Extract the task result. If an unhandled exception occurred,
        # or if the task was cancelled, it will throw the error right here.
        task.result()
        
    except asyncio.CancelledError:
        # --- SCENARIO A: AWS ECS Container Shutdown Triggered ---
        # Bypassed business logic (BaseException). Handled here cleanly.
        job_id = getattr(task, "job_id", "Unknown ID")
        logger.warning(
            "Tracked Task [Job %s] aborted via SIGTERM/Cancellation. "
            "Spawning background state mitigation worker...", job_id
        )
        # Action: Fire an independent fallback task to safely clean up 
        # state in database/AWS without locking the event loop.
        asyncio.create_task(_emergency_shutdown_cleanup(task))

    except Exception as unhandled_infra_crash:
        # --- SCENARIO B: Nested Crash (e.g., Error handler crashed on dead DB) ---
        job_id = getattr(task, "job_id", "Unknown ID")
        logger.critical(
            "CRITICAL SYSTEM BREAKDOWN: Task [Job %s] crashed outside of business logic "
            "boundaries. Exception: %s", job_id, unhandled_infra_crash, exc_info=True
        )
        # Action: Route this directly to your developer alerts rotation!
        _send_alert_to_devops_pager(job_id, unhandled_infra_crash)


async def _emergency_shutdown_cleanup(task: asyncio.Task) -> None:
    """Safely transitions dangling user jobs during container eviction cycles."""
    try:
        job_id = getattr(task, "job_id", None)
        if not job_id:
            return
            
        logger.info("Executing graceful boot-retry flagging for Job %s...", job_id)
        # Action: Use your global application context to reset the user's status 
        # in the database so the next healthy Fargate instance can pick it up.
        # await database.update_status(job_id, status="SIGTERM_CONTAINER_RETRY")
        
    except Exception as e:
        logger.error("Failed to execute emergency database cleanup: %s", e)


def _send_alert_to_devops_pager(job_id: str, error: Exception) -> None:
    """Plug your monitoring clients (Sentry, Slack Webhooks, Datadog) here."""
    # Example: sentry_sdk.capture_exception(error)
    pass
