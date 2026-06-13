
import asyncpg
from src.core.logger import get_logger
from src.core.exceptions import TransientWebhookError
from src.core.database import get_system_transaction

logger = get_logger(__name__)

async def _handle_push(payload: dict) -> None:
    repo_data      = payload.get("repository", {})
    repo_id        = repo_data.get("id")          
    default_branch = repo_data.get("default_branch", "main")
    ref            = payload.get("ref", "")

    if ref != f"refs/heads/{default_branch}":
        return

    if not repo_id:
        raise ValueError("Push webhook payload missing repository.id")

    try:
        async with get_system_transaction() as conn:
            status_tag = await conn.execute(
                """
                UPDATE repos SET
                    index_status = 'stale',
                    updated_at   = NOW()
                WHERE github_repo_id = $1
                  AND index_status   = 'ready'
                """,
                repo_id,
            )
            
            if status_tag == "UPDATE 1":
                logger.info("Repo ID %d marked STALE via push", repo_id)

    except asyncpg.PostgresError as e:
        logger.error("Database failed during push event for repo_id %d: %s", repo_id, e)
        raise TransientWebhookError(f"DB failed during push event: {e}")


async def _handle_pull_request(payload: dict) -> None:
    action = payload.get("action")
    pr     = payload.get("pull_request", {})

    is_merged        = pr.get("merged", False)
    target_branch    = pr.get("base", {}).get("ref", "")
    repo_data        = payload.get("repository", {})
    repo_id          = repo_data.get("id")       
    default_branch   = repo_data.get("default_branch", "main")

    if action != "closed" or not is_merged or target_branch != default_branch:
        return

    if not repo_id:
        raise ValueError("PR webhook payload missing repository.id")

    try:
        async with get_system_transaction() as conn:
            status_tag = await conn.execute(
                """
                UPDATE repos SET
                    index_status = 'stale',
                    updated_at   = NOW()
                WHERE github_repo_id = $1
                  AND index_status   = 'ready'
                """,
                repo_id,
            )
            
            if status_tag == "UPDATE 1":
                logger.info("Repo ID %d marked STALE via merged PR", repo_id)

    except asyncpg.PostgresError as e:
        logger.error("Database failed during PR event for repo_id %d: %s", repo_id, e)
        raise TransientWebhookError(f"DB failed during PR event: {e}")