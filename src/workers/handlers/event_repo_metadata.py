import asyncpg
from src.core.exceptions import TransientWebhookError
from src.core.database import get_system_transaction
from src.core.logger import get_logger


logger = get_logger(__name__)

async def _handle_repository_metadata(payload: dict) -> None:
    """
    Catches repository renames, privacy toggles, and archiving.
    Keeps our database perfectly synced with GitHub's current state
    so our UI and Git Clone commands never break.
    """
    action = payload.get("action")
    repo_data = payload.get("repository", {})
    
    repo_id = repo_data.get("id")
    full_name = repo_data.get("full_name")
    repo_name = repo_data.get("name")
    owner_login = repo_data.get("owner", {}).get("login")
    is_private = repo_data.get("private", True)
    
    # We only care about metadata changes
    relevant_actions = {"renamed", "privatized", "publicized", "archived", "unarchived"}
    if action not in relevant_actions:
        return

    # Poison Pill Check
    if not repo_id or not full_name:
        raise ValueError(f"Repository webhook payload missing ID or name for action {action}")

    try:
        async with get_system_transaction() as conn:
            # We update the name and the privacy flag simultaneously.
            # If a repo is archived, we can mark it stale/inaccessible if desired,
            # but updating the name is the critical part here.
            status_tag = await conn.execute(
                """
                UPDATE repos SET
                    full_name   = $1,
                    repo_name   = $2,
                    owner_login = $3,
                    private     = $4,
                    updated_at  = NOW()
                WHERE github_repo_id = $5
                """,
                full_name,
                repo_name,
                owner_login,
                is_private,
                repo_id,
            )
            
            if status_tag == "UPDATE 1":
                logger.info(
                    "Repo ID %d metadata synced (Action: %s). New name: %s", 
                    repo_id, action, full_name
                )

    except asyncpg.PostgresError as e:
        logger.error("Database failed syncing repo metadata for ID %d: %s", repo_id, e)
        raise TransientWebhookError(f"DB failed during repository metadata sync: {e}")