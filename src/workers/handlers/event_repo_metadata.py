import asyncpg
from src.core.exceptions import TransientWebhookError
from src.core.database import get_system_transaction
from src.core.logger import get_logger


logger = get_logger(__name__)

async def _handle_repository_metadata(payload: dict,delivery_id:str) -> None:
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
    is_archived = repo_data.get("archived", False)
    
    # We only care about metadata changes
    relevant_actions = {"renamed", "privatized", "publicized", "archived", "unarchived"}
    if action not in relevant_actions:
        return

    # Poison Pill Check
    if not repo_id or not full_name:
        raise ValueError(f"Repository webhook payload missing ID or name for action {action} for delivery id:{delivery_id}")

    try:
        async with get_system_transaction() as conn:
            # We update the name and the privacy flag simultaneously.
            # If a repo is archived, we can mark it stale/inaccessible if desired,
            # but updating the name is the critical part here.
            if action == "archived":
                # Lock the repository down so workers stop scanning it
                status_tag = await conn.execute(
                    """
                    UPDATE repos SET
                        private      = $1,
                        is_archived  = TRUE,
                        index_status = 'inaccessible',  -- 👈 Freeze the UI/Pipeline actions
                        updated_at   = NOW()
                    WHERE github_repo_id = $2
                    """,
                    is_private,
                    repo_id,
                )
            
            elif action == "unarchived":
                # Unlock the repository back to a standard ready or not_indexed state
                status_tag = await conn.execute(
                    """
                    UPDATE repos SET
                        private      = $1,
                        is_archived  = FALSE,
                        index_status = 'not_indexed',   -- 👈 Let the user re-index it now
                        updated_at   = NOW()
                    WHERE github_repo_id = $2
                    """,
                    is_private,
                    repo_id,
                )

            else:
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
            
            # Robust check for asyncpg command execution strings (e.g., 'UPDATE 1')
            if status_tag and (status_tag == "UPDATE 1" or str(status_tag).endswith("1")):
                logger.info(
                    "Repo ID %d metadata synced (Action: %s). New name: %s for delivery id: %s", 
                    repo_id, action, full_name, delivery_id
                )
            else:
                logger.warning(
                    "Repo ID %d metadata sync processed but 0 matching records were updated "
                    "in DB (Action: %s) for delivery id: %s",
                    repo_id, action, delivery_id
                )
    
    except asyncpg.PostgresError as e:
        logger.error("Database failed syncing repo metadata for ID %d: %s", repo_id, e)
        raise TransientWebhookError(f"DB failed during repository metadata sync: {e}")
    
    logger.info(
        "Successfully completed repository metadata sync processing loop for delivery id: %s",
        delivery_id
    )