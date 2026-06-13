import asyncpg
from src.core.logger import get_logger
from src.core.exceptions import TransientWebhookError
from src.core.database import get_system_transaction

logger = get_logger(__name__)

async def _handle_app_authorization(payload: dict) -> None:
    """
    Handles 'github_app_authorization' events.
    Fires when a human user revokes their OAuth web login from GitHub settings.
    We must immediately destroy their active sessions and access tokens.
    """
    action = payload.get("action")
    
    # We only care about the revocation event
    if action != "revoked":
        return
        
    sender = payload.get("sender", {})
    github_user_id = sender.get("id")
    
    # Poison Pill Check
    if not github_user_id:
        raise ValueError("github_app_authorization payload missing sender.id")
        
    try:
        # We use the system transaction because the worker needs to find 
        # and update the user row without an active user JWT context.
        async with get_system_transaction() as conn:
            
            # NOTE: Adjust these column names to perfectly match your users table schema.
            # The goal is to wipe out the tokens and force a re-login.
            status_tag = await conn.execute(
                """
                UPDATE users 
                SET 
                    github_oauth_token = NULL,
                    github_refresh_token = NULL,
                    updated_at = NOW()
                WHERE github_id = $1
                """,
                github_user_id
            )
            
            if status_tag == "UPDATE 1":
                logger.info(
                    "OAuth authorization revoked for GitHub User ID %d. Sessions destroyed.", 
                    github_user_id
                )
            else:
                # If they revoke access but never actually completed our DB signup flow
                logger.debug(
                    "OAuth revoked for GitHub ID %d, but user not found in DB.", 
                    github_user_id
                )
                
    except asyncpg.PostgresError as e:
        logger.error("Database failed during auth revoke for user %d: %s", github_user_id, e)
        raise TransientWebhookError(f"DB failed during app authorization revoke: {e}")