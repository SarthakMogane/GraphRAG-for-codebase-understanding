from typing import Optional
from src.core.database import get_system_transaction
from src.core.logger import get_logger
import json

logger = get_logger(__name__)

async def _mark_processed(
    delivery_id: str,
    processed:bool,
    error: Optional[str] = None,
    event_type: Optional[str] = "unknown_recovered_event",
    payload: Optional[dict] = None
) -> None:
    """
    Mark a webhook as processed (or failed) in the idempotency table.
    Called at the end of every handler — success or failure.
    Swallows its own exceptions to avoid masking the original error.
    """
    try:
        # Safely extract fallback data in case this is a brand new insert
        github_install_id = None
        repo_full_name = None
        payload_str = "{}"
        
        if payload:
            github_install_id = (
                payload.get("installation", {}).get("id") 
                or payload.get("installation", {}).get("app_id")
            )
            repo_full_name = payload.get("repository", {}).get("full_name")
            payload_str = json.dumps(payload)

        async with get_system_transaction() as conn:
            
                await conn.execute(
                    """
                INSERT INTO webhooks_received (
                    delivery_id, event_type, github_install_id, 
                    repo_full_name, payload, processed, error, 
                    processed_at, updated_at
                )
                VALUES (
                    $1, $2, $3, $4,$5::jsonb, $6, $7,
                    -- Magic Trick: Set processed_at only if it succeeded!
                    CASE WHEN $6 = TRUE THEN NOW() ELSE NULL END,
                    NOW()
                )
                ON CONFLICT (delivery_id) 
                DO UPDATE SET 
                    processed = EXCLUDED.processed,
                    error = EXCLUDED.error,
                    updated_at = NOW(),
                    -- Keep old processed_at if it was already set, otherwise update it on success
                    processed_at = CASE 
                        WHEN EXCLUDED.processed = TRUE THEN NOW() 
                        ELSE webhooks_received.processed_at 
                    END
                """,
                delivery_id,
                event_type,
                github_install_id,
                repo_full_name,
                payload_str,
                processed,
                error
        
                )

    except Exception as e:
        # Log but don't raise — we're already in a finally-like position
        logger.error(
            "Failed to mark webhook processed — delivery=%s: %s",
            delivery_id, e,
        )

