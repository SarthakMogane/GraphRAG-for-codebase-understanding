from typing import Optional
from src.core.database import get_system_transaction
from src.core.logger import get_logger

logger = get_logger(__name__)

async def _mark_processed(
    delivery_id: str,
    processed:bool,
    error: Optional[str] = None,
) -> None:
    """
    Mark a webhook as processed (or failed) in the idempotency table.
    Called at the end of every handler — success or failure.
    Swallows its own exceptions to avoid masking the original error.
    """
    try:
        async with get_system_transaction() as conn:
            if processed:
                await conn.execute(
                    """
                    UPDATE webhooks_received SET
                        processed    = TRUE,
                        processed_at = NOW(),
                        error        = $2
                    WHERE delivery_id = $1
                    """,
                    delivery_id,
        
                )
            else:
                # Failure: Keep processed=FALSE, record the error text
                await conn.execute(
                    """
                    UPDATE webhooks_received
                    SET 
                        processed = FALSE,
                        error = $2,
                        updated_at = NOW()
                    -- Do not touch processed_at because it hasn't succeeded yet
                    WHERE delivery_id = $1
                    """,
                    delivery_id,
                    error
                )
    except Exception as e:
        # Log but don't raise — we're already in a finally-like position
        logger.error(
            "Failed to mark webhook processed — delivery=%s: %s",
            delivery_id, e,
        )

