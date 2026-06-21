from fastapi import APIRouter, Request, HTTPException, Header , Depends
from src.core.logger import get_logger
from src.utils.services_helpers import get_sqs_client
from src.services.github import GitHubService
from src.core.database import get_system_transaction
from src.core.config import get_settings
import json


logger = get_logger(__name__)
settings = get_settings()
router = APIRouter(prefix="/api/webhooks", tags=["Webhooks"])

@router.post("/github")
async def github_webhook(
    request: Request,
    x_hub_signature_256: str = Header(None),
    x_github_event: str = Header(None),
    x_github_delivery: str   = Header(None),
    sqs_client = Depends(get_sqs_client)
):
    """Listens for GitHub App events (like installs and uninstalls)."""
    
    # 1. SECURITY FIRST: Read the raw bytes of the request body
    payload_bytes = await request.body()
    
    # 2. Cryptographically verify GitHub sent this using your webhook secret
    if not GitHubService.validate_webhook_signature(payload_bytes, x_hub_signature_256):
        logger.error("Webhook signature validation failed!")
         # Still return 200 — GitHub should not retry invalid signatures
        return {"status": "ignored", "reason": "invalid_signature"}
    
    if not x_github_delivery or not x_github_event:
        return {"status": "ignored", "reason": "missing_headers"}

    # 3. Parse the JSON safely now that we know it's authentic
    payload = await request.json()
    action = payload.get("action")
    
    #___idempotency check__
    try:
        async with get_system_transaction() as conn:
            existing = await conn.fetchval(
                "SELECT id FROM webhooks_received WHERE delivery_id = $1",
                x_github_delivery,
            )
            if existing:
                logger.info(
                    "Duplicate webhook — delivery_id=%s already processed",
                    x_github_delivery,
                )
                return {"status": "duplicate", "delivery_id": x_github_delivery}
    except Exception as e:
        # DB check failed — proceed anyway, worker will handle idempotency again
        logger.error("Idempotency check failed: %s — proceeding", e)

      # ── Store raw event for audit + idempotency ───────────────────────────────
    github_install_id = (
        payload.get("installation", {}).get("id")
        or payload.get("installation", {}).get("app_id")
    )

    db_save = False

    try:
        async with get_system_transaction() as conn:
            await conn.execute(
                """
                INSERT INTO webhooks_received
                    (delivery_id, event_type, github_install_id,
                     repo_full_name, payload, processed)
                VALUES ($1, $2, $3, $4, $5::jsonb, FALSE)
                ON CONFLICT (delivery_id) DO NOTHING
                """,
                x_github_delivery,
                x_github_event,
                github_install_id,
                payload.get("repository", {}).get("full_name"),
                json.dumps(payload),
            )
        db_save = True
    except Exception as e:
        logger.error("Failed to store webhook — delivery=%s: %s", x_github_delivery, e)
        # Continue to SQS even if DB insert failed — don't lose the event
 
  
      # ── Enqueue to SQS for async processing ───────────────────────────────────
    try:
        await _enqueue_to_sqs(
            sqs_client= sqs_client,
            event_type=x_github_event,
            delivery_id=x_github_delivery,
            install_id = github_install_id,
            payload=payload,
        )
    
    except Exception as e:
        logger.error(
            "Failed to enqueue webhook to SQS — delivery=%s: %s",
            x_github_delivery, e,
        )
        if db_save:
            return {"status":"saved to db only", "delivery_id":x_github_delivery}
        else:
           raise HTTPException(
                status_code=500, 
                detail="Catastrophic infrastructure failure. Please retry."
            )
 
    return {"status": "queued", "delivery_id": x_github_delivery}



# SQS enqueue
# ─────────────────────────────────────────────────────────────────────────────
 
async def _enqueue_to_sqs(
    sqs_client,
    event_type: str,
    delivery_id: str,
    install_id: int,
    payload: dict,
) -> None:
    """
    Drop the webhook event onto SQS.
    MessageDeduplicationId prevents SQS-level duplicates (FIFO queue).
    MessageGroupId groups by installation_id for ordered processing.
    """
    
    group_id = str(install_id) if install_id else "global"

    message = {
        "event_type":  event_type,
        "delivery_id": delivery_id,
        "payload":     payload,
    }

    await sqs_client.send_message(
        QueueUrl=settings.SQS_WEBHOOK_QUEUE_URL,
        MessageBody=json.dumps(message),
        MessageDeduplicationId=delivery_id,
        MessageGroupId=f"install-{group_id}",
        MessageAttributes={
            "event_type": {
                "DataType":    "String",
                "StringValue": event_type,
            }
        },
    )
 

