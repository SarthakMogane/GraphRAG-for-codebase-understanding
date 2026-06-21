
from src.crud.webhook_ops import _mark_processed
from src.core.logger import get_logger
from src.core.database import get_db_dep
from src.core.exceptions import TransientWebhookError
from src.workers.handlers.event_installation import _handle_installation ,_handle_installation_repos
from src.workers.handlers.event_repo_metadata import _handle_repository_metadata
from src.workers.handlers.event_push import  _handle_pull_request,_handle_push
from src.workers.handlers.event_app_auth import _handle_app_authorization
logger = get_logger(__name__)

async def process_webhook_event(
    delivery_id: str,
    event_type: str,
    payload: dict,
) -> None:
    """
    Entry point for the Celery worker. Routes to the correct handler.
 
    Error handling pattern:
      - Each handler wraps its DB operations in try/except
      - On error: mark webhooks_received.error, log full traceback
      - Never raise — let the worker mark the SQS message as processed
        (if we raise, SQS requeues and retries, which may cause more damage)
    """
    try:
        # Second idempotency check inside the worker
        # (in case two workers raced on the same SQS message)
        async with get_db_dep() as conn:
            already_done = await conn.fetchval(
                "SELECT processed FROM webhooks_received WHERE delivery_id = $1",
                delivery_id,
            )
            if already_done is True:
                logger.info("Webhook already processed — delivery=%s", delivery_id)
                return
 
        # Route to correct handler
        handler = {
            "installation":                _handle_installation,
            "installation_repositories":   _handle_installation_repos,
            "push":                        _handle_push,
            "pull_request":                _handle_pull_request,
            "repository":                  _handle_repository_metadata,
            "github_app_authorization":    _handle_app_authorization,
        }.get(event_type)
 
        if handler:
            await handler(payload)
            await _mark_processed(delivery_id,processed=True,event_type=event_type , payload=payload)
        else:
            logger.debug("Unhandled webhook event type: %s", event_type)
            await _mark_processed(delivery_id,processed=False, error="unhandled_event",event_type=event_type,payload=payload)
    
    except TransientWebhookError as e:
        logger.warning(
            "Transient error for delivery=%s. Returning to SQS queue. Reason: %s", 
            delivery_id, e
        )
        # We save the error so we can see it in the DB, but leave processed=FALSE
        await _mark_processed(delivery_id, processed=False, error=str(e),event_type=event_type , payload = payload)
        # BUBBLE UP: This tells the SQS consumer script NOT to delete the message
        raise e
 
    except Exception as e:
        logger.exception(
            "Permanent fatal error in webhook processor — delivery=%s: %s",
            delivery_id, e,
        )
        await _mark_processed(delivery_id,processed=False, error=str(e),event_type=event_type , payload=payload)


