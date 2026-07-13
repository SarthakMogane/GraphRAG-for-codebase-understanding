# app/workers/sqs_consumer.py
import aioboto3, asyncio, json
from src.core.logger import get_logger
from src.core.exceptions import TransientWebhookError
from botocore.exceptions import ClientError
from src.core.config import get_settings
from src.workers.webhook_processor import process_webhook_event
from src.core.database import create_pools , close_pools

logger = get_logger(__name__)
settings = get_settings()

_shutdown_event = asyncio.Event()

def _trigger_shutdown(sig_name):
    logger.info("Received signal %s. Initiating graceful shutdown...", sig_name)
    _shutdown_event.set()

async def process_single_message(sqs, queue_url, msg, handler):
    """Handles a single webhook message concurrently."""
    receipt_handle = msg["ReceiptHandle"]
    try:
        body_dict = json.loads(msg["Body"])
        delivery_id = body_dict.get("delivery_id")
        
        # Process the event (validating signatures, updating DB, etc.)
        await handler(delivery_id, body_dict.get("event_type"), body_dict.get("payload", {}))
        
        # Success: Delete message
        await sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
        
    except TransientWebhookError as e:
        logger.warning("Transient error for delivery %s: %s. Leaving for redrive.", delivery_id, e)
    except json.JSONDecodeError as e:
        logger.error("Permanently unparseable JSON payload: %s. Deleting.", e)
        await sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
    except Exception as e:
        logger.exception("Unexpected error processing webhook delivery %s", delivery_id)
        # Fail safe: Do not delete, let it DLQ eventually

async def consume(queue_url: str, handler) -> None:
    """
    Asynchronously polls SQS via long-polling and passes events to the handler.
    """
    #boot the database pool
    await create_pools()

    session = aioboto3.Session(
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION
        )
    
    # 1. Open a true async connection to AWS
    async with session.client("sqs") as sqs:
        logger.info("Started async SQS consumer loop for %s", queue_url)
        try: 
            while True:
                try:
                    resp = await sqs.receive_message(
                        QueueUrl=queue_url,
                        MaxNumberOfMessages=10,
                        WaitTimeSeconds=20,   # 20s long-polling
                    )
                    
                    messages = resp.get("Messages", [])
                    if not messages:
                        continue
                        
                    for msg in messages:
                        receipt_handle = msg["ReceiptHandle"]
                        
                        try:
                            # 2. Unpack the exact JSON structure we created in _enqueue_to_sqs
                            body_dict = json.loads(msg["Body"])
                            delivery_id = body_dict.get("delivery_id")
                            event_type  = body_dict.get("event_type")
                            payload     = body_dict.get("payload", {})
                            
                            # 3. Call our master wrapper
                            await handler(delivery_id, event_type, payload)
                            
                            # 4. Success Path (Or Swallowed Poison Pill): Delete the message
                            await sqs.delete_message(
                                QueueUrl=queue_url,
                                ReceiptHandle=receipt_handle
                            )
                            
                        except TransientWebhookError as e:
                            # 5. Temporary Failure: DO NOT DELETE. 
                            # SQS visibility timeout will expire and it will be redelivered.
                            logger.warning("Transient error for delivery %s. Will retry. Error: %s", delivery_id, e)
                            continue 
                            
                        except Exception as e:
                            # 6. Catastrophic JSON Parsing Failure:
                            # If json.loads fails, the message is permanently broken. Delete it.
                            logger.error("Permanently unparseable SQS message. Deleting. Error: %s", e)
                            await sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)

                except ClientError as e:
                    # If AWS networking goes down, don't crash the while loop.
                    # Sleep for 5 seconds and try reconnecting.
                    logger.error("AWS SQS network error: %s", e)
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    logger.critical("Unexpected consumer loop crash: %s", e)
                    await asyncio.sleep(5)

        finally:
            await close_pools()

if __name__ == "__main__":
    
    QUEUE_URL = settings.SQS_WEBHOOK_QUEUE_URL
    import os

    # Temporary debug lines - remove these after fixing!
    print("--- AWS CREDENTIAL CHECK ---")
    print("--- PYDANTIC SETTINGS CHECK ---")
    print(f"Settings Key ID: {settings.AWS_ACCESS_KEY_ID}")
    print(f"Settings Secret: {settings.AWS_SECRET_ACCESS_KEY}")
    print(f"Settings Region: {settings.AWS_REGION}")
    print(f"queue link:{settings.SQS_WEBHOOK_QUEUE_URL}")

    try:
        asyncio.run(consume(
            queue_url=QUEUE_URL, 
            handler=process_webhook_event
        ))
    except KeyboardInterrupt:
        logger.info("Webhook worker shutting down gracefully via keyboard interrupt.")