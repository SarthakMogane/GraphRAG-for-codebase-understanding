# app/workers/sqs_consumer.py
import aioboto3, asyncio, json
from src.core.logger import get_logger
from src.core.exceptions import TransientWebhookError
from botocore.exceptions import ClientError

logger = get_logger(__name__)

async def consume(queue_url: str, handler) -> None:
    """
    Asynchronously polls SQS via long-polling and passes events to the handler.
    """
    session = aioboto3.Session()
    
    # 1. Open a true async connection to AWS
    async with session.client("sqs") as sqs:
        logger.info("Started async SQS consumer loop for %s", queue_url)
        
        while True:
            try:
                # This await allows the worker to do other things while waiting
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