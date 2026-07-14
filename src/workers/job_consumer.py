# app/workers/job_consumer.py
import asyncio
import json
import logging
import sys
import aioboto3
from botocore.exceptions import ClientError

from src.core.config import get_settings
from src.workers.ingestion_worker import IngestionWorker, PermanentJobFailure, TransientJobFailure

logger = logging.getLogger(__name__)
settings = get_settings()

async def extend_visibility_heartbeat(sqs, queue_url: str, receipt_handle: str):
    """
    Background task that pings SQS to keep the message alive during heavy GraphRAG processing.
    Pings every 3 minutes (180s) to grant an additional 5 minutes (300s) of visibility.
    """
    try:
        while True:
            await asyncio.sleep(180)
            try:
                await sqs.change_message_visibility(
                    QueueUrl=queue_url,
                    ReceiptHandle=receipt_handle,
                    VisibilityTimeout=300
                )
                logger.info("Heartbeat success: Extended visibility timeout by 5 minutes.")
            except ClientError as e:
                logger.warning("Heartbeat failed to extend visibility: %s", e)
    except asyncio.CancelledError:
        # Task was cancelled successfully when the main job finished
        logger.debug("Heartbeat task cancelled cleanly.")


async def run_ephemeral_job():
    """
    Polls exactly ONE message. Processes it. Then terminates the container.
    """
    queue_url = settings.SQS_INGESTION_QUEUE_URL
    session = aioboto3.Session(
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION
    )

    async with session.client("sqs") as sqs:
        logger.info("Fargate task booted. Polling for single ingestion job.")
        
        try:
            # 1. Grab exactly ONE message
            resp = await sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=10, # Keep polling brief so empty tasks die fast
                AttributeNames=["ApproximateReceiveCount"],
            )
            
            messages = resp.get("Messages", [])
            if not messages:
                logger.info("Queue is empty. Terminating Fargate task to save compute.")
                sys.exit(0)
                
            msg = messages[0]
            receipt_handle = msg["ReceiptHandle"]
            receive_count = int(msg.get("Attributes", {}).get("ApproximateReceiveCount", "1"))
            
            logger.info("Claimed job message. Attempt %d", receive_count)

            # 2. Start the heartbeat to protect this long-running job
            heartbeat_task = asyncio.create_task(
                extend_visibility_heartbeat(sqs, queue_url, receipt_handle)
            )

            # 3. Execute the heavy lifting (Clone, Tree-sitter AST, Neo4j ingestion)
            try:
                worker = IngestionWorker()
                await worker.handle_message(msg["Body"])
                
                # 4a. Success Path
                await sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
                logger.info("Ingestion complete. Message deleted. Terminating container.")
                sys.exit(0)

            except PermanentJobFailure as e:
                # 4b. Permanent Failure (e.g., Repo 404, Bad Credentials)
                logger.error("Permanent ingestion failure: %s. Deleting message.", e)
                await sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
                sys.exit(1)

            except TransientJobFailure as e:
                # 4c. Transient Failure (e.g., Neo4j connection dropped, GitHub API rate limit)
                logger.warning("Transient failure (attempt %d): %s. Leaving for redrive.", receive_count, e)
                # We do NOT delete the message. It will redeliver after the current visibility expires.
                sys.exit(1)

            except Exception as e:
                # 4d. Unhandled Crash
                logger.exception("Unexpected crash during ingestion: %s", e)
                sys.exit(1)
                
            finally:
                # Always shut down the heartbeat cleanly
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
                    
        except ClientError as e:
            logger.critical("AWS Network Error preventing queue access: %s", e)
            sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(run_ephemeral_job())
    except KeyboardInterrupt:
        logger.info("Task terminated manually.")
        sys.exit(1)