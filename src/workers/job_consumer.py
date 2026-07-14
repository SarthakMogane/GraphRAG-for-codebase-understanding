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


