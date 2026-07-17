# app/workers/job_consumer.py
import asyncio
import ijson
import logging
import sys
import aioboto3
from botocore.exceptions import ClientError
from src.core.database import get_system_transaction
from src.core.config import get_settings
from src.workers.ingestion_worker import IngestionWorker, PermanentJobFailure, TransientJobFailure

logger = logging.getLogger(__name__)
settings = get_settings()

_shutdown_event = asyncio.Event()

def _trigger_shutdown(sig_name):
    logger.info("Received signal %s. Initiating graceful shutdown of Orchestrator...", sig_name)
    _shutdown_event.set()

class TrustedOrchestrator:
    def __init__(self):
        self.session = aioboto3.Session(
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.queue_url = settings.SQS_INGESTION_QUEUE_URL

    async def extend_visibility_heartbeat(sqs, queue_url: str, receipt_handle: str):
        """
        Background task that pings SQS to keep the message alive during heavy GraphRAG processing.
        Pings every 3 minutes (180s) to grant an additional 5 minutes (300s) of visibility.
        """
        try:
            while not _shutdown_event.is_set():
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

    async def update_job_status(self, job_id: str, phase: str, node_count: int = 0):
        """Updates the central database directly from the Trusted Orchestrator."""
        async with get_system_transaction() as conn:
            await conn.execute(
                "UPDATE ingestion_jobs SET phase = $1, node_count = $2 WHERE id = $3",
                phase, node_count, job_id
            )
            logger.info("Job %s transitioned to %s", job_id, phase)

    async def process_single_tenant_job(self, sqs, msg):
        """
        The core orchestration logic for a single repository.
        Runs concurrently alongside other jobs in the ECS event loop.
        """
        receipt_handle = msg["ReceiptHandle"]
        body = json.loads(msg["Body"])
        
        job_id = body["job_id"]
        tenant_id = body["tenant_id"]
        repo_url = body["repo_url"]
        s3_key = f"staging/{tenant_id}/{job_id}_ast.json"

        # 1. Start the Heartbeat
        heartbeat_task = asyncio.create_task(
            self.extend_visibility_heartbeat(sqs, receipt_handle)
        )

        try:
            await self.update_job_status(job_id, phase="ORCHESTRATING")

            async with self.session.client("kms") as kms, \
                       self.session.client("s3") as s3, \
                       self.session.client("lambda") as lambda_client:

                # 2. TRUSTED ZONE: Decrypt BYOK Token in Memory
                decrypted_token_resp = await kms.decrypt(
                    CiphertextBlob=bytes.fromhex(body["encrypted_token"])
                )
                github_token = decrypted_token_resp["Plaintext"].decode('utf-8')

                # 3. TRUSTED ZONE: Generate Pre-Signed S3 PUT URL (Zero-Credential Access for Sandbox)
                presigned_s3_url = await s3.generate_presigned_url(
                    ClientMethod='put_object',
                    Params={'Bucket': settings.STAGING_BUCKET, 'Key': s3_key},
                    ExpiresIn=1800 # 30 minutes
                )

                # 4. UNTRUSTED ZONE: Invoke Sandbox via Secure Response Streaming
                logger.info("Invoking MicroVM Sandbox for Job %s (Tenant: %s)", job_id, tenant_id)
                response = await lambda_client.invoke_with_response_stream(
                    FunctionName=settings.PARSER_LAMBDA_NAME,
                    TenantId=tenant_id,  # CRITICAL: AWS Hardware Tenant Isolation Routing
                    Payload=json.dumps({
                        "repo_url": repo_url,
                        "github_token": github_token,
                        "presigned_s3_url": presigned_s3_url
                    }).encode('utf-8')
                )

                # 5. Consume Real-Time Sandbox Execution Updates
                async for event in response['EventStream']:
                    if 'PayloadChunk' in event:
                        # The MicroVM yielded a status update (e.g., {"phase": "CLONING"})
                        chunk = json.loads(event['PayloadChunk']['Payload'].decode('utf-8'))
                        await self.update_job_status(job_id, phase=chunk.get("phase", "PROCESSING"))
                        
                    if 'InvokeComplete' in event:
                        complete_data = event['InvokeComplete']
                        if 'ErrorCode' in complete_data:
                            raise Exception(f"MicroVM Crash: {complete_data.get('ErrorDetails')}")

                # 6. TRUSTED ZONE: Stream S3 JSON into Neo4j (Memory Safe)
                await self.update_job_status(job_id, phase="INJECTING_GRAPH")
                await self.stream_s3_to_graph(s3, s3_key)

                # 7. Cleanup & Completion
                await s3.delete_object(Bucket=settings.STAGING_BUCKET, Key=s3_key)
                await self.update_job_status(job_id, phase="COMPLETED")
                
                # Delete the job from SQS so it never runs again
                await sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt_handle)
                logger.info("Job %s successfully completed and removed from queue.", job_id)

        except Exception as e:
            logger.exception("Failed to process job %s: %s", job_id, e)
            await self.update_job_status(job_id, phase="FAILED")
            # Do NOT delete the SQS message; let visibility timeout expire for redrive.

        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass


