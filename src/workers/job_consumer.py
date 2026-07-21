# app/workers/job_consumer.py
import asyncio
import ijson , json
from uuid import UUID
from typing import Optional
import signal
import logging
import sys
import aioboto3
from botocore.exceptions import ClientError
from src.core.database import get_system_transaction,create_pools , close_pools
from src.core.config import get_settings
from src.core.crypto import decrypt_token
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

    async def extend_visibility_heartbeat(self, sqs, queue_url: str, receipt_handle: str):
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

    async def update_job_status(self, job_id: UUID, phase: str, node_count: Optional[int]= None):
        """Updates the central database directly from the Trusted Orchestrator."""
        async with get_system_transaction() as conn:
            await conn.execute(
                "UPDATE ingestion_jobs SET phase = $1, node_count = COALESCE ($2, node_count) WHERE id = $3",
                phase, node_count, job_id
            )
            logger.info("Job %s transitioned to %s", job_id, phase)
    
    async def _already_terminal(self, job_id: UUID) -> bool:
        """
        Consumer-side idempotency check — missing from the draft this
        supersedes. Checked before ANY work starts, same discipline
        established for the Fargate-only design.
        """
        async with get_system_transaction() as conn:
            status = await conn.fetchval(
                "SELECT status FROM ingestion_jobs WHERE id = $1", str(job_id)
            )
        return status in ("completed", "failed")

    async def process_single_tenant_job(self, sqs, msg ,kms_client , s3_client , lambda_client):
        """
        Orchestrates one job end to end. Runs concurrently alongside
        other jobs' invocations of this same method — safe because this
        method itself never touches untrusted repo content; it only
        waits on I/O (SQS, KMS, S3, Lambda invoke) and writes to
        Postgres/Neo4j using data the Lambda already validated/filtered.
        """
        receipt_handle = msg["ReceiptHandle"]
        body = json.loads(msg["Body"])
        
        job_id = body["job_id"]
        repo_id = body["repo_id"]
        tenant_id = body["account_id"]
        repo_url = body["repo_url"]  #update : need to add repo url in sqs first . 
        owner = body["owner"] 
        repo = body["repo_name"]
        s3_key = f"staging/{tenant_id}/{job_id}_ast.json"

        if await self._already_terminal(job_id):
            logger.info("Job %s already terminal — skipping (redelivered message)", job_id)
            await sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt_handle)
            return

        heartbeat_task: Optional[asyncio.Task] = None

        try:
            # 1. Start the Heartbeat
            heartbeat_task = asyncio.create_task(
                self.extend_visibility_heartbeat(sqs, self.queue_url ,receipt_handle)
            )

            await self.update_job_status(job_id, phase="ORCHESTRATING")

            installation_id = body["installation_id"]
            github_token =  await self.gh.auth.get_installation_token(installation_id)

            # 3. TRUSTED ZONE: Generate Pre-Signed S3 PUT URL (Zero-Credential Access for Sandbox)
            presigned_s3_url = await s3_client.generate_presigned_url(
                ClientMethod='put_object',
                Params={'Bucket': settings.STAGING_BUCKET, 'Key': s3_key},
                ExpiresIn=1800 # 30 minutes
            )

            await self._update_job_status(job_id, phase="CLONING_REPO")
            logger.info(
                "Invoking sandbox for job=%s owner=%s repo=%s (tenant=%s)",
                job_id, owner, repo, tenant_id,
            )
            # 4. UNTRUSTED ZONE: Invoke Sandbox via Secure Response Streaming
            logger.info("Invoking MicroVM Sandbox for Job %s (Tenant: %s)", job_id, tenant_id)
            response = await lambda_client.invoke_with_response_stream(
                FunctionName=settings.PARSER_LAMBDA_NAME,
                TenantId=str(tenant_id),  # CRITICAL: AWS Hardware Tenant Isolation Routing
                Payload=json.dumps({
                    "job_id":        str(job_id),
                    "owner":         owner,
                    "repo":          repo,
                    "branch":        body.get("branch", "main"),
                    "sparse_dirs":   body.get("selected_subprojects", []),
                    "submodules":    body.get("selected_submodules", []),
                    "github_token":  github_token,
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
                        raise RuntimeError(
                            f"Sandbox error: {complete_data.get('ErrorDetails')}"
                        )

            # 6. TRUSTED ZONE: Stream S3 JSON into Neo4j (Memory Safe)
            await self.update_job_status(job_id, phase="INJECTING_GRAPH")
            manifest_rows, ast_nodes_written =await self.stream_s3_to_graph(s3_client, s3_key,job_id, repo_id)

            # 7. Cleanup & Completion
            await self.update_job_status(job_id, phase="COMPLETED", node_count=ast_nodes_written)
            await self._mark_job_completed(job_id, repo_id, len(manifest_rows))
            await s3_client.delete_object(Bucket=settings.STAGING_BUCKET, Key=s3_key)
            
            
            # Delete the job from SQS so it never runs again
            await sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt_handle)
            logger.info("Job %s completed — %d nodes written", job_id, ast_nodes_written)
            logger.info("Job %s successfully completed and removed from queue.", job_id)

        except Exception as e:
            logger.exception("Failed to process job %s: %s", job_id, e)
            await self.update_job_status(job_id, phase="FAILED")
            await self._mark_job_failed(job_id, repo_id, str(e))
            # Do NOT delete the SQS message; let visibility timeout expire for redrive.

        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass

        # ─────────────────────────────────────────────────────────────────────────
    # Streaming the sandbox's output into Postgres + Neo4j
    # ─────────────────────────────────────────────────────────────────────────
 
    async def _stream_output_and_record(
        self, s3_client, s3_key: str, job_id: UUID, repo_id: UUID,
    ) -> tuple[list[dict], int]:
        """
        Streams the sandbox's NDJSON output (file manifest rows +
        AST/graph nodes interleaved, each line tagged by "kind") using
        ijson so a 500MB payload never loads fully into worker memory.
 
        Manifest rows go to IngestionWorker's existing, tested DB-write
        method — not duplicated here. AST nodes go to Neo4j via batched
        UNWIND + MERGE (idempotent — safe to re-run on a refresh).
        """
        s3_obj = await s3_client.get_object(Bucket=settings.STAGING_BUCKET, Key=s3_key)
        stream = s3_obj["Body"]
 
        manifest_rows: list[dict] = []
        node_buffer: list[dict] = []
        total_nodes = 0
 
        async for item in _aiter_ndjson(stream):
            if item.get("kind") == "manifest_row":
                manifest_rows.append(item["data"])
            elif item.get("kind") == "ast_node":
                node = item["data"]
                node["repo_id"] = str(repo_id)   # enforced on every node, see §5
                node_buffer.append(node)
                if len(node_buffer) >= 1000:
                    total_nodes += await self._write_node_batch(node_buffer)
                    node_buffer.clear()
 
        if node_buffer:
            total_nodes += await self._write_node_batch(node_buffer)
 
        if manifest_rows:
            await self.manifest_writer._write_manifest_rows_raw(
                repo_id=repo_id, job_id=job_id, rows=manifest_rows,
            )
 
        return manifest_rows, total_nodes

    async def _write_node_batch(self, batch: list[dict]) -> int:
        """
        MERGE, not CREATE — idempotent under re-index. Matches on the
        stable identity (repo_id, file_path, symbol_name), so re-running
        this for the same repo updates existing nodes instead of
        duplicating them. repo_id is enforced non-null by a Neo4j
        constraint created once at deploy time (see setup notes) — the
        graph-store equivalent of a Postgres RLS account_id filter.
        """
        from src.services.graph_writer import get_neo4j_driver   # new file, see below
 
        driver = await get_neo4j_driver()
        await driver.execute_query(
            """
            UNWIND $batch AS node
            MERGE (n:ASTNode {repo_id: node.repo_id, file_path: node.file_path, symbol_name: node.symbol_name})
            SET n += node
            """,
            batch=batch,
        )
        return len(batch)
    
     # ─────────────────────────────────────────────────────────────────────────
    # Terminal state writes
    # ─────────────────────────────────────────────────────────────────────────
 
    async def _mark_job_completed(self, job_id: UUID, repo_id: UUID, files_total: int) -> None:
        async with get_system_transaction() as conn:
            await conn.execute(
                "UPDATE ingestion_jobs SET status='completed', completed_at=NOW(), files_total=$2 WHERE id=$1",
                str(job_id), files_total,
            )
            await conn.execute(
                "UPDATE repos SET index_status='ready', last_indexed_at=NOW() WHERE id=$1",
                str(repo_id),
            )
 
    async def _mark_job_failed(self, job_id: UUID, repo_id: UUID, error: str) -> None:
        async with get_system_transaction() as conn:
            await conn.execute(
                "UPDATE ingestion_jobs SET status='failed', completed_at=NOW(), error_message=$2, phase='FAILED' WHERE id=$1",
                str(job_id), error[:2000],
            )


    async def consume_forever(self):
        """The long-running orchestrator loop."""
        await create_pools()
        logger.info("Starting Highly-Concurrent ECS Trusted Orchestrator...")

        async with self.session.client("sqs") as sqs,\
                   self.session.client("kms") as kms, \
                   self.session.client("s3") as s3, \
                   self.session.client("lambda") as lambda_client:
            try:
                while not _shutdown_event.is_set():
                    try:
                        # Pull up to 10 jobs at once (adjust based on ECS instance size)
                        resp = await sqs.receive_message(
                            QueueUrl=self.queue_url,
                            MaxNumberOfMessages=10, 
                            WaitTimeSeconds=20
                        )
                        
                        messages = resp.get("Messages", [])
                        if not messages:
                            continue
                            
                        # Execute all pulled jobs concurrently
                        tasks = [
                            self.process_single_tenant_job(sqs, msg, kms,s3, lambda_client)
                            for msg in messages 
                        ]
                        results = await asyncio.gather(*tasks,return_exceptions=True)

                        for r in results:
                            if isinstance(r, Exception):
                                logger.error("Unexpected task-level exception: %s", r)

                    except ClientError as e:
                        logger.error("AWS SQS network error: %s", e)
                        await asyncio.sleep(5)
                        
            finally:
                logger.info("Draining active jobs and closing database pools.")
                await close_pools()

# ─────────────────────────────────────────────────────────────────────────────
# NDJSON async iterator helper
# ─────────────────────────────────────────────────────────────────────────────
 
async def _aiter_ndjson(stream):
    """Async line-by-line NDJSON reader over an aioboto3 StreamingBody."""
    buffer = b""
    async for chunk in stream.iter_chunks():
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            if line.strip():
                yield json.loads(line)
    if buffer.strip():
        yield json.loads(buffer)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: _trigger_shutdown(s.name))
        
    try:
        orchestrator = TrustedOrchestrator()
        loop.run_until_complete(orchestrator.consume_forever())
    except Exception as e:
        logger.critical("Fatal Orchestrator Crash: %s", e)
        raise
