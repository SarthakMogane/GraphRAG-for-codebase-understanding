# app/workers/job_consumer.py
import asyncio
import dataclasses
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
from src.services.microvm_client import MicroVMClient,MicroVMError
from src.services.github import (
    GitHubAPIError,
    GitHubAuthenticationError,
    GitHubConflictError,
    GitHubService,
    GitHubValidationError,
    RateLimitError,
    RepoAccessError,
    RepoNotFoundError,
)
from src.workers.utils.callback import _active_background_tasks,handle_task_completion
from src.services.clone_strategy import CloneStrategySelector , RepoSizingInfo , CloneConfig
from src.models.database import RepoStatus

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
            region_name=settings.AWS_REGION,
        )
        self.gh = GitHubService()
        self.queue_url = settings.SQS_INGESTION_QUEUE_URL
        self.microvm = MicroVMClient(self.session)
        self.strategy_selector = CloneStrategySelector()

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

    _PHASE_TO_REPO_STATUS = {
        "CLONING_REPO":       RepoStatus.CLONING,
        "FILTERING":          RepoStatus.FILTERING,
        "PARSING_AST":        RepoStatus.MANIFESTING,
        "UPLOADING":          RepoStatus.MANIFESTING,
        "RECORDING_MANIFEST": RepoStatus.MANIFESTING,
        "FAILED":             RepoStatus.FAILED,
        "RETRYING":           RepoStatus.RETRYING,
    }

    async def _update_job_phase(self, job_id: UUID, repo_id: UUID,phase: str, node_count: Optional[int]= None):
        """Updates the central database directly from the Trusted Orchestrator."""
        async with get_system_transaction() as conn:
            status = await conn.execute(
                """ 
                    UPDATE ingestion_jobs SET
                    phase = $1, node_count = COALESCE ($2, node_count) 
                    WHERE id = $3 AND (phase IS DISTINCT FROM $1 OR node_count IS NOT NULL)
                """,
                phase, node_count, job_id
            )
            if status!= "UPDATE 0":
                repo_status = self._PHASE_TO_REPO_STATUS.get(phase)
                if repo_status:
                    await conn.execute(
                        "UPDATE repos SET index_status = $1 WHERE id = $2",
                        repo_status.value, repo_id,
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

     # Clone strategy decision — trusted zone, pure logic (see clone_strategy.py)
    # ─────────────────────────────────────────────────────────────────────────
 
    async def _decide_clone_strategy(
        self, repo_id: UUID, owner: str, repo: str, body: dict,
    ) -> CloneConfig:
        """
        create metadata and calls CloneStrategySelector.select().
        """
        validation=body.get("validation_payload",{})
        sizing = RepoSizingInfo(
            size_kb=validation.get("size_kb",0),
            owner=owner,
            name=repo,
            uses_git_lfs=validation.get("uses_git_lfs",False)
        )

        selection = body.get("selection_payload",{})
        return self.strategy_selector.select(
            metadata=sizing,
            is_monorepo=body.get("is_monorepo",False),
            sparse_dirs=selection.get("selected_subprojects", []),
            total_subprojects_detected=selection.get("total_subprojects",0)
        )
        
    async def process_single_tenant_job(self, sqs, msg ,kms_client , s3_client , microvm):
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

            await self._update_job_phase(job_id,repo_id=repo_id,phase="ORCHESTRATING")

            installation_id = body["installation_id"]
            github_token = getattr(self.gh, "auth", self.gh)
            if hasattr(github_token, "get_installation_token"):
                github_token = await github_token.get_installation_token(installation_id)
            else:
                github_token = await self.gh.get_installation_token(installation_id)
           

            # clone strategy
            clone_config = await self._decide_clone_strategy(repo_id=repo_id, owner=owner ,repo=repo,body=body)
            if clone_config.estimated_disk_mb > settings.CLONE_SANITY_REJECT_MB:
                raise MicroVMError(
                    f"Repo {owner}/{repo} estimated at {clone_config.estimated_disk_mb}MB "
                    f"exceeds the {settings.CLONE_SANITY_REJECT_MB}MB sanity limit",
                    code="SANITY_REPO_TOO_LARGE",
                    retryable=False,
                )

            # 3. TRUSTED ZONE: Generate Pre-Signed S3 PUT URL (Zero-Credential Access for Sandbox)
            try:
                presigned_s3_url = await s3_client.generate_presigned_url(
                    ClientMethod="put_object",
                    Params={"Bucket": settings.STAGING_BUCKET, "Key": s3_key},
                    ExpiresIn=1800,
                )
            except Exception as e:
                raise MicroVMError(
                    f"Failed to sign staging upload URL for key {s3_key}: {e}",
                    code="STAGING_URL_SIGNING_FAILED",
                    retryable=False,  # Configuration/credential issues won't fix themselves on immediate retry
                ) from e

            await self._update_job_phase(job_id,repo_id=repo_id, phase="LAUNCHING_SANDBOX")
            logger.info(
                "Invoking sandbox for job=%s owner=%s repo=%s (tenant=%s)",
                job_id, owner, repo, tenant_id,
            )
            # 4. UNTRUSTED ZONE: Invoke Sandbox via Secure Response Streaming
            microvm_id, endpoint,start,*others= await self.microvm.launch(
                image_identifier=settings.SANDBOX_MICROVM_IMAGE_ARN,
                run_hook_payload={
                    "job_id":        str(job_id),
                    "account_id":    str(tenant_id),
                    "owner":         owner,
                    "repo":          repo,
                    "branch":        body.get("branch", "main"),
                    "clone_config":  dataclasses.asdict(clone_config),
                    "submodules":    body.get("selected_submodules", []),
                    "github_token":  github_token,   # never logged — see redaction note in sandbox_app/app.py
                    "presigned_url": presigned_s3_url,
                    "image_version": str(settings.IMAGE_VERSION),
                },
                egress_connector_name=settings.MICROVM_EGRESS_CONNECTOR_NAME,
                ingress_connector_name = settings.MICROVM_INGRESS_CONNECTOR_NAME,
                execution_role_arn=settings.MICROVM_EXECUTION_ROLE_ARN,
                image_version=settings.IMAGE_VERSION,
                maximum_duration_seconds=settings.MAXIMUM_DURATION_SECONDS,
            )

            try:
                auth_token = await self.microvm.create_auth_token(
                    microvm_id, allowed_ports=[8080],expire=settings.EXPIRE_MICROVM_AUTH_TOKEN
                )
 
                async for event in self.microvm.stream_status(
                    endpoint, auth_token, timeout_seconds=600
                ):
                    phase = event.get("phase","PROCESSING")

                    if phase == "FAILED":
                        error_msg = event.get("error","Unknow_sandbox_error")
                        raise MicroVMError(
                            f"Sandbox Execution Failed :{error_msg}",
                            code="SandboxExecutionFailed",
                            retryable=False,# Code parsing errors will not succeed on SQS retry
                        )
                    if phase!="VM_WORK_COMPLETED":
                        await self._update_job_phase(job_id,repo_id=repo_id,phase=event.get("phase", "PROCESSING"))
                   
 
            finally:
                # Explicit termination is the normal path here — not the
                # idle-policy fallback. See microvm_client.launch()'s
                # docstring for why we don't rely on suspend/resume for
                # this one-shot job model.
                await self.microvm.terminate(microvm_id)

            # 6. TRUSTED ZONE: Stream S3 JSON into Neo4j (Memory Safe)
            await self._update_job_phase(job_id,repo_id=repo_id, phase="RECORDING_MANIFEST")
            manifest_rows, ast_nodes_written =await self._stream_output_and_record(sqs,s3_client, s3_key,job_id, repo_id , tenant_id)

            # 7. Cleanup & Completion
            await self._update_job_phase(job_id,repo_id=repo_id, phase="COMPLETED", node_count=ast_nodes_written)
            await self._mark_job_completed(job_id, repo_id, len(manifest_rows))
            await s3_client.delete_object(Bucket=settings.STAGING_BUCKET, Key=s3_key)
            
            
            # Delete the job from SQS so it never runs again
            await sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt_handle)
            logger.info("Job %s completed — %d nodes written", job_id, ast_nodes_written)
            logger.info("Job %s successfully completed and removed from queue.", job_id)

        # ── GitHub Exceptions Mapping ──────────────────────────────────────────
        except RepoNotFoundError as e:
            logger.error("Job %s Permanent Failure: Repository %s/%s missing.", job_id, owner, repo)
            await self._update_job_phase(job_id, repo_id=repo_id, phase="FAILED")
            await self._mark_job_failed(job_id, repo_id, f"[REPO_NOT_FOUND] {e}")
            await sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt_handle)

        except RepoAccessError as e:
            logger.error("Job %s Permanent Failure: Access forbidden to %s/%s.", job_id, owner, repo)
            await self._update_job_phase(job_id, repo_id=repo_id, phase="FAILED")
            await self._mark_job_failed(job_id, repo_id, f"[GITHUB_AUTH_REVOKED] {e}")
            await sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt_handle)

        except (GitHubAuthenticationError, GitHubValidationError, GitHubConflictError) as e:
            code = getattr(e, "code", "GITHUB_PERMANENT_ERROR")
            logger.error("Job %s Permanent GitHub Error [%s]: %s", job_id, code, e)
            await self._update_job_phase(job_id, repo_id=repo_id, phase="FAILED")
            await self._mark_job_failed(job_id, repo_id, f"[{code}] {e}")
            await sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt_handle)

        except RateLimitError as e:
            logger.warning("Job %s Rate Limit Reached. Setting RETRYING for visibility redrive.", job_id)
            await self._update_job_phase(job_id, repo_id=repo_id, phase="RETRYING")
            await self._mark_job_failed(job_id, repo_id, f"[GITHUB_RATE_LIMIT_EXCEEDED] {e}")

        except GitHubAPIError as e:
            is_transient = e.status_code is None or (e.status_code >= 500)
            if is_transient:
                logger.warning("Job %s Transient GitHub API Error (status=%s). Retrying.", job_id, e.status_code)
                await self._update_job_phase(job_id, repo_id=repo_id, phase="RETRYING")
                await self._mark_job_failed(job_id, repo_id, f"[GITHUB_SERVER_ERROR] {e}")
            else:
                logger.error("Job %s Permanent GitHub API Error (status=%s). Purging.", job_id, e.status_code)
                await self._update_job_phase(job_id, repo_id=repo_id, phase="FAILED")
                await self._mark_job_failed(job_id, repo_id, f"[GITHUB_API_ERROR] {e}")
                await sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt_handle)

        # ── MicroVM & General Exceptions Mapping ────────────────────────────────
        except MicroVMError as e:
            # Structured log including code and retryable status
            logger.error(
                "Job %s failed with MicroVMError [code=%s, retryable=%s]: %s",
                job_id, e.code, e.retryable, e,
                extra={"job_id": job_id, "error_code": e.code, "retryable": e.retryable}
            )
            if e.retryable:
                await self._update_job_phase(job_id, repo_id=repo_id, phase="RETRYING")
                await self._mark_job_failed(job_id, repo_id, f"[{e.code}] {e}")

            # Non-retryable errors are explicitly purged from SQS to avoid deadlocks
            else:
                await self._mark_job_failed(job_id, repo_id, error_message=f"[{e.code}] {e}")
                logger.warning("Deleting non-retryable job %s from SQS queue", job_id)
                await sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt_handle)
                
        except Exception as e:
            logger.exception("Failed to process job %s: %s", job_id, e)
            await self._update_job_phase(job_id,repo_id=repo_id,phase="RETRYING")
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
        self,sqs, s3_client, s3_key: str, job_id: UUID, repo_id: UUID,tenant_id:UUID
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
                # When buffer hits 1000, write to DB, then Fan-Out to AI
                if len(node_buffer) >= 1000:
                    # 1. Guarantee data is in Neo4j FIRST
                    await self._write_node_batch(node_buffer)
                    
                    # 2. Fan-out to AI workers concurrently
                    await self._fan_out_to_ai_queue(sqs, node_buffer, tenant_id, repo_id)
                    
                    total_nodes += len(node_buffer)
                    node_buffer.clear()
 
        # Flush the remainder of the file
        if node_buffer:
            await self._write_node_batch(node_buffer)
            await self._fan_out_to_ai_queue(sqs, node_buffer, tenant_id, repo_id)
            total_nodes += len(node_buffer)
 
        if manifest_rows:
            async with get_system_transaction() as conn:
                for row in manifest_rows:
                    await conn.execute(
                        "INSERT INTO repo_files (repo_id, file_path, file_hash) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
                        str(repo_id), row["file_path"], row.get("file_hash", "")
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
            await conn.execute(
                "UPDATE repos SET index_status='failed' WHERE id=$1",
                repo_id,
            )


    async def consume_forever(self):
        """The long-running orchestrator loop."""
        await create_pools()
        logger.info("Starting Highly-Concurrent ECS Trusted Orchestrator...")

        # Configure your thread pool sizes for your sync boto3 clients
        loop = asyncio.get_running_loop()
        from concurrent.futures import ThreadPoolExecutor

        loop.set_default_executor(ThreadPoolExecutor(max_workers=30))

        async with self.session.client("sqs") as sqs,\
                   self.session.client("kms") as kms, \
                   self.session.client("s3") as s3, \
                   self.session.client("lambda") as lambda_client:

            try:
                while not _shutdown_event.is_set():
                    if len(_active_background_tasks) >= 50:
                        await asyncio.sleep(1)
                        continue
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
                        for msg in messages:
                            body = json.loads(msg["Body"])
                            job_id = body.get("job_id")

                            # 1. Spawn the background task
                            task = asyncio.create_task(
                                self.process_single_tenant_job(sqs, msg, kms, s3)
                            )

                            # 2. CRITICAL STEP: Attach metadata dynamically to the Task instance object
                            # This prevents the task from being anonymous inside the Done-Callback!
                            setattr(task, "job_id", job_id)

                            # 3. Add to memory protection set
                            _active_background_tasks.add(task)

                            # 4. Attach your final production-ready helper callback
                            task.add_done_callback(handle_task_completion)

                    except ClientError as e:
                        logger.error("AWS SQS network error: %s", e)
                        await asyncio.sleep(5)
                        
            finally:
                logger.info("Draining active jobs and closing database pools.")
                await close_pools()


    async def _fan_out_to_ai_queue(self, sqs, nodes: list[dict], tenant_id: str, repo_id: UUID):
        """
        Filters high-value AST nodes and batches them into the SQS AI Enrichment Queue.
        Utilizes asyncio.gather to bypass the SQS 10-message batch limit without blocking.
        """
        # 1. Filter: Only send structural concepts to the LLM to save money and time
        ai_target_types = {"function_definition", "class_definition", "method_definition"}
        enrichment_candidates = [n for n in nodes if n.get("type") in ai_target_types]

        if not enrichment_candidates:
            return

        sqs_batches = []
        current_batch = []

        # 2. Chunk into AWS-mandated 10-message batches
        for idx, node in enumerate(enrichment_candidates):
            current_batch.append({
                'Id': str(idx), # AWS requires a unique ID per message within a batch
                'MessageBody': json.dumps({
                    "tenant_id": str(tenant_id),
                    "repo_id": str(repo_id),
                    "file_path": node["file_path"],
                    "symbol_name": node["symbol_name"],
                    "node_type": node["type"]
                })
            })
            
            if len(current_batch) == 10:
                sqs_batches.append(current_batch)
                current_batch = []
                
        if current_batch:
            sqs_batches.append(current_batch)

        # 3. Fire all HTTP requests to AWS SQS concurrently
        # If we have 100 batches, this executes them in parallel rather than waiting sequentially
        tasks = [
            sqs.send_message_batch(
                QueueUrl=settings.SQS_ENRICHMENT_QUEUE_URL,
                Entries=batch
            )
            for batch in sqs_batches
        ]
        
        try:
            await asyncio.gather(*tasks)
            logger.debug("Successfully fanned out %d micro-tasks to AI Queue", len(enrichment_candidates))
        except ClientError as e:
            logger.error("Failed to fan-out enrichment batch to SQS: %s", e)
            # In a strict environment, you might raise here to fail the job and retry
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
