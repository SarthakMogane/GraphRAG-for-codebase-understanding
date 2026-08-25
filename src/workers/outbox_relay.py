# src/workers/outbox_relay.py

import asyncio
import json
import logging
import aioboto3
import asyncpg
from src.core.logger import get_logger
from src.core.config import get_settings
from src.core.database import get_system_transaction , create_pools, close_pools
from src.services.janitor_service import DatabaseJanitor
from botocore.exceptions import ClientError


logger = get_logger(__name__)
settings = get_settings()

class MaintenanceWorker:
    def __init__(self):
        self.session = aioboto3.Session(
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
        )

    async def start(self):
        """Launches both the Outbox loop and the Janitor loop concurrently."""
        logger.info("🚀 Maintenance Worker Booted (Outbox Relay + DB Janitor).")
        
        # Run both infinite loops side-by-side
        await asyncio.gather(
            self._run_outbox_loop(),
            self._run_janitor_loop()
        )

    async def _run_janitor_loop(self):
        """Sweeps the database every 5 minutes to fix stuck states."""
        while True:
            try:
                await DatabaseJanitor.run_sweep()
            except Exception as e:
                logger.error(f"Janitor sweep failed: {e}")
            await asyncio.sleep(300) # Sleep 5 minutes

    async def _run_outbox_loop(self):
        """Sweeps the database every 2 seconds for pending user jobs."""
        while True:
            try:
                await self._dispatch_pending_jobs()
            except asyncpg.PostgresError as e:
                # Transient Database Error
                logger.warning(f"Relay DB connection hiccup: {e}. Retrying in 2s...")
                
            except ClientError as e:
                # Transient AWS SQS Network Error
                logger.warning(f"Relay AWS network hiccup: {e}. Retrying in 2s...")

            except Exception as e:
                logger.error(f"Fatal Bug:Outbox Relay error: {e}")
            await asyncio.sleep(2)

    async def _dispatch_pending_jobs(self):
        async with get_system_transaction() as conn:
            # SKIP LOCKED guarantees no deadlocks if you boot 5 relay containers
            pending_jobs = await conn.fetch(
                """
                SELECT ij.id, ij.repo_id, ij.account_id, ij.job_type,r.repo_name, r.owner_login,
                       r.default_branch, r.size_kb,r.using_git_lfs,
                       us.selected_subprojects, us.selected_submodules,
                       rsr.head_sha,rsr.scout_json

                FROM ingestion_jobs ij
                JOIN repos r ON r.id = ij.repo_id
                JOIN repo_scout_result rsr ON rsr.repo_id = r.id
                JOIN user_selections us ON us.id = ij.selection_id
                WHERE ij.status = 'dispatch_pending'
                ORDER BY ij.created_at ASC
                LIMIT 10
                FOR UPDATE OF ij SKIP LOCKED;
                """
            )

            if not pending_jobs:
                return

        # Instantly mark them as queued inside the DB and release the row locks
            job_ids = [job["id"] for job in pending_jobs]
            await conn.execute(
                "UPDATE ingestion_jobs SET status = 'queued' WHERE id = ANY($1)", 
                job_ids
            )

            async with self.session.client('sqs') as sqs_client:
                successful_job_ids = []
                scout_json = job["scout_json"] or {}
                monorepo = scout_json.get("is_monorepo",False)
                total_subprojects = len(scout_json.get("subprojects",[]))

                for job in pending_jobs:
                    delivery_id = str(job["id"])
                    group_id = str(job["repo_id"])
                    payload = {
                        "repo_id": str(job["repo_id"]),
                        "account_id": str(job["account_id"]),
                        "repo_name": str(job["repo_name"]),
                        "owner":str(job["owner_name"]),
                        "job_id": str(job["id"]),
                        "job_type": job["job_type"],
                        "head_sha":job["head_sha"],
                        "is_monorepo":monorepo,
                        

                        "selection_payload": {
                            "total_subprojects":total_subprojects,
                            "selected_subprojects": job["selected_subprojects"],
                            "selected_submodules": job["selected_submodules"],
                        },
                        "validation_payload": {
                            "default_branch": job["default_branch"],
                            "size_kb": job["size_kb"] or 0,
                            "uses_git_lfs":job["using_git_lfs"] or False,
                        }
                    }

                    try:
                        response = await sqs_client.send_message(
                            QueueUrl=settings.SQS_INGESTION_QUEUE_URL,
                            MessageBody=json.dumps(payload),
                            MessageDeduplicationId=delivery_id,
                            MessageGroupId=group_id,
                    
                        )
                        
                        sqs_message_id = response["MessageId"]   # for tracing/logs only
                        logger.info(
                            "Ingestion job enqueued: job_id=%s repo_id=%s sqs_message_id=%s",
                            job["id"], job["repo_id"], sqs_message_id,
                        )
                        successful_job_ids.append(job["id"])
                    
                    except Exception as e:
                    # If SQS network call fails, revert this single job back to pending so the next loop tries again
                        logger.error("AWS SQS Error for job %s: %s", job['id'], e)
                        
            if successful_job_ids:
                await conn.execute(
                    "UPDATE ingestion_jobs SET status = 'queued' WHERE id = ANY($1)", 
                    successful_job_ids
                )

async def main():
    await create_pools()

    try:
        worker = MaintenanceWorker()
        await worker.start()
    
    finally:
        await close_pools()
        logger.info("Database pools successfully closed.")

if __name__ == "__main__":
    logger.info("Starting up the Outbox Relay Container...")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Maintenance Worker shutting down gracefully.")