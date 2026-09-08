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
                SELECT ij.id, ij.repo_id, ij.account_id, ij.job_type,
                       r.repo_name, r.owner_login,
                       us.selected_subprojects, us.selected_submodules,
                       rsr.head_sha,rsr.scout_json

                FROM ingestion_jobs ij
                JOIN repos r ON r.id = ij.repo_id
                JOIN user_selections us ON us.id = ij.selection_id
                JOIN repo_scout_result rsr ON rsr.id = us.scout_result_id
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
                

                for job in pending_jobs:
                    scout_json = job["scout_json"] or {}
                    total_subprojects = len(scout_json.get("subprojects",[]))

                    selected_subproject_paths = set(
                        job["selected_subprojects"] or []
                    )

                    selected_submodule_paths = set(
                        job["selected_submodules"] or []
                    )
                    scout_subprojects = scout_json.get("subprojects", [])
                    scout_submodules = scout_json.get("submodules", [])

                    subproject_by_path = {
                        node["path"]: node
                        for node in scout_subprojects if "path" in node
                    }

                    submodule_by_path = {
                        node["path"]: node
                        for node in scout_submodules if "path" in node
                    }

                    for sm in scout_submodules:
                        if "subproject" in sm:
                            for sp in sm["subprojects"]:
                                if "path" in sp:
                                    subproject_by_path[sp["path"]] = sp

                    selected_subprojects = [
                        subproject_by_path[path]
                        for path in selected_subproject_paths
                        if path in subproject_by_path
                    ]

                    selected_submodules = [
                        submodule_by_path[path]
                        for path in selected_submodule_paths
                        if path in submodule_by_path
                    ]

                    missing_subprojects = (
                        selected_subproject_paths - subproject_by_path.keys()
                    )

                    missing_submodules = (
                        selected_submodule_paths - submodule_by_path.keys()
                    )

                    if missing_subprojects or missing_submodules:
                        raise RuntimeError(
                            f"Selection references missing scout nodes: "
                            f"subprojects={sorted(missing_subprojects)}, "
                            f"submodules={sorted(missing_submodules)}"
                        )

                    # FORMAT SELECTED SUBPROJECTS (UNIFORM SCHEMA) ---
                    def map_subproject(node: dict) -> dict:
                        return {
                            "path": node.get("path"),
                            "name": node.get("name"),
                            "composite_score": node.get("composite_score", 0.0),
                            "auto_selected": node.get("auto_selected", False),
                            "source_file_count": node.get("source_file_count", 0),
                            "subproject_byte_count": node.get("subproject_byte_count", 0),
                            "has_entry_point": node.get("has_entry_point", False),
                            "dependent_count": node.get("dependent_count", 0),
                            "recent_commit_count": node.get("recent_commit_count", 0),
                            "skip_reason": node.get("skip_reason"),
                        }

                    # Format the root selected subprojects
                    formatted_selected_subprojects = [map_subproject(sp) for sp in selected_subprojects]
                    formatted_selected_submodules = []
                    for node in selected_submodules:
                        formatted_selected_submodules.append({
                            "path": node["path"],
                            "name": node["name"],
                            "owner": node.get("resolved_owner"),
                            "repo": node.get("resolved_repo"),
                            "url": node.get("resolved_url"),
                            "pinned_sha": node.get("pinned_sha"),
                            "outcome": node.get("outcome"),
                            "is_private": node.get("is_private"),
                            "is_monorepo": node.get("is_monorepo", False),
                            "uses_git_lfs": node.get("uses_git_lfs", False),
                            "complexity_band": node.get("complexity_band"),
                            "estimated_source_files": node.get("estimated_source_files", 0),
                            "estimated_source_bytes": node.get("estimated_source_bytes", 0),
                            
                            # Keep structural mapping identical for subprojects nested under submodules!
                            "subprojects": [
                                map_subproject(sp) for sp in node.get("subprojects", [])
                            ],
                        })
                    #payload
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
                        "is_monorepo":scout_json.get("is_monorepo", False),
                        

                        "selection_payload": {
                            "total_subprojects":total_subprojects,
                            "selected_subprojects": formatted_selected_subprojects,
                            "selected_submodules": formatted_selected_submodules,
                        },
                        "validation_payload": {
                            "default_branch": scout_json.get("default_branch"),
                            "size_kb": scout_json.get("size_kb", 0),
                            "uses_git_lfs":scout_json.get("uses_git_lfs", False),
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