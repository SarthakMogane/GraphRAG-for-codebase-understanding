from src.core.logger import get_logger
from src.core.database import get_system_transaction

logger = get_logger(__name__)

class DatabaseJanitor:
    """
    Self-healing automated cleanup utility.
    Detects and resets zombie database states caused by abrupt worker termination,
    OOM crashes, or network drops.
    """

    @classmethod
    async def run_sweep(cls):
        logger.debug("🧹 Initiating system-wide database zombie sweep...")

        # We use the system transaction because this runs in the background
        # and needs VIP access to sweep the whole database, bypassing RLS.
        async with get_system_transaction() as conn:
            
            # ── 1. Reset Dead API Scouts (Timeout: 5 Minutes) ────────────────
            # DeepScout fetches the file tree. It should take seconds.
            # If it takes > 5 minutes, the FastAPI container likely crashed.
            scout_result = await conn.execute(
                """
                UPDATE repos 
                SET index_status = 'failed', 
                    updated_at = NOW() 
                WHERE index_status = 'scouting' 
                  AND updated_at < NOW() - INTERVAL '5 minutes'
                """
            )
            scouts_cleared = int(scout_result.split(" ")[-1])
            if scouts_cleared > 0:
                logger.warning(f"Janitor cleared {scouts_cleared} dead 'scouting' locks.")

            # ── 2. Reset Dead SQS Ingestion Jobs (Timeout: 1 Hour) ───────────
            # Cloning huge repos and creating embeddings can take time.
            # We give the SQS worker a generous 1-hour window before declaring it dead.
            jobs_result = await conn.execute(
                """
                UPDATE ingestion_jobs 
                SET status = 'failed', 
                    updated_at = NOW()
                WHERE status IN ('dispatch_pending', 'queued', 'processing')
                  AND updated_at < NOW() - INTERVAL '1 hour'
                """
            )
            jobs_cleared = int(jobs_result.split(" ")[-1])

            # ── 3. Reset Repositories Locked by Dead Jobs ────────────────────
            # If a job was marked dead in step 2, the repo itself is still locked 
            # in a 'pending' or 'indexing' state. We must release it so the user 
            # can click the Sync button again.
            repos_result = await conn.execute(
                """
                UPDATE repos 
                SET index_status = 'failed', 
                    updated_at = NOW()
                WHERE index_status IN ('pending', 'indexing')
                  AND updated_at < NOW() - INTERVAL '1 hour'
                """
            )
            repos_cleared = int(repos_result.split(" ")[-1])
            
            if jobs_cleared > 0 or repos_cleared > 0:
                logger.warning(
                    f"Janitor cleared {jobs_cleared} dead jobs and unlocked {repos_cleared} repos."
                )

        logger.debug("✨ Database sweep complete.")