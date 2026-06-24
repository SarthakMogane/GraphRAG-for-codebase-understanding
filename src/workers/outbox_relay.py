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

    
async def main():
    await close_pools()

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