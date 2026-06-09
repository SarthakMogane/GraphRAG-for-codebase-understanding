"""
app/core/database.py
─────────────────────
Production PostgreSQL layer using raw asyncpg (no ORM).

Design decisions:
  - asyncpg for all async FastAPI routes and background tasks
  - psycopg3 for Celery sync workers (different pool, same DB)
  - Every connection sets app.current_account_id for Row-Level Security
  - SSL is mandatory in production — rejected if certificate not verified
  - Connection pool sized per environment (dev=5, prod=50+)
  - Prepared statements cached per connection (asyncpg default)
  - Schema migrations run via raw SQL files, not Alembic

Security layers:
  1. SSL/TLS — encrypted wire between app and RDS
  2. IAM auth on RDS Aurora — no long-lived DB passwords in prod
  3. app.current_account_id session variable — RLS policies enforce isolation
  4. read-only replica pool for SELECT queries (chat, dashboard)
  5. Statement timeout — prevents runaway queries from blocking the pool

Usage:
    # FastAPI dependency
    async with get_db() as conn:
        row = await conn.fetchrow("SELECT * FROM repos WHERE id=$1", repo_id)

    # Background task (Celery)
    with get_sync_db() as conn:
        conn.execute("UPDATE repos SET status=$1 WHERE id=$2", "ready", repo_id)
"""

from __future__ import annotations

import logging
import ssl
import time
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator, Optional
from uuid import UUID

import asyncpg
from asyncpg import Connection, Pool
from fastapi import HTTPException

from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()



# ─────────────────────────────────────────────────────────────────────────────
# SSL configuration
# ─────────────────────────────────────────────────────────────────────────────

def _build_ssl_context() -> ssl.SSLContext:
    """
    Build a strict SSL context for RDS connections.

    In production:
      - Uses AWS RDS CA bundle (downloaded to container at build time)
      - CERT_REQUIRED: rejects connections with invalid or self-signed certs
      - Prevents MITM between app and database

    In development:
      - CERT_NONE if RDS_CA_BUNDLE not set (local postgres without SSL)
    """
    if not settings.RDS_CA_BUNDLE_PATH:
        if settings.APP_ENV == "production":
            raise RuntimeError(
                "RDS_CA_BUNDLE_PATH must be set in production. "
                "Download from https://truststore.pki.rds.amazonaws.com/"
            )
        # Development: no SSL verification
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    ctx = ssl.create_default_context(cafile=settings.RDS_CA_BUNDLE_PATH)
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.check_hostname = True
    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# Connection setup hook
# Runs on every new connection in the pool — sets search_path, timeouts, etc.
# ─────────────────────────────────────────────────────────────────────────────

async def _setup_connection(conn: Connection) -> None:
    """
    Called by asyncpg when a new physical connection is established.
    Runs once per connection lifetime, not per query.

    Sets:
      - search_path to the application schema (prevents schema injection)
      - statement_timeout to prevent runaway queries
      - lock_timeout to prevent connection starvation
      - timezone to UTC for consistent timestamp handling
    """
    await conn.execute("""
        SET search_path TO public;
        SET statement_timeout = '30s';
        SET lock_timeout = '5s';
        SET idle_in_transaction_session_timeout = '60s';
        SET timezone = 'UTC';
        SET application_name = 'repo-chat-api';
    """)

    # Register UUID type codec — asyncpg returns UUIDs as strings by default
    # This makes them return as Python uuid.UUID objects
    await conn.set_type_codec(
        "uuid",
        encoder=str,
        decoder=lambda s: UUID(s) if s else None,
        schema="pg_catalog",
        format="text",
    )

    # Register JSONB codec — return as dict, not string
    await conn.set_type_codec(
        "jsonb",
        encoder=lambda v: __import__("json").dumps(v),
        decoder=lambda v: __import__("json").loads(v),
        schema="pg_catalog",
    )

    logger.debug("Pool connection initialized")


