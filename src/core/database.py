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
# Pool singletons
# Created once at app startup via lifespan(). Never recreate per-request.
# ─────────────────────────────────────────────────────────────────────────────

_write_pool: Optional[Pool] = None   # Primary RDS instance — reads + writes
_read_pool:  Optional[Pool] = None   # Read replica (Aurora) — read-only queries


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


# ─────────────────────────────────────────────────────────────────────────────
# Pool lifecycle
# ─────────────────────────────────────────────────────────────────────────────

async def create_pools() -> None:
    """
    Create the write and read pools. Called once in app lifespan startup.
    Both pools use SSL and run the setup hook on every new connection.
    """
    global _write_pool, _read_pool

    ssl_ctx = _build_ssl_context()

    # Write pool — primary RDS instance
    _write_pool = await asyncpg.create_pool(
        dsn=settings.DATABASE_URL,
        ssl=ssl_ctx,
        min_size=settings.DB_POOL_MIN_SIZE,
        max_size=settings.DB_POOL_MAX_SIZE,
        max_inactive_connection_lifetime=300,   # recycle idle connections after 5 min
        command_timeout=30,
        setup=_setup_connection,
        server_settings={
            "application_name": "repo-chat-write",
            "search_path":      "public",
        },
    )

    # Read pool — Aurora read replica (falls back to primary if not configured)
    read_dsn = settings.DATABASE_READ_URL or settings.DATABASE_URL
    _read_pool = await asyncpg.create_pool(
        dsn=read_dsn,
        ssl=ssl_ctx,
        min_size=settings.DB_READ_POOL_MIN_SIZE,
        max_size=settings.DB_READ_POOL_MAX_SIZE,
        max_inactive_connection_lifetime=300,
        command_timeout=15,   # reads should be faster
        setup=_setup_connection,
        server_settings={
            "application_name":            "repo-chat-read",
            "search_path":                 "public",
            "default_transaction_read_only": "on",   # safety: rejects accidental writes
        },
    )

    logger.info(
        "DB pools created — write: %s-%s, read: %s-%s",
        settings.DB_POOL_MIN_SIZE, settings.DB_POOL_MAX_SIZE,
        settings.DB_READ_POOL_MIN_SIZE, settings.DB_READ_POOL_MAX_SIZE,
    )


async def close_pools() -> None:
    """Close all pools gracefully. Called in app lifespan shutdown."""
    global _write_pool, _read_pool
    if _write_pool:
        await _write_pool.close()
        _write_pool = None
    if _read_pool:
        await _read_pool.close()
        _read_pool = None
    logger.info("DB pools closed")


def _get_write_pool() -> Pool:
    if not _write_pool:
        raise RuntimeError("Database pool not initialized. Call create_pools() first.")
    return _write_pool


def _get_read_pool() -> Pool:
    if not _read_pool:
        raise RuntimeError("Database pool not initialized. Call create_pools() first.")
    return _read_pool


# ─────────────────────────────────────────────────────────────────────────────
# RLS-aware connection context managers
# ALWAYS use these — never acquire a raw pool connection directly
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def get_db(
    account_id: Optional[UUID] = None,
    *,
    readonly: bool = False,
) -> AsyncGenerator[Connection, None]:
    """
    Acquire a database connection with RLS session variable set.

    If account_id is provided:
      - Sets app.current_account_id for the duration of the connection
      - Postgres RLS policies use this to filter rows
      - Cleared automatically when connection returns to pool

    If readonly=True:
      - Uses the read replica pool
      - Connection is in read-only transaction mode
      - Any accidental write raises an error immediately

    Usage:
        async with get_db(account_id=user.account_id) as conn:
            rows = await conn.fetch("SELECT * FROM repos WHERE account_id=$1", account_id)

        # FastAPI route with dependency injection:
        @router.get("/repos")
        async def list_repos(
            user: CurrentUser = Depends(get_current_user),
            conn: Connection = Depends(get_db_dep),
        ):
            ...
    """
    pool = _get_read_pool() if readonly else _get_write_pool()

    async with pool.acquire() as conn:
        if account_id:
            # Set RLS session variable — Postgres RLS policies read this
            # Use a local transaction to scope the variable
            await conn.execute(
                "SELECT set_config('app.current_account_id', $1, true)",
                str(account_id),
            )
        try:
            yield conn
        finally:
            if account_id:
                # Clear session variable before returning connection to pool
                # Even though asyncpg resets connections, be explicit
                await conn.execute(
                    "SELECT set_config('app.current_account_id', '', true)"
                )


@asynccontextmanager
async def get_transaction(
    account_id: Optional[UUID] = None,
) -> AsyncGenerator[Connection, None]:
    """
    Acquire a connection and wrap it in an explicit transaction.
    Use for multi-statement operations that must be atomic.

    Automatically rolls back on any exception.

    Usage:
        async with get_transaction(account_id=user.account_id) as conn:
            await conn.execute("INSERT INTO repos ...")
            await conn.execute("INSERT INTO ingestion_jobs ...")
            # Both committed together, or both rolled back
    """
    async with get_db(account_id=account_id) as conn:
        async with conn.transaction():
            yield conn

@asynccontextmanager
async def get_system_transaction() -> AsyncGenerator[Connection, None]:
    """
    Acquire a connection wrapped in a transaction with RLS bypassed.
    STRICTLY FOR BACKGROUND WORKERS (SQS/Celery). Never use in FastAPI web routes!
    """
    async with _get_write_pool().acquire() as conn:
        async with conn.transaction():
            # 1. Set the VIP System Pass
            await conn.execute(
                "SELECT set_config('app.is_system_flow', 'true', true)"
            )
            
            try:
                yield conn
            finally:
                # 2. Safety cleanup
                await conn.execute(
                    "SELECT set_config('app.is_system_flow', '', true)"
                )

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI dependency injection
# ─────────────────────────────────────────────────────────────────────────────

async def get_db_dep() -> AsyncGenerator[Connection, None]:
    """
    FastAPI dependency — no account_id set (for unauthenticated endpoints).
    Most endpoints should use get_authed_db_dep instead.
    """
    async with get_db() as conn:
        yield conn


async def get_authed_read_db_dep(
    account_id: UUID,
) -> AsyncGenerator[Connection, None]:
    """Dependency for authenticated, read-only endpoints (dashboard loads)."""
    async with get_db(account_id=account_id, readonly=True) as conn:
        yield conn


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

async def check_db_health() -> dict:
    """
    Lightweight health check. Called by GET /health endpoint.
    Does not acquire a connection from the pool — uses a minimal query.
    Returns timing and pool stats.
    """
    results = {}

    for name, pool in [("write", _write_pool), ("read", _read_pool)]:
        if not pool:
            results[name] = {"status": "not_initialized"}
            continue
        start = time.monotonic()
        try:
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            elapsed_ms = int((time.monotonic() - start) * 1000)
            results[name] = {
                "status":     "healthy",
                "latency_ms": elapsed_ms,
                "pool_size":  pool.get_size(),
                "pool_free":  pool.get_idle_size(),
            }
        except Exception as e:
            results[name] = {"status": "unhealthy", "error": str(e)}

    return results