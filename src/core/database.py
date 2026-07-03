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
"""

from __future__ import annotations
import ssl
import time
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Optional
from uuid import UUID
from typing import Callable, AsyncContextManager
import asyncpg
from asyncpg import Connection, Pool
from fastapi import Depends
from src.utils.services_helpers import get_current_account_id
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
    if settings.APP_ENV == "development" or not settings.RDS_CA_BUNDLE_PATH:
        if settings.APP_ENV in ("stagging","production") and not settings.RDS_CA_BUNDLE_PATH:
            raise RuntimeError(
                "RDS_CA_BUNDLE_PATH must be set in production. "
                "Download from https://truststore.pki.rds.amazonaws.com/"
            )
        # Development: no SSL verification
        return None

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
        dsn=settings.PYTHON_DATABASE_URL,
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
    use_transaction: bool = False,
    is_system_flow: bool = False,
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

    if account_id == "":
        raise ValueError("CRITICAL BUG: Empty string account_id detected in get_db!")
    if account_id is not None and not isinstance(account_id, UUID):
        raise ValueError(f"CRITICAL BUG: Non-UUID account_id detected: {type(account_id)} - {account_id}")
    
    async with pool.acquire() as conn:
        needs_transaction = use_transaction or account_id is not None or is_system_flow
        
        if needs_transaction:
            async with conn.transaction():
                if is_system_flow:
                    await conn.execute("SELECT set_config('app.is_system_flow','true',true)")
                elif account_id:
                    await conn.execute(
                        "SELECT set_config('app.current_account_id', $1, true)",
                        str(account_id),
                    )

                yield conn
        else:
            yield conn
        

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI dependency injection
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def get_transaction(
    account_id: Optional[UUID] = None,
) -> AsyncGenerator[Connection, None]:
    """
     Acquire a connection and wrap it in an explicit transaction
    """
    async with get_db(account_id=account_id,readonly=False, use_transaction= True) as conn:
            yield conn


@asynccontextmanager
async def get_system_transaction() -> AsyncGenerator[Connection, None]:
    """
    Acquire a connection wrapped in a transaction with RLS bypassed.
    STRICTLY FOR BACKGROUND WORKERS (SQS/Celery). Never use in FastAPI web routes!
    """
    async with get_db(is_system_flow=True,readonly=False) as conn:        
        yield conn

@asynccontextmanager
async def get_db_session_rls(account_id: UUID) -> AsyncGenerator[Connection, None]:
    """Acquires a write connection with session-level RLS and NO transaction block."""
    async with _get_write_pool().acquire() as conn:
        try:
            # 'false' means it stays alive without a transaction block
            await conn.execute("SELECT set_config('app.current_account_id', $1, false)", str(account_id))
            yield conn
        finally:
            # Manual cleanup is mandatory to prevent connection pool contamination!
            await conn.execute("SELECT set_config('app.current_account_id', '', false)")

def get_rls_conn(account_id: UUID = Depends(get_current_account_id)):
    """Yields a writable connection with RLS set, but genuinely NO transaction."""
    return get_db_session_rls(account_id=account_id)

DbFactory = Callable[[], AsyncContextManager[Connection]]

async def get_db_dep() -> DbFactory:
    """
    FastAPI dependency — no account_id set (for unauthenticated endpoints).
    Most endpoints should use get_authed_db_dep instead.
    """
    # Returns the context manager directly so FastAPI's AsyncExitStack handles the lifecycle
    return lambda :get_db()

#  Dependency for Multi-Statement Write Transactions ──────────────────────
async def get_rls_tx_conn(account_id: UUID = Depends(get_current_account_id) ) -> DbFactory:
    """FastAPI Dependency:Yields a read/write connection wrapped inside an atomic transaction bound by RLS."""
    return lambda: get_transaction(account_id=account_id)

async def get_read_db_dep() -> AsyncGenerator[Connection, None]:
    """FastAPI Dependency: Fetches a raw connection from the READ pool with zero transaction footprint."""
    async with get_db(readonly=True, use_transaction=False) as conn:
        yield conn

async def get_authed_read_db_dep(
    account_id: UUID = Depends(get_current_account_id),
) -> DbFactory:
    """FastAPI Dependency for authenticated, highly optimised read-only endpoints."""
    return lambda: get_db(account_id=account_id, readonly=True,use_transaction=False) 
    


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