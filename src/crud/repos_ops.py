from uuid import UUID
from fastapi import HTTPException
from src.core.logger import get_logger
import asyncpg
from src.services.pre_clone.types import ValidationVerdict, RoutingDecision, ValidationResult

logger = get_logger(__name__)

async def _get_indexed_repos(conn) -> dict[str, int]:
    "return the users indexed repos "
    indexed_rows = await conn.fetch(
            "SELECT owner_login, repo_name, id FROM repos WHERE index_status = 'ready'"
        )

    return {f"{r.github_owner}/{r.github_repo}": r.id for r in indexed_rows}

async def _upsert_repos_in_conn(
    conn,
    repos: list[dict],
    installation_db_id: str,
    account_id: str,
) -> None:
    """
    INSERT repos from a GitHub payload list. Called inside an existing
    transaction — does not open its own transaction.

    Stores BOTH public and private repos to support UI visibility and 
    future repository visibility toggles.
    """
    for repo in repos:
        full_name = repo.get("full_name", "")
        if "/" not in full_name:
            continue

        owner_login, repo_name = full_name.split("/", 1)
        
        # 1. Capture the actual visibility flag from GitHub
        is_private = repo.get("private", True) 

        await conn.execute(
            """
            INSERT INTO repos (
                account_id, installation_id,
                github_repo_id, full_name, owner_login, repo_name,
                private, index_status
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, 'not_indexed')
            ON CONFLICT (github_repo_id)
            DO UPDATE SET
                installation_id = EXCLUDED.installation_id,
                account_id      = EXCLUDED.account_id,
                full_name       = EXCLUDED.full_name,
                
                -- 2. CRITICAL: Update visibility in case it changed 
                -- since the last time this webhook fired
                private         = EXCLUDED.private, 
                
                updated_at      = NOW()
                
                -- Deliberately NOT updating index_status
                -- so existing indexed repos keep their READY/STALE status
            """,
            account_id,
            installation_db_id,
            repo.get("id"),
            full_name,
            owner_login,
            repo_name,
            is_private, # Pass the dynamic boolean here
        )


async def _remove_repos_in_conn(
    conn,  # The live connection passed from the parent's get_transaction()
    repos: list[dict],
    installation_db_id: str,
    account_id: str
) -> None:
    """
    Marks removed repositories as 'inaccessible' instead of hard-deleting them.
    Executes in a single bulk query for maximum performance.
    """
    if not repos:
        return

    # Extract just the raw GitHub repo IDs into a flat Python list
    # e.g., [12345, 67890, 112233]
    repo_ids_to_remove = [
        repo.get("id") for repo in repos if repo.get("id")
    ]

    if not repo_ids_to_remove:
        return

    # Use ANY() to bulk update every repo in that list in one shot
    await conn.execute(
        """
        UPDATE repos
        SET
            index_status = 'inaccessible',
            updated_at = NOW()
        WHERE
            account_id = $1
            AND installation_id = $2
            AND github_repo_id = ANY($3::bigint[])
        """,
        account_id,
        installation_db_id,
        repo_ids_to_remove
    )

# Dependency: verify ownership — prevents IDOR
# ─────────────────────────────────────────────────────────────────────────────

async def _get_owned_repo(
    repo_id: int,
    account_id: UUID,
    conn,  
):
    """
    Returns the Repository record only if it belongs to account_id.
    Always returns 404 (never 403) to prevent existence enumeration.
    """
    row = await conn.fetchrow(
        """
        SELECT 
            r.id, 
            r.account_id, 
            r.repo_name, 
            r.github_repo_id, 
            r.index_status,
            i.github_install_id
        FROM repos r
        INNER JOIN installations i
        ON i.id = r.installation_id 
        WHERE id = $1 
          AND account_id = $2
        """,
        repo_id, 
        account_id,
    )

    if not row:
        # Deliberately 404, not 403 — don't reveal whether repo_id exists
        raise HTTPException(status_code=404, detail="Repository not found")

    return row



async def apply_pipeline_result_to_db(
    conn, 
    repo_id: int, 
    result: ValidationResult
) -> None:
    """
    Translates a ValidationResult directly into a PostgreSQL update.
    Maps routing decisions to string statuses and merges fresh GitHub metadata.
    """
    # ── 1. Translate the Business Decision to a DB String ──────────────
    new_status = "not_indexed" # Safe fallback
    
    if result.verdict == ValidationVerdict.REPO_NOT_FOUND:
        new_status = "inaccessible"
    elif result.routing == RoutingDecision.SERVE_CACHE:
        new_status = "ready"
    elif result.routing in [RoutingDecision.NEW_INGESTION, RoutingDecision.REFRESH]:
        new_status = "pending"

    # ── 2. Execute the Database Update ─────────────────────────────────
    try:
        # We always update the status, but only update metadata if github_id exists
        if result.github_id:
            await conn.execute(
                """
                UPDATE repos 
                SET 
                    index_status = $1,
                    github_repo_id = COALESCE($2, github_repo_id),
                    default_branch = COALESCE($3, default_branch),
                    primary_language = COALESCE($4, primary_language),
                    repo_size_kb = COALESCE($5, repo_size_kb),
                    is_fork = COALESCE($6, is_fork),
                    updated_at = NOW()
                WHERE id = $7
                """,
                new_status,
                result.github_id,
                result.default_branch,
                result.primary_language,
                result.size_kb,
                result.fork_info.is_fork if result.fork_info else None,
                repo_id
            )
        else:
            # If pipeline failed early (no metadata), just update the status
            await conn.execute(
                "UPDATE repos SET index_status = $1, updated_at = NOW() WHERE id = $2",
                new_status,
                repo_id
            )

    except asyncpg.PostgresError as db_err:
        logger.error("Failed to apply pipeline result for repo %d: %s", repo_id, db_err)
        raise HTTPException(status_code=500, detail="Database error during repo sync.")