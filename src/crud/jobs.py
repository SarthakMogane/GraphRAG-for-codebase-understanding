from src.core.config import get_settings
from fastapi import HTTPException
from uuid import UUID 

settings = get_settings()

async def _execute_queue_insertion_raw(
    conn, 
    repo_id: int, 
    selection_id: int, 
    account_id: UUID
) -> int:
    
    "job insertion and queue for selection endpoint "
    #  Lock the repository row immediately to freeze out duplicate button clicks
    repo_state = await conn.fetchrow(
        """
        SELECT index_status, commit_sha 
        FROM repos 
        WHERE id = $1 
        FOR UPDATE NOWAIT
        """,
        repo_id
    )

    if repo_state["index_status"] in ("pending", "indexing"):
        raise HTTPException(
            status_code=409, 
            detail="An ingestion task is already queued or actively processing for this repository."
        )
    
    # Determine job categorization strictly from the locked record truth
    job_type = "refresh" if repo_state["commit_sha"] else "initial"

    job_id = await conn.fetchval(
        """
        INSERT INTO ingestion_jobs (repo_id, account_id, selection_id, job_type, status, created_at, updated_at)
        VALUES ($1, $2, $3, $4, 'dispatch_pending', NOW(), NOW())
        RETURNING id
        """,
        repo_id, account_id, selection_id, job_type
    )

    await conn.execute("UPDATE repos SET index_status = 'pending', updated_at = NOW() WHERE id = $1", repo_id)

    return job_id