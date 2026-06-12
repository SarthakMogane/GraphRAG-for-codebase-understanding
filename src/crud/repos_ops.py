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