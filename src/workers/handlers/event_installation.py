#  installation event
# ─────────────────────────────────────────────────────────────────────────────
from src.core.database import get_system_transaction ,get_transaction
from src.core.logger import get_logger
from src.crud.repos_ops import _upsert_repos_in_conn ,_remove_repos_in_conn
from src.core.exceptions import TransientWebhookError
import asyncpg , asyncio
logger = get_logger(__name__)

async def _handle_installation(payload: dict, delivery_id: str) -> None:
    """
    Handles all installation.* actions.
 
    IDOR-safe account linking:
      installtion_id already saved using user account_id who clicked the installation
 
    Multi-account scenario (USER A public + USER A private):
      User logs in as github_id=111 (personal account)
      Installs app on github_id=222 (work org or secondary account)
      Webhook payload: sender.id=111, installation.account.id=222
      We look up sender.id=111 → get account_id from users table
      Store installation with that account_id
      Result: the installation for "222" belongs to account "111" ✓
    """
    action       = payload.get("action")
    installation = payload.get("installation", {})
    sender       = payload.get("sender", {})
 
    github_install_id = installation.get("id")
    owner_login       = installation.get("account", {}).get("login")
    owner_type        = installation.get("account", {}).get("type", "User")
    owner_github_id   = installation.get("account", {}).get("id")
    sender_github_id  = sender.get("id")
 
    if not github_install_id:
        logger.error("installation event missing installation.id — delivery=%s", delivery_id)
        raise ValueError("Webhook payload is missing installation.id")
 
    if action == "created":
        await _handle_installation_created(
            payload=payload,
            github_install_id=github_install_id,
            owner_login=owner_login,
            owner_type=owner_type,
            owner_github_id=owner_github_id,
            sender_github_id=sender_github_id,
        )
 
    elif action == "deleted":
        await _handle_installation_deleted(
            github_install_id=github_install_id
        )
 
    elif action == "suspend":
        await _handle_installation_suspend(
            github_install_id=github_install_id
        )
 
    elif action == "unsuspend":
        await _handle_installation_unsuspend(
            github_install_id=github_install_id
        )
 
    else:
        logger.debug("Unhandled installation action: %s", action)


async def _handle_installation_repos(
    payload: dict
) -> None:
    """
    Handles 'installation_repositories' events (added or removed).
    """
    action = payload.get("action") # Usually 'added' or 'removed'
    github_install_id = payload.get("installation", {}).get("id")

    # 🚨 POISON PILL CHECK: If GitHub sends a broken payload, let it crash!
    if not github_install_id:
        raise ValueError("Webhook payload is missing installation.id")

    # ── PHASE 1: FETCH INSTALLATION (System VIP Pass) ─────────────────────────
    try:
        async with get_system_transaction() as conn:
            install_row = await conn.fetchrow(
                "SELECT id, account_id FROM installations WHERE github_install_id = $1",
                github_install_id
            )
    except asyncpg.PostgresError as e:
        raise TransientWebhookError(f"DB failed fetching install context: {e}")

    # ── RACE CONDITION CHECK ──────────────────────────────────────────────────
    if not install_row:
        # The frontend redirect hasn't created the installation row yet!
        # We throw this back to SQS to retry in 30 seconds.
        logger.warning(
            "Install row not found for install_id %d. Throwing to SQS for retry.", 
            github_install_id
        )
        raise TransientWebhookError("Installation row not found. Cannot sync repos yet.")

    account_uuid = install_row["account_id"]
    install_db_id = str(install_row["id"])

    repos_added = payload.get("repositories_added", [])
    repos_removed = payload.get("repositories_removed", [])

    # ── PHASE 2: SYNC REPOSITORIES (Secure User Context) ──────────────────────
    if repos_added or repos_removed:
        try:
            # We now know who the user is! Switch to the strict RLS transaction.
            async with get_transaction(account_id=account_uuid) as conn:
                
                # 1. Handle Added Repos
                if repos_added:
                    await _upsert_repos_in_conn(
                        conn=conn,
                        repos=repos_added,
                        installation_db_id=install_db_id,
                        account_id=str(account_uuid)
                    )

                # 2. Handle Removed Repos
                if repos_removed:
                    await _remove_repos_in_conn(
                        conn=conn,
                        repos=repos_removed,
                        installation_db_id=install_db_id,
                        account_id=str(account_uuid)
                    )
                    
        except asyncpg.PostgresError as e:
            raise TransientWebhookError(f"DB failed syncing repositories: {e}")

    logger.info(
        "Successfully synced repos for install_id %d. action:%s , Added: %d, Removed: %d",
        github_install_id,action, len(repos_added), len(repos_removed)
    )

#helpers ------

async def _handle_installation_created(
    payload: dict,
    github_install_id: int,
    owner_login: str,
    owner_type: str,
    owner_github_id: int
) -> None:
    """
    New app installation.
 
    Steps:
      1. Look up sender's account_id via github_id (IDOR-safe)
      2. INSERT installation row (or update if already exists from recovery)
      3. INSERT all repos from the repositories_added payload
      4. Mark webhook processed
 
    All in ONE transaction — either everything saves or nothing does.
    """
 
    try:
        # Use ONE system transaction for the entire operation.
        # This bypasses RLS so we can insert even if account_id is NULL.
        async with get_system_transaction() as conn:
            
            # ── 1. UPSERT Installation ────────────────────────────────────────
            install_row = await conn.fetchrow(
                """
                INSERT INTO installations (
                    github_install_id, owner_login, owner_type, owner_github_id, is_active
                )
                VALUES ($1, $2, $3, $4, TRUE)
                ON CONFLICT (github_install_id) 
                DO UPDATE SET 
                    owner_login = EXCLUDED.owner_login,
                    owner_type = EXCLUDED.owner_type,
                    owner_github_id = EXCLUDED.owner_github_id,
                    is_active = TRUE,
                    updated_at = NOW()
                RETURNING id, account_id;
                """,
                github_install_id,
                owner_login,
                owner_type,
                owner_github_id,
            )

            if not install_row:
                raise TransientWebhookError("Failed to return row during installation UPSERT")

            internal_install_id = install_row["id"]
            account_uuid = install_row["account_id"] # This might be None!

            # ── 2. UPSERT Repositories ────────────────────────────────────────
            repos_added = payload.get("repositories", [])
            
            for repo in repos_added:
                repo_id = repo.get("id")
                full_name = repo.get("full_name")
                repo_name = repo.get("name")
                is_private = repo.get("private", True)
                
                await conn.execute(
                    """
                    INSERT INTO repos (
                        account_id, installation_id, github_repo_id, 
                        full_name, repo_name, owner_login, private, index_status
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, 'not_indexed')
                    ON CONFLICT (github_repo_id) 
                    DO UPDATE SET
                        installation_id = EXCLUDED.installation_id,
                        full_name = EXCLUDED.full_name,
                        repo_name = EXCLUDED.repo_name,
                        private = EXCLUDED.private,
                        index_status = 'not_indexed',
                        updated_at = NOW();
                    """,
                    account_uuid,         # Safe to pass None
                    internal_install_id,  # Guaranteed to exist from Step 1
                    repo_id,
                    full_name,
                    repo_name,
                    owner_login,
                    is_private
                )

        # Transaction automatically commits here upon exiting the 'async with' block.
        logger.info("Successfully UPSERTED install %d with %d repos", github_install_id, len(repos_added))

    except asyncpg.PostgresError as db_err:
        logger.error("DB error during installation transaction: %s", db_err)
        # SQS will catch this, wait, and retry the whole process cleanly.
        raise TransientWebhookError(f"Database failed during installation: {db_err}")
    
                  
 
async def _handle_installation_deleted(
    github_install_id: int
) -> None:
    """
    App uninstalled. Mark installation and all its repos as inaccessible.
    We keep the data — user may re-install and expect their history back.
    """
    try:
        async with get_system_transaction() as conn:
            
                # Mark installation inactive
                await conn.execute(
                    """
                    UPDATE installations
                    SET is_active = FALSE, uninstalled_at = NOW()
                    WHERE github_install_id = $1
                    """,
                    github_install_id,
                )
 
                # Mark all repos covered by this installation as inaccessible
                await conn.execute(
                    """
                    UPDATE repos SET
                        index_status = 'inaccessible',
                        updated_at   = NOW()
                    WHERE installation_id = (
                        SELECT id FROM installations
                        WHERE github_install_id = $1
                    )
                    """,
                    github_install_id,
                )

        logger.info("Installation deleted: install_id=%d", github_install_id)
 
    except asyncpg.PostgresError as e:
        logger.exception(
            "Failed to handle installation.deleted — install_id=%d: %s",
            github_install_id, e,
        )
        raise TransientWebhookError(f"DB failed during installation.deleted: {e}")
 
 
async def _handle_installation_suspend(
    github_install_id: int
) -> None:
    try:
        async with get_system_transaction() as conn:
            await conn.execute(
                """
                UPDATE installations
                SET suspended_at = NOW()
                WHERE github_install_id = $1
                """,
                github_install_id,
            )
        logger.info("Installation suspended: install_id=%d", github_install_id)
    except asyncpg.PostgresError as e:
        raise TransientWebhookError(f"DB failed during suspend_at change:{e}")
 
 
async def _handle_installation_unsuspend(
    github_install_id: int
) -> None:
    try:
        async with get_system_transaction() as conn:
            await conn.execute(
                """
                UPDATE installations
                SET suspended_at = NULL
                WHERE github_install_id = $1
                """,
                github_install_id,
            )
        logger.info("Installation unsuspended: install_id=%d", github_install_id)
    except asyncpg.PostgresError as e:
        logger.exception("Failed to handle unsuspend — install_id=%d: %s", github_install_id, e)
        raise TransientWebhookError(f"DB failed during unsuspending change:{e}")