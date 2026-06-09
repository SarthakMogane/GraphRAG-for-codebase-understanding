# Core DB write — upsert user + account + settings
# This is the function that solves the chicken-and-egg RLS problem
# ─────────────────────────────────────────────────────────────────────────────

from typing import Optional
from datetime import datetime
from uuid import UUID
from src.core.logger import get_logger
from src.core.database import get_transaction

logger = get_logger(__name__)
 
async def _upsert_user(
    github_user: dict,
    primary_email: Optional[str],
    oauth_token_enc: bytes,
    refresh_token_enc: Optional[bytes],
    token_expires: Optional[datetime],
) -> tuple[UUID, UUID, bool]:
    """
    Create or update user + account + settings in one atomic transaction.
 
    Returns: (user_id, account_id, is_new_user)
 
    """
    github_id = github_user["id"]
 
    # get_db() with no account_id → no RLS variable set yet
    # We set is_auth_flow inside the transaction instead
    async with get_transaction() as conn:
 
        # ── Enable auth bypass for this transaction ────────────────────────
        
        await conn.execute(
            "SELECT set_config('app.is_auth_flow', 'true', true)"
        )
 
        # ── Check if user already exists ───────────────────────────────────
        existing_user = await conn.fetchrow(
            """
            SELECT u.id, u.account_id
            FROM users u
            WHERE u.github_id = $1
            """,
            github_id,
        )
 
        is_new = existing_user is None
 
        if is_new:
            # ── Create account first (user references it) ──────────────────
            account_id = await conn.fetchval(
                """
                INSERT INTO accounts (type, plan)
                VALUES ('personal', 'free')
                RETURNING id
                """
            )
 
            # ── Create user ────────────────────────────────────────────────
            user_id = await conn.fetchval(
                """
                INSERT INTO users (
                    account_id,
                    github_id,
                    github_login,
                    github_email,
                    github_avatar_url,
                    github_name,
                    oauth_token_enc,
                    refresh_token_enc,
                    oauth_token_expires,
                    role
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'owner')
                RETURNING id
                """,
                account_id,
                github_id,
                github_user.get("login"),
                primary_email,
                github_user.get("avatar_url"),
                github_user.get("name"),
                oauth_token_enc,
                refresh_token_enc,
                token_expires,
            )
 
            # ── Create user settings with defaults ─────────────────────────
            await conn.execute(
                """
                INSERT INTO user_settings (user_id)
                VALUES ($1)
                """,
                user_id,
            )
 
            logger.info(
                "New user created: github_login=%s account_id=%s",
                github_user.get("login"), account_id,
            )
 
        else:
            # ── Returning user — update mutable fields only ────────────────
            # github_id is stable and never changes
            # Update: avatar, name, email (can change), token (always refresh)
            user_id    = existing_user["id"]
            account_id = existing_user["account_id"]

            await conn.execute(
                "SELECT set_config('app.current_account_id',1$ ,true)",
                str(account_id)
            )

            await conn.execute(
                "SELECT set_config('app.is_auth_flow',false,true)"
            )
#update last seen or updated due to trigger ?
            await conn.execute(
                """
                UPDATE users SET
                    github_login        = $1,
                    github_email        = $2,
                    github_avatar_url   = $3,
                    github_name         = $4,
                    oauth_token_enc     = $5,
                    refresh_token_enc   = $6,
                    oauth_token_expires = $7,
                    last_seen_at        = NOW(), 
                    updated_at          = NOW()   
                WHERE id = $8
                """,
                github_user.get("login"),
                primary_email,
                github_user.get("avatar_url"),
                github_user.get("name"),
                oauth_token_enc,
                refresh_token_enc,
                token_expires,
                user_id,
            )
 
            logger.info(
                "Returning user updated: github_login=%s",
                github_user.get("login"),
            )
 
        # Transaction commits here — is_auth_flow clears automatically
 
    return user_id, account_id, is_new