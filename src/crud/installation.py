from uuid import UUID 
from src.core.logger import get_logger
from src.core.config import get_settings
from src.core.database import get_transaction

logger = get_logger(__name__)
settings = get_settings()

async def _recover_installations(
    account_id: UUID,
    installs: list[dict],
) -> None:
    """
    If the user has already installed the GitHub App on some orgs,
    save those installations to DB now.
 
    Why: User may have installed the app before creating an account
    (e.g. through GitHub Marketplace), or they reinstalled after
    revoking access. We recover at login time so they don't have to
    go through the install flow again.
 
    The installation webhook should also fire, but this is the safety net.
    """
    our_installs = [
        i for i in installs
        if i.get("app_slug") == settings.GITHUB_APP_SLUG
    ]
 
    if not our_installs:
        return
 
    async with get_transaction(account_id=account_id) as conn:
        for install in our_installs:
            await conn.execute(
                """
                INSERT INTO installations (
                    account_id, github_install_id, owner_login,
                    owner_type, owner_github_id, is_active
                )
                VALUES ($1, $2, $3, $4, $5, TRUE)
                ON CONFLICT (github_install_id)
                DO UPDATE SET
                    account_id  = EXCLUDED.account_id,
                    is_active   = TRUE,
                    suspended_at = NULL
                """,
                account_id,
                install["id"],
                install["account"]["login"],
                install["account"]["type"],
                install["account"]["id"],
            )
            logger.info(
                "Installation recovered: org=%s install_id=%d",
                install["account"]["login"], install["id"],
            )