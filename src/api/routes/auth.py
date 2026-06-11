from src.database.mock_db import MOCK_DB
from fastapi import APIRouter, Request, HTTPException ,Depends
from fastapi.responses import RedirectResponse,JSONResponse
from authlib.integrations.starlette_client import OAuth
from src.core.config import get_settings 
from src.core.logger import get_logger
from src.core.crypto import encrypt_token, decrypt_token
from src.utils.services_helpers import get_github_service
from src.utils.auth_helpers import _parse_token_expiry
from src.services.github import GitHubService
from src.services.github_oauth import oauth
from src.crud.user import _upsert_user
from src.crud.installation import _recover_installations ,_save_installation
from src.core.database import get_authed_read_db_dep
from uuid import UUID

import asyncio
import secrets


settings = get_settings()
logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/login")
async def login_via_github(request: Request):
    """
    This endpoint instantly redirects user to GitHub's authorization page.
    """
    request.session.clear()
    redirect_uri = request.url_for('auth_github_callback')
    
    return await oauth.github.authorize_redirect(request, redirect_uri,code_challenge_method='S256')


@router.get("/github/callback", name="auth_github_callback")
async def auth_github_callback(request: Request):
    """
    GitHub sends the user back here with a temporary 'code'.
    We exchange it for a token and fetch their identity.
    """
    error = request.query_params.get('error')
    if error:
        logger.warning("GitHub OAuth error: %s", error)
        redirect_url = (
            f"{settings.FRONTEND_URL}?error=access_denied"
            if error == "access_denied"
            else f"{settings.FRONTEND_URL}?error=oauth_failed"
        )
        return RedirectResponse(url=redirect_url)
    
    try:
        token_data = await oauth.github.authorize_access_token(request)
    except Exception as e:
        logger.error("Token exchange failed: %s", e)
        return RedirectResponse(url=f"{settings.FRONTEND_URL}?error=token_failed")
 
    oauth_token   = token_data.get("access_token")
    refresh_token = token_data.get("refresh_token")

    if not oauth_token:
        return RedirectResponse(url=f"{settings.FRONTEND_URL}?error=no_token")

    
    tasks = [
        oauth.github.get('user', token=token_data),
        oauth.github.get('user/emails', token=token_data),
        oauth.github.get('user/installations', token=token_data)
    ]

    response = await asyncio.gather(*tasks,return_exceptions=True)
    user_resp, email_resp,installations_resp = response

    if isinstance(user_resp,Exception)or user_resp.status_code != 200:
        logger.error("Failed to fetch core GitHub profile data: %s", user_resp)
        return RedirectResponse(url=f"{settings.FRONTEND_URL}?error=profile_failed")
   
    github_user = user_resp.json()

    primary_email = None
    if not isinstance(email_resp, Exception) and email_resp.status_code == 200:
        emails = email_resp.json()
        primary_email = next(
            (
                email["email"] 
                for email in emails 
                if email.get("primary") and email.get("verified")
            ),
            None
        )
    # Fallback: If no verified primary email is found, check the public profile email field
    if not primary_email:
        primary_email = github_user.get("email")

    existing_installs = []
    if not isinstance(installations_resp, Exception) and installations_resp.status_code == 200:
        existing_install = installations_resp.json().get("installations", [])

    # encrypting token 
    oauth_token_enc   = await encrypt_token(oauth_token)
    refresh_token_enc = await encrypt_token(refresh_token) if refresh_token else None
    token_expires     = _parse_token_expiry(token_data)

    #db save
    user_id, account_id, is_new = await _upsert_user(
        github_user=github_user,
        primary_email=primary_email,
        oauth_token_enc=oauth_token_enc,
        refresh_token_enc=refresh_token_enc,
        token_expires=token_expires,
    )
    
     # ── Recover any existing GitHub App installations ─────────────────────────
    if existing_installs:
        await _recover_installations(
            account_id=account_id,
            installs=existing_installs,
        )
    
    request.session["user_id"] = user_id
    request.session["account_id"] = account_id
    request.session["is_new"] = is_new

    redirect_url = (
        f"{settings.FRONTEND_URL}?status=new_user"
        if is_new else
        f"{settings.FRONTEND_URL}?status=returning"
    )

    return RedirectResponse(url=redirect_url , status_code=302)


@router.get("/status")
async def get_auth_status(request: Request):
    """
    Fast session check. Called by frontend to know:
      - Is user logged in?
      - Have they installed the GitHub App?
      - What's their plan?
 
    Only reads from session (no DB query) for the auth check itself.
    Then does ONE DB query for display info (username, plan, install status).
    """
    user_id    = request.session.get("user_id")
    account_id = request.session.get("account_id")
 
    if not user_id or not account_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
 
    try:
        user_uuid    = UUID(user_id)
        account_uuid = UUID(account_id)
    except ValueError:
        # Malformed session value — clear and reject
        request.session.clear()
        raise HTTPException(status_code=401, detail="Invalid session")
 
    # One DB query with join — gets everything needed for the status response
    async with get_authed_read_db_dep(account_id=account_uuid) as conn:
        row = await conn.fetchrow(
            """
            SELECT
                u.id,
                u.github_login,
                u.github_avatar_url,
                a.plan,
                a.payment_status,
                a.queries_this_month,
                a.max_queries_month,
                EXISTS(
                    SELECT 1 FROM installations i
                    WHERE i.account_id = a.id
                      AND i.is_active = TRUE
                ) AS is_installed
            FROM users u
            JOIN accounts a ON a.id = u.account_id
            WHERE u.id = $1
              AND u.account_id = $2
            """,
            user_uuid, account_uuid,
        )
 
    if not row:
        # User in session but not in DB — session is stale
        request.session.clear()
        raise HTTPException(status_code=401, detail="Session expired")
 
    return {
        "authenticated":     True,
        "user_id":           str(row["id"]),
        "username":          row["github_login"],
        "avatar_url":        row["github_avatar_url"],
        "plan":              row["plan"],
        "is_installed":      row["is_installed"],
        "queries_remaining": max(
            0, row["max_queries_month"] - row["queries_this_month"]
        ),
    }

@router.get("/install")
async def redirect_to_github_install(request: Request):
    """The frontend calls this when the user actively clicks 'Connect GitHub'"""
    user_id = request.session.get("user_id")
    if not user_id:
        return RedirectResponse(url=f"{settings.FRONTEND_URL}?error=not_authenticated")
    
    install_state = secrets.token_urlsafe(32)  

    request.session["github_install_state"] = install_state

    install_url = f"https://github.com/apps/{settings.GITHUB_APP_SLUG}/installations/new?state={install_state}"
    
    return RedirectResponse(url=install_url)


@router.get("/github/setup-redirect")
async def github_app_setup_redirect(
    request: Request, 
    installation_id: str = None,
    setup_action: str = "install", 
    state: str = None
):
    # 1. Initial Installation (Triggered from your App)
    if setup_action == "install":
        session_state = request.session.pop("github_install_state", None)
        account_id = request.session.get("account_id")
        account_uuid = UUID(account_id)

        if not session_state or state != session_state:
            logger.warning("CSRF state mismatch on app install redirect")
            request.session.clear()
            return RedirectResponse(url=f"{settings.FRONTEND_URL}?error=csrf_failed")

        is_new_install = await _save_installation(account_id =account_uuid ,installion_id =installation_id)  

        return RedirectResponse(url=f"{settings.FRONTEND_URL}?status=installed&new_install={str(is_new_install).lower()}")

    # 2. Permissions Update (Triggered from GitHub Settings)
    elif setup_action == "update":
        return RedirectResponse(url=f"{settings.FRONTEND_URL}?status=updated")

    # 3. Fallback for anomalous requests
    else:
        # If they somehow got here with missing or weird parameters, 
        # kick them back to the frontend to let the standard auth check handle them.
        return RedirectResponse(url=settings.FRONTEND_URL)

@router.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return JSONResponse(content={"success": True}, status_code=200)



