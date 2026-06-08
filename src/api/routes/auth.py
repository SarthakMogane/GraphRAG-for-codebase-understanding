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

    existing_install = []
    if not isinstance(installations_resp, Exception) and installations_resp.status_code == 200:
        existing_install = installations_resp.json().get("installations", [])

    # encrypting token 
    oauth_token_enc   = await encrypt_token(oauth_token)
    refresh_token_enc = await encrypt_token(refresh_token) if refresh_token else None
    token_expires     = _parse_token_expiry(token_data)

    # ----------------------------------------------------------------------
    # TODO: DATABASE SAVE
    # user_id, account_id, is_new = await _upsert_user(
    #     github_user=github_user,
    #     primary_email=primary_email,
    #     oauth_token_enc=oauth_token_enc,
    #     refresh_token_enc=refresh_token_enc,
    #     token_expires=token_expires,
    # )
    
    

    #update URL 
    response = RedirectResponse(url="http://127.0.0.1:5500/index.html",status_code= 302)
    # response = RedirectResponse(url="/", status_code=302)

    return response


@router.get("/status")
async def get_auth_status(request: Request):
    github_id = request.session.get("auth_user_id")
    # github_id = request.cookies.get("auth_user_id")
    

    # --- OUR DEBUG TRAP ---
    print("\n=== STATUS CHECK ===")
    print(f"1. ID found in Cookie: {github_id}")
    print(f"2. IDs currently in Database: {list(MOCK_DB['users'].keys())}")
    print("====================\n")
    if not github_id:
        raise HTTPException(status_code=401, detail="Browser did not send the cookie")
    
    # if github_id not in MOCK_DB["users"]:
    #     raise HTTPException(status_code=401, detail="Cookie received, but DB is empty!")
    if not github_id or github_id not in MOCK_DB["users"]:
        raise HTTPException(status_code=401, detail="Not authenticated")
        
    user_data = MOCK_DB["users"][github_id]

    app_is_installed = github_id in MOCK_DB.get("installations", {})
    print(app_is_installed)
    print(MOCK_DB)

    return {
        "authenticated": True, 
        "username": user_data["username"],
        "is_installed": app_is_installed # The frontend will use this!
    }

@router.get("/install")
async def redirect_to_github_install(request: Request):
    """The frontend calls this when the user actively clicks 'Connect GitHub'"""
    github_id = request.session.get("auth_user_id")

    if not github_id:
        # update URL 
        return RedirectResponse(url="http://127.0.0.8000/index.html")
        # return RedirectResponse(url='/')
    install_state = secrets.token_urlsafe(32)  

    request.session["github_install_state"] = install_state

    app_slug = "repobeacon"
    install_url = f"https://github.com/apps/{app_slug}/installations/new?state={install_state}"
    
    return RedirectResponse(url=install_url)


@router.get("/github/setup-redirect")
async def github_app_setup_redirect(
    request: Request, 
    setup_action: str = "install", 
    state: str = None
):
    # 1. Initial Installation (Triggered from your App)
    if setup_action == "install":
        session_state = request.session.pop("github_install_state", None)
        
        if not session_state or state != session_state:
            logger.warning("CSRF / State mismatch detected.")
            # We redirect to /login to safely wipe the slate clean
            return RedirectResponse(url="/login")
            
        return RedirectResponse(url=f"{settings.FRONTEND_URL}?status=installed")

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
    response = JSONResponse(content={"status": "logged_out"})
    # response.delete_cookie("auth_user_id")
    return JSONResponse(content={"success": True}, status_code=200)



