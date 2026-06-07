from src.database.mock_db import MOCK_DB
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import RedirectResponse,JSONResponse
from authlib.integrations.starlette_client import OAuth
from src.core.config import get_settings 
from src.core.logger import get_logger
import asyncio
import secrets


settings = get_settings()
logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])

# ── Authlib Setup ─────────────────────────────────────────────────────────────
oauth = OAuth()
oauth.register(
    name='github',
    client_id=settings.GITHUB_CLIENT_ID,
    client_secret=settings.GITHUB_CLIENT_SECRET,
    code_challenge_method='S256',
    access_token_url='https://github.com/login/oauth/access_token',
    authorize_url='https://github.com/login/oauth/authorize',
    api_base_url='https://api.github.com/',
    client_kwargs={'scope': 'user:email'}, 
)

# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/login")
async def login_via_github(request: Request):
    """
    This endpoint instantly redirects user to GitHub's authorization page.
    """
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
        if error == 'access_denied':
            # return RedirectResponse(url="/?error=denied")
            return RedirectResponse(url="http://127.0.0.1:5500/?error=denied")
        raise HTTPException(status_code=400, detail=f"GitHub Error: {error}")
    
    try:
        token_data = await oauth.github.authorize_access_token(request)
        user_oauth_token = token_data.get('access_token')
        
        if not user_oauth_token:
            raise HTTPException(status_code=400, detail="Failed to retrieve access token")

        
        tasks = [
            oauth.github.get('user', token=token_data),
            oauth.github.get('user/emails', token=token_data),
            oauth.github.get('user/installations', token=token_data)
        ]

        response = await asyncio.gather(*tasks)
        user_resp, email_resp,installations_resp = response

        if isinstance(user_resp,Exception):
            raise HTTPException(status_code=500 , detail="Couldn't fetch Github Profile")
        user_resp.raise_for_status()
        github_user = user_resp.json()

        if not isinstance(email_resp,Exception) and email_resp.status_code == 200:
            emails = email_resp.json()
            primary_email = next((email['email'] for email in emails if email['primary']), None)

        if not isinstance(installations_resp, Exception) and installations_resp.status_code == 200:
            install_data = installations_resp.json()
        # ----------------------------------------------------------------------
        # TODO: DATABASE SAVE
        # Here is where you would look up the user in your Postgres/SQLite database.
        # If they don't exist, create them using github_user['id'] and primary_email.
        # Save 'user_oauth_token' to their database record.
        # ----------------------------------------------------------------------
        github_id = str(github_user['id'])
        username = github_user['login']
        # ----------------------------------------------------------------------
        # MOCK DATABASE SAVE
        # ----------------------------------------------------------------------
        MOCK_DB["users"][github_id] = {
            "username": username,
            "email": primary_email,
            "oauth_token": user_oauth_token
        }
        print(f"[DB MOCK] Saved User: {github_user['login']}")
        
        app_slug = "repobeacon" 
        
        # Look through their installed apps to find ours
        existing_install = next(
            (inst for inst in install_data.get('installations', []) if inst['app_slug'] == app_slug), 
            None
        )
        
        if existing_install:
            # They already installed it! Save the ID into our Mock DB
            MOCK_DB["installations"][github_id] = existing_install['id']
            print(f"[DB MOCK] Recovered existing installation ID: {existing_install['id']}")

        request.session["auth_user_id"] = github_id   
        #update URL 
        response = RedirectResponse(url="http://127.0.0.1:5500/index.html",status_code= 302)
        # response = RedirectResponse(url="/", status_code=302)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")


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



