from src.db.mock_db import MOCK_DB
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import RedirectResponse,JSONResponse
from authlib.integrations.starlette_client import OAuth
from src.core.config import get_settings 
import asyncio

settings = get_settings()
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

    return {
        "authenticated": True, 
        "username": user_data["username"],
        "is_installed": app_is_installed # The frontend will use this!
    }

@router.get("/install")
async def redirect_to_github_install(request: Request):
    """The frontend calls this when the user actively clicks 'Connect GitHub'"""
    github_id = request.session.get("auth_user_id")
    # github_id = request.cookies.get("auth_user_id")
    if not github_id:
        # update URL 
        return RedirectResponse(url="http://127.0.0.8000/index.html")
        # return RedirectResponse(url='/')
        
    app_slug = "repobeacon"
    install_url = f"https://github.com/apps/{app_slug}/installations/new"
    
    return RedirectResponse(url=install_url)

@router.post("/logout")
async def logout(request: Request):
    request.session.clear()
    response = JSONResponse(content={"status": "logged_out"})
    # response.delete_cookie("auth_user_id")
    return JSONResponse(content={"success": True}, status_code=200)




# __________________old code ---------------

# from fastapi import APIRouter, Request
# from fastapi.responses import HTMLResponse, RedirectResponse ,JSONResponse
# from authlib.integrations.starlette_client import OAuthError
# # from src.services.github import oauth
# from src.core.security import create_jwt_token

# # import redis 
# # redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# router = APIRouter(tags=["Authentication"])

# @router.get("/login")
# async def login(request: Request):
#     redirect_uri = request.url_for('auth_callback')
#     return await oauth.github.authorize_redirect(request, redirect_uri)

# @router.get("/auth/callback")
# async def auth_callback(request: Request):
#     try:
#         token = await oauth.github.authorize_access_token(request)
#     except OAuthError as error:
#         return HTMLResponse(f'<h1>OAuth Error: {error.error}</h1>')
    
#     resp = await oauth.github.get('user', token=token)
#     user_data = resp.json()  # change:here .
#     username = user_data["login"]
#     github_access_token = token['access_token']


#         # --- STEP 1: STORE GITHUB TOKEN IN REDIS (NOT JWT) ---
#     # Set it to expire in 24 hours to match your JWT
#     # redis_client.setex(
#     #     name=f"gh_token:{username}",
#     #     time=86400, 
#     #     value=github_access_token
#     # ) 
    
#     jwt_token = create_jwt_token(user_data)

#     response = RedirectResponse(url='/')
#     response.set_cookie(
#         key="access_token",
#         value=jwt_token,
#         httponly=True,  # JavaScript CANNOT read this
#         secure=False,   # Set to True in Production (requires HTTPS) change:
#         samesite="lax", # Protects against CSRF attacks
#         max_age=86400   # 24 hours in seconds
#     )
#     return response 


# # --- STEP 3: HOW TO USE THE TOKEN LATER ---
# # @router.get("/api/github/repos")
# # async def get_repos(user: dict = Depends(get_current_user)):
# #     # Get the username from the decoded JWT
# #     username = user["sub"]
    
# #     # Retrieve the secret token from the server-side store
# #     github_token = redis_client.get(f"gh_token:{username}")
    
# #     if not github_token:
# #         raise HTTPException(status_code=401, detail="GitHub session expired")
    
# #     # Now use github_token to call GitHub API...
# #     return {"message": "Success"}

# # @router.get("/api/auth/status")
# # async def check_auth_status(user: dict = Depends(get_current_user)):
# #     """A lightweight endpoint for the frontend to check if the cookie is valid."""
# #     return {"logged_in": True, "username": user["sub"]}

# @router.post("/api/auth/logout")
# async def logout():
#     """Destroys the secure cookie to log the user out."""
#     response = JSONResponse(content={"message": "Logged out"})
#     response.delete_cookie("access_token")

#     # redis_client.delete(f"gh_token:{username}").
#     # change: 
#     return response