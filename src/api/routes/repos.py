from fastapi import APIRouter, HTTPException, Request
from typing import List
from src.services.github import GitHubService
from src.db.mock_db import MOCK_DB

# Notice: Because prefix="/api", your routes automatically become /api/repos
router = APIRouter(prefix="/api", tags=["Repositories"])

github_service = GitHubService()

@router.get("/repos")
async def list_installed_repositories(request: Request):
    """
    Fetches the list of repositories the user has granted the app access to.
    Used to populate the frontend dashboard.
    """
    # 1. Verify the user is logged in via their EXPLICIT cookie
    github_id = request.session.get("auth_user_id")
    if not github_id or github_id not in MOCK_DB["users"]:
        raise HTTPException(status_code=401, detail="Not authenticated")

    github_id = str(github_id)
    # 2. Get their OAuth token and installation ID safely using .get()
    user_oauth_token = MOCK_DB["users"][github_id]["oauth_token"]
    installation_id = MOCK_DB.get("installations", {}).get(github_id)

    # If they haven't installed the app, return an empty list
    if not installation_id:
        return []

    try:
        # 3. Fetch their repos using the new User-to-Server method
        raw_repos = await github_service.get_installed_repositories(
            user_oauth_token=user_oauth_token,
            installation_id=installation_id
        )
        
        # 4. Format the data to match exactly what your index.html expects
        formatted_repos = [
            {
                "name": repo["full_name"],
                "private": repo["private"],
                "url": repo["html_url"]
            }
            for repo in raw_repos
        ]
        
        return formatted_repos
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch repositories: {str(e)}")


# Note: Changed from {full_name} to {owner}/{repo} so FastAPI parses it automatically for us!
@router.get("/repos/{owner}/{repo}/branches", response_model=List[str])
async def list_branches(owner: str, repo: str, request: Request):
    """
    Fetches the branches for a specific repository. 
    Uses the App Installation Token (Server-to-Server) so it can run in the background.
    """
    # 1. Authenticate via explicit cookie (No more JWT Depends!)
    github_id = request.session.get("auth_user_id")
    if not github_id or github_id not in MOCK_DB["users"]:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # 2. Get the Installation ID 
    installation_id = MOCK_DB.get("installations", {}).get(github_id)
    if not installation_id:
        raise HTTPException(status_code=403, detail="GitHub App not installed")

    try:
        # 3. Call the APP-level service (Server-to-Server flow!)
        # We don't pass the user token anymore, just the installation ID
        branches = await github_service.get_repo_branches(
            owner=owner, 
            repo=repo, 
            installation_id=installation_id
        )
        return branches
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch branches: {str(e)}")

from pydantic import BaseModel

class SyncInstallRequest(BaseModel):
    installation_id: int

