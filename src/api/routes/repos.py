from fastapi import APIRouter, HTTPException, Request
from typing import List
from src.services.github import GitHubService
from src.db.mock_db import MOCK_DB , MockRepository
from src.models.database import RepoStatus

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
        print("debug: repo endpoint :",raw_repos)
    except Exception as github_err:
        # Production-grade: Distinct error telling you GitHub service specifically failed
        raise HTTPException(
            status_code=502, 
            detail=f"GitHub API service communication failure: {str(github_err)}"
        )

    try:   
        # 4. Format the data to match exactly what your index.html expects
        formatted_repos = [
            {
                "name": repo["full_name"],
                "private": repo["private"],
                "url": repo["html_url"],
                "visibility":repo["visibility"],
                "default_branch":repo["default_branch"],
                "size":repo["size"]
            }
            for repo in raw_repos
        ]
        repo_table = MOCK_DB["repositories"]
        
        for repo in raw_repos:
            repo_name = repo["full_name"]
            if repo_name not in repo_table:
                repo_table[repo_name] = MockRepository(

                        name = repo_name,
                        private= repo["private"],
                        url= repo["html_url"],
                        visibility = repo["visibility"],
                        default_branch = repo["default_branch"],
                        size = repo["size"],
                        status = None,
                        language =repo["language"]
                )

        return formatted_repos

    except KeyError as key_err:
        # Production-grade: Catches if GitHub unexpectedly changes their JSON payload structure
        raise HTTPException(
            status_code=500, 
            detail=f"Data structure mismatch. Missing expected key: {str(key_err)}"
        )
       
    except Exception as processing_error:
        raise HTTPException(status_code=500, detail=f"Failed to fetch repositories: {str(processing_error)}")


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


@router.post("/repos/status")
async def get_repos_status(
    repo_names: list[str],   # ["myorg/repo1", "myorg/repo2"]
    # db: AsyncSession = Depends(get_db),
):
    # result = await db.execute(
    #     select(Repository).where(
    #         Repository.full_name.in_(repo_names)
    #     )
    # )
    # repos = {r.full_name: r for r in result.scalars().all()}
    mock_repo_tables = MOCK_DB["repositories"]
    print("debug : repo inside DB : \n",mock_repo_tables)
    print("debug : repo list gotten by UI \n ",repo_names)
    result = {}
    for name in repo_names:
        if name not in mock_repo_tables:
            result[name] = {"status": "never_indexed", "action": "index"}
            continue
        
        repo = mock_repo_tables[name]
        status = repo.status
        print(status)
        if status is None:
            result[name] = {"status": "never_indexed", "action": "index"}

        elif status == RepoStatus.READY:
            # Check webhook-flagged staleness (DB only, no API call)
            if repo.is_stale:  # <--- Checked directly from your new DB column!
                result[name] = {
                    "status": "stale",
                    "commits_since":repo.stale_commit_count,
                    "last_indexed_at": repo.last_ingested_at,
                    "last_indexed_sha": repo.last_ingested_sha,
                    "action": "reindex",
                }
            else:
                result[name] = {
                    "status": "ready",
                    "last_indexed_at": repo.last_ingested_at,
                    "last_indexed_sha": repo.last_ingested_sha,
                    "action": "view",
            }
        elif status == RepoStatus.AWAITING_UI:
            result[name] = {"status": "setup_paused", "action": "resume"}
        elif status in (RepoStatus.CLONING, RepoStatus.FILTERING,
                        RepoStatus.MANIFESTING, RepoStatus.SUBMODULES):
            result[name] = {"status": "indexing", "action": "none"}
        elif status == RepoStatus.FAILED:
            result[name] = {"status": "failed", "action": "retry"}
        elif status == RepoStatus.STALE:
            result[name] = {
                "status": "stale",
                "commits_since": repo.stale_commit_count,
                "action": "reindex",
            }
        else:
            result[name] = {"status": "pending", "action": "none"}

    return result