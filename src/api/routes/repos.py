from fastapi import APIRouter, HTTPException, Request , Depends , Body
from fastapi.responses import RedirectResponse
from typing import List
from src.services.github import GitHubService
from src.database.mock_db import MOCK_DB , MockRepository
from src.models.database import RepoStatus
from src.schemas.responses import IndexResponse
from src.crud.repos_ops import _get_owned_repo ,apply_pipeline_result_to_db
from uuid import UUID
from src.core.database import get_transaction
from src.services.pre_clone.pipeline import PreClonePipeline
from src.services.pre_clone.types import ValidationVerdict,RoutingDecision
from src.utils.services_helpers import get_github_service
from src.services.scout.check_structural_changes import _handle_refresh
from src.utils.services_helpers import get_github_service , get_current_account_id
from src.core.database import get_transaction ,get_rls_tx_conn ,get_authed_read_db_dep
from src.core.config import get_settings
from src.core.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

# Notice: Because prefix="/api", your routes automatically become /api/repos
router = APIRouter(prefix="/api", tags=["Repositories"])

@router.get("/repos")
async def get_user_repos(
    account_id :UUID = Depends(get_current_account_id),
    conn = Depends(get_authed_read_db_dep)
):
    """
    Purely observes the database state. Does NOT contact GitHub.
    """
    
    async with get_authed_read_db_dep(account_id=account_id) as conn:
        repos = await conn.fetch(
            "SELECT * FROM repos WHERE account_id = $1 ORDER BY updated_at DESC", 
            account_id
        )
        return repos

# # Note: Changed from {full_name} to {owner}/{repo} so FastAPI parses it automatically for us!
# @router.get("/repos/{owner}/{repo}/branches", response_model=List[str])
# async def list_branches(owner: str,
#                         repo: str, 
#                         request: Request,
#                         github_service: GitHubService = Depends(get_github_service)):
#     """
#     Fetches the branches for a specific repository. 
#     Uses the App Installation Token (Server-to-Server) so it can run in the background.
#     """
#     # 1. Authenticate via explicit cookie (No more JWT Depends!)
#     github_id = request.session.get("auth_user_id")
#     if not github_id or github_id not in MOCK_DB["users"]:
#         raise HTTPException(status_code=401, detail="Not authenticated")

#     # 2. Get the Installation ID 
#     installation_id = MOCK_DB.get("installations", {}).get(github_id)
#     if not installation_id:
#         raise HTTPException(status_code=403, detail="GitHub App not installed")

#     try:
#         # 3. Call the APP-level service (Server-to-Server flow!)
#         # We don't pass the user token anymore, just the installation ID
#         branches = await github_service.get_repo_branches(
#             owner=owner, 
#             repo=repo, 
#             installation_id=installation_id
#         )
#         return branches
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to fetch branches: {str(e)}")


# @router.post("/repos/status")
# async def get_repos_status(
#     repo_names: list[str],   # ["myorg/repo1", "myorg/repo2"]
#     # db: AsyncSession = Depends(get_db),
# ):
#     # result = await db.execute(
#     #     select(Repository).where(
#     #         Repository.full_name.in_(repo_names)
#     #     )
#     # )
#     # repos = {r.full_name: r for r in result.scalars().all()}
#     mock_repo_tables = MOCK_DB["repositories"]
#     print("debug : repo inside DB : \n",mock_repo_tables)
#     print("debug : repo list gotten by UI \n ",repo_names)
#     result = {}
#     for name in repo_names:
#         if name not in mock_repo_tables:
#             result[name] = {"status": "never_indexed", "action": "index"}
#             continue
        
#         repo = mock_repo_tables[name]
#         status = repo.status
#         print(status)
#         if status is None:
#             result[name] = {"status": "never_indexed", "action": "index"}

#         elif status == RepoStatus.READY:
#             # Check webhook-flagged staleness (DB only, no API call)
#             if repo.is_stale:  # <--- Checked directly from your new DB column!
#                 result[name] = {
#                     "status": "stale",
#                     "commits_since":repo.stale_commit_count,
#                     "last_indexed_at": repo.last_ingested_at,
#                     "last_indexed_sha": repo.last_ingested_sha,
#                     "action": "reindex",
#                 }
#             else:
#                 result[name] = {
#                     "status": "ready",
#                     "last_indexed_at": repo.last_ingested_at,
#                     "last_indexed_sha": repo.last_ingested_sha,
#                     "action": "view",
#             }
#         elif status == RepoStatus.AWAITING_UI:
#             result[name] = {"status": "setup_paused", "action": "resume"}
#         elif status in (RepoStatus.CLONING, RepoStatus.FILTERING,
#                         RepoStatus.MANIFESTING, RepoStatus.SUBMODULES):
#             result[name] = {"status": "indexing", "action": "none"}
#         elif status == RepoStatus.FAILED:
#             result[name] = {"status": "failed", "action": "retry"}
#         elif status == RepoStatus.STALE:
#             result[name] = {
#                 "status": "stale",
#                 "commits_since": repo.stale_commit_count,
#                 "action": "reindex",
#             }
#         else:
#             result[name] = {"status": "pending", "action": "none"}

#     return result


@router.post("repos/{repo_id}/index",response_model=IndexResponse)
async def index_repo(
    repo_id : int,
    request:Request,
    db = Depends(get_rls_tx_conn()),
    _gh = Depends(get_github_service())
    ):
    """
    Entry point for all indexing actions from the dashboard.
 
    Called when user clicks:
      "Index"     → status was not_indexed
      "Re-index"  → status was stale
      "Retry"     → status was failed
      "Resume"    → status was awaiting_ui (treated as new_ingestion)
 
    Returns next action for the frontend to navigate to.
    """
    user_id    = request.session.get("user_id")
    account_id = request.session.get("account_id")
    
    if not user_id or not account_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        account_uuid = UUID(account_id)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid session")

    #IDOR prevention
    repo = await _get_owned_repo(repo_id, account_uuid, db) 

     # ── Reject if already actively processing ──────────────────────────────
    # Prevents duplicate concurrent ingestion jobs for the same repo.
    # AWAITING_UI is allowed through — user may want to re-submit selections.
    BLOCKING_STATUSES = {
        RepoStatus.SCOUTING.value,
        RepoStatus.CLONING.value,
        RepoStatus.FILTERING.value,
        RepoStatus.SUBMODULES.value,
        RepoStatus.MANIFESTING.value,
    }
    current_status = repo["index_status"]
    if current_status in BLOCKING_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=(
                f"This repository is currently {current_status}. "
                f"Wait for it to finish or check job status."
            ),
        )
    
    installation_id = repo["github_install_id"]
    if not installation_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "No GitHub App installation found for this repository. "
                "Please reinstall the app and grant access to this repo."
            ),
        )
    
    # Pipeline validates: URL, rate limit, DB cache, GitHub API exists/private,
    # not archived, not empty, fork check. Returns ValidationResult.
    pipeline = PreClonePipeline(
        github_service=_gh,
        conn=db,
        installation_id=installation_id,
        force_refresh=False,
    )
    github_url = f"https://github.com/{repo.github_owner}/{repo.github_repo}"
    result = await pipeline.run(github_url)

    # ── 2. Persist the Result to PostgreSQL ─────────────────────────────
    # This single helper automatically translates the verdict/routing into the 
    # correct string ('inaccessible', 'ready', 'pending') and merges the GitHub metadata.
    await apply_pipeline_result_to_db(conn=db, repo_id=repo_id, result=result)

    # ── 3. Handle Rejections (Early Exit) ───────────────────────────────
    if result.verdict != ValidationVerdict.APPROVED:
        status_code = _verdict_to_http_status(result.verdict)
        raise HTTPException(
            status_code=status_code,
            detail=result.message or f"Pipeline rejected: {result.verdict.value}",
        )

    # ── 4. Handle Approvals (Routing logic) ─────────────────────────────
    if result.routing == RoutingDecision.SERVE_CACHE:
        return IndexResponse(
            repo_id=repo_id,
            next="none",
            message=f"{repo['full_name']} is up to date. Wiki is ready.",
            wiki_url=f"/wiki/{repo_id}",
        )

    if result.routing == RoutingDecision.NEW_INGESTION:
        return IndexResponse(
            repo_id=repo_id,
            next="scout",
            message=f"{repo['full_name']} approved. Starting structure scan.",
        )

    if result.routing == RoutingDecision.REFRESH:
        return await _handle_refresh(
            repo=repo,                         # Pass the asyncpg record dict
            repo_id=repo_id,
            result=result,
            installation_id=installation_id,
            conn=db,                         
        )

    # ── 5. Fallback ─────────────────────────────────────────────────────
    logger.error("Unhandled routing decision: %s", result.routing)
    raise HTTPException(status_code=500, detail="Unexpected routing state")


# HTTP status mapping for pipeline verdicts
# ─────────────────────────────────────────────────────────────────────────────
 
def _verdict_to_http_status(verdict: ValidationVerdict) -> int:
    return {
        ValidationVerdict.URL_PARSE_ERROR:      400,
        ValidationVerdict.RATE_LIMIT_BLOCKED:   429,
        ValidationVerdict.REPO_NOT_FOUND:       404,
        ValidationVerdict.REPO_PRIVATE:         403,
        ValidationVerdict.REPO_ARCHIVED:        422,
        ValidationVerdict.REPO_DISABLED:        422,
        ValidationVerdict.REPO_EMPTY:           422,
        ValidationVerdict.REPO_FORK_REDIRECTED: 422,
        ValidationVerdict.ALREADY_PROCESSING:   409,
    }.get(verdict, 400)

