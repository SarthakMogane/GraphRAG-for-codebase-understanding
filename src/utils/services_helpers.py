from src.services.github import GitHubService
from src.services.scout.deep_scout import RepoScoutResult as ScoutResult
from fastapi import Request , HTTPException
import dataclasses
from uuid import UUID


async def get_current_account_id(request: Request) -> UUID:
    """
    Extracts the tenant identity parameter directly from the active HTTP session.
    Keeps the database framework decoupled from high-level auth utilities.
    """
    account_id_str = request.session.get("account_id")
    if not account_id_str:
        raise HTTPException(
            status_code=401, 
            detail="Access Denied: Missing active authentication session context."
        )
    try:
        return UUID(account_id_str)
    except ValueError:
        raise HTTPException(
            status_code=400, 
            detail="Malformed Session Token: Multi-tenancy key type verification failure."
        )
    
def get_github_service(request: Request) -> GitHubService:
    """
    Dependency to inject the singleton GitHubService into routes.
    Extracts the active instance from the application state.
    """
    return request.app.state.github_service


def get_sqs_client(request: Request):
    """Dependency provider for the shared async SQS connection pool."""
    return request.app.state.typed.sqs_client


def _serialize_scout(result: ScoutResult) -> dict:
    """Convert RepoScoutResult dataclass tree to a JSON-safe dict."""
    def _convert(obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, list):
            return [_convert(i) for i in obj]
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if hasattr(obj, "value"):   # Enum → string
            return obj.value
        return obj
    return _convert(result)