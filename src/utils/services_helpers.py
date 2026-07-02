from src.services.github import GitHubService
from src.services.scout.deep_scout import RepoScoutResult as ScoutResult
from fastapi import Request , HTTPException , Depends
from pydantic import BaseModel
import dataclasses
from uuid import UUID

class AuthSession(BaseModel):
    user_id: UUID
    account_id: UUID

def is_valid_uuid(val: str | None) -> bool:
    """Helper to ensure we only pass strictly formatted UUIDs to Postgres."""
    if not val:
        return False
    try:
        UUID(str(val))
        return True
    except ValueError:
        return False
    
# 2. The Core Dependency (Does the heavy lifting and validation)
def get_auth_session(request: Request) -> AuthSession:
    user_id_raw = request.session.get("user_id")
    account_id_raw = request.session.get("account_id")
    
    if not is_valid_uuid(user_id_raw) or not is_valid_uuid(account_id_raw):
        request.session.clear()
        raise HTTPException(status_code=401, detail="Not authenticated")
        
    return AuthSession(
            user_id=UUID(user_id_raw),
            account_id=UUID(account_id_raw)
        )
    

def get_current_account_id(session: AuthSession = Depends(get_auth_session)) -> UUID:
    """
    Endpoints in database.py, repos.py, etc., can keep calling Depends(get_current_account_id).
    FastAPI will automatically run get_auth_session first, then just return the account_id!
    """
    return session.account_id

def get_current_user_id(session: AuthSession = Depends(get_auth_session)) -> UUID:
    """
    Just in case you ever have an endpoint that ONLY needs the user_id.
    """
    return session.user_id
    
def get_github_service(request: Request) -> GitHubService:
    """
    Dependency to inject the singleton GitHubService into routes.
    Extracts the active instance from the application state.
    """
    return request.app.state.github_service


async def get_sqs_client(request: Request):
    """Dependency provider for the shared async SQS connection pool."""
    session = request.app.state.aws_session
    config = request.app.state.boto_config
    
    # This safely draws from a managed pool capped at 50 connections max
    async with session.client("sqs", config=config) as client:
        yield client


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