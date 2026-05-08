"""
app/services/github_service.py
───────────────────────────────
All GitHub API interactions in one place.

Responsibilities:
  - GitHub App authentication (Multi-tenant JWT -> installation token caching)
  - User-to-Server API calls (OAuth token)
  - Repository metadata validation & File tree fetching
  - Webhook signature validation
  - Rate limit awareness with buffer protection
"""

import time
import hmac
import hashlib
import logging
from datetime import datetime
from typing import Optional

import httpx
import jwt                          # PyJWT — for GitHub App JWT generation
from pydantic import BaseModel, Field, ConfigDict
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from src.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Data Classes ──────────────────────────────────────────────────────────────

class RepoMetadata(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")
    owner: str
    name: str
    github_id: int
    default_branch: str
    primary_language: Optional[str]
    size_kb: int = Field(alias="size", ge=0)
    is_fork: bool
    is_private: bool = Field(alias="private")
    visibility:str
    is_archived: bool = Field(alias="archived")
    is_empty: bool
    has_submodules: bool
    uses_git_lfs: bool
    description: Optional[str]
    topics: list[str]


class TreeEntry(BaseModel):
    path: str
    type: str                       # "blob" (file) or "tree" (directory)
    sha: str
    size: Optional[int] = None      # bytes — only present for blobs


class CommitSummary(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")
    sha: str
    message: str
    author: str
    timestamp: str
    files_changed: list[str]


# ── Custom Exceptions ─────────────────────────────────────────────────────────

class RateLimitError(Exception):
    """Raised when GitHub API rate limit buffer is reached."""
    pass

class RepoNotFoundError(Exception):
    """Raised when a repository does not exist or is inaccessible (404)."""
    pass

class RepoAccessError(Exception):
    """Raised when permissions are insufficient (non-rate-limit 403)."""
    pass

# ── Retry Logic ───────────────────────────────────────────────────────────────

def should_retry_httpx_error(exception: BaseException) -> bool:
    # Always retry our custom RateLimitError
    if isinstance(exception, RateLimitError):
        return True
    
    if isinstance(exception, httpx.HTTPStatusError):
        status = exception.response.status_code
        
        # Retry on standard server hiccups and standard rate limits
        if status in (429, 500, 502, 503, 504):
            return True
            
        # Handle the ambiguous 403
        if status == 403:
            resp_text = exception.response.text.lower()
            headers = exception.response.headers
            
            # Check for Primary or Secondary Rate Limits
            is_rate_limit = (
                "rate limit" in resp_text or 
                "secondary rate" in resp_text or
                headers.get("x-ratelimit-remaining") == "0" or
                "retry-after" in headers
            )
            return is_rate_limit
            
        # Do not retry on 401 (token expired), 404 (not found), or 422 (validation)
        return False
        
    # Retry on network timeouts, connection resets, or DNS issues
    return isinstance(exception, httpx.RequestError)


# ── GitHub App Authentication (Multi-Tenant) ──────────────────────────────────

class GitHubAuthManager:
    """
    Manages GitHub App authentication lifecycle for MULTIPLE installations.
    Caches installation access tokens per installation_id.
    """

    def __init__(self, http_client: httpx.AsyncClient):
        self.client = http_client
        # Cache format: { installation_id: (token, expires_at_timestamp) }
        self._installation_tokens: dict[int, tuple[str, float]] = {}

    def _generate_jwt(self) -> str:
        """Generate a short-lived JWT signed with the App's private key."""
        now = int(time.time())
        payload = {
            "iat": now - 60,        # Issued 60s ago (clock skew tolerance)
            "exp": now + 540,       # Expires in 9 minutes
            "iss": settings.GITHUB_APP_ID,
        }
        
        # Ensure newlines in the .env private key are formatted correctly
        private_key = settings.GITHUB_PRIVATE_KEY.replace("\\n", "\n")
        
        return jwt.encode(payload, private_key, algorithm="RS256")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry(should_retry_httpx_error),
    )
    async def get_installation_token(self, installation_id: int) -> str:
        """
        Return a valid installation access token for a SPECIFIC user's installation.
        Refreshes automatically 5 minutes before expiry.
        """
        now = time.time()
        
        # Check if we have a valid cached token for this specific installation
        if installation_id in self._installation_tokens:
            token, expires_at = self._installation_tokens[installation_id]
            if now < expires_at - 300:
                return token

        # If not, authenticate as the App and request a new installation token
        app_jwt = self._generate_jwt()
        url = f"/app/installations/{installation_id}/access_tokens"
        
        resp = await self.client.post(
            url,
            headers={
                "Authorization": f"Bearer {app_jwt}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )
        resp.raise_for_status()
        data = resp.json()

        # Parse expiry ("2024-01-15T10:00:00Z") using modern Python datetime
        expires_at = datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00")).timestamp()
        
        # Cache and return
        self._installation_tokens[installation_id] = (data["token"], expires_at)
        logger.info(f"GitHub installation token refreshed for installation {installation_id}")
        return data["token"]


# ── Main GitHub Service ───────────────────────────────────────────────────────

class GitHubService:
    def __init__(self):
        self._base_url = "https://api.github.com"
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
        timeout = httpx.Timeout(15.0, connect=5.0)
        self.client = httpx.AsyncClient(
            base_url=self._base_url,
            limits=limits,
            timeout=timeout,
        )
        self.auth = GitHubAuthManager(self.client)

    async def close(self):
        await self.client.aclose()

    async def _get_as_app(self, path: str, installation_id: int, params: dict = None) -> dict:
        """Authenticated GET using the App Installation Token (Server-to-Server)."""
        token = await self.auth.get_installation_token(installation_id)
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        return await self._execute_request(path, headers, params)

    async def _get_as_user(self, path: str, user_oauth_token: str ,params: dict = None) -> dict:
        """Authenticated GET using the User's OAuth Token (User-to-Server)."""
        headers = {
            "Authorization": f"Bearer {user_oauth_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        return await self._execute_request(path, headers, params)

    async def _execute_request(self, path: str, headers: dict, params: dict = None) -> dict:
        """Internal method to handle rate limits and request execution."""
        resp = await self.client.get(path, headers=headers, params=params or {})

        remaining = int(resp.headers.get("X-RateLimit-Remaining", 9999))
        if remaining < settings.GITHUB_API_RATE_LIMIT_BUFFER:
            reset_at = int(resp.headers.get("X-RateLimit-Reset", 0))
            wait_seconds = max(0, reset_at - int(time.time()))
            logger.warning(f"GitHub rate limit buffer reached. Reset in {wait_seconds}s. Remaining: {remaining}")
            raise RateLimitError(f"Rate limit buffer reached. {remaining} calls remaining.")

        resp.raise_for_status()
        return resp.json()

    # ── User API Calls (OAuth Token) ──────────────────────────────────────────

# ── User API Calls (OAuth Token) ──────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5), retry=retry(should_retry_httpx_error))
    async def get_installed_repositories(self, user_oauth_token: str, installation_id: int) -> list:
        """
        Fetches ONLY the repositories the user explicitly granted access to 
        during the GitHub App installation.
        """
        data = await self._get_as_user(
            f"user/installations/{installation_id}/repositories", 
            user_oauth_token, 
            params={"per_page": 100} # Grab up to 100 permitted repos at once
        )
        return data.get("repositories", [])


    # ── App API Calls (Installation Token) ────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5), retry=retry(should_retry_httpx_error))
    async def get_repo_branches(self, owner: str, repo: str, installation_id: int) -> list[str]:
        """
        Fetches branches of a repository using the App Installation token.
        Moved to Server-to-Server so background workers can fetch branches 
        without needing an active human session.
        """
        branches = await self._get_as_app(f"repos/{owner}/{repo}/branches", installation_id)
        return [b["name"] for b in branches]

    # ── App API Calls (Installation Token) ────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), retry=retry(should_retry_httpx_error),reraise = True )
    async def get_repo_metadata(self, owner: str, repo: str, installation_id: int) -> RepoMetadata:
        data = await self._get_as_app(f"/repos/{owner}/{repo}", installation_id)
        branch = data.get("default_branch")
        root_files = await self._get_root_file_list(owner, repo, branch, installation_id)
        
        return RepoMetadata(
            owner=data["owner"]["login"],
            name=data["name"],
            github_id=data["id"],
            default_branch=data["default_branch"],
            primary_language=data.get("language"),
            size_kb=data["size"],
            is_fork=data["fork"],
            is_private=data["private"],
            # Add this! Tells you if it is 'public', 'private', or 'internal'
            visibility=data.get("visibility", "private"),
            is_archived=data["archived"],
            is_empty=data["size"] == 0,
            has_submodules=".gitmodules" in root_files,
            uses_git_lfs=".gitattributes" in root_files,
            description=data.get("description"),
            topics=data.get("topics", []),
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5), retry=retry(should_retry_httpx_error))
    async def _get_root_file_list(self, owner: str, repo: str, branch: str, installation_id: int) -> set[str]:
        "return the root file list."
        if not branch:
            return set()
        data = await self._get_as_app(f"/repos/{owner}/{repo}/git/trees/{branch}", installation_id)
        return {entry["path"] for entry in data.get("tree", [])}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5), retry=retry(should_retry_httpx_error))
    async def get_full_tree(self, owner: str, repo: str, sha: str, installation_id: int) -> list[TreeEntry]:
        data = await self._get_as_app(
            f"/repos/{owner}/{repo}/git/trees/{sha}",
            installation_id,
            params={"recursive": "1"},
        )
        if data.get("truncated"):
            logger.warning("Tree response truncated for %s/%s", owner, repo)

        return [
            TreeEntry(
                path=entry["path"],
                type=entry["type"],
                sha=entry["sha"],
                size=entry.get("size"),
            ) for entry in data.get("tree", [])
        ]

    # ── Webhook Payload Validation ────────────────────────────────────────────

    @staticmethod
    def validate_webhook_signature(payload_bytes: bytes, signature_header: str) -> bool:
        """
        Validate the X-Hub-Signature-256 header on incoming webhook payloads.
        Rejects any payload not signed by our webhook secret.
        """
        if not signature_header or not signature_header.startswith("sha256="):
            return False
            
        expected = hmac.new(
            settings.GITHUB_WEBHOOK_SECRET.encode(),
            payload_bytes,
            hashlib.sha256,
        ).hexdigest()
        
        received = signature_header[len("sha256="):]
        return hmac.compare_digest(expected, received)