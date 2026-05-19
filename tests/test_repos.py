import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import HTTPException, Request
from src.db.mock_db import MOCK_DB

# Import your actual code components here. 
# (Assuming they are imported like this from your application structure)
# from your_app.main import list_installed_repositories, MOCK_DB, MockRepository, RepoStatus

# Setup reusable mock fixtures for tests
@pytest.fixture(autouse=True)
def reset_mock_db():
    """Resets the mock database before every single test run."""
    
    MOCK_DB.clear()
    MOCK_DB.update({
        "users": {
            "12345": {"oauth_token": "mock_token_abc"}
        },
        "installations": {
            "12345": 98765
        },
        "repositories": {}
    })
    yield

@pytest.fixture
def mock_request():
    """Creates a mock FastAPI/Starlette request with an active session."""
    request = MagicMock(spec=Request)
    request.session = {"auth_user_id": 12345}
    return request

@pytest.fixture
def mock_github_service(monkeypatch):
    """Mocks the external GitHub service API wrapper."""
    service_mock = AsyncMock()
    # Replace 'your_module.github_service' with the actual path where it's used
    # monkeypatch.setattr("your_module.github_service", service_mock)
    return service_mock


# --- THE TESTS ---

@pytest.mark.asyncio
async def test_list_installed_repositories_success(mock_request, mock_github_service, monkeypatch):
    """Verifies fresh repositories are successfully saved and formatted."""
    # 1. Arrange: Setup the payload exactly as GitHub returns it
    mock_github_service.get_installed_repositories.return_return_value = [
        {
            "name": "myorg/repo1",
            "private": False,
            "html_url": "https://github.com",
            "visibility": "public",
            "default_branch": "main",
            "size": 1200
        }
    ]
    monkeypatch.setattr("src.api.routes.repos.GitHubService", mock_github_service)

    # 2. Act: Call the endpoint function
    from src.api.routes.repos import list_installed_repositories
    from src.db.mock_db import MOCK_DB
    response = await list_installed_repositories(request=mock_request)

    # 3. Assert: Verify endpoint returned the clean formatted payload
    assert len(response) == 1
    assert response[0]["name"] == "myorg/repo1"
    assert response[0]["size"] == 1200

    # 4. Assert: Verify it permanently wrote to the database dictionary
    assert "myorg/repo1" in MOCK_DB["repositories"]
    saved_repo = MOCK_DB["repositories"]["myorg/repo1"]
    assert saved_repo.status is None  # Initial default state
    assert saved_repo.url == "https://github.com"


@pytest.mark.asyncio
async def test_list_installed_repositories_preserves_active_status(mock_request, mock_github_service, monkeypatch):
    """Verifies page reloads DO NOT overwrite ongoing processing statuses."""
    from src.api.routes.repos import list_installed_repositories
    from src.db.mock_db import MOCK_DB, MockRepository
    from src.models.database import RepoStatus
    
    # 1. Arrange: Pre-populate DB with a repository currently in 'CLONING' state
    MOCK_DB["repositories"]["myorg/repo1"] = MockRepository(
        name="myorg/repo1",
        private=False,
        url="https://github.com",
        visibility="public",
        default_branch="main",
        size="1200",
        status=RepoStatus.CLONING # Active task
    )

    mock_github_service.get_installed_repositories.return_value = [
        {
            "name": "myorg/repo1",
            "private": False,
            "html_url": "https://github.com",
            "visibility": "public",
            "default_branch": "main",
            "size": 1200
        }
    ]
    monkeypatch.setattr("src.api.routes.repos.GitHubService", mock_github_service)

    # 2. Act: Trigger the reload request simulation
    await list_installed_repositories(request=mock_request)

    # 3. Assert: Confirm that the active status was NOT wiped back to None
    saved_repo = MOCK_DB["repositories"]["myorg/repo1"]
    assert saved_repo.status == RepoStatus.CLONING


@pytest.mark.asyncio
async def test_list_installed_repositories_unauthenticated(mock_request):
    """Verifies unauthorized requests drop a clean 401 error."""
    from src.api.routes.repos import list_installed_repositories
    mock_request.session = {"auth_user_id": None} # Malicious or timed out session

    with pytest.raises(HTTPException) as exc_info:
        await list_installed_repositories(request=mock_request)
        
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Not authenticated"


@pytest.mark.asyncio
async def test_list_installed_repositories_github_failure(mock_request, mock_github_service, monkeypatch):
    """Verifies production-grade fail-safes catch downstream API crashes with a 502."""
    from src.api.routes.repos import list_installed_repositories
    
    # Simulate a network timeout or raw connection drop from GitHub servers
    mock_github_service.get_installed_repositories.side_effect = Exception("Connection closed by remote host")
    monkeypatch.setattr("src.api.routes.repos.GitHubService", mock_github_service)

    with pytest.raises(HTTPException) as exc_info:
        await list_installed_repositories(request=mock_request)
        
    assert exc_info.value.status_code == 502
    assert "GitHub API service communication failure" in exc_info.value.detail
