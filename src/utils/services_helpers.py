from src.services.github import GitHubService
from fastapi import Request


def get_github_service(request: Request) -> GitHubService:
    """
    Dependency to inject the singleton GitHubService into routes.
    Extracts the active instance from the application state.
    """
    return request.app.state.github_service


def get_sqs_client(request: Request):
    """Dependency provider for the shared async SQS connection pool."""
    return request.app.state.typed.sqs_client