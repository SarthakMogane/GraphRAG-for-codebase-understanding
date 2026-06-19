from datetime import datetime
from typing import Any, Optional,List
from pydantic import BaseModel, Field, HttpUrl, field_validator
import re

class SelectionPayload(BaseModel):
    selected_subprojects:   list[str] = []
    selected_submodules:    list[str] = []
    deselected_subprojects: list[str] = []
    deselected_submodules:  list[str] = []
    start_immediately: bool = True

# class IngestRequest(BaseModel):
#     repo_url: HttpUrl = Field(..., description="The GitHub repository URL")
#     branch: str = Field(..., description="Repository branch name")
#     openai_api_key: str = Field(..., min_length=40, description="User's OpenAI Key")
    
# class IndexRequest(BaseModel):
#     repo_name: str
#     openai_key: str
    
# class ChatRequest(BaseModel):
#     task_id: str
#     message: str = Field(..., min_length=1, max_length=1000)

# class QueryRequest(BaseModel):
#     """Query request schema"""
#     query: str = Field(..., description="Natural language query about codebase", min_length=3)
#     top_k: int = Field(10, description="Number of results to retrieve", ge=1, le=50)
#     max_hops: int = Field(2, description="Maximum graph traversal hops", ge=1, le=5)
#     enable_verification: bool = Field(True, description="Enable SelfCheckGPT verification")
#     stream: bool = Field(False, description="Stream response in real-time")



# #new code form here . 
# class SubmitRepoRequest(BaseModel):
#     """POST /api/v1/repos — submit a repo for ingestion."""
#     github_url: str = Field(
#         ...,
#         examples=["https://github.com/owner/repo"],
#         description="Full GitHub repository URL"
#     )
#     force_refresh: bool = Field(
#         default=False,
#         description="Force re-ingestion even if already indexed"
#     )

#     @field_validator("github_url")
#     @classmethod
#     def validate_github_url(cls, v: str) -> str:
#         pattern = r"^https://github\.com/([a-zA-Z0-9._-]+)/([a-zA-Z0-9._-]+?)(?:\.git)?/?$"
#         if not re.match(pattern, v):
#             raise ValueError(
#                 "URL must be a valid GitHub repo URL: https://github.com/owner/repo"
#             )
#         return v.rstrip("/").removesuffix(".git")

#     def parse_owner_repo(self) -> tuple[str, str]:
#         parts = self.github_url.replace("https://github.com/", "").split("/")
#         return parts[0], parts[1]