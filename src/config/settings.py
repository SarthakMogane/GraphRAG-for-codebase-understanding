"""
Modern configuration using Pydantic Settings
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Configuration
    GOOGLE_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.0-flash-exp"
    GEMINI_EMBEDDING_MODEL: str = "models/text-embedding-004"
    
    # Neo4j Configuration
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "graphrag2025"
    
    # Application Configuration
    APP_NAME: str = "agents"
    LOG_LEVEL: str = "INFO"
    
    # Graph Settings
    MAX_TRAVERSAL_DEPTH: int = 3
    VECTOR_SEARCH_TOP_K: int = 10
    GRAPH_EXPANSION_LIMIT: int = 20
    
    # Embedding Settings
    EMBEDDING_DIMENSION: int = 768
    EMBEDDING_BATCH_SIZE: int = 50
    
    # Repository Settings
    REPO_CACHE_DIR: Path = Path("./data/repositories")
    SUPPORTED_LANGUAGES: List[str] = ["python", "javascript", "typescript"]
    
    # Model Configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.REPO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        Path("./data/evaluation").mkdir(parents=True, exist_ok=True)
        Path("./logs").mkdir(parents=True, exist_ok=True)



# Global settings instance
settings = Settings()