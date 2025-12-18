#!/usr/bin/env python3
"""
Modern GraphRAG Application using ADK Runner
"""
import asyncio
from loguru import logger
import sys
import time
import threading
from pathlib import Path

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from workflows.indexing_workflow import indexing_workflow
from src.workflows.agent import query_workflow
from config.settings import settings

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level=settings.LOG_LEVEL
)
logger.add(
    "logs/graphrag.log",
    rotation="100 MB",
    retention="7 days"
)

class GraphRAGApp:
    """Main application using ADK Runner and Sessions"""
    
    def __init__(self):
        # Create session service
        self.session_service = InMemorySessionService()
        
        # Create runner for indexing
        self.indexing_runner = Runner(
            agent=indexing_workflow,
            app_name=settings.APP_NAME,
            session_service=self.session_service
        )
        
        # Create runner for querying
        self.query_runner = Runner(
            agent=query_workflow,
            app_name=settings.APP_NAME,
            session_service=self.session_service
        )
        
        logger.info("GraphRAG Application initialized")
    
    async def create_session(self, user_id: str = "default_user"):
        """Create a new session for a user"""
        import uuid
        session_id = str(uuid.uuid4())
        
        session = await self.session_service.create_session(
            app_name=settings.APP_NAME,
            user_id=user_id,
            session_id=session_id
        )
        
        return user_id, session_id
    
    async def index_repository(
        self,
        repo_url: str,
        branch: str = "main",
        user_id: str = "default_user"
    ):
        """
        Index a GitHub repository
        
        Args:
            repo_url: GitHub repository URL
            branch: Branch to index
            user_id: User identifier
        """
        logger.info(f"Starting indexing: {repo_url}")
        
        # Create session
        user_id, session_id = await self.create_session(user_id)
        
        # Create message content
        message_text = f"Index this repository: {repo_url} (branch: {branch})"
        content = types.Content(
            role='user',
            parts=[types.Part(text=message_text)]
        )
        
        # Run indexing workflow
        logger.info("Executing indexing workflow...")
        events = self.indexing_runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=content
        )
        
        # Process events - collect all, display only final
        result = None
        last_event = None
        for event in events:
            if hasattr(event, 'agent_name') and event.agent_name:
                logger.info(f"Agent: {event.agent_name}")
            
            if hasattr(event, 'content') and event.content and event.content.parts:
                last_event = event

        if last_event:
            result = last_event.content.parts[0].text
            logger.success("Workflow complete!")
            print("\n" + "="*80)
            print("RESULT:")
            print("="*80)
            print(result)
            print("="*80 + "\n")

        return result
    
    async def query(
        self,
        query: str,
        user_id: str = "default_user"
    ):
        """
        Query the codebase
        
        Args:
            query: User question
            user_id: User identifier
        """
        logger.info(f"Query: {query}")
        
        # Create session
        user_id, session_id = await self.create_session(user_id)
        
        # Create message content
        content = types.Content(
            role='user',
            parts=[types.Part(text=query)]
        )
        
        # Run query workflow
        logger.info("Executing query workflow...")
        events = self.query_runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=content
        )
        
        # Process events - collect all, display only final
        result = None
        last_event = None
        for event in events:
            if hasattr(event, 'agent_name') and event.agent_name:
                logger.info(f"Agent: {event.agent_name}")
            
            if hasattr(event, 'content') and event.content and event.content.parts:
                last_event = event

        if last_event:
            result = last_event.content.parts[0].text
            logger.success("Workflow complete!")
            print("\n" + "=" * 80)
            print("RESULT:")
            print("=" * 80)
            print(result)
            print("=" * 80 + "\n")

        return result


async def main():
    """Main demo function"""
    
    app = GraphRAGApp()
    
    print("\n" + "🚀 "*20)
    print("    GRAPHRAG - Modern ADK Implementation")
    print("🚀 "*20 + "\n")
    
    try:
        # Demo 1: Index a repository
        print("📦 STEP 1: Indexing Repository")
        print("-" * 80)
        
        repo_url = "https://github.com/SarthakMogane/SMS-Spam-VotingClassifier-"
        
        await app.index_repository(
            repo_url=repo_url,
            branch="main"
        )
        
        # Demo 2: Query the codebase
        print("\n💬 STEP 2: Querying Codebase")
        print("-" * 80)
        
        queries = [
            "What functions are in this repository?",
            "Show me the main code files and what they do",
        ]
        
        for query in queries:
            await app.query(query)
            await asyncio.sleep(2)  # Rate limiting
        
        print("\n✨ Demo Complete!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # Wait for background ADK Runner threads to finish
        active_threads = [t for t in threading.enumerate() if t.name.startswith('Thread')]
        for thread in active_threads:
            if thread.is_alive():
                thread.join(timeout=5)