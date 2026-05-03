"""
FastAPI Production API for Hybrid GraphRAG
Provides REST endpoints with streaming, caching, and monitoring
"""

import sys
from pathlib import Path
import time
from typing import Optional, List
import pickle
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
import uvicorn
from loguru import logger
import uuid
import json
import os
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException


from src.api.routes import auth, chat, evaluation , webhooks,repos
from src.utils.llm_client import LangChainClient
from src.indexing.vector_store import VectorStore
from src.indexing.community_detection import CommunityDetector
from src.retrieval.graph_workflow import RAGPipeline
from src.generation.streaming_generator import StreamingGenerator, stream_to_sse

# ============================================
# Pydantic Models (Request/Response schemas)
# ============================================

class QueryRequest(BaseModel):
    """Query request schema"""
    query: str = Field(..., description="Natural language query about codebase", min_length=3)
    top_k: int = Field(10, description="Number of results to retrieve", ge=1, le=50)
    max_hops: int = Field(2, description="Maximum graph traversal hops", ge=1, le=5)
    enable_verification: bool = Field(True, description="Enable SelfCheckGPT verification")
    stream: bool = Field(False, description="Stream response in real-time")


class QueryResponse(BaseModel):
    """Query response schema"""
    query: str
    answer: str
    query_type: str  # 'global' or 'local'
    confidence: float
    sources: List[dict]
    verified: bool
    timing: dict
    request_id: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    components: dict


class MetricsResponse(BaseModel):
    """Metrics response"""
    total_queries: int
    avg_latency: float
    cache_hit_rate: float
    error_rate: float


# ============================================
# Application State
# ============================================

class AppState:
    """Global application state"""
    pipeline: Optional[RAGPipeline] = None
    streaming_generator: Optional[StreamingGenerator] = None
    metrics: dict = {
        "total_queries": 0,
        "total_latency": 0.0,
        "cache_hits": 0,
        "errors": 0
    }


state = AppState()


from dotenv import load_dotenv
import os

# # 1. Force override existing variables
# load_dotenv(override=True) 

# # 2. Add this Debug Block IMMEDIATELY after loading
# # (This acts as a Truth Serum)
# project = os.environ.get("LANGCHAIN_PROJECT")
# key = os.environ.get("LANGCHAIN_API_KEY")
# endpoint = os.environ.get("LANGCHAIN_ENDPOINT")

# print("\n--- DEBUG: LANGSMITH CONFIG ---")
# print(f"Project:  '{project}'")
# print(f"Endpoint: '{endpoint}'")
# print(f"Key Ends: '...{key[-4:] if key else 'NONE'}'")
# print("-------------------------------\n")

# from langchain_google_genai import ChatGoogleGenerativeAI
# key = os.getenv("GOOGLE_API_KEY")   

# llm = ChatGoogleGenerativeAI(api_key=key,model="gemini-1.5-flash") # Use a dummy key if just testing the trace connection
# try:
#     llm.invoke("Hello LangSmith")
#     print("Successfully traced run!")
# except Exception as e:
#     print(f"Error: {e}")

# ============================================
# FastAPI App
# ============================================
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.services.github import GitHubService

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup
    app.state.github_service = GitHubService()
    yield
    # Teardown
    await app.state.github_service.close()

# app = FastAPI(lifespan=lifespan) 



import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from starlette.middleware.sessions import SessionMiddleware # Still needed by Authlib
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from src.core.config import get_settings
from src.api.routes import repos


settings = get_settings()
# Initialize FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production API for intelligent codebase Q&A with GraphRAG",
    lifespan=lifespan
)

# 2. THE SECURITY MIDDLEWARE STACK 
# (Order matters: These wrap the application from the outside in)

# Shield 1: Prevent Host Header Injection Attacks
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "your-production-domain.com"]
)

# Shield 2: Strict Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "https://your-production-domain.com"],
    allow_credentials=True, # MUST be True so the browser sends our httpOnly JWT cookie
    allow_methods=["GET", "POST", "OPTIONS"], # Explicitly deny PUT/DELETE if not used
    allow_headers=["*"],
)

# Shield 3: OAuth Handshake Security (Required by Authlib)
app.add_middleware(
    SessionMiddleware, 
    secret_key=settings.SESSION_SECRET_KEY,
    max_age=300 # Drops the session cookie after 5 minutes to minimize attack surface
)

# Shield 4: The Custom Auditor (Performance & Logging)
@app.middleware("http")
async def add_process_time_header_and_log(request: Request, call_next):
    start_time = time.time()
    
    # Let the request pass through to the routers
    response = await call_next(request)
    
    # Calculate execution time
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    
    # Log the audit trail safely (Avoid logging raw query params if they contain sensitive data)
    logger.info(
        f"Client: {request.client.host} | "
        f"{request.method} {request.url.path} | "
        f"Status: {response.status_code} | "
        f"Latency: {process_time:.3f}s"
    )
    
    return response
# Inside your main.py



# Add this near your other app.include_router calls

# Include all modular routers
app.include_router(auth.router)
app.include_router(repos.router)
app.include_router(chat.router)
app.include_router(evaluation.router)
app.include_router(webhooks.router)

@app.get("/", response_class=FileResponse, tags=["UI"])
async def serve_frontend():
    """Serves the static HTML SPA."""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    html_file_path = os.path.join(BASE_DIR, "index.html")

    if not os.path.exists(html_file_path):
        return HTMLResponse(content="<h1>Error: index.html not found</h1>", status_code=404)
    
    return FileResponse(html_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)




@app.get("/logout")
async def logout(request: Request):
    """Clears the session"""
    request.session.clear()
    return RedirectResponse(url='/')

# ============================================
# Startup/Shutdown Events
# ============================================

# @app.on_event("startup")
# async def startup_event():
#     """Initialize components on startup"""
#     logger.info("Starting Hybrid GraphRAG API...")
    
#     try:
#         # Load graph
#         logger.info("Loading graph...")
#         with open('data/graphs/code_graph_enriched.pkl', 'rb') as f:
#             graph = pickle.load(f)
        
#         # Load vector store
#         logger.info("Loading vector store...")
#         vector_store = VectorStore.load('data/graphs/vector_store')
        
#         # Load communities
#         logger.info("Loading communities...")
#         communities = CommunityDetector.load('data/graphs/communities.json')
        
#         # Initialize LLM
#         logger.info("Initializing LLM...")
#         llm = LangChainClient()
        
#         # Create pipeline
#         logger.info("Creating RAG pipeline...")
#         state.pipeline = RAGPipeline(
#             graph=graph,
#             vector_store=vector_store,
#             community_detector=communities,
#             llm_client=llm,
#             enable_verification=True
#         )
        
#         # Create streaming generator
#         state.streaming_generator = StreamingGenerator(llm)
        
#         logger.success("✓ API started successfully")
        
#     except Exception as e:
#         logger.error(f"Startup failed: {e}")
#         raise


# @app.on_event("shutdown")
# async def shutdown_event():
#     """Cleanup on shutdown"""
#     logger.info("Shutting down Hybrid GraphRAG API...")
#     # Add cleanup logic here (close DB connections, etc.)


# # ============================================
# # API Endpoints
# # ============================================

# @app.get("/", response_model=dict)
# async def root():
#     """Root endpoint"""
#     return {
#         "message": "Hybrid GraphRAG API",
#         "version": "1.0.0",
#         "docs": "/docs",
#         "health": "/health"
#     }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    # 1. Safely check LLM status
    llm_status = "unknown"
    if state.pipeline and state.pipeline.llm:
        # Convert the object to a simple string description
        llm_status = f"active ({state.pipeline.llm.model_name})"
    elif state.pipeline:
        llm_status = "missing"

    return HealthResponse(
        status="healthy" if state.pipeline else "initializing",
        version="1.0.0",
        components={
            "pipeline": state.pipeline is not None,
            "streaming": state.streaming_generator is not None,
            "llm": llm_status  # <--- FIX: Passing a string
        }
    )

# @app.post("/query", response_model=QueryResponse)
# async def query(request: QueryRequest, background_tasks: BackgroundTasks):
#     """
#     Query the codebase
    
#     **Example:**
#     ```json
#     {
#       "query": "How does authentication work?",
#       "top_k": 10,
#       "enable_verification": true
#     }
#     ```
#     """
#     if not state.pipeline:
#         raise HTTPException(status_code=503, detail="Service not ready")
    
#     start_time = time.time()
#     request_id = f"req_{int(time.time() * 1000)}"
    
#     try:
#         # Execute query
#         result = state.pipeline.query(
#             question=request.query,
#             top_k=request.top_k,
#             max_hops=request.max_hops,
#             verbose=False
#         )
        
#         elapsed = time.time() - start_time
        
#         # Update metrics (background task)
#         background_tasks.add_task(update_metrics, elapsed, False)
        
#         return QueryResponse(
#             query=request.query,
#             answer=result['answer'],
#             query_type=result['query_type'],
#             confidence=result['confidence'],
#             sources=result['sources'],
#             verified=result['verified'],
#             timing={"total": elapsed, **result['timing']},
#             request_id=request_id
#         )
    
#     except Exception as e:
#         logger.error(f"Query failed: {e}")
#         background_tasks.add_task(update_metrics, 0, True)
#         raise HTTPException(status_code=500, detail=str(e))


# # main.py

# import json
# import uuid
# from fastapi import HTTPException
# from sse_starlette.sse import EventSourceResponse

# # ... existing imports ...

# @app.post("/query/stream")
# async def query_stream(request: QueryRequest):
#     """
#     Agentic Streaming: Shows status updates, sources, tokens, and verification
#     """
#     if not state.pipeline:
#         raise HTTPException(status_code=503, detail="Service not initialized")

#     # 1. Generate or retrieve thread_id for memory
#     # In production, you might get this from headers
#     thread_id = str(uuid.uuid4())

#     async def event_generator():
#         start_time = time.time()
#         is_error = False
        
#         # 2. Setup the Config for Checkpointing (Memory)
#         config = {
#             "configurable": {
#                 "thread_id": thread_id
#             }
#         }

#         # 3. Prepare Inputs
#         initial_state = {
#             "query": request.query,
#             "top_k": request.top_k,
#             "max_hops": request.max_hops,
#             "verification_enabled": request.enable_verification
#         }

#         try:
#             # 4. Stream Events from LangGraph
#             # We use version="v1" as it often handles custom node streaming smoother
#             async for event in state.pipeline.app.astream_events(
#                 initial_state, 
#                 config=config,
#                 version="v1",
#                 stream_mode = "messages"
#             ):
#                 kind = event["event"]
                
#                 # --- A. STATUS UPDATES (UI Pills) ---
#                 if kind == "on_chain_start":
#                     node_name = event["name"]
                    
#                     if node_name == "classify_query":
#                         yield format_sse("status", "🤔 Classifying query...")
#                     elif node_name == "retrieve_local":
#                         yield format_sse("status", "🔍 Searching codebase...")
#                     elif node_name == "generate_answer":
#                         yield format_sse("status", "💡 Generating solution...")
#                     # FIX: Only show status if verification is ENABLED
#                     elif node_name == "verify_answer" and request.enable_verification:
#                         yield format_sse("status", "🛡️ Verifying facts...")

#                 # --- B. THE TOKEN STREAM (Typewriter Effect) ---
#                 elif kind == "on_chat_model_stream":
#                     # We filter by the TAG we added in nodes.py
#                     tags = event.get("tags", [])
                    
#                     if "GENERATE_ANSWER_NODE" in tags:
#                         chunk = event["data"]["chunk"]
#                         # Handle both LangChain object chunks and string chunks
#                         content = chunk.content if hasattr(chunk, "content") else str(chunk)
                        
#                         if content:
#                             yield format_sse("token", content)

#                 # --- C. SOURCES (When Retrieval Finishes) ---
#                 elif kind == "on_chain_end":
#                     # [NEW] When generation finishes, send the CITED text
#                 # This replaces the raw stream with the version containing [Node] links
#                     if node_name == "generate_answer":
#                         output = event["data"].get("output")
#                         if output and "answer" in output:
#                             yield format_sse("answer_update", output["answer"])

#                     elif event["name"] == "format_context":
#                         outputs = event["data"].get("output")
#                         if outputs and "context" in outputs:
#                             # Send simplified source list and metadata.
#                             sources = [
#                                 {
#                                     "name": item["name"], 
#                                     "type": item["node_type"],
#                                     "score": item.get("score", 0)
#                                 } 
#                                 for item in outputs["context"][:5]
#                             ]
#                             yield format_sse("sources", json.dumps(sources))

#                     # --- D. VERIFICATION RESULTS ---
#                     elif event["name"] == "verify_answer":
#                         outputs = event["data"].get("output")
#                         if outputs:
#                             # 1. Send the Verification Data (Score, Flags)
#                             result = {
#                                 "verified": outputs.get("verified"),
#                                 "score": 1.0 - outputs.get("hallucination_rate", 0),
#                                 "flags": outputs.get("hallucination_flags", [])
#                             }
#                             yield format_sse("verification", json.dumps(result))
                            
#                             # 2. CRITICAL FIX: Send the POLISHED text (with citations/badges)
#                             # This replaces the raw text on the client side
#                             if "answer" in outputs:
#                                 yield format_sse("answer_update", outputs["answer"])
                                
#         except Exception as e:
#             logger.error(f"Stream Error: {e}")
#             yield format_sse("error", str(e))
#             is_error = True
        
#         finally:
#             # Metrics
#             elapsed = time.time() - start_time
#             state.metrics['total_queries'] += 1
#             state.metrics['total_latency'] += elapsed
#             if is_error:
#                 state.metrics['errors'] += 1
                
#             # === CRITICAL FIX: [DONE] is now OUTSIDE the loop ===
#             yield format_sse("done", "[DONE]")

#     return EventSourceResponse(event_generator())


# # Helper function (place this at the bottom of main.py or top level)
# def format_sse(event_type: str, data: str) -> dict:
#     """Helper to format SSE messages consistently"""
#     return {
#         "event": event_type,
#         "data": data
#     }




# # @app.post("/query/stream")
# # async def query_stream(request: QueryRequest):
# #     """
# #     Stream query response in real-time
    
# #     Returns Server-Sent Events (SSE) stream
# #     """
# #     if not state.pipeline or not state.streaming_generator:
# #         raise HTTPException(status_code=503, detail="Service not ready")
    
# #     async def event_generator():
# #         """Generate SSE events"""
# #         try:
# #             # This is simplified - full implementation would integrate with pipeline
# #             context = "Sample context"  # Get from pipeline
            
# #             stream = state.streaming_generator.generate_stream(
# #                 query=request.query,
# #                 context=context,
# #                 include_metadata=True
# #             )
            
# #             for sse_data in stream_to_sse(stream):
# #                 yield sse_data
        
# #         except Exception as e:
# #             logger.error(f"Streaming failed: {e}")
# #             yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
    
# #     return EventSourceResponse(event_generator())


# @app.get("/metrics", response_model=MetricsResponse)
# async def get_metrics():
#     """Get API metrics"""
#     total = state.metrics['total_queries']

#     safe_total = max(total, 1)
    
#     return MetricsResponse(
#         total_queries=total,
#         avg_latency=state.metrics['total_latency'] / safe_total,
#         cache_hit_rate=state.metrics['cache_hits'] / safe_total,
#         error_rate=state.metrics['errors'] / safe_total
#     )


# @app.get("/communities")
# async def list_communities():
#     """List detected code communities"""
#     if not state.pipeline:
#         raise HTTPException(status_code=503, detail="Service not ready")
    
#     communities = state.pipeline.communities.communities
    
#     return {
#         "total": len(communities),
#         "communities": [
#             {
#                 "id": comm.id,
#                 "level": comm.level,
#                 "size": len(comm.nodes),
#                 "summary": comm.summary,
#                 "key_entities": comm.key_entities[:5]
#             }
#             for comm in list(communities.values())[:20]  # First 20
#         ]
#     }


# @app.post("/index/repository")
# async def index_repository(repo_path: str, background_tasks: BackgroundTasks):
#     """
#     Index a new repository (background task)
    
#     **Note:** This is a long-running operation
#     """
#     # In production, this would:
#     # 1. Validate repo path
#     # 2. Run indexing pipeline in background
#     # 3. Return job ID for status tracking
#     # 4. Update state when complete
    
#     return {
#         "message": "Indexing started",
#         "repo_path": repo_path,
#         "status": "queued"
#     }


# # ============================================
# # Helper Functions
# # ============================================

# def update_metrics(latency: float, is_error: bool):
#     """Update metrics (called as background task)"""
#     state.metrics['total_queries'] += 1
#     state.metrics['total_latency'] += latency
#     if is_error:
#         state.metrics['errors'] += 1


# # ============================================
# # Main Entry Point
# # ============================================

# if __name__ == "__main__":
#     uvicorn.run(
#         "main:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True,  # Development only
#         log_level="info"
#     )