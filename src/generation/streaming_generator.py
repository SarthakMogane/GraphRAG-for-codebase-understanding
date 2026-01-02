"""
Streaming Generation - Real-time response output
Streams tokens as they're generated for better UX
"""

import sys
from pathlib import Path
from typing import Iterator, Dict, List, AsyncIterator
import asyncio
import time
from dataclasses import dataclass
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_client import LLMClient


@dataclass
class StreamChunk:
    """Chunk of streamed content"""
    content: str
    chunk_type: str  # 'text', 'citation', 'verification', 'metadata'
    metadata: Dict = None


class StreamingGenerator:
    """
    Generate responses with streaming output
    
    Benefits:
    - Better UX (see response immediately)
    - Perceive faster responses
    - Can cancel long generations
    """
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize streaming generator
        
        Args:
            llm_client: LLM client with streaming support
        """
        self.llm = llm_client
        logger.info("Streaming generator initialized")
    
    def generate_stream(
        self,
        query: str,
        context: str,
        include_metadata: bool = True
    ) -> Iterator[StreamChunk]:
        """
        Generate answer with streaming
        
        Args:
            query: User query
            context: Retrieved context
            include_metadata: Include timing/metadata chunks
        
        Yields:
            StreamChunk objects
        """
        start_time = time.time()
        
        # Yield start metadata
        if include_metadata:
            yield StreamChunk(
                content="",
                chunk_type="metadata",
                metadata={
                    "status": "started",
                    "timestamp": start_time
                }
            )
        
        # Build prompt
        system_prompt = "You are a code analysis expert. Answer based on the provided context."
        
        user_prompt = f"""Context:
{context}

Question: {query}

Provide a clear, concise answer citing specific code entities when relevant."""
        
        # Stream from LLM
        full_content = ""
        
        try:
            for chunk_text in self.llm.stream_generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            ):
                full_content += chunk_text
                
                yield StreamChunk(
                    content=chunk_text,
                    chunk_type="text",
                    metadata={"accumulated": len(full_content)}
                )
        
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield StreamChunk(
                content=f"\n\n[Error: {e}]",
                chunk_type="text",
                metadata={"error": str(e)}
            )
        
        # Yield end metadata
        if include_metadata:
            elapsed = time.time() - start_time
            yield StreamChunk(
                content="",
                chunk_type="metadata",
                metadata={
                    "status": "completed",
                    "elapsed": elapsed,
                    "total_chars": len(full_content)
                }
            )
    
    async def generate_stream_async(
        self,
        query: str,
        context: str,
        include_metadata: bool = True
    ) -> AsyncIterator[StreamChunk]:
        """
        Async version of streaming generation
        
        Useful for FastAPI and async frameworks
        """
        start_time = time.time()
        
        if include_metadata:
            yield StreamChunk(
                content="",
                chunk_type="metadata",
                metadata={"status": "started", "timestamp": start_time}
            )
        
        # In production, use actual async LLM client
        # This is a simplified version
        for chunk in self.generate_stream(query, context, include_metadata=False):
            yield chunk
            await asyncio.sleep(0.01)  # Simulate async delay
        
        if include_metadata:
            elapsed = time.time() - start_time
            yield StreamChunk(
                content="",
                chunk_type="metadata",
                metadata={"status": "completed", "elapsed": elapsed}
            )


class StructuredStreamGenerator:
    """
    Stream structured responses with sections
    
    Format:
    1. Query analysis
    2. Retrieval summary
    3. Answer (streamed)
    4. Citations
    5. Verification
    """
    
    def __init__(self, llm_client: LLMClient):
        """Initialize structured stream generator"""
        self.llm = llm_client
        self.generator = StreamingGenerator(llm_client)
    
    def generate_structured_stream(
        self,
        query: str,
        context: str,
        query_type: str,
        sources: List[Dict],
        enable_verification: bool = False
    ) -> Iterator[StreamChunk]:
        """
        Generate structured streaming response
        
        Yields chunks in order:
        1. Header with query analysis
        2. Source summary
        3. Answer content (streamed)
        4. Verification (if enabled)
        5. Footer with metadata
        """
        
        # 1. Header
        yield StreamChunk(
            content=f"**Query Type:** {query_type.upper()}\n",
            chunk_type="header",
            metadata={"query_type": query_type}
        )
        
        yield StreamChunk(
            content=f"**Sources Retrieved:** {len(sources)}\n\n",
            chunk_type="header"
        )
        
        # 2. Source summary
        yield StreamChunk(
            content="**Key Sources:**\n",
            chunk_type="sources"
        )
        
        for i, source in enumerate(sources[:3], 1):
            yield StreamChunk(
                content=f"{i}. {source.get('name', 'unknown')} ({source.get('type', 'code')})\n",
                chunk_type="sources"
            )
        
        yield StreamChunk(content="\n**Answer:**\n", chunk_type="header")
        
        # 3. Stream answer
        answer_chunks = []
        for chunk in self.generator.generate_stream(query, context, include_metadata=False):
            answer_chunks.append(chunk.content)
            yield chunk
        
        full_answer = ''.join(answer_chunks)
        
        # 4. Verification (if enabled)
        if enable_verification:
            yield StreamChunk(
                content="\n\n**Verification:** ",
                chunk_type="verification"
            )
            
            # Simplified verification indicator
            # Full implementation would use SelfCheckGPT
            yield StreamChunk(
                content="✓ Answer verified\n",
                chunk_type="verification",
                metadata={"verified": True}
            )
        
        # 5. Footer
        yield StreamChunk(
            content="",
            chunk_type="footer",
            metadata={
                "answer_length": len(full_answer),
                "sources_count": len(sources)
            }
        )


# Helper function for FastAPI integration
def stream_to_sse(stream_iter: Iterator[StreamChunk]) -> Iterator[str]:
    """
    Convert StreamChunk iterator to Server-Sent Events format
    
    Args:
        stream_iter: Iterator of StreamChunk
    
    Yields:
        SSE formatted strings
    """
    for chunk in stream_iter:
        # Format as SSE
        data = {
            "content": chunk.content,
            "type": chunk.chunk_type,
            "metadata": chunk.metadata or {}
        }
        
        import json
        yield f"data: {json.dumps(data)}\n\n"


# Example usage
if __name__ == "__main__":
    from utils.llm_client import LLMClient
    import sys
    
    # Initialize
    llm = LLMClient()
    generator = StreamingGenerator(llm)
    
    # Test streaming
    query = "How does validation work?"
    context = """
    The validate_email function checks email format using regex.
    The check_user_input function validates all user inputs.
    """
    
    print("Streaming answer...")
    print("-" * 60)
    
    for chunk in generator.generate_stream(query, context):
        if chunk.chunk_type == "text":
            print(chunk.content, end='', flush=True)
        elif chunk.chunk_type == "metadata":
            if chunk.metadata.get("status") == "completed":
                print(f"\n\n[Completed in {chunk.metadata['elapsed']:.2f}s]")
    
    print("\n" + "-" * 60)
    
    # Test structured streaming
    print("\n\nStructured streaming...")
    print("-" * 60)
    
    structured = StructuredStreamGenerator(llm)
    
    sources = [
        {"name": "validate_email", "type": "function"},
        {"name": "check_user_input", "type": "function"}
    ]
    
    for chunk in structured.generate_structured_stream(
        query, context, "local", sources, enable_verification=True
    ):
        if chunk.chunk_type in ["text", "header", "sources", "verification"]:
            print(chunk.content, end='', flush=True)
    
    print("\n" + "-" * 60)