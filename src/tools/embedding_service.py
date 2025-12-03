"""
Embedding generation using Gemini (ADK tools)
All functions return JSON strings for ADK compatibility
"""
import json
from typing import List, Dict
import google.generativeai as genai
from loguru import logger

from src.config.settings import settings

# Configure Gemini
genai.configure(api_key=settings.GOOGLE_API_KEY)

def embed_text(text: str, task_type: str = "retrieval_document") -> str:
    """
    Generate embedding for a single text.
    
    Args:
        text: Input text to embed
        task_type: Embedding task type (retrieval_document, retrieval_query, etc.)
        
    Returns:
        JSON string with embedding vector
    """
    try:
        result = genai.embed_content(
            model=settings.GEMINI_EMBEDDING_MODEL,
            content=text,
            task_type=task_type
        )
        
        return json.dumps({
            "embedding": result['embedding'],
            "dimension": len(result['embedding'])
        })
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return json.dumps({"error": str(e)})

def embed_code_function(
    function_name: str,
    code: str,
    docstring: str = "",
    parameters: List[str] = None
) -> str:
    """
    Generate specialized embedding for a code function.
    
    Args:
        function_name: Name of the function
        code: Function source code
        docstring: Function documentation
        parameters: Function parameters
        
    Returns:
        JSON string with embedding vector
    """
    # Construct rich representation
    parts = []
    
    if docstring:
        parts.append(f"Description: {docstring}")
    
    parts.append(f"Function: {function_name}")
    
    if parameters:
        params_str = ", ".join(parameters)
        parts.append(f"Parameters: {params_str}")
    
    # Include first few lines of code
    code_lines = code.split('\n')[:5]
    code_snippet = '\n'.join(code_lines)
    parts.append(f"Code:\n{code_snippet}")
    
    combined_text = '\n\n'.join(parts)
    
    return embed_text(combined_text, task_type="retrieval_document")

def embed_query(query: str) -> str:
    """
    Generate embedding for a search query.
    
    Args:
        query: Search query text
        
    Returns:
        JSON string with embedding vector
    """
    return embed_text(query, task_type="retrieval_query")

def embed_batch(texts: List[str], task_type: str = "retrieval_document") -> str:
    """
    Generate embeddings for multiple texts in batch.
    
    Args:
        texts: List of texts to embed
        task_type: Embedding task type
        
    Returns:
        JSON string with list of embeddings
    """
    try:
        embeddings = []
        
        # Process in batches for rate limiting
        batch_size = settings.EMBEDDING_BATCH_SIZE
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            result = genai.embed_content(
                model=settings.GEMINI_EMBEDDING_MODEL,
                content=batch,
                task_type=task_type
            )
            
            # Handle both single and batch responses
            if isinstance(result['embedding'][0], list):
                embeddings.extend(result['embedding'])
            else:
                embeddings.append(result['embedding'])
        
        return json.dumps({
            "embeddings": embeddings,
            "count": len(embeddings),
            "dimension": len(embeddings[0]) if embeddings else 0
        })
        
    except Exception as e:
        logger.error(f"Batch embedding failed: {e}")
        return json.dumps({"error": str(e)})

def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> str:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        JSON string with similarity score
    """
    import numpy as np
    
    try:
        a = np.array(embedding1)
        b = np.array(embedding2)
        
        similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        return json.dumps({
            "similarity": float(similarity),
            "distance": float(1 - similarity)
        })
        
    except Exception as e:
        return json.dumps({"error": str(e)})