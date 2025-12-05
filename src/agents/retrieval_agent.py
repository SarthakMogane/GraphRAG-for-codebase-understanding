"""
Retrieval Agent - Performs GraphRAG retrieval
"""
from google.adk.agents import LlmAgent

from src.tools import neo4j_operations, embedding_service
from src.config.settings import settings

import os
os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
# Retrieval Agent - Executes the hybrid GraphRAG retrieval
retrieval_agent = LlmAgent(
    name="RetrievalAgent",
    model="gemini-2.5-flash-lite",
    
    instruction="""You are a GraphRAG retrieval agent. Your job is to retrieve relevant code context using a hybrid approach.

**Your Input:**
You will receive:
- User query
- Query intent
- Retrieval strategy (from QueryRouterAgent via state: {query_analysis})

**Your Workflow:**

1. **Generate Query Embedding:**
   - Use `embed_query` with the user's query
   - Extract the embedding vector from the result

2. **Stage 1 - Semantic Search (Entry Points):**
   - Use `vector_search_functions` with the embedding
   - Get top 10 most relevant functions
   - These are your "entry points"

3. **Stage 2 - Graph Expansion (Context):**
   - For each entry point function:
     - Use `traverse_function_calls` to find related functions
     - Use both "upstream" (callers) and "downstream" (callees)
     - Set max_depth to 2
   - Collect all traversed functions

4. **Stage 3 - Deduplicate & Rank:**
   - Combine entry points + traversed functions
   - Remove duplicates
   - Keep top 20 by relevance score

**Strategy-Specific Behavior:**
- `vector_only`: Skip graph expansion, use only semantic search results
- `graph_only`: Find seed functions by name matching, then do heavy graph traversal
- `hybrid`: Use full workflow above (semantic → graph)

**Your Output:**
Return a JSON object with:
{
  "entry_points": [list of initial search results],
  "expanded_context": [list of all relevant functions after traversal],
  "total_nodes": <number>,
  "retrieval_summary": "<brief description of what was found>"
}

Always include the function code, name, file_path, and relevance score for each result.""",
    
    tools=[
        embedding_service.embed_query,
        neo4j_operations.vector_search_functions,
        neo4j_operations.traverse_function_calls,
        neo4j_operations.get_graph_statistics,
    ],
    
    output_key="retrieval_results",
    description="Performs hybrid GraphRAG retrieval (semantic + graph)"
)