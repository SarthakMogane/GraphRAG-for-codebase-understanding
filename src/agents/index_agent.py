"""
Index Agent - Responsible for repository indexing
Uses LlmAgent from ADK with tool functions
"""
from google.adk.agents import LlmAgent
import os 

from src.tools import (
    github_operations,
    code_parser,
    neo4j_operations,
    embedding_service
)
from src.config.settings import settings
os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY

# Index Agent - Orchestrates the indexing pipeline
index_agent = LlmAgent(
    name="IndexAgent",
    model="gemini-2.5-flash-lite",
    instruction="""You are an expert code indexing agent. Your job is to:

1. Clone a GitHub repository
2. Parse all source files to extract functions, classes, and imports
3. Build a knowledge graph in Neo4j
4. Generate embeddings for all code entities

**Workflow:**
1. Use `clone_repository` to clone the repo
2. Use `get_source_files` to list all source files
3. Use `read_file_content` to read each file
4. Use `parse_python_file` to parse Python files
5. Use `initialize_graph_schema` to set up the graph
6. Use `create_function_node` and `create_file_node` to create nodes
7. Use `create_relationship` to link nodes (CONTAINS, CALLS relationships)
8. Use `embed_code_function` to generate embeddings
9. Update function nodes with embeddings

Always:
- Report progress after each major step
- Handle errors gracefully
- Provide statistics about indexed content
- Be efficient with API calls

When you're done, provide a summary with:
- Total files parsed
- Total functions indexed
- Total classes indexed
- Total relationships created
- Any errors encountered""",
    
    tools=[
        # GitHub operations
        github_operations.clone_repository,
        github_operations.get_source_files,
        github_operations.read_file_content,
        github_operations.get_repository_stats,
        
        # Code parsing
        code_parser.parse_python_file,
        code_parser.generate_node_id,
        
        # Neo4j operations
        neo4j_operations.initialize_graph_schema,
        neo4j_operations.create_function_node,
        neo4j_operations.create_file_node,
        neo4j_operations.create_relationship,
        neo4j_operations.get_graph_statistics,
        
        # Embeddings
        embedding_service.embed_code_function,
        embedding_service.embed_batch,
    ],
    
    output_key="indexing_result",
    description="Indexes GitHub repositories into a knowledge graph with embeddings"
)