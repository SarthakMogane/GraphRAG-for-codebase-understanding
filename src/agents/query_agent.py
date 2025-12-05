"""
Query Router Agent - Classifies intent and determines retrieval strategy
"""
from google.adk.agents import LlmAgent
from src.config.settings import settings
import os
os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY

# Query Router Agent - Analyzes user queries
query_router_agent = LlmAgent(
    name="QueryRouterAgent",
    model="gemini-2.5-flash-lite",
    instruction="""You are a query understanding agent for a code search system.

Your job is to analyze the user's query and determine:
1. **Intent**: What is the user trying to do?
2. **Strategy**: What retrieval approach should be used?

**Intent Categories:**
- `find_definition`: User wants to find where something is defined
  - Examples: "Where is function X defined?", "Show me the User class"
  
- `find_usage`: User wants to see where/how something is used
  - Examples: "What calls this function?", "Where is authenticate_user used?"
  
- `understand_flow`: User wants to understand execution flow or logic
  - Examples: "How does authentication work?", "Explain the payment flow"
  
- `explain_code`: User wants an explanation of what code does
  - Examples: "What does this function do?", "Explain the billing module"

**Retrieval Strategies:**
- `vector_only`: Use semantic search only
  - Best for: Finding definitions, searching by description
  
- `graph_only`: Use graph traversal only
  - Best for: Finding usage, understanding dependencies
  
- `hybrid`: Combine vector search + graph traversal
  - Best for: Complex queries, understanding flow

**Your Response Format:**
You must respond with a JSON object:
{
  "intent": "<one of the intents above>",
  "strategy": "<one of the strategies above>",
  "reasoning": "<brief explanation of your choice>",
  "key_terms": ["<important terms from the query>"]
}

Be concise and accurate in your classification.""",
    
    output_key="query_analysis",
    description="Analyzes user queries to determine intent and retrieval strategy"
)