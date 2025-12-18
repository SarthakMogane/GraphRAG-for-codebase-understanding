"""
Query Workflow - Sequential pipeline for question answering
"""
from google.adk.agents import SequentialAgent

from src.agents.query_agent import query_router_agent
from src.agents.retrieval_agent import retrieval_agent
from src.agents.synthesis_agent import synthesis_agent

# Query Workflow - Multi-stage question answering pipeline
root_agent = SequentialAgent(
    name="QueryWorkflow",
    sub_agents=[
        query_router_agent,   # Stage 1: Analyze query → output_key="query_analysis"
        retrieval_agent,      # Stage 2: Retrieve context → output_key="retrieval_results"
        synthesis_agent,      # Stage 3: Generate answer → output_key="final_answer"
    ],
    description="""Complete pipeline for answering questions about code.

**Workflow:**
1. QueryRouterAgent: Classifies intent and determines strategy
2. RetrievalAgent: Performs GraphRAG retrieval (semantic + graph)
3. SynthesisAgent: Generates final answer from context

**State Flow:**
- Query goes through each agent sequentially
- Each agent's output is stored in state with its output_key
- Later agents access previous outputs via state

**Input:** User query
**Output:** Comprehensive answer with code context"""
)