"""
LangGraph State Schema
Defines the state that flows through the RAG workflow
"""

from typing import List, Dict, Optional, Literal, TypedDict, Annotated
from dataclasses import dataclass, field
import operator


class GraphState(TypedDict):
    """
    State that flows through the LangGraph workflow
    
    This represents all information needed across the pipeline:
    - Input query
    - Classification results
    - Retrieved context
    - Generated answer
    - Verification results
    """
    
    # Input
    query: str  # User's natural language query
    top_k: int  # Number of results to retrieve
    max_hops: int  # Max graph traversal distance
    
    # Classification
    query_type: Optional[Literal["global", "local"]]  # Determined by classifier
    confidence: Optional[float]  # Classification confidence (0-1)
    reasoning: Optional[str]  # Why this classification?
    
    # Retrieval
    retrieved_nodes: Annotated[List[str], operator.add]  # Node IDs (accumulates)
    context: Annotated[List[Dict], operator.add]  # Retrieved context (accumulates)
    retrieval_path: Optional[str]  # Which path was taken
    
    # Community-specific (for global queries)
    relevant_communities: Optional[List[int]]  # Community IDs
    community_summaries: Optional[List[str]]  # Pre-generated summaries
    
    # Graph-specific (for local queries)
    anchor_nodes: Optional[List[str]]  # Starting points for traversal
    subgraph_nodes: Optional[List[str]]  # Nodes in extracted subgraph
    
    # Generation
    formatted_context: Optional[str]  # Context prepared for LLM
    answer: Optional[str]  # Generated answer
    sources: Optional[List[str]]  # Source node IDs cited
    
    # Verification (SelfCheckGPT)
    verification_enabled: bool  # Whether to verify
    verification_samples: Optional[List[str]]  # N alternative answers generated
    consistency_scores: Optional[List[float]]  # Per-sentence scores (0-1)
    hallucination_flags: Optional[List[bool]]  # Flagged sentences
    hallucination_rate: Optional[float]  # Overall rate (0-1)
    verification_samples_count: Optional[int]  # Number of samples generated
    verified: Optional[bool]  # Overall verification result
    
    # Metadata
    steps_taken: Annotated[List[str], operator.add]  # Track workflow steps
    timing: Optional[Dict[str, float]]  # Time spent in each node
    tokens_used: Optional[Dict[str, int]]  # Token usage per LLM call
    errors: Annotated[List[str], operator.add]  # Any errors encountered


@dataclass
class NodeResult:
    """
    Result from a workflow node
    Used to update the state
    """
    updates: Dict  # State updates to apply
    next_node: Optional[str] = None  # Override next node (for conditional routing)
    metadata: Dict = field(default_factory=dict)  # Additional info


@dataclass
class RetrievalContext:
    """
    Retrieved context item
    """
    node_id: str
    node_type: str  # function, class, etc.
    name: str
    code: str
    summary: str
    tags: List[str]
    score: float  # Relevance score
    source: str  # "vector_search", "graph_traversal", "community"


@dataclass
class VerificationResult:
    """
    Result from SelfCheckGPT verification
    """
    sentence: str
    consistency_score: float
    is_hallucination: bool
    supporting_samples: List[str]


# Helper functions for state manipulation

def add_step(state: GraphState, step: str) -> Dict:
    """Add a step to the workflow history"""
    return {"steps_taken": [step]}


def add_context(state: GraphState, context_item: RetrievalContext) -> Dict:
    """Add a context item"""
    return {
        "context": [{
            "node_id": context_item.node_id,
            "node_type": context_item.node_type,
            "name": context_item.name,
            "code": context_item.code,
            "summary": context_item.summary,
            "tags": context_item.tags,
            "score": context_item.score,
            "source": context_item.source
        }],
        "retrieved_nodes": [context_item.node_id]
    }


def add_error(state: GraphState, error: str) -> Dict:
    """Add an error message"""
    return {"errors": [error]}


def initialize_state(
    query: str,
    top_k: int = 10,
    max_hops: int = 2,
    verification_enabled: bool = True
) -> GraphState:
    """
    Create initial state for a query
    
    Args:
        query: User's query
        top_k: Number of results to retrieve
        max_hops: Max graph traversal distance
        verification_enabled: Enable SelfCheckGPT verification
    
    Returns:
        Initialized GraphState
    """
    return GraphState(
        # Input
        query=query,
        top_k=top_k,
        max_hops=max_hops,
        
        # Classification
        query_type=None,
        confidence=None,
        reasoning=None,
        
        # Retrieval
        retrieved_nodes=[],
        context=[],
        retrieval_path=None,
        
        # Community
        relevant_communities=None,
        community_summaries=None,
        
        # Graph
        anchor_nodes=None,
        subgraph_nodes=None,
        
        # Generation
        formatted_context=None,
        answer=None,
        sources=None,
        
        # Verification
        verification_enabled=verification_enabled,
        verification_samples=None,
        consistency_scores=None,
        hallucination_flags=None,
        verified=None,
        
        # Metadata
        steps_taken=[],
        timing={},
        tokens_used={},
        errors=[]
    )


def format_state_summary(state: GraphState) -> str:
    """Format state for logging/debugging"""
    
    summary = f"""
=== Workflow State Summary ===
Query: {state['query']}
Type: {state.get('query_type', 'unknown')} (confidence: {state.get('confidence', 0):.2f})
Steps taken: {' → '.join(state.get('steps_taken', []))}
Context retrieved: {len(state.get('context', []))} items
Answer generated: {'Yes' if state.get('answer') else 'No'}
Verified: {state.get('verified', 'N/A')}
Errors: {len(state.get('errors', []))}
"""
    
    return summary.strip()


# Example usage
if __name__ == "__main__":
    # Create initial state
    state = initialize_state(
        query="What is the authentication system?",
        top_k=5,
        verification_enabled=True
    )
    
    print("Initial state:")
    print(format_state_summary(state))
    
    # Simulate updates
    state["steps_taken"].append("classify_query")
    state["query_type"] = "global"
    state["confidence"] = 0.95
    
    state["steps_taken"].append("retrieve_communities")
    state["retrieved_nodes"].extend(["node1", "node2"])
    
    print("\nAfter updates:")
    print(format_state_summary(state))