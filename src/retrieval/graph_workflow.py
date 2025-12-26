"""
LangGraph RAG Workflow
Main orchestration graph for intelligent code Q&A
"""

import sys
from pathlib import Path
from typing import Dict, Literal
from loguru import logger

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.state import GraphState, initialize_state, format_state_summary
from retrieval.nodes import WorkflowNodes
from utils.llm_client import LangChainClient
from indexing.vector_store import VectorStore
from indexing.community_detection import CommunityDetector


def create_rag_workflow(
    graph,
    vector_store: VectorStore,
    community_detector: CommunityDetector,
    llm_client: LangChainClient = None,
    enable_verification: bool = True,
    enable_optimization: bool = True
):
    """
    Create the complete RAG workflow using LangGraph
    
    Args:
        graph: NetworkX graph with code entities
        vector_store: FAISS vector store
        community_detector: Community detector with summaries
        llm_client: LLM client (default: create new)
        enable_verification: Enable answer verification
    
    Returns:
        Compiled LangGraph workflow
    """
    
    logger.info("Creating LangGraph RAG workflow")
    
    # Initialize workflow nodes
    nodes = WorkflowNodes(
        graph=graph,
        vector_store=vector_store,
        community_detector=community_detector,
        llm_client=llm_client
    )

    if enable_optimization:
        from sentence_transformers import SentenceTransformer
        from retrieval.context_pruner import ContextPruner
        from retrieval.reranker import Reranker
        from retrieval.query_expansion import QueryExpander
        
        logger.info("Enabling Week 4 optimizations...")
        
        # Load embedding model for pruning
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Add pruner
        nodes.pruner = ContextPruner(
            graph=graph,
            embedding_model=embedding_model,
            strategy='hybrid'
        )
        
        # Add reranker
        nodes.reranker = Reranker(model_name='fast')
        
        # Add query expander
        nodes.query_expander = QueryExpander(llm_client=llm_client)
        
        logger.info("✓ Optimizations enabled: pruning, reranking, query expansion")


    
    # Create the state graph
    workflow = StateGraph(GraphState)
    
    # ========================================
    # Add nodes to the graph
    # ========================================
    
    workflow.add_node("classify_query", nodes.classify_query)
    workflow.add_node("retrieve_global", nodes.retrieve_global)
    workflow.add_node("retrieve_local", nodes.retrieve_local)
    workflow.add_node("format_context", nodes.format_context)
    workflow.add_node("generate_answer", nodes.generate_answer)
    
    if enable_verification:
        workflow.add_node("verify_answer", nodes.verify_answer)

    
    # Create the state graph
    workflow = StateGraph(GraphState)
    
    # ========================================
    # Add nodes to the graph
    # ========================================
    
    workflow.add_node("classify_query", nodes.classify_query)
    workflow.add_node("retrieve_global", nodes.retrieve_global)
    workflow.add_node("retrieve_local", nodes.retrieve_local)
    workflow.add_node("format_context", nodes.format_context)
    workflow.add_node("generate_answer", nodes.generate_answer)
    
    if enable_verification:
        workflow.add_node("verify_answer", nodes.verify_answer)
    
    # ========================================
    # Define the workflow edges
    # ========================================
    
    # Start with classification
    workflow.set_entry_point("classify_query")
    
    # Conditional routing based on query type
    def route_by_query_type(state: GraphState) -> Literal["retrieve_global", "retrieve_local"]:
        """Route to global or local retrieval based on classification"""
        query_type = state.get("query_type", "local")
        logger.info(f"Routing to {query_type} retrieval")
        return f"retrieve_{query_type}"
    
    workflow.add_conditional_edges(
        "classify_query",
        route_by_query_type,
        {
            "retrieve_global": "retrieve_global",
            "retrieve_local": "retrieve_local"
        }
    )
    
    # Both retrieval paths lead to formatting
    workflow.add_edge("retrieve_global", "format_context")
    workflow.add_edge("retrieve_local", "format_context")
    
    # Format leads to generation
    workflow.add_edge("format_context", "generate_answer")
    
    # Generation leads to verification or end
    if enable_verification:
        workflow.add_edge("generate_answer", "verify_answer")
        workflow.add_edge("verify_answer", END)
    else:
        workflow.add_edge("generate_answer", END)
    
    # ========================================
    # Compile the graph
    # ========================================
    
    # Add memory checkpointer (optional - allows resuming)
    memory = MemorySaver()
    
    app = workflow.compile(checkpointer=memory)
    
    logger.success("LangGraph workflow compiled successfully")
    
    return app


def visualize_workflow(workflow) -> str:
    """
    Generate mermaid diagram of the workflow
    
    Returns:
        Mermaid diagram as string
    """
    
    mermaid = """
graph TD
    START([User Query]) --> CLASSIFY[Classify Query]
    
    CLASSIFY -->|Global| GLOBAL[Retrieve Global<br/>Community Summaries]
    CLASSIFY -->|Local| LOCAL[Retrieve Local<br/>Vector + Graph]
    
    GLOBAL --> FORMAT[Format Context]
    LOCAL --> FORMAT
    
    FORMAT --> GENERATE[Generate Answer<br/>with LLM]
    
    GENERATE --> VERIFY[Verify Answer<br/>SelfCheckGPT]
    
    VERIFY --> END([Return Answer])
    
    style CLASSIFY fill:#e1f5ff
    style GLOBAL fill:#c8e6c9
    style LOCAL fill:#fff9c4
    style GENERATE fill:#ffccbc
    style VERIFY fill:#f8bbd0
"""
    
    return mermaid


class RAGPipeline:
    """
    High-level interface for the RAG system
    Wraps the LangGraph workflow with convenience methods
    """
    
    def __init__(
        self,
        graph,
        vector_store: VectorStore,
        community_detector: CommunityDetector,
        llm_client: LangChainClient = None,
        enable_verification: bool = True
    ):
        """Initialize RAG pipeline"""
        
        self.graph = graph
        self.vector_store = vector_store
        self.communities = community_detector
        self.llm = llm_client or LangChainClient()
        
        # Create workflow
        self.workflow = create_rag_workflow(
            graph=graph,
            vector_store=vector_store,
            community_detector=community_detector,
            llm_client=self.llm,
            enable_verification=enable_verification
        )
        
        logger.info("RAG Pipeline initialized")
    
    def query(
        self,
        question: str,
        top_k: int = 10,
        max_hops: int = 2,
        verbose: bool = False
    ) -> Dict:
        """
        Query the codebase
        
        Args:
            question: Natural language question
            top_k: Number of results to retrieve
            max_hops: Max graph traversal distance
            verbose: Print intermediate steps
        
        Returns:
            Dictionary with answer and metadata
        """
        
        logger.info(f"Processing query: {question}")
        
        # Initialize state
        initial_state = initialize_state(
            query=question,
            top_k=top_k,
            max_hops=max_hops,
            verification_enabled=True
        )
        
        # Run workflow
        try:
            # Execute with thread ID for memory
            config = {"configurable": {"thread_id": "default"}}
            final_state = self.workflow.invoke(initial_state, config)
            
            if verbose:
                print("\n" + format_state_summary(final_state))
            
            # Extract result
            result = {
                "answer": final_state.get("answer", "No answer generated"),
                "query_type": final_state.get("query_type"),
                "confidence": final_state.get("confidence"),
                "sources": [
                    {
                        "name": item["name"],
                        "type": item["node_type"],
                        "score": item["score"]
                    }
                    for item in final_state.get("context", [])[:5]
                ],
                "verified": final_state.get("verified"),
                "steps": final_state.get("steps_taken", []),
                "timing": final_state.get("timing", {}),
                "errors": final_state.get("errors", [])
            }
            
            logger.success("Query processed successfully")
            
            return result
        
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "answer": f"Error processing query: {e}",
                "error": str(e)
            }
    
    def stream_query(self, question: str, **kwargs):
        """
        Stream query execution (shows intermediate steps)
        
        Yields state after each node
        """
        
        initial_state = initialize_state(query=question, **kwargs)
        config = {"configurable": {"thread_id": "stream"}}
        
        for state in self.workflow.stream(initial_state, config):
            yield state
    
    def get_graph_visualization(self) -> str:
        """Get mermaid diagram of workflow"""
        return visualize_workflow(self.workflow)


# Example usage
if __name__ == "__main__":
    import pickle
    
    # This is a demo - actual usage in scripts/query_with_langgraph.py
    
    print("="*70)
    print("LangGraph RAG Workflow")
    print("="*70)
    
    print("\nWorkflow Structure:")
    print(visualize_workflow(None))
    
    print("\n" + "="*70)
    print("To use:")
    print("  from retrieval.graph_workflow import RAGPipeline")
    print("  pipeline = RAGPipeline(graph, vector_store, communities)")
    print("  result = pipeline.query('What is the authentication system?')")
    print("="*70)