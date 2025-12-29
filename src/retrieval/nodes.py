"""
LangGraph Workflow Nodes
Each node is a function that operates on the GraphState
"""

import sys
from pathlib import Path
import time
from typing import Dict, List
import json
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_client import LangChainClient
from retrieval.state import GraphState, add_step, add_context, RetrievalContext
from indexing.vector_store import VectorStore
from indexing.community_detection import CommunityDetector


class WorkflowNodes:
    """
    Collection of workflow nodes for the RAG pipeline
    Each method is a node in the LangGraph
    """
    
    def __init__(
        self,
        graph,
        vector_store: VectorStore,
        community_detector: CommunityDetector,
        llm_client: LangChainClient = None
    ):
        """
        Initialize workflow nodes
        
        Args:
            graph: NetworkX graph with code entities
            vector_store: FAISS vector store
            community_detector: Community detector with summaries
            llm_client: LLM client for generation
        """
        self.graph = graph
        self.vector_store = vector_store
        self.communities = community_detector
        self.llm = llm_client or LangChainClient()        
        logger.info("Workflow nodes initialized")
    
    # ========================================
    # Node 1: Query Classification
    # ========================================
    
    def classify_query(self, state: GraphState) -> Dict:
        """
        Classify query as GLOBAL or LOCAL
        
        Global: Architecture, design, high-level questions
        Local: Specific functions, implementation details
        """
        start_time = time.time()
        logger.info(f"Classifying query: {state['query']}")
        
        classification_prompt = f"""Classify this code repository query as GLOBAL or LOCAL.

**Query:** {state['query']}

**Classification Rules:**

GLOBAL queries ask about:
- Overall architecture or design
- Main components or modules
- High-level flow or patterns
- General purpose of the system
- Multiple related entities
Examples: "What's the architecture?", "Main modules?", "How is data processed?"

LOCAL queries ask about:
- Specific functions or classes
- Implementation details
- Debugging or tracing
- Individual code entities
Examples: "How does login() work?", "What does validate_email do?", "Where is X called?"

**Respond with JSON:**
{{
  "classification": "GLOBAL" or "LOCAL",
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation"
}}
"""
        
        try:
            result = self.llm.generate_json(
                prompt=classification_prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "classification": {"type": "string", "enum": ["GLOBAL", "LOCAL"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["classification", "confidence", "reasoning"]
                }
            )
            
            query_type = result['classification'].lower()
            confidence = result['confidence']
            reasoning = result['reasoning']
            
            logger.info(f"Classification: {query_type.upper()} ({confidence:.2f})")
            
            elapsed = time.time() - start_time
            
            return {
                **add_step(state, "classify_query"),
                "query_type": query_type,
                "confidence": confidence,
                "reasoning": reasoning,
                "timing": {**state.get("timing", {}), "classify": elapsed}
            }
        
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            # Default to local on error
            return {
                **add_step(state, "classify_query_failed"),
                "query_type": "local",
                "confidence": 0.5,
                "reasoning": f"Classification failed, defaulting to local: {e}",
                "errors": [str(e)]
            }
    
    # ========================================
    # Node 2a: Global Retrieval (Community Summaries)
    # ========================================
    
    def retrieve_global(self, state: GraphState) -> Dict:
        """
        Retrieve relevant community summaries for global queries
        """
        start_time = time.time()
        logger.info("Executing global retrieval (community summaries)")
        
        query = state['query']
        
        # Use vector search to find relevant communities
        # Search community summaries directly
        relevant_comms = []
        
        for comm_id, community in self.communities.communities.items():
            # Create searchable text from community
            comm_text = f"{community.summary} {community.description}"
            
            # Simple relevance check (in production, embed community summaries)
            # For now, use keyword matching as fallback
            relevant_comms.append((comm_id, community))
            print(f"Checking community {comm_id}: {comm_text}")
        
        # Take top communities (simplified - should use semantic search)
        top_communities = relevant_comms[:state.get('top_k', 3)]
        
        # Gather context from communities
        context_items = []
        comm_ids = []
        
        for comm_id, community in top_communities:
            # Add community summary as context
            context_item = {
                "node_id": f"community_{comm_id}",
                "node_type": "community",
                "name": f"Community {comm_id}",
                "code": "",
                "summary": community.summary,
                "tags": [],
                "score": 1.0,
                "source": "community",
                "description": community.description,
                "purpose": community.purpose,
                "key_entities": community.key_entities
            }
            context_items.append(context_item)
            comm_ids.append(comm_id)
        
        elapsed = time.time() - start_time
        logger.info(f"Retrieved {len(context_items)} community summaries")
        
        return {
            **add_step(state, "retrieve_global"),
            "retrieval_path": "global",
            "relevant_communities": comm_ids,
            "context": context_items,
            "timing": {**state.get("timing", {}), "retrieve_global": elapsed}
        }
    
    # ========================================
    # Node 2b: Local Retrieval (Vector + Graph)
    # ========================================
    
    def retrieve_local(self, state: GraphState) -> Dict:
            """
            Retrieve relevant code using vector search + graph traversal + pruning + reranking
            """
            start_time = time.time()
            logger.info("Executing local retrieval (vector + graph + optimization)")
            
            query = state['query']
            top_k = state.get('top_k', 10)
            max_hops = state.get('max_hops', 2)
            
            # Step 1: Query expansion (optional, for better recall)
            if hasattr(self, 'query_expander'):
                expanded = self.query_expander.expand(query, num_expansions=2)
                all_queries = [query] + expanded.expansions
            else:
                all_queries = [query]
            
            # Step 2: Vector search with expanded queries
            all_anchor_nodes = set()
            for q in all_queries[:2]:  # Use top 2 queries
                search_results = self.vector_store.search(q, top_k=top_k)
                all_anchor_nodes.update([r.node_id for r in search_results])
            
            anchor_nodes = list(all_anchor_nodes)[:top_k]
            logger.info(f"Found {len(anchor_nodes)} anchor nodes via vector search")
            
            # Step 3: Graph traversal from anchors
            expanded_nodes = set(anchor_nodes)
            
            for anchor in anchor_nodes[:3]:  # Expand from top 3
                if anchor in self.graph:
                    neighbors = self._get_k_hop_neighbors(anchor, max_hops)
                    expanded_nodes.update(neighbors)
            
            logger.info(f"Expanded to {len(expanded_nodes)} nodes via graph traversal")
            
            # Step 4: Prune context (NEW - Week 4)
            if hasattr(self, 'pruner') and len(expanded_nodes) > top_k:
                pruned = self.pruner.prune(
                    candidate_nodes=list(expanded_nodes),
                    query=query,
                    target_count=top_k * 2  # Get 2x for reranking
                )
                selected_nodes = pruned.nodes
                logger.info(f"Pruned to {len(selected_nodes)} nodes")
            else:
                selected_nodes = list(expanded_nodes)[:top_k * 2]
            
            # Step 5: Re-rank with cross-encoder
            if hasattr(self, 'reranker'):
                reranked = self.reranker.rerank(
                    query=query,
                    nodes=selected_nodes,
                    graph=self.graph,
                    top_k=top_k
                )
                final_nodes = [r.node_id for r in reranked]
                scores_dict = {r.node_id: r.score for r in reranked}
                logger.info(f"Re-ranked to top {len(final_nodes)} nodes")
            else:
                final_nodes = selected_nodes[:top_k]
                scores_dict = {n: 0.8 for n in final_nodes}
            
            # Step 6: Build context items
            context_items = []
            for node_id in final_nodes:
                if node_id in self.graph.nodes:
                    attrs = self.graph.nodes[node_id]
                    
                    context_item = {
                        "node_id": node_id,
                        "node_type": attrs.get('type', 'unknown'),
                        "name": attrs.get('name', 'unknown'),
                        "code": attrs.get('code', '')[:500],  # Truncate
                        "summary": attrs.get('summary', ''),
                        "tags": attrs.get('tags', []),
                        "score": scores_dict.get(node_id, 0.5),
                        "source": "optimized_retrieval"
                    }
                    context_items.append(context_item)
            
            elapsed = time.time() - start_time
            logger.info(f"Retrieved {len(context_items)} optimized entities in {elapsed:.2f}s")
            
            return {
                **add_step(state, "retrieve_local_optimized"),
                "retrieval_path": "local",
                "anchor_nodes": anchor_nodes,
                "subgraph_nodes": list(expanded_nodes),
                "context": context_items,
                "timing": {**state.get("timing", {}), "retrieve_local": elapsed}
            }
    
    def _get_k_hop_neighbors(self, node_id: str, k: int) -> List[str]:
        """Get all nodes within k hops"""
        visited = {node_id}
        current_level = {node_id}
        
        for _ in range(k):
            next_level = set()
            for node in current_level:
                # Get both successors and predecessors
                if node in self.graph:
                    next_level.update(self.graph.successors(node))
                    next_level.update(self.graph.predecessors(node))
            
            next_level -= visited
            visited.update(next_level)
            current_level = next_level
            
            if not current_level:
                break
        
        return list(visited)
    
    # ========================================
    # Node 3: Format Context
    # ========================================
    
    def format_context(self, state: GraphState) -> Dict:
        """
        Format retrieved context for LLM consumption
        """
        logger.info("Formatting context for LLM")
        
        context_items = state.get('context', [])
        query_type = state.get('query_type', 'local')
        
        if query_type == 'global':
            # Format community summaries
            formatted = "# Architecture Overview\n\n"
            
            for item in context_items:
                formatted += f"## {item['name']}\n"
                formatted += f"**Summary:** {item['summary']}\n\n"
                formatted += f"**Description:** {item.get('description', '')}\n\n"
                formatted += f"**Purpose:** {item.get('purpose', '')}\n\n"
                
                if item.get('key_entities'):
                    formatted += f"**Key entities:** {', '.join(item['key_entities'])}\n\n"
                
                formatted += "---\n\n"
        
        else:
            # Format code entities
            formatted = "# Relevant Code Entities\n\n"
            
            for i, item in enumerate(context_items, 1):
                formatted += f"## {i}. {item['name']} ({item['node_type']})\n"
                formatted += f"**Score:** {item['score']:.2f} | **Source:** {item['source']}\n\n"
                
                if item['summary']:
                    formatted += f"**Summary:** {item['summary']}\n\n"
                
                if item['tags']:
                    formatted += f"**Tags:** {', '.join(item['tags'][:5])}\n\n"
                
                if item['code']:
                    formatted += f"**Code:**\n```python\n{item['code'][:300]}\n```\n\n"
                
                formatted += "---\n\n"
        
        return {
            **add_step(state, "format_context"),
            "formatted_context": formatted
        }
    
    # ========================================
    # Node 4: Generate Answer
    # ========================================
    
    def generate_answer(self, state: GraphState) -> Dict:
        """
        Generate answer using LLM
        """
        start_time = time.time()
        logger.info("Generating answer with LLM")
        
        query = state['query']
        context = state.get('formatted_context', '')
        query_type = state.get('query_type', 'local')
        
        if query_type == 'global':
            system_prompt = """You are a code architecture expert. Answer questions about system architecture and design using the provided module summaries. Focus on high-level structure and relationships."""
        else:
            system_prompt = """You are a code analysis expert. Answer questions about specific code implementation using the provided code entities. Be precise and cite specific functions/classes."""
        
        user_prompt = f"""Based on the following context, answer this question:

**Question:** {query}

**Context:**
{context}

**Instructions:**
1. Answer the question directly and clearly
2. Use information from the context
3. Cite specific entities when relevant
4. If information is insufficient, say so
5. Keep the answer concise but complete
"""
        
        try:
            response = self.llm.generate(
           
                system=system_prompt,
                prompt=user_prompt,
                temperature=0.3
            )
            
            answer = response.content
            tokens = response.usage_metadata
            
            elapsed = time.time() - start_time
            logger.success(f"Answer generated ({tokens['output_tokens']} tokens)")
            
            return {
                **add_step(state, "generate_answer"),
                "answer": answer,
                "tokens_used": {**state.get("tokens_used", {}), "generation": tokens},
                "timing": {**state.get("timing", {}), "generate": elapsed}
            }
        
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {
                **add_step(state, "generate_answer_failed"),
                "answer": f"Failed to generate answer: {e}",
                "errors": [str(e)]
            }
    
    # ========================================
    # Node 5: Verify Answer (Simplified)
    # ========================================
    
    def verify_answer(self, state: GraphState) -> Dict:
        """
        Verify answer for hallucinations (simplified SelfCheckGPT)
        """
        if not state.get('verification_enabled', False):
            logger.info("Verification disabled, skipping")
            return {
                **add_step(state, "verification_skipped"),
                "verified": True
            }
        
        logger.info("Verifying answer (simplified check)")
        
        # Simplified verification: check if answer references context
        answer = state.get('answer', '')
        context_items = state.get('context', [])
        
        # Simple heuristic: does answer mention entities from context?
        entity_names = [item['name'] for item in context_items]
        mentions = sum(1 for name in entity_names if name.lower() in answer.lower())
        
        grounded_ratio = mentions / max(len(entity_names), 1)
        verified = grounded_ratio > 0.2  # At least 20% of entities mentioned
        
        logger.info(f"Verification: {mentions}/{len(entity_names)} entities mentioned")
        
        return {
            **add_step(state, "verify_answer"),
            "verified": verified,
            "consistency_scores": [grounded_ratio]
        }


# Example usage
if __name__ == "__main__":
    print("Workflow nodes defined. Use within graph_workflow.py")