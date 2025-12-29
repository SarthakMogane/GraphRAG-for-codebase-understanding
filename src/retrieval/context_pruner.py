"""
Context Pruning - Remove irrelevant information before LLM generation
Uses multiple strategies: PageRank, embedding similarity, and heuristics
"""

import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple
import networkx as nx
import numpy as np
from loguru import logger
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class PrunedContext:
    """Result of context pruning"""
    nodes: List[str]  # Pruned node IDs
    scores: Dict[str, float]  # Node ID -> relevance score
    pruned_count: int  # Number of nodes removed
    strategy: str  # Pruning strategy used


class ContextPruner:
    """
    Intelligent context pruning using multiple strategies
    
    Strategies:
    1. PageRank - Graph-based importance
    2. Embedding Similarity - Semantic relevance to query
    3. Degree Centrality - Connection-based importance
    4. Hybrid - Combination of multiple signals
    """
    
    def __init__(
        self,
        graph: nx.DiGraph,
        embedding_model=None,
        strategy: str = 'hybrid'
    ):
        """
        Initialize context pruner
        
        Args:
            graph: NetworkX graph
            embedding_model: Sentence transformer model (for similarity)
            strategy: Pruning strategy ('pagerank', 'similarity', 'hybrid')
        """
        self.graph = graph
        self.embedding_model = embedding_model
        self.strategy = strategy
        
        # Pre-compute PageRank for efficiency
        self._pagerank_scores = None
        
        logger.info(f"Context pruner initialized (strategy={strategy})")
    
    def prune(
        self,
        candidate_nodes: List[str],
        query: str = None,
        target_count: int = 10,
        min_score: float = 0.3
    ) -> PrunedContext:
        """
        Prune candidate nodes to most relevant subset
        
        Args:
            candidate_nodes: Initial set of nodes to prune
            query: User query (for similarity-based pruning)
            target_count: Target number of nodes to keep
            min_score: Minimum relevance score threshold
        
        Returns:
            PrunedContext with selected nodes
        """
        logger.info(f"Pruning {len(candidate_nodes)} nodes to ~{target_count}")
        
        if len(candidate_nodes) <= target_count:
            logger.info("Already below target, no pruning needed")
            return PrunedContext(
                nodes=candidate_nodes,
                scores={node: 1.0 for node in candidate_nodes},
                pruned_count=0,
                strategy='none'
            )
        
        # Apply pruning strategy
        if self.strategy == 'pagerank':
            scores = self._prune_by_pagerank(candidate_nodes)
        elif self.strategy == 'similarity':
            scores = self._prune_by_similarity(candidate_nodes, query)
        elif self.strategy == 'degree':
            scores = self._prune_by_degree(candidate_nodes)
        else:  # hybrid
            scores = self._prune_hybrid(candidate_nodes, query)
        
        # Filter and sort
        filtered = {
            node: score 
            for node, score in scores.items() 
            if score >= min_score
        }
        
        sorted_nodes = sorted(
            filtered.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:target_count]
        
        pruned_nodes = [node for node, _ in sorted_nodes]
        final_scores = dict(sorted_nodes)
        
        pruned_count = len(candidate_nodes) - len(pruned_nodes)
        
        logger.info(f"Pruned {pruned_count} nodes, kept {len(pruned_nodes)}")
        
        return PrunedContext(
            nodes=pruned_nodes,
            scores=final_scores,
            pruned_count=pruned_count,
            strategy=self.strategy
        )
    
    def _prune_by_pagerank(self, nodes: List[str]) -> Dict[str, float]:
        """Prune using PageRank centrality"""
        
        # Compute PageRank if not cached
        if self._pagerank_scores is None:
            logger.info("Computing PageRank scores...")
            self._pagerank_scores = nx.pagerank(self.graph, alpha=0.85)
        
        # Get scores for candidate nodes
        scores = {}
        for node in nodes:
            scores[node] = self._pagerank_scores.get(node, 0.0)
        
        # Normalize
        max_score = max(scores.values()) if scores else 1.0
        scores = {node: score / max_score for node, score in scores.items()}
        
        return scores
    
    def _prune_by_similarity(
        self, 
        nodes: List[str], 
        query: str
    ) -> Dict[str, float]:
        """Prune using embedding similarity to query"""
        
        if not self.embedding_model or not query:
            logger.warning("No embedding model or query, using uniform scores")
            return {node: 0.5 for node in nodes}
        
        # Get query embedding
        query_emb = self.embedding_model.encode([query])[0]
        print(query_emb)
        
        # Get node embeddings and compute similarity
        scores = {}
        for node in nodes:
            if node not in self.graph.nodes:
                scores[node] = 0.0
                continue
            
            attrs = self.graph.nodes[node]
            # Create text representation
            text = f"{attrs.get('name', '')} {attrs.get('summary', '')} {attrs.get('code', '')[:200]}"
            
            # Compute similarity
            node_emb = self.embedding_model.encode([text])[0]
            print(node_emb)
            similarity = np.dot(query_emb, node_emb)
            scores[node] = float(similarity)
        
        return scores
    
    def _prune_by_degree(self, nodes: List[str]) -> Dict[str, float]:
        """Prune using degree centrality"""
        
        # Create subgraph with candidate nodes
        subgraph = self.graph.subgraph(nodes)
        
        # Compute degree centrality
        centrality = nx.degree_centrality(subgraph)
        
        return centrality
    
    def _prune_hybrid(
        self, 
        nodes: List[str], 
        query: str = None
    ) -> Dict[str, float]:
        """
        Hybrid pruning combining multiple signals
        
        Combines:
        - PageRank (40%) - Graph importance
        - Similarity (40%) - Query relevance
        - Degree (20%) - Local connectivity
        """
        
        pagerank_scores = self._prune_by_pagerank(nodes)
        degree_scores = self._prune_by_degree(nodes)
        
        if query and self.embedding_model:
            similarity_scores = self._prune_by_similarity(nodes, query)
            weights = {'pagerank': 0.4, 'similarity': 0.4, 'degree': 0.2}
        else:
            similarity_scores = {node: 0.0 for node in nodes}
            weights = {'pagerank': 0.5, 'similarity': 0.0, 'degree': 0.5}
        
        # Combine scores
        hybrid_scores = {}
        for node in nodes:
            hybrid_scores[node] = (
                weights['pagerank'] * pagerank_scores.get(node, 0.0) +
                weights['similarity'] * similarity_scores.get(node, 0.0) +
                weights['degree'] * degree_scores.get(node, 0.0)
            )
        
        return hybrid_scores
    
    def prune_with_diversity(
        self,
        candidate_nodes: List[str],
        query: str = None,
        target_count: int = 10,
        diversity_weight: float = 0.3
    ) -> PrunedContext:
        """
        Prune while maintaining diversity (MMR-style)
        
        Maximal Marginal Relevance: Balance relevance and diversity
        """
        
        # Get base scores
        base_scores = self._prune_hybrid(candidate_nodes, query)
        
        selected = []
        remaining = set(candidate_nodes)
        
        # Select first (highest score)
        first_node = max(remaining, key=lambda n: base_scores[n])
        selected.append(first_node)
        remaining.remove(first_node)
        
        # Iteratively select diverse nodes
        while len(selected) < target_count and remaining:
            best_node = None
            best_score = -1
            
            for node in remaining:
                # Relevance score
                relevance = base_scores[node]
                
                # Diversity penalty (similarity to selected)
                diversity_penalty = 0
                if node in self.graph.nodes:
                    for selected_node in selected:
                        if selected_node in self.graph.nodes:
                            # Penalize if directly connected
                            if self.graph.has_edge(node, selected_node) or \
                               self.graph.has_edge(selected_node, node):
                                diversity_penalty += 0.2
                
                # MMR score
                mmr_score = (1 - diversity_weight) * relevance - \
                           diversity_weight * diversity_penalty
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_node = node
            
            if best_node:
                selected.append(best_node)
                remaining.remove(best_node)
        
        final_scores = {node: base_scores[node] for node in selected}
        
        return PrunedContext(
            nodes=selected,
            scores=final_scores,
            pruned_count=len(candidate_nodes) - len(selected),
            strategy='mmr'
        )


class TokenBudgetPruner:
    """
    Prune based on token budget constraints
    Ensures context fits within LLM's context window
    """
    
    def __init__(self, max_tokens: int = 3000):
        """
        Initialize token budget pruner
        
        Args:
            max_tokens: Maximum tokens for context
        """
        self.max_tokens = max_tokens
        logger.info(f"Token budget pruner initialized (max={max_tokens})")
    
    def prune_to_budget(
        self,
        nodes: List[str],
        graph: nx.DiGraph,
        scores: Dict[str, float]
    ) -> List[str]:
        """
        Prune nodes to fit within token budget
        
        Args:
            nodes: Candidate nodes (sorted by score)
            graph: NetworkX graph
            scores: Node relevance scores
        
        Returns:
            Subset of nodes that fit budget
        """
        
        # Sort by score
        sorted_nodes = sorted(nodes, key=lambda n: scores.get(n, 0), reverse=True)
        
        selected = []
        total_tokens = 0
        
        for node in sorted_nodes:
            if node not in graph.nodes:
                continue
            
            attrs = graph.nodes[node]
            
            # Estimate tokens (rough: 1 token ≈ 4 chars)
            code = attrs.get('code', '')
            summary = attrs.get('summary', '')
            node_tokens = (len(code) + len(summary)) // 4
            
            if total_tokens + node_tokens <= self.max_tokens:
                selected.append(node)
                total_tokens += node_tokens
            else:
                break
        
        logger.info(f"Token pruning: {len(selected)}/{len(nodes)} nodes "
                   f"({total_tokens}/{self.max_tokens} tokens)")
        
        return selected


# Example usage
if __name__ == "__main__":
    import pickle
    from sentence_transformers import SentenceTransformer
    
    # Load graph
    with open('data/graphs/code_graph_enriched.pkl', 'rb') as f:
        graph = pickle.load(f)
    
    # Initialize pruner
    model = SentenceTransformer('all-MiniLM-L6-v2')
    pruner = ContextPruner(graph, embedding_model=model, strategy='hybrid')
    
    # Test pruning
    all_nodes = list(graph.nodes())[:30]  # Take 30 nodes
    
    result = pruner.prune(
        candidate_nodes=all_nodes,
        query="function that validates data",
        target_count=10
    )
    
    print(f"\n=== Pruning Result ===")
    print(f"Input: {len(all_nodes)} nodes")
    print(f"Output: {len(result.nodes)} nodes")
    print(f"Pruned: {result.pruned_count} nodes")
    print(f"Strategy: {result.strategy}")
    
    print(f"\n=== Top Nodes ===")
    for i, (node, score) in enumerate(sorted(
        result.scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:5], 1):
        node_name = graph.nodes[node].get('name', node)
        print(f"{i}. {node_name} - {score:.3f}")