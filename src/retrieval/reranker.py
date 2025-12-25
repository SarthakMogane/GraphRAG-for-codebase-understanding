"""
Re-ranking Module - Improve retrieval quality with cross-encoders
More accurate than bi-encoders for relevance scoring
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
from loguru import logger
from sentence_transformers import CrossEncoder
import torch


sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class RerankedResult:
    """Re-ranked result"""
    node_id: str
    score: float
    original_rank: int
    new_rank: int
    score_change: float


class Reranker:
    """
    Re-rank retrieved nodes using cross-encoder
    
    Cross-encoders are more accurate than bi-encoders because they
    process query and document together, capturing interaction.
    
    Trade-off: Slower (can't pre-compute), but better quality.
    """
    
    # Recommended cross-encoder models
    MODELS = {
        'fast': 'cross-encoder/ms-marco-MiniLM-L-6-v2',  # Fast, decent
        'balanced': 'cross-encoder/ms-marco-MiniLM-L-12-v2',  # Good balance
        'accurate': 'cross-encoder/ms-marco-electra-base'  # Best quality
    }
    
    def __init__(
        self,
        model_name: str = 'fast',
        device: str = None,
        batch_size: int = 16
    ):
        """
        Initialize reranker
        
        Args:
            model_name: Model size ('fast', 'balanced', 'accurate')
            device: 'cuda' or 'cpu' (auto-detect if None)
            batch_size: Batch size for scoring
        """
        model_path = self.MODELS.get(model_name, model_name)
        
        logger.info(f"Loading cross-encoder: {model_path}")
        
        # Detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = CrossEncoder(model_path, device=device)
        self.batch_size = batch_size
        
        logger.info(f"Reranker initialized (device={device})")
    
    def rerank(
        self,
        query: str,
        nodes: List[str],
        graph,
        top_k: int = None
    ) -> List[RerankedResult]:
        """
        Re-rank nodes based on relevance to query
        
        Args:
            query: User query
            nodes: List of node IDs to re-rank
            graph: NetworkX graph (for node attributes)
            top_k: Return only top K (default: all)
        
        Returns:
            List of RerankedResult sorted by score
        """
        logger.info(f"Re-ranking {len(nodes)} nodes")
        
        # Prepare query-document pairs
        pairs = []
        valid_nodes = []
        
        for node_id in nodes:
            if node_id not in graph.nodes:
                continue
            
            attrs = graph.nodes[node_id]
            
            # Create document text
            doc_text = self._create_document_text(attrs)
            pairs.append([query, doc_text])
            valid_nodes.append(node_id)
        
        if not pairs:
            logger.warning("No valid nodes to rerank")
            return []
        
        # Score with cross-encoder
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        
        # Create results
        results = []
        for i, (node_id, score) in enumerate(zip(valid_nodes, scores)):
            results.append(RerankedResult(
                node_id=node_id,
                score=float(score),
                original_rank=i,
                new_rank=-1,  # Will be set after sorting
                score_change=0.0
            ))
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks and score changes
        for new_rank, result in enumerate(results):
            result.new_rank = new_rank
            result.score_change = result.score - (1.0 - result.original_rank / len(results))
        
        # Apply top_k filter
        if top_k:
            results = results[:top_k]
        
        logger.info(f"Re-ranking complete, top score: {results[0].score:.3f}")
        
        return results
    
    def rerank_with_original_scores(
        self,
        query: str,
        nodes_with_scores: List[Tuple[str, float]],
        graph,
        alpha: float = 0.7,
        top_k: int = None
    ) -> List[RerankedResult]:
        """
        Re-rank combining original scores with cross-encoder scores
        
        Args:
            query: User query
            nodes_with_scores: List of (node_id, original_score) tuples
            graph: NetworkX graph
            alpha: Weight for cross-encoder score (1-alpha for original)
            top_k: Return only top K
        
        Returns:
            List of RerankedResult with combined scores
        """
        logger.info(f"Re-ranking {len(nodes_with_scores)} nodes (alpha={alpha})")
        
        nodes = [node_id for node_id, _ in nodes_with_scores]
        original_scores = {node_id: score for node_id, score in nodes_with_scores}
        
        # Get cross-encoder scores
        ce_results = self.rerank(query, nodes, graph, top_k=None)
        
        # Normalize cross-encoder scores to [0, 1]
        ce_scores = {r.node_id: r.score for r in ce_results}
        max_ce = max(ce_scores.values()) if ce_scores else 1.0
        min_ce = min(ce_scores.values()) if ce_scores else 0.0
        ce_range = max_ce - min_ce if max_ce > min_ce else 1.0
        
        ce_scores_norm = {
            node_id: (score - min_ce) / ce_range
            for node_id, score in ce_scores.items()
        }
        
        # Combine scores
        combined_results = []
        for node_id in nodes:
            orig_score = original_scores.get(node_id, 0.0)
            ce_score = ce_scores_norm.get(node_id, 0.0)
            
            combined_score = alpha * ce_score + (1 - alpha) * orig_score
            
            combined_results.append(RerankedResult(
                node_id=node_id,
                score=combined_score,
                original_rank=nodes.index(node_id),
                new_rank=-1,
                score_change=ce_score - orig_score
            ))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for new_rank, result in enumerate(combined_results):
            result.new_rank = new_rank
        
        if top_k:
            combined_results = combined_results[:top_k]
        
        return combined_results
    
    def _create_document_text(self, node_attrs: Dict) -> str:
        """Create text representation for cross-encoder"""
        
        parts = []
        
        # Name and type
        name = node_attrs.get('name', 'unknown')
        node_type = node_attrs.get('type', 'code')
        parts.append(f"{node_type}: {name}")
        
        # Summary
        summary = node_attrs.get('summary', '')
        if summary:
            parts.append(summary)
        
        # Tags
        tags = node_attrs.get('tags', [])
        if tags:
            parts.append(f"Tags: {', '.join(tags[:5])}")
        
        # Code snippet (first 200 chars)
        code = node_attrs.get('code', '')
        if code:
            parts.append(code[:200])
        
        return ' '.join(parts)
    
    def batch_rerank(
        self,
        queries: List[str],
        node_lists: List[List[str]],
        graph,
        top_k: int = 10
    ) -> List[List[RerankedResult]]:
        """
        Re-rank multiple queries in batch
        
        Args:
            queries: List of queries
            node_lists: List of node lists (one per query)
            graph: NetworkX graph
            top_k: Top K per query
        
        Returns:
            List of re-ranked results (one list per query)
        """
        logger.info(f"Batch re-ranking {len(queries)} queries")
        
        results = []
        for query, nodes in zip(queries, node_lists):
            query_results = self.rerank(query, nodes, graph, top_k=top_k)
            results.append(query_results)
        
        return results


class HybridScorer:
    """
    Combine multiple scoring signals
    
    Signals:
    - Vector similarity (bi-encoder)
    - Cross-encoder relevance
    - Graph importance (PageRank)
    - Recency/popularity (if available)
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize hybrid scorer
        
        Args:
            weights: Dictionary of signal weights
        """
        self.weights = weights or {
            'vector': 0.3,
            'cross_encoder': 0.5,
            'graph': 0.2
        }
        
        logger.info(f"Hybrid scorer initialized: {self.weights}")
    
    def score(
        self,
        node_id: str,
        signals: Dict[str, float]
    ) -> float:
        """
        Compute hybrid score
        
        Args:
            node_id: Node ID
            signals: Dictionary of signal_name -> score
        
        Returns:
            Combined score
        """
        score = 0.0
        total_weight = 0.0
        
        for signal_name, weight in self.weights.items():
            if signal_name in signals:
                score += weight * signals[signal_name]
                total_weight += weight
        
        # Normalize by actual weight used
        if total_weight > 0:
            score /= total_weight
        
        return score


# Example usage
if __name__ == "__main__":
    import pickle
    from sentence_transformers import SentenceTransformer
    
    # Load graph
    with open('data/graphs/code_graph_enriched.pkl', 'rb') as f:
        graph = pickle.load(f)
    
    # Get some candidate nodes
    nodes = list(graph.nodes())[:20]
    
    # Initialize reranker
    reranker = Reranker(model_name='fast')
    
    # Test re-ranking
    query = "function that validates email addresses"
    
    results = reranker.rerank(query, nodes, graph, top_k=5)
    
    print(f"\n=== Re-ranking Results ===")
    print(f"Query: {query}\n")
    
    for i, result in enumerate(results, 1):
        node_name = graph.nodes[result.node_id].get('name', 'unknown')
        print(f"{i}. {node_name}")
        print(f"   Score: {result.score:.3f}")
        print(f"   Rank change: {result.original_rank} → {result.new_rank}")
        print()