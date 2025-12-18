"""
Evaluation metrics for GraphRAG system
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import json
from pathlib import Path
from loguru import logger

@dataclass
class EvaluationResult:
    """Single evaluation result"""
    query: str
    expected_answer: str
    generated_answer: str
    retrieved_nodes: List[str]
    expected_nodes: List[str]
    
    # Metrics
    retrieval_precision: float
    retrieval_recall: float
    retrieval_f1: float
    answer_relevance: float
    
    def to_dict(self) -> Dict:
        return {
            'query': self.query,
            'expected_answer': self.expected_answer,
            'generated_answer': self.generated_answer,
            'retrieved_nodes': self.retrieved_nodes,
            'expected_nodes': self.expected_nodes,
            'metrics': {
                'retrieval_precision': self.retrieval_precision,
                'retrieval_recall': self.retrieval_recall,
                'retrieval_f1': self.retrieval_f1,
                'answer_relevance': self.answer_relevance
            }
        }

class GraphRAGEvaluator:
    """Evaluate GraphRAG system performance"""
    
    def __init__(self):
        self.results: List[EvaluationResult] = []
    
    def evaluate_retrieval(
        self,
        retrieved_nodes: List[str],
        expected_nodes: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality
        
        Metrics:
        - Precision: How many retrieved nodes are relevant?
        - Recall: How many relevant nodes were retrieved?
        - F1: Harmonic mean of precision and recall
        """
        if not retrieved_nodes and not expected_nodes:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        
        if not retrieved_nodes:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        if not expected_nodes:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Calculate overlap
        retrieved_set = set(retrieved_nodes)
        expected_set = set(expected_nodes)
        
        true_positives = len(retrieved_set & expected_set)
        
        precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
        recall = true_positives / len(expected_set) if expected_set else 0.0
        
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3)
        }
    
    def evaluate_answer_relevance(
        self,
        query: str,
        generated_answer: str,
        expected_answer: str
    ) -> float:
        """
        Evaluate answer relevance (simplified)
        
        In production, use:
        - BLEU/ROUGE scores
        - Semantic similarity (embedding-based)
        - LLM-as-judge
        """
        # Simplified: Check for keyword overlap
        query_words = set(query.lower().split())
        generated_words = set(generated_answer.lower().split())
        expected_words = set(expected_answer.lower().split())
        
        # How many expected keywords appear in generated answer?
        keyword_overlap = len(expected_words & generated_words) / len(expected_words) if expected_words else 0.0
        
        return round(keyword_overlap, 3)
    
    def add_evaluation(
        self,
        query: str,
        expected_answer: str,
        generated_answer: str,
        retrieved_nodes: List[str],
        expected_nodes: List[str]
    ):
        """Add a single evaluation case"""
        
        retrieval_metrics = self.evaluate_retrieval(retrieved_nodes, expected_nodes)
        answer_relevance = self.evaluate_answer_relevance(
            query, generated_answer, expected_answer
        )
        
        result = EvaluationResult(
            query=query,
            expected_answer=expected_answer,
            generated_answer=generated_answer,
            retrieved_nodes=retrieved_nodes,
            expected_nodes=expected_nodes,
            retrieval_precision=retrieval_metrics['precision'],
            retrieval_recall=retrieval_metrics['recall'],
            retrieval_f1=retrieval_metrics['f1'],
            answer_relevance=answer_relevance
        )
        
        self.results.append(result)
    
    def get_aggregate_metrics(self) -> Dict[str, float]:
        """Calculate aggregate metrics across all evaluations"""
        
        if not self.results:
            return {}
        
        metrics = {
            'avg_retrieval_precision': sum(r.retrieval_precision for r in self.results) / len(self.results),
            'avg_retrieval_recall': sum(r.retrieval_recall for r in self.results) / len(self.results),
            'avg_retrieval_f1': sum(r.retrieval_f1 for r in self.results) / len(self.results),
            'avg_answer_relevance': sum(r.answer_relevance for r in self.results) / len(self.results)
        }
        
        return {k: round(v, 3) for k, v in metrics.items()}
    
    def save_results(self, output_path: str):
        """Save evaluation results to JSON"""
        
        data = {
            'total_evaluations': len(self.results),
            'aggregate_metrics': self.get_aggregate_metrics(),
            'individual_results': [r.to_dict() for r in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_path}")
    
    def print_report(self):
        """Print evaluation report"""
        
        print("\n" + "="*80)
        print("GRAPHRAG EVALUATION REPORT")
        print("="*80)
        
        if not self.results:
            print("No evaluation results available")
            return
        
        metrics = self.get_aggregate_metrics()
        
        print(f"\nTotal Evaluations: {len(self.results)}")
        print("\nAggregate Metrics:")
        print(f"  Retrieval Precision: {metrics['avg_retrieval_precision']:.1%}")
        print(f"  Retrieval Recall:    {metrics['avg_retrieval_recall']:.1%}")
        print(f"  Retrieval F1:        {metrics['avg_retrieval_f1']:.1%}")
        print(f"  Answer Relevance:    {metrics['avg_answer_relevance']:.1%}")
        
        print("\nPer-Query Results:")
        print("-"*80)
        
        for i, result in enumerate(self.results, 1):
            print(f"\n{i}. {result.query}")
            print(f"   Precision: {result.retrieval_precision:.1%} | "
                  f"Recall: {result.retrieval_recall:.1%} | "
                  f"F1: {result.retrieval_f1:.1%}")
            print(f"   Answer Relevance: {result.answer_relevance:.1%}")
        
        print("\n" + "="*80 + "\n")


# Example ground truth data
EXAMPLE_EVALUATION_DATA = [
    {
        "query": "How does authentication work?",
        "expected_nodes": ["authenticate_user", "verify_token", "check_credentials"],
        "expected_answer": "Authentication is handled through token verification and credential checking."
    },
    {
        "query": "What functions handle file uploads?",
        "expected_nodes": ["upload_file", "validate_file", "save_to_storage"],
        "expected_answer": "File uploads are processed through validation and storage functions."
    },
    {
        "query": "Show me the database connection logic",
        "expected_nodes": ["create_connection", "execute_query", "close_connection"],
        "expected_answer": "Database connections are managed through create, execute, and close operations."
    }
]


def create_ground_truth_dataset(repo_name: str) -> str:
    """
    Create a ground truth dataset for evaluation
    
    Returns path to created JSON file
    """
    output_path = f"data/evaluation/{repo_name}_ground_truth.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(EXAMPLE_EVALUATION_DATA, f, indent=2)
    
    logger.info(f"Ground truth dataset created: {output_path}")
    return output_path