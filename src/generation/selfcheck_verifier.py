"""
SelfCheckGPT - Complete Implementation
Zero-resource hallucination detection through sampling consistency
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
import re
from dataclasses import dataclass
from loguru import logger
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_client import LangChainClient


@dataclass
class SentenceVerification:
    """Verification result for a single sentence"""
    sentence: str
    consistency_score: float  # 0-1, higher = more consistent
    is_hallucination: bool  # True if likely hallucinated
    supporting_samples: List[int]  # Indices of supporting samples
    evidence: str  # Brief explanation


class SelfCheckGPT:
    """
    Complete SelfCheckGPT implementation
    
    Method: Generate N alternative answers with high temperature,
    then check if original answer's claims appear in samples.
    
    Intuition: If model is certain (not hallucinating), all samples
    will contain similar information. If hallucinating, samples diverge.
    """
    
    def __init__(
        self,
        llm_client: LangChainClient,
        n_samples: int = 5,
        temperature: float = 1.0,
        consistency_threshold: float = 0.5
    ):
        """
        Initialize SelfCheckGPT verifier
        
        Args:
            llm_client: LLM client for generation
            n_samples: Number of alternative samples to generate
            temperature: Sampling temperature (higher = more random)
            consistency_threshold: Threshold for hallucination detection
        """
        self.llm = llm_client
        self.n_samples = n_samples
        self.temperature = temperature
        self.threshold = consistency_threshold
        
        logger.info(f"SelfCheckGPT initialized (n={n_samples}, T={temperature})")
    
    def verify(
        self,
        query: str,
        context: str,
        answer: str
    ) -> List[SentenceVerification]:
        """
        Verify answer for hallucinations
        
        Args:
            query: Original query
            context: Context used for generation
            answer: Generated answer to verify
        
        Returns:
            List of SentenceVerification for each sentence
        """
        logger.info("Running SelfCheckGPT verification")
        
        # Step 1: Split answer into sentences
        sentences = self._split_sentences(answer)
        logger.info(f"Verifying {len(sentences)} sentences")
        
        # Step 2: Generate N alternative samples
        samples = self._generate_samples(query, context)
        logger.info(f"Generated {len(samples)} alternative samples")
        
        # Step 3: Check each sentence
        verifications = []
        for i, sentence in enumerate(sentences):
            verification = self._verify_sentence(
                sentence, 
                samples, 
                i, 
                len(sentences)
            )
            verifications.append(verification)
        
        # Log summary
        hallucinated = sum(1 for v in verifications if v.is_hallucination)
        logger.info(f"Verification complete: {hallucinated}/{len(sentences)} "
                   f"potential hallucinations")
        
        return verifications
    
    def _generate_samples(self, query: str, context: str) -> List[str]:
        """Generate N alternative answers with high temperature"""
        
        samples = []
        
        system_prompt = "You are a code analysis expert. Answer based on the provided context."
        
        user_prompt = f"""Context:
{context}

Question: {query}

Provide a concise answer based only on the context."""
        
        for i in range(self.n_samples):
            try:
                response = self.llm.generate(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,  # High for diversity
                    # max_tokens=500
                )
                samples.append(response.content)
                
            except Exception as e:
                logger.error(f"Sample {i+1} generation failed: {e}")
        
        return samples
    
    def _verify_sentence(
        self,
        sentence: str,
        samples: List[str],
        sentence_idx: int,
        total_sentences: int
    ) -> SentenceVerification:
        """
        Verify a single sentence against samples
        
        Uses LLM to check if sentence information appears in samples
        """
        
        # Build verification prompt
        verification_prompt = f"""You are verifying factual consistency.

**Sentence to verify:**
{sentence}

**Reference samples (generated independently):**
"""
        
        for i, sample in enumerate(samples, 1):
            verification_prompt += f"\nSample {i}:\n{sample}\n"
        
        verification_prompt += """
**Task:**
Does the information in the sentence appear in the reference samples?

Respond with JSON:
{
  "appears_in_samples": [list of sample numbers, 1-5],
  "consistency_score": 0.0 to 1.0,
  "explanation": "brief reasoning"
}

Score guide:
- 1.0: Information clearly in all/most samples
- 0.5: Partially mentioned or implied
- 0.0: Not mentioned or contradicted
"""
        
        try:
            result = self.llm.generate_json(
                prompt=verification_prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "appears_in_samples": {
                            "type": "array",
                            "items": {"type": "integer"}
                        },
                        "consistency_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "explanation": {"type": "string"}
                    },
                    "required": ["appears_in_samples", "consistency_score", "explanation"]
                },
                temperature=0.3  # Lower for consistent verification
            )
            
            consistency_score = result['consistency_score']
            supporting_samples = result['appears_in_samples']
            evidence = result['explanation']
            
        except Exception as e:
            logger.warning(f"Verification failed for sentence {sentence_idx+1}: {e}")
            # Fallback: simple heuristic
            consistency_score = self._heuristic_check(sentence, samples)
            supporting_samples = []
            evidence = "Fallback heuristic check"
        
        # Determine if hallucination
        is_hallucination = consistency_score < self.threshold
        
        return SentenceVerification(
            sentence=sentence,
            consistency_score=consistency_score,
            is_hallucination=is_hallucination,
            supporting_samples=supporting_samples,
            evidence=evidence
        )
    
    def _heuristic_check(self, sentence: str, samples: List[str]) -> float:
        """
        Fallback heuristic: count how many samples contain key terms
        """

        if not samples:
            logger.warning("No samples available for heuristic check. Assuming uncertain (0.5).")
            return 0.5
        # Extract key terms (simple: words longer than 4 chars)
        words = sentence.lower().split()
        key_terms = [w.strip('.,!?;:') for w in words if len(w) > 4]
        
        if not key_terms:
            return 0.5
        
        # Count samples containing key terms
        matches = 0
        for sample in samples:
            sample_lower = sample.lower()
            if any(term in sample_lower for term in key_terms):
                matches += 1
        
        return matches / len(samples)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitter
        # In production, use spacy or nltk
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def get_verification_summary(
        self,
        verifications: List[SentenceVerification]
    ) -> Dict:
        """Get summary statistics of verification"""
        
        total = len(verifications)
        hallucinated = sum(1 for v in verifications if v.is_hallucination)
        
        scores = [v.consistency_score for v in verifications]
        
        return {
            'total_sentences': total,
            'hallucinated_count': hallucinated,
            'hallucination_rate': hallucinated / total if total > 0 else 0,
            'avg_consistency': np.mean(scores) if scores else 0,
            'min_consistency': np.min(scores) if scores else 0,
            'verified': hallucinated == 0
        }
    
    def format_verified_answer(
        self,
        answer: str,
        verifications: List[SentenceVerification],
        include_scores: bool = False
    ) -> str:
        """
        Format answer with verification indicators
        
        Args:
            answer: Original answer
            verifications: Verification results
            include_scores: Include consistency scores
        
        Returns:
            Formatted answer with indicators
        """
        sentences = self._split_sentences(answer)
        
        formatted = []
        for sentence, verification in zip(sentences, verifications):
            if verification.is_hallucination:
                # Mark as potentially unreliable
                formatted.append(f"⚠️ {sentence}")
                if include_scores:
                    formatted.append(f"   (Consistency: {verification.consistency_score:.2f})")
            else:
                formatted.append(f"✓ {sentence}")
                if include_scores:
                    formatted.append(f"   (Consistency: {verification.consistency_score:.2f})")
        
        return '\n'.join(formatted)


class CitationExtractor:
    """
    Extract citations from answer and link to source code
    """
    
    def __init__(self, graph):
        """
        Initialize citation extractor
        
        Args:
            graph: NetworkX graph with code entities
        """
        self.graph = graph
        
        # Build entity name index
        self.entity_index = {
            attrs.get('name', '').lower(): node_id
            for node_id, attrs in graph.nodes(data=True)
        }
        
        logger.info(f"Citation extractor initialized ({len(self.entity_index)} entities)")
    
    def extract_citations(self, answer: str, context_nodes: List[str]) -> Dict:
        """
        Extract citations from answer
        
        Args:
            answer: Generated answer
            context_nodes: Node IDs that were in context
        
        Returns:
            Dictionary mapping sentences to cited entities
        """
        citations = {}
        
        sentences = self._split_sentences(answer)
        
        for i, sentence in enumerate(sentences):
            cited_entities = self._find_entities_in_sentence(sentence)
            
            # Filter to only entities that were in context
            cited_entities = [
                entity for entity in cited_entities
                if entity in context_nodes
            ]
            
            if cited_entities:
                citations[i] = {
                    'sentence': sentence,
                    'entities': cited_entities
                }
        
        return citations
    
    def _find_entities_in_sentence(self, sentence: str) -> List[str]:
        """Find entity mentions in sentence"""
        
        sentence_lower = sentence.lower()
        mentioned_entities = []
        
        for entity_name, node_id in self.entity_index.items():
            if entity_name in sentence_lower:
                mentioned_entities.append(node_id)
        
        return mentioned_entities
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def format_with_citations(
        self,
        answer: str,
        citations: Dict,
        graph
    ) -> str:
        """Format answer with inline citations"""
        
        sentences = self._split_sentences(answer)
        formatted = []
        
        for i, sentence in enumerate(sentences):
            if i in citations:
                entities = citations[i]['entities']
                entity_names = [
                    graph.nodes[node_id].get('name', 'unknown')
                    for node_id in entities[:3]  # Top 3
                ]
                citation_str = f" [{', '.join(entity_names)}]"
                formatted.append(sentence + citation_str)
            else:
                formatted.append(sentence)
        
        return ' '.join(formatted)


# Example usage
if __name__ == "__main__":
    from utils.llm_client import LangChainClient
    
    # Initialize
    llm = LangChainClient(temperature=0.3)
    verifier = SelfCheckGPT(llm, n_samples=3)  # 3 for demo
    
    # Test data
    query = "How does validation work?"
    context = """
    The validate_email function checks email format using regex.
    The check_user_input function validates all user inputs.
    """
    
    answer = """The system validates data through multiple functions. 
    The validate_email function uses regex patterns to check email format. 
    The check_user_input function validates all user inputs including email, 
    phone numbers, and addresses. 
    The system also uses machine learning for advanced validation."""
    
    # Verify
    print("Running SelfCheckGPT verification...\n")
    verifications = verifier.verify(query, context, answer)
    
    # Show results
    for i, v in enumerate(verifications, 1):
        print(f"{i}. {v.sentence}")
        print(f"   Consistency: {v.consistency_score:.2f}")
        print(f"   Hallucination: {v.is_hallucination}")
        print(f"   Evidence: {v.evidence}\n")
    
    # Summary
    summary = verifier.get_verification_summary(verifications)
    print(f"Summary:")
    print(f"  Total: {summary['total_sentences']}")
    print(f"  Hallucinated: {summary['hallucinated_count']}")
    print(f"  Rate: {summary['hallucination_rate']:.1%}")
    print(f"  Avg consistency: {summary['avg_consistency']:.2f}")

    # Show the "Traffic Light" formatted output
    print("\n--- Verified Output (Traffic Light) ---")
    formatted_text = verifier.format_verified_answer(
        answer, 
        verifications, 
        include_scores=True
    )
    print(formatted_text)