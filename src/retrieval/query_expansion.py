"""
Query Expansion - Generate query variations for better recall
Uses LLM to create semantically similar queries
"""

import sys
from pathlib import Path
from typing import List, Dict, Set
from dataclasses import dataclass
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_client import LangChainClient
from pydantic import BaseModel,Field

import nltk
from nltk.corpus import stopwords

class QueryVariations(BaseModel):
    """Structure For LLM Output """
    queries: List[str] = Field(description="List of alternative search queries")

@dataclass
class ExpandedQuery:
    """Expanded query with variations"""
    original: str
    expansions: List[str]
    strategy: str


class QueryExpander:
    """
    Generate query variations to improve recall
    
    Strategies:
    1. Synonym replacement
    2. Paraphrasing
    3. Specificity variation (more/less specific)
    4. Multi-perspective (different angles)
    """
    
    def __init__(self, llm_client: LangChainClient):
        """
        Initialize query expander
        
        Args:
            llm_client: LLM client for expansion
        """
        self.llm = llm_client
        logger.info("Query expander initialized")
    
    def expand(
        self,
        query: str,
        num_expansions: int = 3,
        strategy: str = 'paraphrase'
    ) -> ExpandedQuery:
        """
        Expand query with variations
        
        Args:
            query: Original query
            num_expansions: Number of variations to generate
            strategy: Expansion strategy
        
        Returns:
            ExpandedQuery with variations
        """
        logger.info(f"Expanding query: {query}")
        
        if strategy == 'paraphrase':
            expansions = self._expand_paraphrase(query, num_expansions)
        elif strategy == 'specificity':
            expansions = self._expand_specificity(query, num_expansions)
        elif strategy == 'perspective':
            expansions = self._expand_perspective(query, num_expansions)
        else:
            expansions = self._expand_paraphrase(query, num_expansions)
        
        logger.info(f"Generated {len(expansions)} query variations")
        
        return ExpandedQuery(
            original=query,
            expansions=expansions,
            strategy=strategy
        )
    
    def _expand_paraphrase(self, query: str, num: int) -> List[str]:
        """Generate paraphrased variations"""
        
        prompt = f"""Generate {num} different ways to phrase this code search query.
Keep the same meaning but use different words and structures.

Original query: {query}

Requirements:
- Maintain the technical meaning
- Use code-related terminology
- Keep queries concise (under 15 words)
- Make them diverse

Provide as a JSON array of strings."""
        
        try:
            result = self.llm.generate_json(
                prompt=prompt,
                schema=QueryVariations
            )
            return result['queries']
        except:
            # Fallback: simple variations
            return [
                query.replace("function", "method"),
                query.replace("how does", "what is the implementation of"),
                f"code for {query}"
            ][:num]
    
    def _expand_specificity(self, query: str, num: int) -> List[str]:
        """Generate more/less specific variations"""
        
        prompt = f"""For this code search query, generate variations at different specificity levels.

Original query: {query}

Generate {num} variations:
1. More specific (add technical details)
2. Less specific (broader/higher-level)
3. Alternative angle (same specificity, different focus)

Provide as JSON array."""
        
        try:
            result = self.llm.generate_json(
                prompt=prompt,
                schema=QueryVariations
            )
            return result['queries'][:num]
        except:
            return [query] * num
    
    def _expand_perspective(self, query: str, num: int) -> List[str]:
        """Generate different perspective variations"""
        
        prompt = f"""Rephrase this code query from different perspectives.

Original: {query}

Generate {num} variations:
- From a developer's perspective (implementation)
- From an architect's perspective (design)
- From a user's perspective (behavior)

Keep them technical and code-focused.

JSON array format."""
        
        try:
            result = self.llm.generate_json(prompt=prompt,schema = QueryVariations)
            if 'queries' in result:
                return result['queries'][:num]
            return [query] * num
        except:
            return [query] * num
    
    def expand_multi_strategy(
        self,
        query: str,
        num_per_strategy: int = 2
    ) -> List[str]:
        """
        Expand using multiple strategies
        
        Args:
            query: Original query
            num_per_strategy: Expansions per strategy
        
        Returns:
            List of all expansions
        """
        all_expansions = [query]  # Include original
        
        strategies = ['paraphrase', 'specificity', 'perspective']
        
        for strategy in strategies:
            try:
                expanded = self.expand(query, num_per_strategy, strategy)
                all_expansions.extend(expanded.expansions)
            except Exception as e:
                logger.error(f"Expansion strategy {strategy} failed: {e}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for q in all_expansions:
            q_lower = q.lower()
            if q_lower not in seen:
                seen.add(q_lower)
                unique.append(q)
        
        return unique


class QueryRewriter:
    """
    Rewrite queries to be more effective for code search
    
    - Remove filler words
    - Add code-specific keywords
    - Normalize terminology
    """
    
    def __init__(self):
        """Initialize query rewriter"""

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        self.stop_words = set(stopwords.words('english'))
        
        code_keywords = {
            'if', 'else', 'while', 'for', 'return', 'yield', 
            'break', 'continue', 'try', 'except', 'raise', 
            'class', 'def', 'from', 'import', 'as', 'is', 
            'not', 'and', 'or', 'in', 'with'
        }

        self.stop_words = self.stop_words - code_keywords
        # Code-specific term mappings
        self.term_mappings = {
            'method': 'function',
            'procedure': 'function',
            'routine': 'function',
            'subroutine': 'function',
            'func': 'function',
            'params': 'parameters',
            'args': 'arguments'
        }
        
        logger.info(f"Query rewriter initialized with {len(self.stop_words)} NLTK stopwords")
    
    def rewrite(self, query: str) -> str:
        """
        Rewrite query for better code search
        
        Args:
            query: Original query
        
        Returns:
            Rewritten query
        """
        # Lowercase
        query = query.lower()
        
        # Remove stop words
        words = query.split()
        words = [w for w in words if w not in self.stop_words]
        
        # Apply term mappings
        words = [self.term_mappings.get(w, w) for w in words]
        
        # Rejoin
        rewritten = ' '.join(words)
        
        logger.debug(f"Rewrote: '{query}' → '{rewritten}'")
        
        return rewritten
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract key search terms from query
        
        Args:
            query: Query string
        
        Returns:
            List of keywords
        """
        # Remove stop words and common phrases
        words = query.lower().split()
        keywords = [
            w for w in words 
            if w not in self.stop_words and len(w) > 2
        ]
        
        return keywords


# Example usage
if __name__ == "__main__":
    from utils.llm_client import LangChainClient
    
    # Initialize
    llm = LangChainClient()
    expander = QueryExpander(llm_client=llm)
    rewriter = QueryRewriter()
    
    # Test query
    # query = "How does the function that validates user email addresses work?"
    query = "tell me about fetch_password_function ?"
    
    print(f"Original query: {query}\n")
    
    # Rewrite
    rewritten = rewriter.rewrite(query)
    print(f"Rewritten: {rewritten}\n")
    
    # Extract keywords
    keywords = rewriter.extract_keywords(query)
    print(f"Keywords: {', '.join(keywords)}\n")
    
    # Expand
    expanded = expander.expand(query, num_expansions=3)
    
    print(f"Expansions ({expanded.strategy}):")
    for i, expansion in enumerate(expanded.expansions, 1):
        print(f"  {i}. {expansion}")
    
    # Multi-strategy
    print(f"\nMulti-strategy expansion:")
    all_expansions = expander.expand_multi_strategy(query, num_per_strategy=2)
    for i, exp in enumerate(all_expansions, 1):
        print(f"  {i}. {exp}")