"""
Semantic Enrichment - Add LLM-generated summaries and metadata to graph nodes
Uses latest LLM APIs with structured output and batch processing
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import networkx as nx
from tqdm import tqdm
from loguru import logger
import json
# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pydantic import BaseModel, Field

from utils.llm_client import LangChainClient


class EnrichedNode(BaseModel):
    """Enriched node data"""
    node_id: str
    summary: str = Field(description="One-sentence summary of what this code does")
    description: str = Field(description="2-3 sentence detailed explanation")
    tags: List[str] = Field(description="Domain tags (e.g., authentication, database)")
    complexity: str = Field(description="Code complexity: low, medium, or high")
    purpose: str = Field(description="Primary purpose or role in the system")


class SemanticEnricher:
    """
    Enrich graph nodes with LLM-generated semantic information
    
    Features:
    - Structured summaries
    - Domain tagging
    - Complexity assessment
    - Batch processing
    - Progress tracking
    """
    
    def __init__(self,
        llm_client: LangChainClient,
        batch_size: int = 10,
        skip_types: List[str] = None):
        """
        Initialize semantic enricher
        
        Args:
            llm_client: LLM client instance (default: create new)
            batch_size: Number of nodes to process in parallel
            skip_types: Node types to skip (e.g., ['import', 'file'])
        """
        self.llm = llm_client
        self.batch_size = batch_size
        self.skip_types = skip_types or ['import','file']
        
        logger.info(f"Semantic enricher initialized (batch_size={batch_size})")
    
    def enrich_graph(
        self,
        graph: nx.DiGraph,
        skip_existing: bool = True,
        save_path: str = None
    ) -> nx.DiGraph:
        """
        Enrich all nodes in graph
        
        Args:
            graph: Input NetworkX graph
            skip_existing: Skip nodes that already have summaries
        
        Returns:
            Enriched graph
        """
        # Filter nodes to enrich
        nodes_to_enrich = []
        for node_id, attrs in graph.nodes(data=True):
            node_type = attrs.get('type', 'unknown')
            
            # Skip certain types
            if node_type in self.skip_types:
                continue
            
            # Skip if already enriched
            if skip_existing and attrs.get('summary'):
                continue
            
            nodes_to_enrich.append((node_id, attrs))
        
        logger.info(f"Enriching {len(nodes_to_enrich)} nodes...")
        
        # Process in batches
        enriched_count = 0
        failed_count = 0
        
        for i in tqdm(range(0, len(nodes_to_enrich), self.batch_size)):
            batch = nodes_to_enrich[i:i + self.batch_size]
            
            for node_id, attrs in batch:
                try:
                    enriched = self._enrich_node(node_id, attrs)
                    
                    # Update graph
                    graph.nodes[node_id]['summary'] = enriched.summary
                    graph.nodes[node_id]['description'] = enriched.description
                    graph.nodes[node_id]['tags'] = enriched.tags
                    graph.nodes[node_id]['complexity'] = enriched.complexity
                    graph.nodes[node_id]['purpose'] = enriched.purpose
                    graph.nodes[node_id]['enriched'] = True
                    
                    enriched_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to enrich {node_id}: {e}")
                    failed_count += 1

            if save_path:
                import pickle
                with open(save_path, 'wb') as f:
                    pickle.dump(graph, f)
        logger.success(
            f"Enrichment complete: {enriched_count} enriched, {failed_count} failed"
        )
        
        return graph
    
    def _enrich_node(self, node_id: str, attrs: Dict) -> EnrichedNode:
        """Enrich a single node"""
        
        # Build prompt
        prompt_text = self._build_enrichment_prompt(node_id, attrs)
        
        # Generate structured response
        try:
            response = self.llm.generate_json(
                prompt=prompt_text,
                schema=EnrichedNode
            )
            
            return response
        
        except Exception as e:
            logger.warning(f"Structured output failed, using fallback for {node_id}")
            return self._enrich_node_fallback(node_id, attrs)
    
    def _build_enrichment_prompt(self, node_id: str, attrs: Dict) -> str:
        """Build prompt for node enrichment"""
        
        node_type = attrs.get('type', 'unknown')
        name = attrs.get('name', 'unknown')
        code = attrs.get('code', '')
        docstring = attrs.get('docstring', '')
        file_path = attrs.get('file_path', '')
        
        prompt = f"""Analyze this {node_type} and provide structured information.

**Code Entity:**
- Name: {name}
- Type: {node_type}
- File: {Path(file_path).name if file_path else 'unknown'}

**Code:**
```
{code[:800]}  {'...' if len(code) > 800 else ''}
```
"""
        
        if docstring:
            prompt += f"""
**Docstring:**
{docstring}
"""
        
        prompt += """
**Task:**
Provide a JSON response with:
1. **summary**: One clear sentence explaining what this code does
2. **description**: 2-3 sentences with more detail about functionality
3. **tags**: List of relevant domain tags (e.g., ["authentication", "database", "validation"])
4. **complexity**: Rate as "low", "medium", or "high"
5. **purpose**: The main purpose/role in the system

Focus on clarity and accuracy. Base your analysis only on the provided code.
"""
        
        return prompt
    
    def _enrich_node_fallback(self, node_id: str, attrs: Dict) -> EnrichedNode:
        """Fallback enrichment without structured output"""
        
        prompt = self._build_enrichment_prompt(node_id, attrs)
        prompt += "\nProvide your response as JSON."
        
        response = self.llm.generate(
            prompt=prompt
        )
        
        # Try to parse JSON from response
        try:
            # Extract JSON if wrapped in markdown
            content = response.content
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            
            data = json.loads(content.strip())
            
            return EnrichedNode(
                node_id=node_id,
                summary=data.get('summary', 'No summary available'),
                description=data.get('description', ''),
                tags=data.get('tags', []),
                complexity=data.get('complexity', 'medium'),
                purpose=data.get('purpose', '')
            )
        
        except Exception as e:
            logger.error(f"Failed to parse fallback response: {e}")
            # Return minimal enrichment
            return EnrichedNode(
                node_id=node_id,
                summary=f"{attrs.get('type', 'code')} named {attrs.get('name', 'unknown')}",
                description="",
                tags=[],
                complexity="unknown",
                purpose=""
            )
    
    def enrich_single(self, node_id: str, attrs: Dict) -> EnrichedNode:
        """
        Enrich a single node (useful for testing)
        
        Args:
            node_id: Node identifier
            attrs: Node attributes
        
        Returns:
            EnrichedNode object
        """
        return self._enrich_node(node_id, attrs)
    
    def get_enrichment_stats(self, graph: nx.DiGraph) -> Dict:
        """Get statistics about enrichment coverage"""
        
        total_nodes = graph.number_of_nodes()
        enriched_nodes = sum(
            1 for _, attrs in graph.nodes(data=True)
            if attrs.get('enriched', False)
        )
        
        # Tag distribution
        all_tags = []
        for _, attrs in graph.nodes(data=True):
            all_tags.extend(attrs.get('tags', []))
        
        from collections import Counter
        tag_counts = Counter(all_tags)
        
        return {
            'total_nodes': total_nodes,
            'enriched_nodes': enriched_nodes,
            'enrichment_rate': enriched_nodes / max(total_nodes, 1),
            'top_tags': tag_counts.most_common(10),
            'total_unique_tags': len(tag_counts)
        }


def enrich_from_pickle(
    input_path: str,
    output_path: str = None,
    llm_provider: str = "google"
) -> nx.DiGraph:
    """
    Enrich graph from pickle file
    
    Args:
        input_path: Input graph pickle
        output_path: Output path (default: add _enriched suffix)
        llm_provider: 'openai' or 'anthropic'
    
    Returns:
        Enriched graph
    """
    import pickle
    import os
    
    if output_path is None:
        final_path = input_path.replace('.pkl', '_enriched.pkl')
    else:
        final_path = output_path
    
    if os.path.exists(final_path):
        logger.info(f"Found existing progress at {final_path}. Resuming...")
        load_path = final_path
    else:
        logger.info(f"Starting fresh from {input_path}")
        load_path = input_path

    # Load graph
    logger.info(f"Loading graph from {load_path}")
    with open(load_path, 'rb') as f:
        graph = pickle.load(f)
    
    logger.info(f"Loaded graph: {graph.number_of_nodes()} nodes")
    
    # Enrich
    llm = LangChainClient(provider=llm_provider)
    enricher = SemanticEnricher(llm_client=llm)
    enriched_graph = enricher.enrich_graph(graph,save_path=final_path)
    
    # Save
    with open(final_path, 'wb') as f:
        pickle.dump(enriched_graph, f)
    
    logger.success(f"Enriched graph saved to {output_path}")
    
    # Show stats
    stats = enricher.get_enrichment_stats(enriched_graph)
    logger.info(f"\nEnrichment Stats:")
    logger.info(f"  Enriched: {stats['enriched_nodes']}/{stats['total_nodes']} "
                f"({stats['enrichment_rate']:.1%})")
    logger.info(f"  Unique tags: {stats['total_unique_tags']}")
    logger.info(f"  Top tags: {', '.join(t[0] for t in stats['top_tags'][:5])}")
    
    return enriched_graph


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python semantic_enrichment.py <graph_pickle>")
        print("Example: python semantic_enrichment.py data/graphs/code_graph.pkl")
        sys.exit(1)
    
    graph_path = sys.argv[1]
    provider = sys.argv[2] if len(sys.argv) > 2 else None
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    enriched = enrich_from_pickle(graph_path, llm_provider=provider)
    
    print("\n=== Sample Enriched Node ===")
    # Show one enriched node
    for node_id, attrs in enriched.nodes(data=True):
        if attrs.get('enriched'):
            print(f"\nNode: {attrs['name']} ({attrs['type']})")
            print(f"Summary: {attrs.get('summary', 'N/A')}")
            print(f"Tags: {', '.join(attrs.get('tags', []))}")
            print(f"Complexity: {attrs.get('complexity', 'N/A')}")
            break