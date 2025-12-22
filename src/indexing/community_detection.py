"""
Community Detection - Find natural module boundaries in code graphs
Uses Louvain algorithm for hierarchical clustering
"""

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import networkx as nx
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from loguru import logger
import json
import pickle

# Community detection algorithms
import community as community_louvain  # python-louvain
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm_client import LangChainClient


@dataclass
class Community:
    """Represents a detected community"""
    id: int
    level: int  # Hierarchy level (0=finest, higher=coarser)
    nodes: Set[str]
    summary: str = ""
    description: str = ""
    purpose: str = ""
    key_entities: List[str] = None
    subcommunities: List[int] = None
    
    def __post_init__(self):
        if self.key_entities is None:
            self.key_entities = []
        if self.subcommunities is None:
            self.subcommunities = []


class CommunityDetector:
    """
    Detect and summarize code communities
    
    Features:
    - Louvain algorithm for modularity optimization
    - Hierarchical community structure
    - LLM-generated community summaries
    - Multi-level granularity
    """
    
    def __init__(
        self,
        resolution: float = 1.0,
        min_community_size: int = 3,
        llm_client: LangChainClient = None
    ):
        """
        Initialize community detector
        
        Args:
            resolution: Louvain resolution parameter (higher=more communities)
            min_community_size: Minimum nodes per community
            llm_client: LLM for generating summaries
        """
        self.resolution = resolution
        self.min_community_size = min_community_size
        self.llm = llm_client or LangChainClient()        
        self.communities: Dict[int, Community] = {}
        self.hierarchy: Dict[int, List[int]] = {}  # level -> community_ids
        
        logger.info(f"Community detector initialized (resolution={resolution})")
    
    def detect_communities(
        self,
        graph: nx.DiGraph,
        hierarchical: bool = True
    ) -> Dict[int, Community]:
        """
        Detect communities in graph
        
        Args:
            graph: Input NetworkX graph
            hierarchical: Generate multiple hierarchy levels
        
        Returns:
            Dictionary of community_id -> Community
        """
        logger.info(f"Detecting communities in graph ({graph.number_of_nodes()} nodes)")
        
        # Convert to undirected for community detection
        undirected = graph.to_undirected()
        
        if hierarchical:
            # Get hierarchical partition
            dendogram = community_louvain.generate_dendrogram(
                undirected,
                resolution=self.resolution,
                random_state=42
            )
            
            # Process each level
            logger.info(f"Found {len(dendogram)} hierarchy levels")
            
            community_id = 0
            for level in range(len(dendogram)):
                partition = community_louvain.partition_at_level(dendogram, level)
                
                # Group nodes by community
                level_communities = defaultdict(set)
                for node, comm_id in partition.items():
                    level_communities[comm_id].add(node)
                
                # Create Community objects
                level_comm_ids = []
                for local_id, nodes in level_communities.items():
                    if len(nodes) >= self.min_community_size:
                        comm = Community(
                            id=community_id,
                            level=level,
                            nodes=nodes
                        )
                        self.communities[community_id] = comm
                        level_comm_ids.append(community_id)
                        community_id += 1
                
                self.hierarchy[level] = level_comm_ids
                logger.info(f"  Level {level}: {len(level_comm_ids)} communities")
        
        else:
            # Single level partition
            partition = community_louvain.best_partition(
                undirected,
                resolution=self.resolution,
                random_state=42
            )
            
            # Group nodes by community
            communities_dict = defaultdict(set)
            for node, comm_id in partition.items():
                communities_dict[comm_id].add(node)
            
            # Create Community objects
            community_id = 0
            for local_id, nodes in communities_dict.items():
                if len(nodes) >= self.min_community_size:
                    comm = Community(
                        id=community_id,
                        level=0,
                        nodes=nodes
                    )
                    self.communities[community_id] = comm
                    community_id += 1
            
            self.hierarchy[0] = list(self.communities.keys())
            logger.info(f"Detected {len(self.communities)} communities")
        
        # Calculate community statistics
        self._calculate_statistics(graph)
        
        return self.communities
    
    def _calculate_statistics(self, graph: nx.DiGraph):
        """Calculate statistics for each community"""
        
        for comm_id, community in self.communities.items():
            # Get node types
            node_types = []
            node_names = []
            
            for node_id in community.nodes:
                if node_id in graph.nodes:
                    attrs = graph.nodes[node_id]
                    node_types.append(attrs.get('type', 'unknown'))
                    node_names.append(attrs.get('name', 'unknown'))
            
            # Find most important entities (by degree)
            subgraph = graph.subgraph(community.nodes)
            degrees = dict(subgraph.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            
            community.key_entities = [
                graph.nodes[node_id].get('name', node_id)
                for node_id, _ in top_nodes
                if node_id in graph.nodes
            ]
    
    def summarize_communities(
        self,
        graph: nx.DiGraph,
        batch_size: int = 5
    ):
        """
        Generate LLM summaries for all communities
        
        Args:
            graph: Input graph with node attributes
            batch_size: Communities to process in parallel
        """
        logger.info(f"Generating summaries for {len(self.communities)} communities")
        
        for comm_id, community in tqdm(self.communities.items(), desc="Summarizing"):
            try:
                summary = self._summarize_community(community, graph)
                community.summary = summary['summary']
                community.description = summary['description']
                community.purpose = summary['purpose']
            except Exception as e:
                logger.error(f"Failed to summarize community {comm_id}: {e}")
                community.summary = f"Community with {len(community.nodes)} entities"
        
        logger.success("Community summarization complete")
    
    def _summarize_community(
        self,
        community: Community,
        graph: nx.DiGraph
    ) -> Dict:
        """Generate summary for a single community"""
        
        # Gather community information
        node_info = []
        for node_id in list(community.nodes)[:20]:  # Limit to 20 for context
            if node_id in graph.nodes:
                attrs = graph.nodes[node_id]
                info = {
                    'name': attrs.get('name', 'unknown'),
                    'type': attrs.get('type', 'unknown'),
                    'summary': attrs.get('summary', '')
                }
                node_info.append(info)
        
        # Build prompt
        prompt = self._build_community_summary_prompt(community, node_info)
        
        # JSON schema for structured output
        schema = {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "One sentence describing the module's purpose"
                },
                "description": {
                    "type": "string",
                    "description": "2-3 sentences with architectural details"
                },
                "purpose": {
                    "type": "string",
                    "description": "The main responsibility of this module"
                }
            },
            "required": ["summary", "description", "purpose"]
        }
        
        try:
            result = self.llm.generate_json(prompt=prompt, schema=schema)
            return result
        except:
            # Fallback without structured output
            response = self.llm.generate(prompt=prompt + "\nProvide JSON.")
            import json
            content = response.content
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            return json.loads(content.strip())
    
    def _build_community_summary_prompt(
        self,
        community: Community,
        node_info: List[Dict]
    ) -> str:
        """Build prompt for community summarization"""
        
        prompt = f"""Analyze this code module and provide a high-level summary.

**Module Information:**
- Size: {len(community.nodes)} entities
- Level: {community.level} (0=detailed, higher=abstract)
- Key entities: {', '.join(community.key_entities[:5])}

**Entities in Module:**
"""
        
        for info in node_info[:15]:
            prompt += f"\n- {info['type']}: {info['name']}"
            if info['summary']:
                prompt += f" - {info['summary'][:80]}"
        
        if len(node_info) > 15:
            prompt += f"\n... and {len(node_info) - 15} more entities"
        
        prompt += """

**Task:**
Provide a JSON response with:
1. **summary**: One sentence describing what this module does
2. **description**: 2-3 sentences explaining the module's architecture and key components
3. **purpose**: The main responsibility or role of this module in the system

Focus on the MODULE LEVEL, not individual functions. Think about:
- What is the shared purpose of these entities?
- What domain or feature does this module represent?
- How does it fit into the overall architecture?
"""
        
        return prompt
    
    def get_community_for_node(self, node_id: str, level: int = 0) -> int:
        """Get community ID for a node at specific hierarchy level"""
        
        for comm_id in self.hierarchy.get(level, []):
            if node_id in self.communities[comm_id].nodes:
                return comm_id
        return None
    
    def get_community_summary(self, comm_id: int) -> str:
        """Get formatted summary for a community"""
        
        if comm_id not in self.communities:
            return "Community not found"
        
        comm = self.communities[comm_id]
        
        summary = f"**{comm.summary}**\n\n"
        summary += f"{comm.description}\n\n"
        summary += f"*Purpose*: {comm.purpose}\n\n"
        summary += f"*Key entities*: {', '.join(comm.key_entities[:5])}\n"
        summary += f"*Size*: {len(comm.nodes)} entities"
        
        return summary
    
    def get_statistics(self) -> Dict:
        """Get community detection statistics"""
        
        stats = {
            'total_communities': len(self.communities),
            'hierarchy_levels': len(self.hierarchy),
            'communities_per_level': {
                level: len(comm_ids)
                for level, comm_ids in self.hierarchy.items()
            },
            'avg_community_size': np.mean([
                len(c.nodes) for c in self.communities.values()
            ]),
            'size_distribution': {}
        }
        
        # Size distribution
        sizes = [len(c.nodes) for c in self.communities.values()]
        stats['size_distribution'] = {
            'min': min(sizes) if sizes else 0,
            'max': max(sizes) if sizes else 0,
            'median': np.median(sizes) if sizes else 0
        }
        
        return stats
    
    def save(self, output_path: str):
        """Save communities to file"""
        
        data = {
            'communities': {
                comm_id: {
                    'id': comm.id,
                    'level': comm.level,
                    'nodes': list(comm.nodes),
                    'summary': comm.summary,
                    'description': comm.description,
                    'purpose': comm.purpose,
                    'key_entities': comm.key_entities
                }
                for comm_id, comm in self.communities.items()
            },
            'hierarchy': {
                level: comm_ids
                for level, comm_ids in self.hierarchy.items()
            },
            'config': {
                'resolution': self.resolution,
                'min_community_size': self.min_community_size
            }
        }
        
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.success(f"Communities saved to {output_path}")
    
    @classmethod
    def load(cls, input_path: str) -> 'CommunityDetector':
        """Load communities from file"""
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Recreate detector
        detector = cls(
            resolution=data['config']['resolution'],
            min_community_size=data['config']['min_community_size']
        )
        
        # Restore communities
        for comm_id_str, comm_data in data['communities'].items():
            comm_id = int(comm_id_str)
            detector.communities[comm_id] = Community(
                id=comm_data['id'],
                level=comm_data['level'],
                nodes=set(comm_data['nodes']),
                summary=comm_data['summary'],
                description=comm_data['description'],
                purpose=comm_data['purpose'],
                key_entities=comm_data['key_entities']
            )
        
        # Restore hierarchy
        detector.hierarchy = {
            int(level): comm_ids
            for level, comm_ids in data['hierarchy'].items()
        }
        
        logger.info(f"Loaded {len(detector.communities)} communities")
        
        return detector


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python community_detection.py <enriched_graph.pkl>")
        sys.exit(1)
    
    graph_path = sys.argv[1]
    
    # Load graph
    logger.info(f"Loading graph from {graph_path}")
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    
    # Detect communities
    detector = CommunityDetector(resolution=1.0)
    communities = detector.detect_communities(graph, hierarchical=True)
    
    # Show statistics
    stats = detector.get_statistics()
    print(f"\n=== Community Detection Results ===")
    print(f"Total communities: {stats['total_communities']}")
    print(f"Hierarchy levels: {stats['hierarchy_levels']}")
    print(f"Avg community size: {stats['avg_community_size']:.1f}")
    
    # Summarize
    detector.summarize_communities(graph)
    
    # Save
    output_path = Path(graph_path).parent / 'communities.json'
    detector.save(str(output_path))
    
    # Show sample
    print(f"\n=== Sample Community ===")
    sample_comm = list(communities.values())[0]
    print(detector.get_community_summary(sample_comm.id))