"""
Graph Builder - Constructs Knowledge Graph from Code Entities
Handles relationships: CALLS, IMPORTS, DEFINES, INHERITS
"""
import os 
from pathlib import Path
import re
import networkx as nx
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from loguru import logger
from collections import defaultdict

from ast_parser import CodeEntity


@dataclass
class GraphEdge:
    """Represents an edge in the code graph"""
    source: str  # Source node ID
    target: str  # Target node ID
    type: str    # Edge type: CALLS, IMPORTS, etc.
    weight: float = 1.0
    properties: Dict = None


class GraphBuilder:
    """Builds a NetworkX graph from parsed code entities"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entities_map: Dict[str, CodeEntity] = {}
        self.file_to_entities: Dict[str, List[str]] = defaultdict(list)
    
    def build_graph(self, entities: List[CodeEntity]) -> nx.DiGraph:
        """
        Build complete graph from code entities
        
        Args:
            entities: List of parsed code entities
        
        Returns:
            NetworkX directed graph
        """
        logger.info(f"Building graph from {len(entities)} entities")
        
        # Step 1: Add all nodes
        self._add_nodes(entities)
        
        # Step 2: Extract and add edges
        self._add_structural_edges(entities)
        self._add_import_edges(entities)
        self._add_call_edges(entities)
        self._add_file_hierarchy(entities)
        
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, "
                   f"{self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def _add_nodes(self, entities: List[CodeEntity]):
        """Add all entities as nodes in the graph"""
        for entity in entities:
            node_id = self._get_node_id(entity)
            self.entities_map[node_id] = entity
            
            # Add node with attributes
            self.graph.add_node(
                node_id,
                name=entity.name,
                type=entity.type,
                code=entity.code[:500],  # Truncate for storage
                docstring=entity.docstring,
                file_path=entity.file_path,
                start_line=entity.start_line,
                end_line=entity.end_line,
                language=entity.language
            )
            
            # Track entities per file
            self.file_to_entities[entity.file_path].append(node_id)
        
        logger.info(f"Added {len(self.entities_map)} nodes")
    
    def _add_structural_edges(self, entities: List[CodeEntity]):
        """Add edges for structural relationships (class contains method)"""
        for entity in entities:
            if entity.parent:
                parent_id = self._find_entity_id(entity.parent, entity.file_path)
                if parent_id:
                    self._add_edge(
                        parent_id,
                        self._get_node_id(entity),
                        'DEFINES'
                    )
    
    def _add_import_edges(self, entities: List[CodeEntity]):
        """Add IMPORTS edges between files/modules"""
        import_entities = [e for e in entities if e.type == 'import']
        
        for import_entity in import_entities:
            source_file = import_entity.file_path
            imported_module = import_entity.name
            
            # Find all nodes in source file
            for source_node in self.file_to_entities.get(source_file, []):
                # Try to find imported entity
                target_node = self._find_imported_entity(imported_module)
                
                if target_node:
                    self._add_edge(source_node, target_node, 'IMPORTS')
        
        logger.info(f"Added import edges")
    
    def _add_call_edges(self, entities: List[CodeEntity]):
        """
        Add CALLS edges by analyzing function calls in code
        Uses regex pattern matching
        """
        function_entities = [e for e in entities 
                           if e.type in ['function', 'method']]
        
        # Build function name to ID map
        func_map = {e.name: self._get_node_id(e) for e in function_entities}
        
        for entity in function_entities:
            caller_id = self._get_node_id(entity)
            
            # Find function calls in code
            called_funcs = self._extract_function_calls(entity.code)
            
            for called_func in called_funcs:
                if called_func in func_map:
                    callee_id = func_map[called_func]
                    if caller_id != callee_id:  # Avoid self-loops
                        self._add_edge(caller_id, callee_id, 'CALLS')
        
        logger.info(f"Added call edges")
    
    def _add_file_hierarchy(self, entities: List[CodeEntity]):
        """Add file nodes and hierarchy"""
        files = set(e.file_path for e in entities)
        
        for file_path in files:
            # Add file node
            file_id = f"file:{file_path}"
            self.graph.add_node(
                file_id,
                name=file_path.split('/')[-1],
                type='file',
                path=file_path
            )
            
            # Connect file to its entities
            for entity_id in self.file_to_entities[file_path]:
                self._add_edge(file_id, entity_id, 'CONTAINS')
    
    def _extract_function_calls(self, code: str) -> Set[str]:
        """
        Extract function names being called in code
        Uses simple regex - can be improved
        """
        # Pattern: function_name( or .function_name(
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        matches = re.findall(pattern, code)
        
        # Filter out keywords and common built-ins
        keywords = {'if', 'for', 'while', 'def', 'class', 'return', 
                   'print', 'len', 'range', 'str', 'int', 'list', 'dict'}
        
        return set(m for m in matches if m not in keywords)
    
    def _find_imported_entity(self, module_name: str) -> str:
        """
        Find entity ID for imported module
        Simplified: looks for file or class with matching name
        """
        # Try to find file
        for file_path in self.file_to_entities.keys():
            if module_name in file_path or file_path.endswith(f"{module_name}.py"):
                return f"file:{file_path}"
        
        # Try to find class/function
        for node_id, entity in self.entities_map.items():
            if entity.name == module_name:
                return node_id
        
        return None
    
    def _find_entity_id(self, name: str, file_path: str) -> str:
        """Find entity ID by name in specific file"""
        for node_id in self.file_to_entities.get(file_path, []):
            entity = self.entities_map[node_id]
            if entity.name == name:
                return node_id
        return None
    
    def _get_node_id(self, entity: CodeEntity) -> str:
        """Generate unique node ID for entity"""
        return f"{entity.type}:{entity.file_path}:{entity.name}:{entity.start_line}"
    
    def _add_edge(self, source: str, target: str, edge_type: str, weight: float = 1.0):
        """Add edge to graph"""
        if source in self.graph and target in self.graph:
            self.graph.add_edge(source, target, type=edge_type, weight=weight)
    
    def get_graph_stats(self) -> Dict:
        """Return statistics about the graph"""
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph),
        }
        
        # Node type distribution
        node_types = {}
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        stats['node_types'] = node_types
        
        # Edge type distribution
        edge_types = {}
        for u, v in self.graph.edges():
            edge_type = self.graph[u][v].get('type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        stats['edge_types'] = edge_types
        
        return stats
    
    def save_graph(self, output_path: str):
        """Save graph to file"""
        import os
        output_dir = os.path.dirname(output_path)
    
    # Check if the directory is specified AND does not exist
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)


        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(self.graph, f)
        logger.info(f"Graph saved to {output_path}")
    
    @staticmethod
    def load_graph(input_path: str) -> nx.DiGraph:
        """Load graph from file"""
        import pickle
        with open(input_path, 'rb') as f:
            graph = pickle.load(f)
        logger.info(f"Graph loaded from {input_path}")
        return graph


def visualize_graph(graph: nx.DiGraph, output_html: str = "graph.html"):
    """
    Create interactive visualization using pyvis
    
    Args:
        graph: NetworkX graph
        output_html: Output HTML file path
    """
    
    from pyvis.network import Network

    output_dir = os.path.dirname(output_html)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Create pyvis network
    net = Network(height="750px", width="100%", directed=True,
                  notebook=False, cdn_resources='remote')
    
    pyvis_path = Path("D:/kaggle_project/GraphRAG/myenv/Lib/site-packages/pyvis")
    template_path = pyvis_path / "templates" / "template.html"

    if not template_path.exists():
        logger.error(f"Template file not found at: {template_path}")

    net.html_template = str(template_path)
    # Color map for node types
    color_map = {
        'function': '#97C2FC',
        'class': '#FB7E81',
        'file': '#7BE141',
        'import': '#FFA807',
        'method': '#6E6EFD'
    }
    
    # Add nodes
    for node in graph.nodes():
        node_data = graph.nodes[node]
        node_type = node_data.get('type', 'unknown')
        
        net.add_node(
            node,
            label=node_data.get('name', node)[:30],
            title=f"{node_type}: {node_data.get('name', '')}",
            color=color_map.get(node_type, '#DDDDDD'),
            size=20
        )
    
    # Add edges
    for u, v in graph.edges():
        edge_data = graph[u][v]
        edge_type = edge_data.get('type', '')
        
        net.add_edge(u, v, title=edge_type, 
                    color={'color': '#848484'})
    
    # Configure physics
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {"iterations": 150}
      }
    }
    """)
    
    net.show(output_html,notebook=False)
    logger.info(f"Visualization saved to {output_html}")


# Example usage
if __name__ == "__main__":
    import sys
    from ast_parser import ASTParser
    
    if len(sys.argv) < 2:
        print("Usage: python graph_builder.py <repo_path>")
        sys.exit(1)
    
    repo_path = sys.argv[1]

    # repo_path = 'd:/kaggle_project/GraphRAG/data/repositories/SMS-Spam-VotingClassifier-'
    
    # Parse repository
    parser = ASTParser(repo_path)
    entities = parser.parse_repository()
    
    # Build graph
    builder = GraphBuilder()
    graph = builder.build_graph(entities)
    
    # Show statistics
    stats = builder.get_graph_stats()
    print(f"\n=== Graph Statistics ===")
    print(f"Nodes: {stats['num_nodes']}")
    print(f"Edges: {stats['num_edges']}")
    print(f"Density: {stats['density']:.4f}")
    print(f"Connected: {stats['is_connected']}")
    
    print(f"\n=== Node Types ===")
    for node_type, count in stats['node_types'].items():
        print(f"{node_type}: {count}")
    
    print(f"\n=== Edge Types ===")
    for edge_type, count in stats['edge_types'].items():
        print(f"{edge_type}: {count}")
    
    # Save graph
    builder.save_graph('data/graphs/code_graph.pkl')
    
    # Visualize (optional)
    if '--visualize' in sys.argv:
        visualize_graph(graph, 'data/outputs/graph_visualization.html')
        print(f"\nVisualization saved to data/outputs/graph_visualization.html")