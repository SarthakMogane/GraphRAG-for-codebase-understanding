"""
Neo4j Loader - Load NetworkX Graph into Neo4j Database
Handles batch operations and efficient loading
"""

import os
from typing import Dict, List
import networkx as nx
from neo4j import GraphDatabase
from loguru import logger
from tqdm import tqdm
import time


class Neo4jLoader:
    """Load code graph into Neo4j database"""
    
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        """
        Initialize Neo4j connection
        
        Args:
            uri: Neo4j connection URI (default: from env)
            user: Database user (default: from env)
            password: Database password (default: from env)
        """
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = user or os.getenv('NEO4J_USER', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', 'graphrag2025')
        
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def load_graph(self, graph: nx.DiGraph, clear_existing: bool = True,
                   batch_size: int = 100):
        """
        Load NetworkX graph into Neo4j
        
        Args:
            graph: NetworkX graph to load
            clear_existing: Whether to clear existing data
            batch_size: Number of nodes/edges per batch
        """
        with self.driver.session() as session:
            if clear_existing:
                logger.info("Clearing existing data...")
                session.run("MATCH (n) DETACH DELETE n")
            
            # Create constraints for performance
            self._create_constraints(session)
            
            # Load nodes in batches
            logger.info(f"Loading {graph.number_of_nodes()} nodes...")
            self._load_nodes_batch(session, graph, batch_size)
            
            # Load edges in batches
            logger.info(f"Loading {graph.number_of_edges()} edges...")
            self._load_edges_batch(session, graph, batch_size)
            
            logger.info("Graph loaded successfully!")
    
    def _create_constraints(self, session):
        """Create indexes and constraints for better performance"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:CodeEntity) REQUIRE n.id IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (n:CodeEntity) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:CodeEntity) ON (n.type)",
            "CREATE INDEX IF NOT EXISTS FOR (n:File) ON (n.path)",
        ]
        
        for constraint in constraints:
            try:
                session.run(constraint)
            except Exception as e:
                logger.warning(f"Constraint creation warning: {e}")
    
    def _load_nodes_batch(self, session, graph: nx.DiGraph, batch_size: int):
        """Load nodes in batches"""
        nodes = list(graph.nodes(data=True))
        
        for i in tqdm(range(0, len(nodes), batch_size), desc="Loading nodes"):
            batch = nodes[i:i + batch_size]
            
            # Prepare batch data
            node_data = []
            for node_id, attrs in batch:
                node_dict = {
                    'id': node_id,
                    'name': attrs.get('name', 'unknown'),
                    'type': attrs.get('type', 'unknown'),
                    'code': attrs.get('code', ''),
                    'docstring': attrs.get('docstring', ''),
                    'file_path': attrs.get('file_path', ''),
                    'start_line': attrs.get('start_line', 0),
                    'end_line': attrs.get('end_line', 0),
                    'language': attrs.get('language', 'unknown')
                }
                node_data.append(node_dict)
            
            # Batch insert
            query = """
            UNWIND $nodes AS node
            MERGE (n:CodeEntity {id: node.id})
            SET n.name = node.name,
                n.type = node.type,
                n.code = node.code,
                n.docstring = node.docstring,
                n.file_path = node.file_path,
                n.start_line = node.start_line,
                n.end_line = node.end_line,
                n.language = node.language
            """
            
            session.run(query, nodes=node_data)
    
    def _load_edges_batch(self, session, graph: nx.DiGraph, batch_size: int):
        """Load edges in batches"""
        edges = list(graph.edges(data=True))
        
        for i in tqdm(range(0, len(edges), batch_size), desc="Loading edges"):
            batch = edges[i:i + batch_size]
            
            # Prepare batch data
            edge_data = []
            for source, target, attrs in batch:
                edge_dict = {
                    'source': source,
                    'target': target,
                    'type': attrs.get('type', 'RELATES_TO'),
                    'weight': attrs.get('weight', 1.0)
                }
                edge_data.append(edge_dict)
            
            # Batch insert - use dynamic relationship types
            query = """
            UNWIND $edges AS edge
            MATCH (source:CodeEntity {id: edge.source})
            MATCH (target:CodeEntity {id: edge.target})
            CALL apoc.create.relationship(source, edge.type, 
                {weight: edge.weight}, target) YIELD rel
            RETURN count(rel)
            """
            
            # Fallback if APOC not available
            try:
                session.run(query, edges=edge_data)
            except Exception as e:
                logger.warning("APOC not available, using MERGE")
                self._load_edges_without_apoc(session, edge_data)
    
    def _load_edges_without_apoc(self, session, edge_data: List[Dict]):
        """Fallback edge loading without APOC"""
        for edge in edge_data:
            edge_type = edge['type']
            query = f"""
            MATCH (source:CodeEntity {{id: $source}})
            MATCH (target:CodeEntity {{id: $target}})
            MERGE (source)-[r:{edge_type}]->(target)
            SET r.weight = $weight
            """
            session.run(query, 
                       source=edge['source'],
                       target=edge['target'],
                       weight=edge['weight'])
    
    def get_graph_stats(self) -> Dict:
        """Get statistics about the loaded graph"""
        with self.driver.session() as session:
            stats = {}
            
            # Count nodes
            result = session.run("MATCH (n:CodeEntity) RETURN count(n) as count")
            stats['num_nodes'] = result.single()['count']
            
            # Count edges
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            stats['num_edges'] = result.single()['count']
            
            # Node types
            result = session.run("""
                MATCH (n:CodeEntity) 
                RETURN n.type as type, count(n) as count
                ORDER BY count DESC
            """)
            stats['node_types'] = {record['type']: record['count'] 
                                  for record in result}
            
            # Edge types
            result = session.run("""
                MATCH ()-[r]->() 
                RETURN type(r) as type, count(r) as count
                ORDER BY count DESC
            """)
            stats['edge_types'] = {record['type']: record['count'] 
                                  for record in result}
            
            return stats
    
    def query_neighbors(self, node_id: str, max_depth: int = 2) -> List[Dict]:
        """
        Query k-hop neighbors of a node
        
        Args:
            node_id: Starting node ID
            max_depth: Maximum traversal depth
        
        Returns:
            List of neighbor nodes with their properties
        """
        with self.driver.session() as session:
            query = """
            MATCH path = (start:CodeEntity {id: $node_id})-[*1..%d]-(neighbor)
            RETURN DISTINCT neighbor.id as id, 
                   neighbor.name as name,
                   neighbor.type as type,
                   length(path) as distance
            ORDER BY distance, name
            """ % max_depth
            
            result = session.run(query, node_id=node_id)
            neighbors = [dict(record) for record in result]
            
            return neighbors
    
    def find_path(self, source_id: str, target_id: str) -> List[str]:
        """
        Find shortest path between two nodes
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
        
        Returns:
            List of node IDs in the path
        """
        with self.driver.session() as session:
            query = """
            MATCH path = shortestPath(
                (start:CodeEntity {id: $source})-[*]-(end:CodeEntity {id: $target})
            )
            RETURN [node in nodes(path) | node.id] as path
            """
            
            result = session.run(query, source=source_id, target=target_id)
            record = result.single()
            
            return record['path'] if record else []


# Example usage
if __name__ == "__main__":
    import sys
    from graph_builder import GraphBuilder
    
    if len(sys.argv) < 2:
        print("Usage: python neo4j_loader.py <graph_pickle_path>")
        sys.exit(1)
    
    graph_path = sys.argv[1]
    
    # Load graph
    logger.info(f"Loading graph from {graph_path}")
    graph = GraphBuilder.load_graph(graph_path)
    
    # Load into Neo4j
    loader = Neo4jLoader()
    
    try:
        loader.load_graph(graph, clear_existing=True)
        
        # Show stats
        stats = loader.get_graph_stats()
        print(f"\n=== Neo4j Graph Statistics ===")
        print(f"Nodes: {stats['num_nodes']}")
        print(f"Edges: {stats['num_edges']}")
        
        print(f"\n=== Node Types ===")
        for node_type, count in stats['node_types'].items():
            print(f"{node_type}: {count}")
        
        print(f"\n=== Edge Types ===")
        for edge_type, count in stats['edge_types'].items():
            print(f"{edge_type}: {count}")
        
        print(f"\n✓ Data loaded into Neo4j successfully!")
        print(f"View at: http://localhost:7474")
        
    finally:
        loader.close()