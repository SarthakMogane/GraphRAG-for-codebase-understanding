"""
Neo4j operations as plain Python functions (ADK tools)
All functions return JSON strings for ADK compatibility
"""
import json
from typing import List, Dict, Optional
from neo4j import GraphDatabase
from loguru import logger

from src.config.settings import settings

# Global Neo4j driver
_driver = None

def get_driver():
    """Get or create Neo4j driver"""
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
    return _driver

def initialize_graph_schema() -> str:
    """
    Initialize Neo4j schema with indexes and constraints.
    
    Returns:
        JSON string with initialization status
    """
    driver = get_driver()
    
    schema_queries = [
        "CREATE CONSTRAINT function_id IF NOT EXISTS FOR (f:Function) REQUIRE f.id IS UNIQUE",
        "CREATE CONSTRAINT class_id IF NOT EXISTS FOR (c:Class) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT file_id IF NOT EXISTS FOR (f:File) REQUIRE f.id IS UNIQUE",
        "CREATE INDEX function_name IF NOT EXISTS FOR (f:Function) ON (f.name)",
        "CREATE INDEX file_path IF NOT EXISTS FOR (f:File) ON (f.file_path)",
        """CREATE VECTOR INDEX function_embeddings IF NOT EXISTS
           FOR (f:Function) ON (f.embedding)
           OPTIONS {indexConfig: {
               `vector.dimensions`: 768,
               `vector.similarity_function`: 'cosine'
           }}""",
    ]
    
    with driver.session() as session:
        for query in schema_queries:
            try:
                session.run(query)
            except Exception as e:
                logger.warning(f"Schema query warning: {e}")
    
    return json.dumps({"status": "success", "message": "Schema initialized"})

def create_function_node(
    node_id: str,
    name: str,
    file_path: str,
    code: str,
    docstring: Optional[str] = None,
    parameters: Optional[List[str]] = None,
    embedding: Optional[List[float]] = None
) -> str:
    """
    Create a function node in the graph.
    
    Args:
        node_id: Unique identifier
        name: Function name
        file_path: File path
        code: Function code
        docstring: Function documentation
        parameters: Function parameters
        embedding: Vector embedding
        
    Returns:
        JSON string with created node ID
    """
    driver = get_driver()
    
    with driver.session() as session:
        query = """
        MERGE (f:Function {id: $id})
        SET f.name = $name,
            f.file_path = $file_path,
            f.code = $code,
            f.docstring = $docstring,
            f.parameters = $parameters,
            f.embedding = $embedding
        RETURN f.id as id
        """
        
        result = session.run(
            query,
            id=node_id,
            name=name,
            file_path=file_path,
            code=code,
            docstring=docstring,
            parameters=parameters or [],
            embedding=embedding
        )
        
        created_id = result.single()['id']
        return json.dumps({"node_id": created_id, "type": "Function"})

def create_file_node(
    node_id: str,
    name: str,
    file_path: str,
    language: str,
    imports: Optional[List[str]] = None
) -> str:
    """
    Create a file node in the graph.
    
    Args:
        node_id: Unique identifier
        name: File name
        file_path: Full file path
        language: Programming language
        imports: List of imports
        
    Returns:
        JSON string with created node ID
    """
    driver = get_driver()
    
    with driver.session() as session:
        query = """
        MERGE (f:File {id: $id})
        SET f.name = $name,
            f.file_path = $file_path,
            f.language = $language,
            f.imports = $imports
        RETURN f.id as id
        """
        
        result = session.run(
            query,
            id=node_id,
            name=name,
            file_path=file_path,
            language=language,
            imports=imports or []
        )
        
        created_id = result.single()['id']
        return json.dumps({"node_id": created_id, "type": "File"})

def create_relationship(
    source_id: str,
    target_id: str,
    relationship_type: str
) -> str:
    """
    Create a relationship between two nodes.
    
    Args:
        source_id: Source node ID
        target_id: Target node ID
        relationship_type: Type of relationship (CALLS, CONTAINS, etc.)
        
    Returns:
        JSON string with relationship status
    """
    driver = get_driver()
    
    with driver.session() as session:
        query = f"""
        MATCH (a {{id: $source_id}})
        MATCH (b {{id: $target_id}})
        MERGE (a)-[r:{relationship_type}]->(b)
        RETURN type(r) as type
        """
        
        result = session.run(
            query,
            source_id=source_id,
            target_id=target_id
        )
        
        rel = result.single()
        return json.dumps({
            "source": source_id,
            "target": target_id,
            "type": rel['type'] if rel else relationship_type
        })

def vector_search_functions(query_embedding: List[float], top_k: int = 10) -> str:
    """
    Search functions using vector similarity.
    
    Args:
        query_embedding: Query vector embedding
        top_k: Number of results to return
        
    Returns:
        JSON string with search results
    """
    driver = get_driver()
    
    with driver.session() as session:
        query = """
        CALL db.index.vector.queryNodes(
            'function_embeddings',
            $k,
            $embedding
        )
        YIELD node, score
        RETURN 
            node.id as id,
            node.name as name,
            node.file_path as file_path,
            node.code as code,
            node.docstring as docstring,
            score
        ORDER BY score DESC
        """
        
        result = session.run(
            query,
            k=top_k,
            embedding=query_embedding
        )
        
        results = [dict(record) for record in result]
        return json.dumps(results)

def traverse_function_calls(
    function_id: str,
    direction: str = "downstream",
    max_depth: int = 3
) -> str:
    """
    Traverse function call graph.
    
    Args:
        function_id: Starting function ID
        direction: "downstream" (what it calls) or "upstream" (who calls it)
        max_depth: Maximum traversal depth
        
    Returns:
        JSON string with traversal results
    """
    driver = get_driver()
    
    if direction == "downstream":
        query = """
        MATCH path = (f:Function {id: $func_id})-[:CALLS*1..$depth]->(target)
        RETURN 
            target.id as id,
            target.name as name,
            target.file_path as file_path,
            target.code as code,
            length(path) as depth
        ORDER BY depth ASC
        LIMIT 20
        """
    else:  # upstream
        query = """
        MATCH path = (caller)-[:CALLS*1..$depth]->(f:Function {id: $func_id})
        RETURN 
            caller.id as id,
            caller.name as name,
            caller.file_path as file_path,
            caller.code as code,
            length(path) as depth
        ORDER BY depth ASC
        LIMIT 20
        """
    
    with driver.session() as session:
        result = session.run(
            query,
            func_id=function_id,
            depth=max_depth
        )
        
        results = [dict(record) for record in result]
        return json.dumps(results)

def get_graph_statistics() -> str:
    """
    Get graph statistics.
    
    Returns:
        JSON string with graph statistics
    """
    driver = get_driver()
    
    with driver.session() as session:
        stats = {}
        
        # Count nodes
        result = session.run("MATCH (f:Function) RETURN count(f) as count")
        stats['functions'] = result.single()['count']
        
        result = session.run("MATCH (c:Class) RETURN count(c) as count")
        stats['classes'] = result.single()['count']
        
        result = session.run("MATCH (f:File) RETURN count(f) as count")
        stats['files'] = result.single()['count']
        
        # Count relationships
        result = session.run("MATCH ()-[r:CALLS]->() RETURN count(r) as count")
        stats['calls'] = result.single()['count']
        
        return json.dumps(stats)

def clear_graph() -> str:
    """
    Clear all nodes and relationships from the graph.
    WARNING: This deletes everything!
    
    Returns:
        JSON string with deletion status
    """
    driver = get_driver()
    
    with driver.session() as session:
        result = session.run("MATCH (n) DETACH DELETE n")
        return json.dumps({"status": "success", "message": "Graph cleared"})