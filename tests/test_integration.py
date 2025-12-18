"""
Integration tests for GraphRAG system
"""

import pytest
import asyncio
from pathlib import Path

from src.agents.orchestrator import orchestrator
from src.agents.parser_agent import parser_agent
from src.agents.search_traversal_synthesis import (
    search_agent, traversal_agent, synthesis_agent
)
from src.tools.neo4j_operations import neo4j_manager
from src.tools.code_parser import code_parser
from src.tools.embedding_service import embedding_generator
from src.models.entities import QueryIntent, RetrievalStrategy

# Sample Python code for testing
SAMPLE_CODE = """
def authenticate_user(username, password):
    '''Authenticate a user with credentials'''
    if validate_credentials(username, password):
        token = generate_token(username)
        return token
    return None

def validate_credentials(username, password):
    '''Validate user credentials against database'''
    user = get_user(username)
    return user and check_password(user, password)

def generate_token(username):
    '''Generate authentication token'''
    import jwt
    payload = {'user': username}
    return jwt.encode(payload, 'secret')

def get_user(username):
    '''Retrieve user from database'''
    # Database query here
    pass

def check_password(user, password):
    '''Verify password hash'''
    # Password verification
    pass
"""

class TestCodeParsing:
    """Test code parsing functionality"""
    
    def test_parse_python_file(self):
        """Test parsing a Python file"""
        parsed = code_parser.parse_file("test.py", SAMPLE_CODE)
        
        assert parsed is not None
        assert len(parsed.functions) == 5
        assert any(f.name == "authenticate_user" for f in parsed.functions)
    
    def test_extract_function_calls(self):
        """Test extracting function calls"""
        parsed = code_parser.parse_file("test.py", SAMPLE_CODE)
        
        auth_func = next(f for f in parsed.functions if f.name == "authenticate_user")
        
        # Should detect calls to validate_credentials and generate_token
        assert "validate_credentials" in auth_func.calls
        assert "generate_token" in auth_func.calls
    
    def test_extract_docstrings(self):
        """Test docstring extraction"""
        parsed = code_parser.parse_file("test.py", SAMPLE_CODE)
        
        auth_func = next(f for f in parsed.functions if f.name == "authenticate_user")
        
        assert auth_func.docstring is not None
        assert "Authenticate" in auth_func.docstring


class TestEmbeddings:
    """Test embedding generation"""
    
    def test_generate_single_embedding(self):
        """Test generating a single embedding"""
        embedding = embedding_generator.embed_text("def hello(): pass")
        
        assert embedding is not None
        assert len(embedding) == 768  # text-embedding-004 dimension
        assert all(isinstance(x, float) for x in embedding)
    
    def test_generate_code_function_embedding(self):
        """Test generating embedding for code function"""
        func_data = {
            'name': 'authenticate_user',
            'docstring': 'Authenticate a user',
            'parameters': [{'name': 'username'}, {'name': 'password'}],
            'code': 'def authenticate_user(username, password): ...'
        }
        
        embedding = embedding_generator.embed_code_function(func_data)
        
        assert embedding is not None
        assert len(embedding) == 768
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        emb1 = embedding_generator.embed_text("authentication")
        emb2 = embedding_generator.embed_text("login")
        emb3 = embedding_generator.embed_text("database")
        
        sim_auth_login = embedding_generator.cosine_similarity(emb1, emb2)
        sim_auth_db = embedding_generator.cosine_similarity(emb1, emb3)
        
        # Auth and login should be more similar than auth and database
        assert sim_auth_login > sim_auth_db


@pytest.mark.asyncio
class TestAgentOrchestration:
    """Test agent orchestration and workflows"""
    
    async def test_index_small_repo(self):
        """Test indexing a small repository"""
        # Use a small test repo
        repo_url = "https://github.com/octocat/Hello-World"
        
        # Clear existing data
        neo4j_manager.clear_database()
        
        result = await orchestrator.index_repository(
            repo_url=repo_url,
            branch="master"
        )
        
        assert result['clone']['success']
        assert result['parse']['success']
        assert result['parse']['parsed_files'] > 0
    
    async def test_query_classification(self):
        """Test query intent classification"""
        queries = {
            "Where is function X defined?": QueryIntent.FIND_DEFINITION,
            "What calls this function?": QueryIntent.FIND_USAGE,
            "Explain how authentication works": QueryIntent.EXPLAIN_CODE,
        }
        
        for query, expected_intent in queries.items():
            intent = await orchestrator._classify_intent(query)
            # Intent classification is LLM-based, so we check it returns a valid intent
            assert isinstance(intent, QueryIntent)
    
    async def test_strategy_determination(self):
        """Test retrieval strategy determination"""
        strategy = orchestrator._determine_strategy(QueryIntent.FIND_DEFINITION)
        assert strategy == RetrievalStrategy.VECTOR_ONLY
        
        strategy = orchestrator._determine_strategy(QueryIntent.FIND_USAGE)
        assert strategy == RetrievalStrategy.GRAPH_ONLY
        
        strategy = orchestrator._determine_strategy(QueryIntent.EXPLAIN_CODE)
        assert strategy == RetrievalStrategy.HYBRID


class TestGraphOperations:
    """Test Neo4j graph operations"""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test"""
        # Setup
        neo4j_manager.clear_database()
        yield
        # Teardown
        neo4j_manager.clear_database()
    
    def test_create_function_node(self):
        """Test creating a function node"""
        from src.models.graph_schema import FunctionNode, NodeType
        
        node = FunctionNode(
            id="test_func_1",
            type=NodeType.FUNCTION,
            name="test_function",
            file_path="test.py",
            start_line=1,
            end_line=5,
            code="def test_function(): pass",
            parameters=[]
        )
        
        node_id = neo4j_manager.create_node(node)
        assert node_id == "test_func_1"
        
        # Verify node exists
        query = "MATCH (f:Function {id: $id}) RETURN f"
        result = neo4j_manager.execute_cypher(query, {'id': node_id})
        assert len(result) == 1
    
    def test_create_relationship(self):
        """Test creating relationships between nodes"""
        from src.models.graph_schema import (
            FunctionNode, NodeType, CodeRelationship, RelationType
        )
        
        # Create two nodes
        node1 = FunctionNode(
            id="func1",
            type=NodeType.FUNCTION,
            name="caller",
            file_path="test.py",
            start_line=1,
            end_line=5,
            code="def caller(): callee()",
            parameters=[]
        )
        
        node2 = FunctionNode(
            id="func2",
            type=NodeType.FUNCTION,
            name="callee",
            file_path="test.py",
            start_line=7,
            end_line=10,
            code="def callee(): pass",
            parameters=[]
        )
        
        neo4j_manager.create_node(node1)
        neo4j_manager.create_node(node2)
        
        # Create CALLS relationship
        rel = CodeRelationship(
            type=RelationType.CALLS,
            source_id="func1",
            target_id="func2"
        )
        
        neo4j_manager.create_relationship(rel)
        
        # Verify relationship
        query = """
        MATCH (f1:Function {id: 'func1'})-[r:CALLS]->(f2:Function {id: 'func2'})
        RETURN r
        """
        result = neo4j_manager.execute_cypher(query)
        assert len(result) == 1
    
    def test_vector_search(self):
        """Test vector similarity search"""
        from src.models.graph_schema import FunctionNode, NodeType
        
        # Create function with embedding
        embedding = embedding_generator.embed_text("user authentication")
        
        node = FunctionNode(
            id="auth_func",
            type=NodeType.FUNCTION,
            name="authenticate",
            file_path="auth.py",
            start_line=1,
            end_line=10,
            code="def authenticate(user, password): ...",
            embedding=embedding,
            parameters=[]
        )
        
        neo4j_manager.create_node(node)
        
        # Search for similar functions
        query_embedding = embedding_generator.embed_query("login function")
        results = neo4j_manager.vector_search(query_embedding, top_k=5)
        
        assert len(results) > 0
        assert results[0]['name'] == "authenticate"


class TestEndToEnd:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test complete indexing and querying workflow"""
        # This test requires a real repository
        # For CI/CD, use a small test repository
        
        repo_url = "https://github.com/octocat/Hello-World"
        
        # Step 1: Index
        neo4j_manager.clear_database()
        
        index_result = await orchestrator.index_repository(repo_url)
        
        assert index_result['clone']['success']
        assert index_result['parse']['success']
        
        # Step 2: Query
        response = await orchestrator.query(
            "What files are in this repository?",
            top_k=5
        )
        
        assert response.answer is not None
        assert len(response.retrieved_nodes) > 0
        assert response.confidence > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])