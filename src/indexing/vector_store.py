"""
Vector Store - Create and manage FAISS index for semantic search
Uses latest sentence-transformers with optimized embedding models
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pickle
import json
from dataclasses import dataclass

import faiss
from tqdm import tqdm
from loguru import logger
import networkx as nx

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings


@dataclass
class SearchResult:
    """Vector search result"""
    node_id: str
    score: float
    name: str
    type: str
    summary: str
    tags: List[str]


class VectorStore:
    """
    FAISS-based vector store for semantic code search
    
    Features:
    - Latest embedding models (all-MiniLM-L6-v2, gte-small)
    - Multiple index types (Flat, IVF, HNSW)
    - Efficient batch embedding
    - Metadata storage
    - Save/load functionality
    """
    
    # Recommended embedding models (2024)
    EMBEDDING_MODELS = {
        'mini': 'sentence-transformers/all-MiniLM-L6-v2',  # 384d, fast
        'small': 'sentence-transformers/all-MiniLM-L12-v2',  # 384d, better
        'gte': 'thenlper/gte-small',  # 384d, SOTA small model
        'large': 'sentence-transformers/all-mpnet-base-v2'  # 768d, best quality
    }
    
    def __init__(
        self,
        provider: str = "huggingface",  # 'huggingface', 'google', 'openai'
        model_name: str = "all-MiniLM-L6-v2",
        index_type: str = 'Flat'
    ):
        """
        Initialize vector store
        
        Args:
            provider: huggingface , google , openai
            index_type: FAISS index type ('Flat', 'IVF', 'HNSW')
            device: 'cuda' or 'cpu' (default: auto-detect)
        """
        self.index_type = index_type
        self.provider = provider


        logger.info(f"Initializing {provider} embeddings with model {model_name}")
        
        self.embedding_model: Embeddings = None
        
        if provider == "huggingface":
            # Runs locally (Free, Fast)
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=f"sentence-transformers/{model_name}"
            )
        elif provider == "google":
            # Gemini Embeddings (Fast, Cheap)
            self.embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004"
            )
        elif provider == "openai":
            # OpenAI Embeddings (Standard)
            self.embedding_model = OpenAIEmbeddings(
                model="text-embedding-3-small"
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # Initialize Index
        self.index = None
        self.node_ids = []
        self.metadata = {}
        
        # We need to know the dimension (e.g., 384 or 1536)
        # We do a test run to find out
        test_embed = self.embedding_model.embed_query("test")
        self.embedding_dim = len(test_embed)
        
        logger.info(f"Vector store initialized (dim={self.embedding_dim}, "
                   f"type={index_type})")
    
    def build_from_graph(
        self,
        graph: nx.DiGraph,
        text_field: str = 'combined',
        batch_size: int = 32
    ) -> 'VectorStore':
        """
        Build vector index from graph
        
        Args:
            graph: Input NetworkX graph
            text_field: Which text to embed ('code', 'summary', 'combined')
            batch_size: Embedding batch size
        
        Returns:
            Self (for chaining)
        """
        logger.info(f"Building vector index from {graph.number_of_nodes()} nodes")
        
        # Collect texts and metadata
        texts = []
        node_ids = []
        
        for node_id, attrs in tqdm(graph.nodes(data=True), desc="Preparing texts"):
            # Get text to embed
            text = self._get_node_text(attrs, text_field)
            
            if text:
                texts.append(text)
                node_ids.append(node_id)
                
                # Store metadata
                self.metadata[node_id] = {
                    'name': attrs.get('name', ''),
                    'type': attrs.get('type', ''),
                    'summary': attrs.get('summary', ''),
                    'tags': attrs.get('tags', []),
                    'file_path': attrs.get('file_path', ''),
                    'complexity': attrs.get('complexity', '')
                }
        
        logger.info(f"Collected {len(texts)} texts for embedding")
        
        # 2. Generate Embeddings via LangChain
        # LangChain's embed_documents handles batching automatically for APIs,
        # but for local HF, we still batch manually to show progress
        # Generate embeddings in batches
        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i + batch_size]

            batch_embeddings = self.embedding_model.embed_documents(batch)
            embeddings.extend(batch_embeddings)
        
        embeddings_np = np.array(embeddings).astype('float32')
        logger.info(f"Generated embeddings: {embeddings_np.shape}")
        
        # Build FAISS index
        self._build_index(embeddings_np)
        
        # Store mappings
        self.node_ids = node_ids
        
        logger.success(f"Vector index built: {len(node_ids)} vectors")
        
        return self
    
    def _get_node_text(self, attrs: Dict, field: str) -> str:
        """Extract text from node attributes"""
        
        if field == 'code':
            return attrs.get('code', '')
        
        elif field == 'summary':
            return attrs.get('summary', attrs.get('docstring', ''))
        
        elif field == 'combined':
            # Combine multiple fields for richer representation
            parts = []
            
            # Name and type
            parts.append(f"{attrs.get('type', '')} {attrs.get('name', '')}")
            
            # Summary or docstring
            summary = attrs.get('summary') or attrs.get('docstring', '')
            if summary:
                parts.append(summary)
            
            # Tags
            tags = attrs.get('tags', [])
            if tags:
                parts.append(f"Tags: {', '.join(tags)}")
            
            # Short code snippet
            code = attrs.get('code', '')
            if code:
                parts.append(code[:300])
            
            return '\n'.join(parts)
        
        else:
            return attrs.get(field, '')
    
    def _build_index(self, embeddings: np.ndarray):
        """Build FAISS index from embeddings"""
        
        n_vectors, dim = embeddings.shape
        
        if self.index_type == 'Flat':
            # Flat index (exact search, best quality)
            self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine with normalized vectors)
        
        elif self.index_type == 'IVF':
            # IVF index (faster for large datasets)
            n_clusters = min(int(np.sqrt(n_vectors)), 256)
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, n_clusters)
            
            # Train index
            logger.info(f"Training IVF index with {n_clusters} clusters...")
            self.index.train(embeddings)
            self.index.nprobe = min(10, n_clusters)  # Search 10 clusters
        
        elif self.index_type == 'HNSW':
            # HNSW index (best speed/quality tradeoff)
            M = 32  # Number of connections
            self.index = faiss.IndexHNSWFlat(dim, M)
            self.index.hnsw.efConstruction = 200  # Build quality
            self.index.hnsw.efSearch = 100  # Search quality
        
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Add vectors
        logger.info("Adding vectors to index...")
        self.index.add(embeddings)
        
        logger.info(f"Index built: {self.index.ntotal} vectors")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_type: str = None
    ) -> List[SearchResult]:
        """
        Search for similar code
        
        Args:
            query: Search query
            top_k: Number of results
            filter_type: Filter by node type (e.g., 'function')
        
        Returns:
            List of SearchResult objects
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_from_graph() first.")
        
        # Embed query
        query_vec = self.embedding_model.embed_query(query)

        # Reshape for FAISS (1, dim)
        query_np = np.array([query_vec]).astype('float32')
        
        # Search
        scores, indices = self.index.search(query_np, top_k * 2)  # Get extra for filtering
        
        # Convert to results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            node_id = self.node_ids[idx]
            meta = self.metadata[node_id]
            
            # Apply filter
            if filter_type and meta['type'] != filter_type:
                continue
            
            results.append(SearchResult(
                node_id=node_id,
                score=float(score),
                name=meta['name'],
                type=meta['type'],
                summary=meta['summary'],
                tags=meta['tags']
            ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def search_by_embedding(
        self,
        embedding: np.ndarray,
        top_k: int = 10
    ) -> List[SearchResult]:
        """Search using pre-computed embedding"""
        
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        embedding = embedding.astype('float32')
        
        scores, indices = self.index.search(embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            
            node_id = self.node_ids[idx]
            meta = self.metadata[node_id]
            
            results.append(SearchResult(
                node_id=node_id,
                score=float(score),
                name=meta['name'],
                type=meta['type'],
                summary=meta['summary'],
                tags=meta['tags']
            ))
        
        return results
    
    def find_similar_nodes(
        self,
        node_id: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """Find nodes similar to a given node"""
        
        # Get node embedding
        try:
            idx = self.node_ids.index(node_id)
        except ValueError:
            raise ValueError(f"Node {node_id} not in index")
        
        # Reconstruct embedding (FAISS doesn't store original embeddings by default)
        # We need to re-embed the node text
        meta = self.metadata[node_id]
        text = f"{meta['type']} {meta['name']} {meta['summary']}"
        
        query_vec = self.embedding_model.embed_query(text)
        embedding = np.array([query_vec]).astype('float32')
        
        # Search (skip first result as it's the node itself)
        results = self.search_by_embedding(embedding, top_k + 1)
        return [r for r in results if r.node_id != node_id][:top_k]
    
    def save(self, path: str):
        """Save vector store to disk"""
        
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(output_path / 'index.faiss'))
        
        # Save metadata
        with open(output_path / 'data.pkl', 'wb') as f:
            pickle.dump({
                'node_ids': self.node_ids,
                'metadata': self.metadata,
                'config': {'provider': self.provider, 'index_type': self.index_type}
            }, f)
        logger.success(f"Saved to {output_path}")
        
        logger.success(f"Vector store saved to {output_path}")
    
    @classmethod
    def load(cls, path: str, model_name: str = 'mini') -> 'VectorStore':
        """Load vector store from disk"""
        
        input_path = Path(path)
        
        with open(input_path / 'data.pkl', 'rb') as f:
            data = pickle.load(f)
            
        store = cls(
            provider=data['config']['provider'],
            index_type=data['config']['index_type']
        )
        
        store.index = faiss.read_index(str(input_path / 'index.faiss'))
        store.node_ids = data['node_ids']
        store.metadata = data['metadata']
        
        logger.info(f"Vector store loaded: {len(store.node_ids)} vectors")
        
        return store


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python vector_store.py <enriched_graph.pkl>")
        sys.exit(1)
    
    graph_path = sys.argv[1]
    
    # Load graph
    logger.info(f"Loading graph from {graph_path}")
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    
    # Build vector store
    store = VectorStore(provider='huggingface',model_name="all-MiniLM-L6-v2" ,index_type='Flat')
    store.build_from_graph(graph, text_field='combined')
    
    # Save
    output_path = Path(graph_path).parent / 'vector_store'
    if not output_path.exists():
        output_path.mkdir()
    store.save(str(output_path))
    
    # Test search
    print("\n=== Testing Search ===")
    results = store.search("How can i connect to database", top_k=5)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.name} ({result.type}) - Score: {result.score:.3f}")
        print(f"   Summary: {result.summary}")
        print(f"   Tags: {', '.join(result.tags)}")