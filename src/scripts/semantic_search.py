#!/usr/bin/env python3
"""
Semantic Search CLI - Query your codebase using natural language
"""

import sys
import argparse
from pathlib import Path
from loguru import logger
import pickle

# Add src to path
# sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.indexing.vector_store import VectorStore


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Search codebase using natural language"
    )
    
    parser.add_argument(
        'query',
        type=str,
        nargs='?',
        help='Search query (interactive mode if not provided)'
    )
    
    parser.add_argument(
        '--vector-store',
        type=str,
        default='data/graphs/vector_store',
        help='Path to vector store directory'
    )
    
    parser.add_argument(
        '--graph',
        type=str,
        default='data/graphs/code_graph_enriched.pkl',
        help='Path to enriched graph (for showing code)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of results to return'
    )
    
    parser.add_argument(
        '--filter-type',
        type=str,
        choices=['function', 'class', 'method', 'file'],
        help='Filter results by node type'
    )
    
    parser.add_argument(
        '--show-code',
        action='store_true',
        help='Show code snippets in results'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='mini',
        choices=['mini', 'small', 'gte', 'large'],
        help='Embedding model (must match what was used to build index)'
    )
    
    return parser.parse_args()


def load_graph(graph_path: str):
    """Load enriched graph"""
    try:
        with open(graph_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logger.warning(f"Graph file not found: {graph_path}")
        return None


def display_results(results, graph=None, show_code=False):
    """Display search results"""
    
    if not results:
        print("\n❌ No results found")
        return
    
    print(f"\n{'='*80}")
    print(f"Found {len(results)} results")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.name} ({result.type})")
        print(f"   Score: {result.score:.3f}")
        
        if result.summary:
            print(f"   Summary: {result.summary}")
        
        if result.tags:
            print(f"   Tags: {', '.join(result.tags)}")
        
        # Show code if available and requested
        if show_code and graph:
            try:
                node_attrs = graph.nodes[result.node_id]
                code = node_attrs.get('code', '')
                if code:
                    # Show first few lines
                    code_lines = code.split('\n')[:10]
                    print(f"\n   Code preview:")
                    for line in code_lines:
                        print(f"   │ {line}")
                    if len(code.split('\n')) > 10:
                        print(f"   │ ...")
            except KeyError:
                pass
        
        print()


def interactive_mode(vector_store, graph, args):
    """Interactive search mode"""
    
    print("\n" + "="*80)
    print("🔍 Semantic Code Search - Interactive Mode")
    print("="*80)
    print("\nEnter queries to search your codebase.")
    print("Commands:")
    print("  /help     - Show this help")
    print("  /stats    - Show index statistics")
    print("  /quit     - Exit")
    print("\nExample queries:")
    print("  - 'function that validates email addresses'")
    print("  - 'code that handles authentication'")
    print("  - 'database connection setup'")
    print()
    
    while True:
        try:
            query = input("🔍 > ").strip()
            
            if not query:
                continue
            
            if query == '/quit' or query == 'exit':
                print("\n👋 Goodbye!")
                break
            
            elif query == '/help':
                print("\nAvailable commands:")
                print("  /help     - Show this help")
                print("  /stats    - Show index statistics")
                print("  /quit     - Exit")
                continue
            
            elif query == '/stats':
                print(f"\nIndex Statistics:")
                print(f"  Total vectors: {len(vector_store.node_ids)}")
                print(f"  Embedding dimension: {vector_store.embedding_dim}")
                print(f"  Index type: {vector_store.index_type}")
                
                # Type distribution
                from collections import Counter
                types = Counter(
                    vector_store.metadata[nid]['type'] 
                    for nid in vector_store.node_ids
                )
                print(f"\n  Node types:")
                for node_type, count in types.most_common():
                    print(f"    {node_type}: {count}")
                continue
            
            # Perform search
            results = vector_store.search(
                query,
                top_k=args.top_k,
                filter_type=args.filter_type
            )
            
            display_results(results, graph, args.show_code)
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        
        except Exception as e:
            logger.error(f"Search error: {e}")


def single_query_mode(query, vector_store, graph, args):
    """Single query mode"""
    
    print(f"\n🔍 Searching for: '{query}'")
    
    results = vector_store.search(
        query,
        top_k=args.top_k,
        filter_type=args.filter_type
    )
    
    display_results(results, graph, args.show_code)


def main():
    """Main execution"""
    args = parse_args()
    
    # Configure logger
    logger.remove()
    logger.add(
        sys.stderr,
        format="<level>{message}</level>",
        level="INFO"
    )
    
    # Load vector store
    vector_store_path = Path(args.vector_store)
    
    if not vector_store_path.exists():
        print(f"\n❌ Vector store not found: {vector_store_path}")
        print("\nDid you run Week 2 pipeline?")
        print("  python scripts/enrich_and_vectorize.py data/graphs/code_graph.pkl")
        sys.exit(1)
    
    print("\n📚 Loading vector store...")
    vector_store = VectorStore.load(str(vector_store_path), model_name=args.model)
    
    # Load graph (optional, for showing code)
    graph = None
    if args.show_code:
        graph = load_graph(args.graph)
    
    print(f"✓ Loaded {len(vector_store.node_ids)} code entities")
    
    # Run in appropriate mode
    if args.query:
        single_query_mode(args.query, vector_store, graph, args)
    else:
        interactive_mode(vector_store, graph, args)


if __name__ == "__main__":
    main()