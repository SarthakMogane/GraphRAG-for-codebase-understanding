#!/usr/bin/env python3
"""
Interactive Query CLI with LangGraph
Query your codebase using the intelligent workflow
"""

import sys
import argparse
from pathlib import Path
from loguru import logger
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.utils.llm_client import LangChainClient
from src.indexing.vector_store import VectorStore
from src.indexing.community_detection import CommunityDetector
from src.retrieval.graph_workflow import RAGPipeline


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Query codebase with LangGraph RAG pipeline"
    )
    
    parser.add_argument(
        'query',
        type=str,
        nargs='?',
        help='Query (interactive mode if not provided)'
    )
    
    parser.add_argument(
        '--graph',
        type=str,
        default='data/graphs/code_graph_enriched.pkl',
        help='Path to enriched graph'
    )
    
    parser.add_argument(
        '--vector-store',
        type=str,
        default='data/graphs/vector_store',
        help='Path to vector store'
    )
    
    parser.add_argument(
        '--communities',
        type=str,
        default='data/graphs/communities.json',
        help='Path to communities file'
    )
    
    parser.add_argument(
        '--provider',
        type=str,
        choices=['google','openai', 'anthropic'],
        default='google',
        help='LLM provider'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of results to retrieve'
    )
    
    parser.add_argument(
        '--max-hops',
        type=int,
        default=2,
        help='Max graph traversal hops'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed workflow execution'
    )
    
    parser.add_argument(
        '--stream',
        action='store_true',
        help='Stream intermediate results'
    )
    
    return parser.parse_args()


def load_pipeline(args) -> RAGPipeline:
    """Load and initialize RAG pipeline"""
    
    print("\n📚 Loading RAG pipeline...")
    
    # Load graph
    with open(args.graph, 'rb') as f:
        graph = pickle.load(f)
    print(f"✓ Graph: {graph.number_of_nodes()} nodes")
    
    # Load vector store
    vector_store = VectorStore.load(args.vector_store)
    print(f"✓ Vector store: {len(vector_store.node_ids)} vectors")
    
    # Load communities
    community_detector = CommunityDetector.load(args.communities)
    print(f"✓ Communities: {len(community_detector.communities)} detected")
    
    # Initialize LLM
    llm_client = LangChainClient(provider=args.provider)
    
    # Create pipeline
    pipeline = RAGPipeline(
        graph=graph,
        vector_store=vector_store,
        community_detector=community_detector,
        llm_client=llm_client
    )
    
    print("✓ Pipeline ready!\n")
    
    return pipeline


def display_result(result: dict, verbose: bool = False):
    """Display query result"""
    
    print(f"\n{'='*80}")
    print(f"📊 Query Analysis")
    print(f"{'='*80}")
    print(f"Type: {result['query_type'].upper()} (confidence: {result.get('confidence', 0):.2%})")
    print(f"Verified: {'✓' if result.get('verified') else '✗'}")
    print(f"Workflow: {' → '.join(result.get('steps', []))}")
    
    if verbose and result.get('timing'):
        print(f"\nTiming:")
        for step, duration in result['timing'].items():
            print(f"  {step}: {duration:.2f}s")
    
    print(f"\n{'='*80}")
    print(f"💬 Answer")
    print(f"{'='*80}")
    print(result['answer'])
    
    if result.get('sources'):
        print(f"\n{'='*80}")
        print(f"📚 Sources ({len(result['sources'])} total)")
        print(f"{'='*80}")
        for i, source in enumerate(result['sources'][:5], 1):
            print(f"{i}. {source['name']} ({source['type']}) - Score: {source['score']:.2f}")
    
    if result.get('errors'):
        print(f"\n⚠️  Errors encountered:")
        for error in result['errors']:
            print(f"  - {error}")
    
    print()


def interactive_mode(pipeline: RAGPipeline, args):
    """Interactive query mode"""
    
    print("\n" + "="*80)
    print("🤖 LangGraph RAG - Interactive Mode")
    print("="*80)
    print("\nAsk questions about your codebase.")
    print("\nCommands:")
    print("  /help     - Show this help")
    print("  /stats    - Show pipeline statistics")
    print("  /workflow - Show workflow diagram")
    print("  /quit     - Exit")
    print("\nExample queries:")
    print("  • What is the overall architecture?")
    print("  • How does authentication work?")
    print("  • What does the save_to_file function do?")
    print("  • Where is input validation performed?")
    print()
    
    while True:
        try:
            query = input("🔍 > ").strip()
            
            if not query:
                continue
            
            if query == '/quit':
                print("\n👋 Goodbye!")
                break
            
            elif query == '/help':
                print("\nAvailable commands:")
                print("  /help     - Show this help")
                print("  /stats    - Show statistics")
                print("  /workflow - Show workflow diagram")
                print("  /quit     - Exit")
                continue
            
            elif query == '/stats':
                print(f"\nPipeline Statistics:")
                print(f"  Graph nodes: {pipeline.graph.number_of_nodes()}")
                print(f"  Graph edges: {pipeline.graph.number_of_edges()}")
                print(f"  Vector store: {len(pipeline.vector_store.node_ids)} vectors")
                print(f"  Communities: {len(pipeline.communities.communities)}")
                stats = pipeline.communities.get_statistics()
                print(f"  Hierarchy levels: {stats['hierarchy_levels']}")
                continue
            
            elif query == '/workflow':
                print("\nWorkflow Diagram:")
                print(pipeline.get_graph_visualization())
                continue
            
            # Process query
            print("\n⏳ Processing...")
            result = pipeline.query(
                query,
                top_k=args.top_k,
                max_hops=args.max_hops,
                verbose=args.verbose
            )
            
            display_result(result, args.verbose)
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.error(f"Query error: {e}")


def stream_mode(pipeline: RAGPipeline, query: str, args):
    """Stream query execution"""
    
    print(f"\n🔍 Query: {query}")
    print("="*80)
    print("Streaming workflow execution...\n")
    
    step_count = 0
    for state in pipeline.stream_query(
        query,
        top_k=args.top_k,
        max_hops=args.max_hops
    ):
        step_count += 1
        # State is a dict with one key (the node that just executed)
        node_name = list(state.keys())[0]
        print(f"✓ Step {step_count}: {node_name}")
    
    print("\n" + "="*80)
    print("Workflow complete!")


def single_query_mode(pipeline: RAGPipeline, query: str, args):
    """Single query mode"""
    
    print(f"\n🔍 Query: {query}")
    
    if args.stream:
        stream_mode(pipeline, query, args)
    else:
        result = pipeline.query(
            query,
            top_k=args.top_k,
            max_hops=args.max_hops,
            verbose=args.verbose
        )
        
        display_result(result, args.verbose)


def main():
    """Main execution"""
    args = parse_args()
    
    # Configure logger
    logger.remove()
    logger.add(
        sys.stderr,
        format="<level>{message}</level>",
        level="DEBUG" if args.verbose else "INFO"
    )
    
    # Check if prerequisites exist
    for path in [args.graph, args.vector_store, args.communities]:
        if not Path(path).exists():
            print(f"\n❌ Not found: {path}")
            print("\nDid you run Week 3 pipeline?")
            print("  python scripts/build_communities_and_workflow.py")
            sys.exit(1)
    
    # Load pipeline
    try:
        pipeline = load_pipeline(args)
    except Exception as e:
        print(f"\n❌ Failed to load pipeline: {e}")
        sys.exit(1)
    
    # Run in appropriate mode
    if args.query:
        single_query_mode(pipeline, args.query, args)
    else:
        interactive_mode(pipeline, args)


if __name__ == "__main__":
    main()