#!/usr/bin/env python3
"""
Week 3 Complete Pipeline: Community Detection + LangGraph Workflow
Builds on Week 2 enriched graph to create intelligent routing
"""

import sys
import argparse
import time
from pathlib import Path
from loguru import logger
import pickle


from src.utils.llm_client import LangChainClient
from src.indexing.community_detection import CommunityDetector
from src.indexing.vector_store import VectorStore
from src.retrieval.graph_workflow import RAGPipeline


def setup_logging(verbose: bool = False):
    """Configure logging"""
    logger.remove()
    
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level
    )
    
    logger.add(
        "logs/week3_pipeline.log",
        rotation="10 MB",
        level="DEBUG"
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Week 3: Build communities and LangGraph workflow"
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
        help='Path to vector store directory'
    )
    
    parser.add_argument(
        '--resolution',
        type=float,
        default=1.0,
        help='Louvain resolution (higher=more communities)'
    )
    
    parser.add_argument(
        '--min-community-size',
        type=int,
        default=3,
        help='Minimum nodes per community'
    )
    
    parser.add_argument(
        '--skip-communities',
        action='store_true',
        help='Skip community detection (load existing)'
    )
    
    parser.add_argument(
        '--test-queries',
        action='store_true',
        help='Run test queries through workflow'
    )
    
    parser.add_argument(
        '--llm-provider',
        type=str,
        choices=['google','openai', 'anthropic'],
        default='google',
        help='LLM provider'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main execution"""
    args = parse_args()
    setup_logging(args.verbose)
    
    logger.info("="*70)
    logger.info("WEEK 3: COMMUNITY DETECTION & LANGGRAPH WORKFLOW")
    logger.info("="*70)
    
    # ========================================
    # STEP 1: Load Prerequisites
    # ========================================
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Loading Graph and Vector Store")
    logger.info("="*70)
    
    # Load graph
    graph_path = Path(args.graph)
    if not graph_path.exists():
        logger.error(f"Graph not found: {graph_path}")
        logger.info("Run Week 2 pipeline first:")
        logger.info("  python scripts/enrich_and_vectorize.py data/graphs/code_graph.pkl")
        sys.exit(1)
    
    logger.info(f"Loading graph from {graph_path}")
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    
    logger.success(f"✓ Graph loaded: {graph.number_of_nodes()} nodes")
    
    # Load vector store
    vector_store_path = Path(args.vector_store)
    if not vector_store_path.exists():
        logger.error(f"Vector store not found: {vector_store_path}")
        sys.exit(1)
    
    logger.info(f"Loading vector store from {vector_store_path}")
    vector_store = VectorStore.load(str(vector_store_path))
    
    logger.success(f"✓ Vector store loaded: {len(vector_store.node_ids)} vectors")
    
    # ========================================
    # STEP 2: Community Detection
    # ========================================
    
    communities_path = graph_path.parent / 'communities.json'
    
    if args.skip_communities and communities_path.exists():
        logger.info("\n" + "="*70)
        logger.info("STEP 2: Loading Existing Communities")
        logger.info("="*70)
        
        community_detector = CommunityDetector.load(str(communities_path))
        logger.success(f"✓ Loaded {len(community_detector.communities)} communities")
    
    else:
        logger.info("\n" + "="*70)
        logger.info("STEP 2: Detecting Communities")
        logger.info("="*70)
        
        start_time = time.time()
        
        # Initialize detector
        llm_client = LangChainClient(provider=args.llm_provider)
        community_detector = CommunityDetector(
            resolution=args.resolution,
            min_community_size=args.min_community_size,
            llm_client=llm_client
        )
        
        # Detect communities
        communities = community_detector.detect_communities(
            graph,
            hierarchical=True
        )
        
        detection_time = time.time() - start_time
        logger.success(f"✓ Community detection complete in {detection_time:.1f}s")
        
        # Show statistics
        stats = community_detector.get_statistics()
        logger.info(f"\nCommunity Statistics:")
        logger.info(f"  Total communities: {stats['total_communities']}")
        logger.info(f"  Hierarchy levels: {stats['hierarchy_levels']}")
        logger.info(f"  Avg size: {stats['avg_community_size']:.1f} nodes")
        logger.info(f"  Size range: {stats['size_distribution']['min']} - {stats['size_distribution']['max']}")
        
        logger.info(f"\nCommunities per level:")
        for level, count in stats['communities_per_level'].items():
            logger.info(f"  Level {level}: {count} communities")
        
        # ========================================
        # STEP 3: Generate Community Summaries
        # ========================================
        logger.info("\n" + "="*70)
        logger.info("STEP 3: Generating Community Summaries")
        logger.info("="*70)
        
        start_time = time.time()
        
        community_detector.summarize_communities(graph)
        
        summary_time = time.time() - start_time
        logger.success(f"✓ Summarization complete in {summary_time:.1f}s")
        
        # Show sample
        logger.info(f"\n=== Sample Community ===")
        sample_comm = list(communities.values())[0]
        logger.info(community_detector.get_community_summary(sample_comm.id))
        
        # Save communities
        community_detector.save(str(communities_path))
        logger.success(f"✓ Communities saved to {communities_path}")
    
    # ========================================
    # STEP 4: Build LangGraph Workflow
    # ========================================
    logger.info("\n" + "="*70)
    logger.info("STEP 4: Building LangGraph Workflow")
    logger.info("="*70)
    
    llm_client = LangChainClient(provider=args.llm_provider)
    
    pipeline = RAGPipeline(
        graph=graph,
        vector_store=vector_store,
        community_detector=community_detector,
        llm_client=llm_client,
        enable_verification=True
    )
    
    logger.success("✓ LangGraph workflow created")
    
    # Show workflow diagram
    logger.info("\n=== Workflow Structure ===")
    print(pipeline.get_graph_visualization())
    
    # ========================================
    # STEP 5: Test Queries (Optional)
    # ========================================
    if args.test_queries:
        logger.info("\n" + "="*70)
        logger.info("STEP 5: Testing Queries")
        logger.info("="*70)
        
        test_queries = [
            ("What is the overall architecture?", "global"),
            ("How does the save_to_file function work?", "local"),
            ("What are the main modules?", "global"),
            ("Where is validation performed?", "local")
        ]
        
        for query, expected_type in test_queries:
            logger.info(f"\n{'='*70}")
            logger.info(f"Query: {query}")
            logger.info(f"Expected type: {expected_type.upper()}")
            logger.info(f"{'='*70}")
            
            result = pipeline.query(query, verbose=True)
            
            print(f"\n📊 Result:")
            print(f"  Type: {result['query_type'].upper()} (confidence: {result['confidence']:.2f})")
            print(f"  Verified: {result['verified']}")
            print(f"  Steps: {' → '.join(result['steps'])}")
            print(f"\n💬 Answer:")
            print(f"  {result['answer']}\n")
            
            if result['sources']:
                print(f"📚 Sources:")
                for source in result['sources'][:3]:
                    print(f"  - {source['name']} ({source['type']}) - {source['score']:.2f}")
            
            print()
    
    # ========================================
    # STEP 6: Summary
    # ========================================
    logger.info("\n" + "="*70)
    logger.info("WEEK 3 COMPLETE!")
    logger.info("="*70)
    
    logger.info(f"\n✓ Communities detected and summarized")
    logger.info(f"✓ LangGraph workflow built")
    logger.info(f"✓ Intelligent query routing enabled")
    
    logger.info(f"\nYour system now has:")
    logger.info(f"  ✓ Hierarchical code organization")
    logger.info(f"  ✓ Module-level summaries")
    logger.info(f"  ✓ Automatic query classification")
    logger.info(f"  ✓ Adaptive retrieval (global vs local)")
    logger.info(f"  ✓ Production-grade orchestration")
    
    logger.info(f"\nQuery your codebase:")
    logger.info(f"  python scripts/query_with_langgraph.py 'What is the architecture?'")
    logger.info(f"  python scripts/query_with_langgraph.py 'How does login() work?'")
    
    logger.info(f"\nNext: Week 4 - Advanced retrieval optimization")


if __name__ == "__main__":
    main()