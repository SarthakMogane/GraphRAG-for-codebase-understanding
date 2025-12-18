#!/usr/bin/env python3
"""
Complete Indexing Pipeline (Week 1)
Orchestrates: Parsing → Graph Building → Neo4j Loading → Visualization
"""

import sys
import argparse
from pathlib import Path
import time
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from indexing.ast_parser import ASTParser
from indexing.graph_builder import GraphBuilder, visualize_graph
from indexing.neo4j_loader import Neo4jLoader


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
        "logs/indexing.log",
        rotation="10 MB",
        level="DEBUG"
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Index a code repository into GraphRAG"
    )
    
    parser.add_argument(
        'repo_path',
        type=str,
        help='Path to code repository'
    )
    
    parser.add_argument(
        '--languages',
        nargs='+',
        default=['python'],
        help='Languages to parse (default: python)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/graphs/code_graph.pkl',
        help='Output path for graph pickle'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate HTML visualization'
    )
    
    parser.add_argument(
        '--neo4j',
        action='store_true',
        help='Load graph into Neo4j'
    )
    
    parser.add_argument(
        '--neo4j-uri',
        type=str,
        default='bolt://localhost:7687',
        help='Neo4j connection URI'
    )
    
    parser.add_argument(
        '--neo4j-user',
        type=str,
        default='neo4j',
        help='Neo4j username'
    )
    
    parser.add_argument(
        '--neo4j-password',
        type=str,
        default='password123',
        help='Neo4j password'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()
    setup_logging(args.verbose)
    
    logger.info("="*60)
    logger.info("GRAPHRAG INDEXING PIPELINE - WEEK 1")
    logger.info("="*60)
    
    # Validate repository path
    repo_path = Path(args.repo_path)
    if not repo_path.exists():
        logger.error(f"Repository path not found: {repo_path}")
        sys.exit(1)
    
    logger.info(f"Repository: {repo_path.absolute()}")
    logger.info(f"Languages: {', '.join(args.languages)}")
    
    # ========================================
    # STEP 1: Parse Code Repository
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Parsing Code Repository")
    logger.info("="*60)
    
    start_time = time.time()
    
    parser = ASTParser(str(repo_path), languages=args.languages)
    entities = parser.parse_repository()
    
    parse_time = time.time() - start_time
    
    logger.success(f"✓ Parsed {len(entities)} entities in {parse_time:.2f}s")
    
    # Show entity breakdown
    from collections import Counter
    entity_types = Counter(e.type for e in entities)
    logger.info("\nEntity breakdown:")
    for entity_type, count in entity_types.most_common():
        logger.info(f"  {entity_type}: {count}")
    
    if not entities:
        logger.error("No entities found. Check repository path and file extensions.")
        sys.exit(1)
    
    # ========================================
    # STEP 2: Build Graph
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Building Knowledge Graph")
    logger.info("="*60)
    
    start_time = time.time()
    
    builder = GraphBuilder()
    graph = builder.build_graph(entities)
    
    build_time = time.time() - start_time
    
    logger.success(f"✓ Graph built in {build_time:.2f}s")
    
    # Show graph statistics
    stats = builder.get_graph_stats()
    logger.info(f"\nGraph statistics:")
    logger.info(f"  Nodes: {stats['num_nodes']}")
    logger.info(f"  Edges: {stats['num_edges']}")
    logger.info(f"  Density: {stats['density']:.4f}")
    logger.info(f"  Connected: {stats['is_connected']}")
    
    logger.info(f"\nNode types:")
    for node_type, count in stats['node_types'].items():
        logger.info(f"  {node_type}: {count}")
    
    logger.info(f"\nEdge types:")
    for edge_type, count in stats['edge_types'].items():
        logger.info(f"  {edge_type}: {count}")
    
    # ========================================
    # STEP 3: Save Graph
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Saving Graph")
    logger.info("="*60)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    builder.save_graph(str(output_path))
    logger.success(f"✓ Graph saved to {output_path}")
    
    # ========================================
    # STEP 4: Visualize (Optional)
    # ========================================
    if args.visualize:
        logger.info("\n" + "="*60)
        logger.info("STEP 4: Creating Visualization")
        logger.info("="*60)
        
        viz_path = output_path.parent / "graph_visualization.html"
        visualize_graph(graph, str(viz_path))
        logger.success(f"✓ Visualization saved to {viz_path}")
        logger.info(f"  Open in browser: file://{viz_path.absolute()}")
    
    # ========================================
    # STEP 5: Load into Neo4j (Optional)
    # ========================================
    if args.neo4j:
        logger.info("\n" + "="*60)
        logger.info("STEP 5: Loading into Neo4j")
        logger.info("="*60)
        
        try:
            loader = Neo4jLoader(
                uri=args.neo4j_uri,
                user=args.neo4j_user,
                password=args.neo4j_password
            )
            
            start_time = time.time()
            loader.load_graph(graph, clear_existing=True)
            load_time = time.time() - start_time
            
            logger.success(f"✓ Graph loaded to Neo4j in {load_time:.2f}s")
            
            # Get Neo4j stats
            neo4j_stats = loader.get_graph_stats()
            logger.info(f"\nNeo4j statistics:")
            logger.info(f"  Nodes: {neo4j_stats['num_nodes']}")
            logger.info(f"  Edges: {neo4j_stats['num_edges']}")
            
            logger.info(f"\nAccess Neo4j Browser:")
            logger.info(f"  URL: http://localhost:7474")
            logger.info(f"  Username: {args.neo4j_user}")
            logger.info(f"  Password: {args.neo4j_password}")
            
            loader.close()
            
        except Exception as e:
            logger.error(f"Failed to load into Neo4j: {e}")
            logger.info("Make sure Neo4j is running: bash scripts/setup_neo4j.sh")
    
    # ========================================
    # Summary
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("INDEXING COMPLETE!")
    logger.info("="*60)
    
    total_time = parse_time + build_time
    logger.info(f"\nTotal time: {total_time:.2f}s")
    logger.info(f"  Parsing: {parse_time:.2f}s")
    logger.info(f"  Graph building: {build_time:.2f}s")
    
    logger.info(f"\n✓ Repository indexed successfully!")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Explore graph: jupyter notebook notebooks/01_structural_extraction.ipynb")
    logger.info(f"  2. Add semantic enrichment (Week 2)")
    logger.info(f"  3. Build retrieval system (Week 4)")


if __name__ == "__main__":
    main()