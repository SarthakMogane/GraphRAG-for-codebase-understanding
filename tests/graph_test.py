# Load enriched graph
import pickle
with open('data/graphs/code_graph_enriched.pkl', 'rb') as f:
    graph = pickle.load(f)

# Check random nodes
import random
nodes = list(graph.nodes(data=True))
for _, attrs in random.sample(nodes, 5):
    if 'summary' in attrs:
        print(f"\nFunction: {attrs['name']}")
        print(f"Summary: {attrs['summary']}")
        print(f"Tags: {attrs.get('tags', [])}")