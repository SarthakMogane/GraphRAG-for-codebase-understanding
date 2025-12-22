import time
from src.indexing.vector_store import VectorStore

store = VectorStore.load('data/graphs/vector_store')

# Measure search latency
queries = ["validation", "database", "configuration"]
latencies = []

for query in queries:
    start = time.time()
    results = store.search(query, top_k=10)
    latency = (time.time() - start) * 1000
    latencies.append(latency)
    print(f"{query}: {latency:.1f}ms")

print(f"Average: {sum(latencies)/len(latencies):.1f}ms")