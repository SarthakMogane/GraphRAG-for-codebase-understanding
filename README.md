# Hybrid GraphRAG for Code Understanding
> A production-grade system combining graph-based retrieval, community detection, and hallucination verification for intelligent codebase Q&A

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)]() 
[![Neo4j](https://img.shields.io/badge/Neo4j-5.14-green)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

## 🎯 Project Overview

This project implements a **Hybrid GraphRAG architecture** that intelligently answers questions about codebases by:

1. **Building a knowledge graph** from code structure (AST parsing)
2. **Enriching with semantics** using LLMs and embeddings
3. **Smart retrieval** via community detection (global) or graph traversal (local)
4. **Verified generation** using SelfCheckGPT to reduce hallucinations

### Why This Project?

Traditional code understanding tools fail because:
- ❌ Vector search misses structural relationships
- ❌ LLMs hallucinate without proper grounding
- ❌ Keyword search misses semantic meaning

**Our solution combines the best of all worlds:**
- ✅ Graph structure captures code relationships
- ✅ Vector search finds semantically similar code
- ✅ LLM generation with verification prevents hallucinations
- ✅ Adaptive retrieval (global vs local) for efficiency

---

## 🏗️ Architecture

```
┌─────────────┐
│ Code Repo   │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  G-INDEXING      │
│  • AST Parsing   │
│  • Graph Build   │
│  • Semantic      │
│  • Communities   │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  G-RETRIEVAL     │
│  • Query Class   │
│  • Global/Local  │
│  • Graph Traverse│
│  • Context Prune │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  G-GENERATION    │
│  • LLM Generate  │
│  • SelfCheckGPT  │
│  • Verification  │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Verified Answer  │
└──────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker (for Neo4j)
- OpenAI or Anthropic API key

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/hybrid-graphrag.git
cd hybrid-graphrag

# Run setup
bash setup.sh

# Activate environment
source venv/bin/activate

# Configure API keys
cp .env.example .env
# Edit .env with your keys
```

### Index Your First Repository

```bash
# Start Neo4j (optional)
bash scripts/setup_neo4j.sh

# Create sample repo
python scripts/create_sample_repo.py

# Index it
python scripts/index_repository.py \
    data/sample_repos/simple_python \
    --visualize \
    --neo4j
```

### Explore in Jupyter

```bash
jupyter notebook notebooks/01_structural_extraction.ipynb
```

---

## 📊 Current Progress (Week 1 ✅)

### Completed Features

- [x] **AST Parser** - Multi-language support (Python, JS)
- [x] **Graph Builder** - Extracts CALLS, IMPORTS, DEFINES edges
- [x] **Neo4j Integration** - Efficient batch loading
- [x] **Graph Visualization** - Interactive HTML viewer
- [x] **CLI Tool** - Complete indexing pipeline

### Sample Output

```
=== Graph Statistics ===
Nodes: 47
Edges: 63
Density: 0.0291

=== Node Types ===
function: 18
class: 3
file: 5
import: 21

=== Edge Types ===
CALLS: 24
IMPORTS: 21
DEFINES: 12
CONTAINS: 6
```

---

## 📅 Roadmap

| Week | Phase | Status |
|------|-------|--------|
| 1 | Structural Indexing | ✅ Complete |
| 2 | Semantic Enrichment | 🔄 In Progress |
| 3 | Community Detection | ⏳ Planned |
| 4 | Retrieval System | ⏳ Planned |
| 5 | Generation & Verification | ⏳ Planned |
| 6 | Demo & Evaluation | ⏳ Planned |

### Upcoming (Week 2)
- LLM-based node summarization
- Embedding generation (all-MiniLM-L6-v2)
- FAISS vector index
- Semantic similarity search

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Parsing** | Tree-sitter, Python AST |
| **Graph DB** | Neo4j 5.14 |
| **Vector Store** | FAISS |
| **Embeddings** | Sentence-Transformers |
| **LLM** | OpenAI GPT-4 / Claude Sonnet |
| **Visualization** | Pyvis, Plotly |
| **Framework** | NetworkX, LangChain |

---

## 📖 Documentation

- [Architecture Guide](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Deployment Guide](docs/deployment_guide.md)
- [Evaluation Results](docs/evaluation_results.md) (Coming Week 6)

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest --cov=src tests/

# Specific module
pytest tests/test_ast_parser.py
```

---

## 📈 Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Indexing Speed | <5s/1000 LOC | TBD |
| Query Latency | <2s | TBD |
| Precision@5 | >0.7 | TBD |
| Hallucination Rate | <15% | TBD |

---

## 🤝 Contributing

This is a portfolio project, but feedback is welcome!

1. Fork the repository
2. Create your feature branch
3. Commit changes
4. Push to the branch
5. Open a Pull Request

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

- **Microsoft GraphRAG** - Community detection approach
- **SelfCheckGPT** - Hallucination detection method
- **Neo4j Labs** - LLM Graph Builder inspiration
- **Tree-sitter** - Robust code parsing

---

## 📬 Contact

**Your Name** - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/hybrid-graphrag](https://github.com/yourusername/hybrid-graphrag)

---

## 🎓 Learning Resources

- [GraphRAG Paper](https://arxiv.org/abs/2404.16130)
- [SelfCheckGPT Paper](https://arxiv.org/abs/2303.08896)
- [Neo4j Graph Data Science](https://neo4j.com/docs/graph-data-science/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)

---

**⭐ Star this repo if you find it useful!**