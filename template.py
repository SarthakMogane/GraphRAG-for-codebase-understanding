import os
from pathlib import Path
import logging

# logging format
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "graphrag-codebase-understanding"

list_of_files = [

    # ---- SRC ----
    f"src/agents/__init__.py",
    f"src/agents/parser_agent.py",
    f"src/agents/search_agent.py",
    f"src/agents/traversal_agent.py",
    f"src/agents/synthesis_agent.py",
    f"src/agents/orchestrator.py",

    f"src/tools/__init__.py",
    f"src/tools/neo4j_tools.py",
    f"src/tools/parser_tools.py",
    f"src/tools/embedding_tools.py",
    f"src/tools/github_tools.py",

    f"src/models/__init__.py",
    f"src/models/graph_schema.py",
    f"src/models/code_entities.py",

    f"src/utils/__init__.py",
    f"src/utils/config.py",
    f"src/utils/logger.py",
    f"src/utils/metrics.py",

    f"src/main.py",

    # ---- DATA ----
    f"data/repositories/.gitkeep",
    f"data/evaluation/.gitkeep",

    # ---- NOTEBOOKS ----
    f"notebooks/01_setup_neo4j.ipynb",
    f"notebooks/02_parse_repository.ipynb",
    f"notebooks/03_test_retrieval.ipynb",
    f"notebooks/04_evaluation.ipynb",

    # ---- TESTS ----
    f"tests/test_parsing.py",
    f"tests/test_graph_ops.py",
    f"tests/test_agents.py",

    # ---- DOCKER ----
    f"docker/Dockerfile",
    f"docker/docker-compose.yml",

    # ---- ROOT FILES ----
    f"requirements.txt",
    f".env.example",
    f"README.md",
    f"setup.py"
]


# Create structure
for filepath in list_of_files:
    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir:
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir}")

    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
