#!/bin/bash

echo "🚀 Setting up GraphRAG Codebase Understanding System"
echo "=================================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Setup environment file
echo ""
echo "⚙️  Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓ Created .env file - Please add your API keys!"
else
    echo "✓ .env file already exists"
fi

# Start Neo4j with Docker
echo ""
echo "🐳 Starting Neo4j database..."
cd docker
docker-compose up -d
cd ..

echo ""
echo "⏳ Waiting for Neo4j to start (30 seconds)..."
sleep 30

# Test Neo4j connection
echo ""
echo "🔌 Testing Neo4j connection..."
python3 -c "
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
user = os.getenv('NEO4J_USER', 'neo4j')
password = os.getenv('NEO4J_PASSWORD', 'graphrag2025')

try:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    print('✓ Neo4j connection successful!')
    driver.close()
except Exception as e:
    print(f'✗ Neo4j connection failed: {e}')
    print('  Please check your Neo4j configuration')
"

echo ""
echo "=================================================="
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Add your GOOGLE_API_KEY to .env file"
echo "  2. Activate virtual environment: source venv/bin/activate"
echo "  3. Run the application: python src/main.py"
echo ""
echo "📖 Neo4j Browser: http://localhost:7474"
echo "   Username: neo4j"
echo "   Password: graphrag2025"
echo ""