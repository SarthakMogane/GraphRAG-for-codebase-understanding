#!/bin/bash

# Setup Neo4j using Docker
# This script starts a Neo4j container with APOC plugin

echo "🚀 Setting up Neo4j..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Stop existing container if running
if [ "$(docker ps -q -f name=neo4j-graphrag)" ]; then
    echo "Stopping existing Neo4j container..."
    docker stop neo4j-graphrag
fi

# Remove existing container
if [ "$(docker ps -aq -f name=neo4j-graphrag)" ]; then
    echo "Removing existing Neo4j container..."
    docker rm neo4j-graphrag
fi

# Create data directory
mkdir -p $HOME/neo4j-graphrag/data
mkdir -p $HOME/neo4j-graphrag/logs
mkdir -p $HOME/neo4j-graphrag/plugins

# Pull latest Neo4j image
echo "Pulling Neo4j Docker image..."
docker pull neo4j:5.14.0

# Start Neo4j container
echo "Starting Neo4j container..."
docker run \
    --name neo4j-graphrag \
    -p 7474:7474 \
    -p 7687:7687 \
    -d \
    -e NEO4J_AUTH=neo4j/password123 \
    -e NEO4J_PLUGINS='["apoc"]' \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_apoc_import_file_use__neo4j__config=true \
    -e NEO4J_dbms_security_procedures_unrestricted=apoc.* \
    -v $HOME/neo4j-graphrag/data:/data \
    -v $HOME/neo4j-graphrag/logs:/logs \
    -v $HOME/neo4j-graphrag/plugins:/plugins \
    neo4j:5.14.0

# Wait for Neo4j to start
echo "Waiting for Neo4j to start (this may take 30-60 seconds)..."
sleep 10

# Check if Neo4j is ready
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if docker logs neo4j-graphrag 2>&1 | grep -q "Started."; then
        echo "✓ Neo4j is ready!"
        break
    fi
    attempt=$((attempt + 1))
    sleep 2
    echo -n "."
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ Neo4j failed to start. Check logs with: docker logs neo4j-graphrag"
    exit 1
fi

echo ""
echo "========================================="
echo "✓ Neo4j Setup Complete!"
echo "========================================="
echo ""
echo "Neo4j Browser: http://localhost:7474"
echo "Bolt URI: bolt://localhost:7687"
echo "Username: neo4j"
echo "Password: password123"
echo ""
echo "To stop Neo4j:"
echo "  docker stop neo4j-graphrag"
echo ""
echo "To start Neo4j again:"
echo "  docker start neo4j-graphrag"
echo ""
echo "To view logs:"
echo "  docker logs neo4j-graphrag"
echo ""