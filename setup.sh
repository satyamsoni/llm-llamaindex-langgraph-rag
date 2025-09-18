#!/bin/bash

# Set your environment name
ENV_NAME="venv"

# List of Python packages to install
PACKAGES=(
    llama-index 
    langgraph 
    milvus
    pymilvus
    ollama
    llama-index-embeddings-ollama
    llama-index-llms-ollama
    llama-index-vector-stores-milvus
)

echo "Creating virtual environment: $ENV_NAME"
python3 -m venv "$ENV_NAME"

chmod +x "$ENV_NAME/bin/activate"

echo "Activating virtual environment..."
. "$ENV_NAME/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing packages..."
for package in "${PACKAGES[@]}"; do
    echo "Installing $package..."
    pip install "$package"
done
curl -fsSL https://ollama.com/install.sh | sh

ollama serve
ollama pull mistral
docker pull milvusdb/milvus:v2.6.0
docker compose up -d