#!/bin/bash
# Setup script for Audiobook Generator

set -e

echo "=========================================="
echo "  Audiobook Generator Setup"
echo "=========================================="

# Check for conda
if command -v conda &> /dev/null; then
    echo "Creating conda environment..."

    # Create environment
    conda create -n audiobook-generator python=3.11 -y

    # Activate
    eval "$(conda shell.bash hook)"
    conda activate audiobook-generator

    echo "Installing dependencies..."

    # Install core dependencies
    pip install langchain>=0.3.0 langgraph>=0.2.0 langchain-ollama>=0.2.0

    # Document processing
    pip install striprtf markdownify python-docx beautifulsoup4 lxml

    # Audio processing
    pip install soundfile librosa numpy scipy

    # Speech-to-text
    pip install faster-whisper

    # CLI and utilities
    pip install typer rich tqdm pydantic python-dotenv

    # Install qwen-tts from parent directory
    pip install -e ..

    # Install this package
    pip install -e .

    echo ""
    echo "=========================================="
    echo "  Setup Complete!"
    echo "=========================================="
    echo ""
    echo "To activate the environment:"
    echo "  conda activate audiobook-generator"
    echo ""
    echo "To generate an audiobook:"
    echo "  audiobook-generator generate <input_file>"
    echo ""
    echo "Make sure Ollama is running for QA agents:"
    echo "  ollama serve"
    echo "  ollama pull llama3"

else
    echo "Conda not found. Using pip..."

    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip

    # Install dependencies
    pip install langchain>=0.3.0 langgraph>=0.2.0 langchain-ollama>=0.2.0
    pip install striprtf markdownify python-docx beautifulsoup4 lxml
    pip install soundfile librosa numpy scipy
    pip install faster-whisper
    pip install typer rich tqdm pydantic python-dotenv

    # Install qwen-tts
    pip install -e ..

    # Install this package
    pip install -e .

    echo ""
    echo "=========================================="
    echo "  Setup Complete!"
    echo "=========================================="
    echo ""
    echo "To activate the environment:"
    echo "  source venv/bin/activate"
    echo ""
    echo "To generate an audiobook:"
    echo "  audiobook-generator generate <input_file>"
fi
