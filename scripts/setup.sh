#!/bin/bash
# Quick setup script for Kerala Ayurveda RAG System

set -e  # Exit on error

echo "🌿 Kerala Ayurveda RAG System - Setup Script"
echo "=============================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV is not installed!"
    echo ""
    echo "Install UV with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    exit 1
fi

echo "✅ UV found: $(uv --version)"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    uv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi
echo ""

# Sync dependencies
echo "📥 Installing dependencies with uv sync..."
uv sync
echo "✅ Dependencies installed"
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your OPENAI_API_KEY"
    echo ""
else
    echo "✅ .env file already exists"
    echo ""
fi

# Create data directories if they don't exist
echo "📁 Setting up data directories..."
mkdir -p data/processed data/indexes
touch data/processed/.gitkeep data/indexes/.gitkeep
echo "✅ Data directories ready"
echo ""

# Check if kerala-data exists
if [ ! -d "kerala-data" ]; then
    echo "⚠️  WARNING: kerala-data/ directory not found!"
    echo "   Make sure to add your Kerala Ayurveda documents before running the pipeline."
    echo ""
else
    echo "✅ kerala-data/ directory found"
    echo ""
fi

echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your OPENAI_API_KEY"
echo "  2. Activate the virtual environment: source .venv/bin/activate"
echo "  3. Run the demo: uv run main_pipeline.py"
echo ""
echo "Or run directly with uv:"
echo "  uv run main_pipeline.py"
echo ""
