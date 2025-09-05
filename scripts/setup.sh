#!/bin/bash

echo "🚀 Setting up AQP Development Environment"
echo "========================================"

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
if (( $(echo "$python_version < 3.11" | bc -l) )); then
    echo "❌ Python 3.11+ required. Current version: $python_version"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Copy environment file
if [ ! -f .env ]; then
    echo "📋 Creating environment file..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your API keys"
fi

# Create necessary directories
mkdir -p data/{market_data,strategies,backtests,models}
mkdir -p logs/{application,trading,monitoring,errors}

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your API keys"
echo "2. Run: source venv/bin/activate"
echo "3. Test: python src/aqp_master_engine.py status"
echo "4. Deploy: python src/aqp_master_engine.py full-auto --target-sharpe 2.0"
