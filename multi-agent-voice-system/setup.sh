#!/bin/bash

echo "=========================================="
echo "Multi-Agent Voice System Setup"
echo "=========================================="
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python $PYTHON_VERSION found"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

echo "✅ Virtual environment created"
echo ""

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install PyTorch (CPU version for compatibility)
echo "🔥 Installing PyTorch..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install requirements
echo "📥 Installing dependencies (this may take several minutes)..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✅ Dependencies installed successfully"
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/sales
mkdir -p data/research
mkdir -p data/chroma
mkdir -p models
mkdir -p logs

echo "✅ Directories created"
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Please edit the .env file and add your API keys:"
    echo "   - LIVEKIT_API_KEY"
    echo "   - LIVEKIT_API_SECRET"
    echo "   - LIVEKIT_URL"
    echo "   - CEREBRAS_API_KEY"
    echo "   - DEEPGRAM_API_KEY"
    echo "   - CARTESIA_API_KEY"
    echo ""
else
    echo "ℹ️  .env file already exists, skipping..."
    echo ""
fi

# Initialize knowledge base with sample data
echo "📚 Initializing knowledge base with sample data..."
python scripts/load_knowledge.py --initialize-sample

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo "3. (Optional) Train intent classifier:"
echo "   python scripts/train_classifier.py --use-sample"
echo "4. Run the system:"
echo "   python main.py"
echo ""
echo "For more information, see README.md"
echo ""
