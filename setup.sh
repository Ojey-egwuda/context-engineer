#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh — One-command environment setup for Context Engineer
#
# Run this once after cloning:
#   chmod +x setup.sh && ./setup.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo ""
echo "======================================================"
echo "  Context Engineer — Environment Setup"
echo "======================================================"
echo ""

# 1. Create virtual environment
echo "[ 1/5 ] Creating virtual environment..."
python3 -m venv venv
echo "        Done: venv/"

# 2. Activate and install dependencies
echo "[ 2/5 ] Installing dependencies..."
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "        Done: all packages installed"

# 3. Copy .env if it doesn't exist
echo "[ 3/5 ] Setting up .env file..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "        Created .env — ADD YOUR API KEYS before running the app"
else
    echo "        .env already exists, skipping"
fi

# 4. Create data directory for SQLite
echo "[ 4/5 ] Creating data directory..."
mkdir -p data
echo "        Done: data/"

# 5. Create __init__.py files
echo "[ 5/5 ] Initialising Python packages..."
touch src/__init__.py
touch src/context/__init__.py
touch src/graph/__init__.py
touch src/agents/__init__.py
touch tests/__init__.py
echo "        Done"

echo ""
echo "======================================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Open .env and add your ANTHROPIC_API_KEY"
echo "  2. Optionally add LANGSMITH_API_KEY for tracing"
echo "  3. Activate venv:  source venv/bin/activate"
echo "  4. Run the app:    streamlit run app.py"
echo "  5. Run tests:      pytest tests/ -v"
echo "======================================================"
echo ""
