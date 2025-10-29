#!/bin/bash
set -e

# Universal Runtime Startup Script
# Ensures dependencies are installed before starting the server

# Configuration with defaults
UNIVERSAL_PORT="${UNIVERSAL_PORT:-11540}"
UNIVERSAL_HOST="${UNIVERSAL_HOST:-127.0.0.1}"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Universal Runtime Server ===${NC}"
echo "Port: $UNIVERSAL_PORT"
echo "Host: $UNIVERSAL_HOST"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}ERROR: uv is not installed${NC}"
    echo ""
    echo "Install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    exit 1
fi

# Ensure Python 3.12 is installed
echo -e "${GREEN}Checking Python 3.12...${NC}"
if ! uv python list 2>/dev/null | grep -q "cpython-3.12"; then
    echo -e "${YELLOW}Installing Python 3.12...${NC}"
    uv python install 3.12
fi

# Create .python-version file if it doesn't exist (ensures uv uses correct Python)
if [ ! -f .python-version ]; then
    echo "3.12" > .python-version
    echo "Created .python-version file"
fi

# Run uv sync to ensure all dependencies are installed
echo -e "${GREEN}Ensuring dependencies are installed...${NC}"
uv sync

echo ""
echo -e "${GREEN}Starting Universal Runtime Server...${NC}"
echo ""
echo "Once started, the server will be available at:"
echo "  http://$UNIVERSAL_HOST:$UNIVERSAL_PORT"
echo ""
echo "OpenAI-compatible endpoints:"
echo "  POST http://$UNIVERSAL_HOST:$UNIVERSAL_PORT/v1/chat/completions"
echo "  POST http://$UNIVERSAL_HOST:$UNIVERSAL_PORT/v1/embeddings"
echo "  POST http://$UNIVERSAL_HOST:$UNIVERSAL_PORT/v1/images/generations"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start the server
exec uv run python server.py
