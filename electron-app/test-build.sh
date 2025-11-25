#!/usr/bin/env bash

# Test build script for LlamaFarm Desktop
# This script validates the build setup and dependencies

set -e

echo "🦙 LlamaFarm Desktop - Build Test"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Node.js
echo -n "Checking Node.js... "
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    # Extract major version number (e.g., "v18.0.0" -> "18")
    NODE_MAJOR=$(echo "$NODE_VERSION" | sed 's/v\([0-9]*\).*/\1/')

    if [ "$NODE_MAJOR" -lt 18 ]; then
        echo -e "${RED}✗${NC} $NODE_VERSION (requires Node.js 18+)"
        echo "Please install Node.js 18+ from https://nodejs.org"
        exit 1
    fi

    echo -e "${GREEN}✓${NC} $NODE_VERSION"
else
    echo -e "${RED}✗${NC} Node.js not found"
    echo "Please install Node.js 18+ from https://nodejs.org"
    exit 1
fi

# Check npm
echo -n "Checking npm... "
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    echo -e "${GREEN}✓${NC} $NPM_VERSION"
else
    echo -e "${RED}✗${NC} npm not found"
    exit 1
fi

# Check package.json
echo -n "Checking package.json... "
if [ -f "package.json" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC} package.json not found"
    exit 1
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
if npm install; then
    echo -e "${GREEN}✓${NC} Dependencies installed"
else
    echo -e "${RED}✗${NC} Failed to install dependencies"
    exit 1
fi

# Check TypeScript compilation
echo ""
echo "Testing TypeScript compilation..."
if npx tsc --noEmit; then
    echo -e "${GREEN}✓${NC} TypeScript compilation successful"
else
    echo -e "${RED}✗${NC} TypeScript compilation failed"
    exit 1
fi

# Check electron-vite config
echo ""
echo -n "Checking electron-vite config... "
if [ -f "electron.vite.config.ts" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC} electron.vite.config.ts not found"
    exit 1
fi

# Test build
echo ""
echo "Testing build..."
if npm run build; then
    echo -e "${GREEN}✓${NC} Build successful"
else
    echo -e "${RED}✗${NC} Build failed"
    exit 1
fi

# Check build output
echo ""
echo -n "Checking build output... "
if [ -d "dist" ]; then
    echo -e "${GREEN}✓${NC} dist/ directory created"
    echo ""
    echo "Build artifacts:"
    ls -lh dist/
else
    echo -e "${RED}✗${NC} dist/ directory not found"
    exit 1
fi

# Summary
echo ""
echo "=================================="
echo -e "${GREEN}✓ All tests passed!${NC}"
echo ""
echo "Next steps:"
echo "  1. Build designer: cd ../designer && npm run build"
echo "  2. Run in dev mode: npm run dev"
echo "  3. Build for macOS: npm run dist:mac"
echo ""
