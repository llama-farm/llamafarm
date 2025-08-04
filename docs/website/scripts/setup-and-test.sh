#!/bin/bash

# Docusaurus Setup and Test Script
# This script sets up dependencies, starts the server, and runs tests

set -e

echo "🚀 Docusaurus Setup and Test Script"
echo "===================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ $2${NC}"
    else
        echo -e "${RED}✗ $2${NC}"
        exit 1
    fi
}

# 1. Check prerequisites
echo -e "\n${YELLOW}1. Checking prerequisites...${NC}"

if command_exists node; then
    NODE_VERSION=$(node -v)
    echo -e "${GREEN}✓ Node.js installed: $NODE_VERSION${NC}"
    
    # Check if Node version is >= 18
    NODE_MAJOR=$(node -v | cut -d. -f1 | sed 's/v//')
    if [ "$NODE_MAJOR" -lt 18 ]; then
        echo -e "${RED}✗ Node.js version must be 18.0 or higher${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Node.js is not installed${NC}"
    exit 1
fi

if command_exists yarn; then
    YARN_VERSION=$(yarn -v)
    echo -e "${GREEN}✓ Yarn installed: $YARN_VERSION${NC}"
else
    echo -e "${RED}✗ Yarn is not installed${NC}"
    exit 1
fi

# 2. Install dependencies
echo -e "\n${YELLOW}2. Installing dependencies...${NC}"
yarn install
print_status $? "Dependencies installed"

# 3. Run type checking
echo -e "\n${YELLOW}3. Running type check...${NC}"
yarn typecheck
print_status $? "TypeScript check passed"

# 4. Build the project
echo -e "\n${YELLOW}4. Building the project...${NC}"
yarn build
print_status $? "Build completed successfully"

# 5. Start the server in background for testing
echo -e "\n${YELLOW}5. Starting development server for testing...${NC}"
yarn start &
SERVER_PID=$!
echo "Server starting with PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for server to be ready..."
RETRIES=30
while [ $RETRIES -gt 0 ]; do
    if curl -s http://localhost:3000 > /dev/null; then
        echo -e "${GREEN}✓ Server is ready!${NC}"
        break
    fi
    RETRIES=$((RETRIES-1))
    sleep 1
    echo -n "."
done

if [ $RETRIES -eq 0 ]; then
    echo -e "${RED}✗ Server failed to start${NC}"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# 6. Run basic health checks
echo -e "\n${YELLOW}6. Running health checks...${NC}"

# Check homepage
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000)
if [ "$HTTP_STATUS" = "200" ]; then
    echo -e "${GREEN}✓ Homepage returns 200 OK${NC}"
else
    echo -e "${RED}✗ Homepage returned status: $HTTP_STATUS${NC}"
fi

# Check if homepage contains expected content
if curl -s http://localhost:3000 | grep -q "Docusaurus"; then
    echo -e "${GREEN}✓ Homepage contains Docusaurus content${NC}"
else
    echo -e "${RED}✗ Homepage missing expected content${NC}"
fi

# Check documentation page
DOC_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/docs/intro)
if [ "$DOC_STATUS" = "200" ]; then
    echo -e "${GREEN}✓ Documentation page returns 200 OK${NC}"
else
    echo -e "${RED}✗ Documentation page returned status: $DOC_STATUS${NC}"
fi

# Check static assets
IMG_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/img/logo.svg)
if [ "$IMG_STATUS" = "200" ]; then
    echo -e "${GREEN}✓ Static assets are served correctly${NC}"
else
    echo -e "${RED}✗ Static assets returned status: $IMG_STATUS${NC}"
fi

# 7. Cleanup
echo -e "\n${YELLOW}7. Cleaning up...${NC}"
kill $SERVER_PID 2>/dev/null
echo -e "${GREEN}✓ Server stopped${NC}"

echo -e "\n${GREEN}✅ All tests passed!${NC}"
echo "You can now run 'yarn start' to run the development server"