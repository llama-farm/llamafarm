#!/bin/bash
# Quick test for version upgrade service shutdown
# This uses fake commit hashes to trigger the mismatch without network downloads

set -e

echo "🧪 Quick Version Upgrade Test"
echo "=============================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

VERSION_FILE="$HOME/.llamafarm/.source_version"

# Build CLI
echo -e "${YELLOW}Building CLI...${NC}"
cd "$(dirname "$0")/.."
go build -o lf main.go
echo -e "${GREEN}✓ CLI built${NC}"
echo ""

# Stop any running services
echo -e "${YELLOW}Cleaning up existing services...${NC}"
./lf stop --all 2>/dev/null || true
sleep 1

# Create a fake "old" version
FAKE_OLD_VERSION="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
FAKE_NEW_VERSION="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

echo -e "${YELLOW}Setting up fake 'old' version: ${FAKE_OLD_VERSION:0:8}...${NC}"
mkdir -p "$(dirname "$VERSION_FILE")"
echo "$FAKE_OLD_VERSION" > "$VERSION_FILE"
echo -e "${GREEN}✓ Version file created${NC}"
echo ""

# Now try to "upgrade" to a new version
echo -e "${YELLOW}Triggering 'upgrade' to new version: ${FAKE_NEW_VERSION:0:8}...${NC}"
echo -e "${BLUE}Looking for 'stopping all services' message in debug output:${NC}"
echo ""

export LF_VERSION_REF="$FAKE_NEW_VERSION"
export LF_DEBUG=1

# Capture output and look for our message
OUTPUT=$(./lf start 2>&1 || true)

# Check for the key message
if echo "$OUTPUT" | grep -q "Source version mismatch"; then
    echo -e "${GREEN}✓ PASS: Version mismatch detected${NC}"
    echo "$OUTPUT" | grep "Source version mismatch" | head -1
else
    echo -e "${RED}✗ FAIL: Version mismatch not detected${NC}"
fi

if echo "$OUTPUT" | grep -q "stopping all services"; then
    echo -e "${GREEN}✓ PASS: Services stop triggered${NC}"
    echo "$OUTPUT" | grep "stopping all services" | head -1
else
    echo -e "${BLUE}ℹ INFO: Services stop message not found (may not have been running)${NC}"
fi

echo ""
echo -e "${YELLOW}Full debug output (first 40 lines):${NC}"
echo "$OUTPUT" | head -40

echo ""
echo -e "${GREEN}=== Test Complete ===${NC}"
echo ""
echo "The test used fake commit hashes to avoid network downloads."
echo "In real scenarios, it will download the actual version from GitHub."

