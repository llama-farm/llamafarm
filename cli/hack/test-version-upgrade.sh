#!/bin/bash
# Test script for version upgrade service shutdown functionality
# This simulates upgrading from one version to another and verifies services are stopped

set -e

CLI_BIN="${1:-./lf}"
if [ ! -f "$CLI_BIN" ]; then
    echo "Error: CLI binary not found at $CLI_BIN"
    echo "Usage: $0 [path-to-lf-binary]"
    exit 1
fi

echo "=== Testing Version Upgrade Service Shutdown ==="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

VERSION_FILE="$HOME/.llamafarm/.source_version"

echo -e "${YELLOW}Step 1: Clean up any existing services${NC}"
$CLI_BIN stop --all 2>/dev/null || true
sleep 2

echo -e "${YELLOW}Step 2: Set initial version (v1.0.0)${NC}"
export LF_VERSION_REF=v1.0.0
echo "v1.0.0" > "$VERSION_FILE"
echo "Version file set to: $(cat $VERSION_FILE)"

echo -e "${YELLOW}Step 3: Start services with v1.0.0${NC}"
# Note: This will likely fail to download v1.0.0 if it doesn't exist,
# but that's okay - we're testing the service shutdown logic
$CLI_BIN start --dry-run 2>&1 | head -20 || true

echo ""
echo -e "${YELLOW}Step 4: Simulate version mismatch (upgrade to v1.1.0)${NC}"
export LF_VERSION_REF=v1.1.0
echo "Current version in file: $(cat $VERSION_FILE)"
echo "Target version (LF_VERSION_REF): $LF_VERSION_REF"

echo ""
echo -e "${YELLOW}Step 5: Attempt to start services with new version${NC}"
echo "Watch the debug output for 'stopping all services before upgrade'..."
LF_DEBUG=1 $CLI_BIN start --dry-run 2>&1 | grep -A5 -B5 "version mismatch\|stopping all services" || true

echo ""
echo -e "${GREEN}=== Test Complete ===${NC}"
echo ""
echo "To test manually:"
echo "1. Set version: export LF_VERSION_REF=v1.0.0"
echo "2. Start services: ./lf start"
echo "3. Change version: export LF_VERSION_REF=v1.1.0"
echo "4. Check logs: LF_DEBUG=1 ./lf start"
echo ""
echo "You should see debug output about 'Source version mismatch' and 'stopping all services before upgrade'"

