#!/bin/bash
#
# Run All Database Demos
#
# Demonstrates the Embedded Trinity Memory System components:
# - DuckDB Store (time-series, spatial)
# - Graph Store (relationships, path finding)
# - Working Memory (TTL buffer)
# - Linkage Table (cross-database linking)
# - UnifiedDatasetStore (Phase 3 Unified Dataset Architecture)
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RAG_DIR="$PROJECT_DIR/rag"

echo -e "${CYAN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Embedded Trinity Memory System - All Demos            ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

cd "$RAG_DIR"

# Track results
PASSED=0
FAILED=0

run_demo() {
    local name="$1"
    local script="$2"

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Running: $name${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    if uv run python "$script"; then
        echo ""
        echo -e "${GREEN}✓ $name - PASSED${NC}"
        PASSED=$((PASSED + 1))
    else
        echo ""
        echo -e "${RED}✗ $name - FAILED${NC}"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# Run all demos
run_demo "DuckDB Store (Time-Series)" "$SCRIPT_DIR/demo_duckdb_store.py"
run_demo "Graph Store (Relationships)" "$SCRIPT_DIR/demo_graph_store.py"
run_demo "Working Memory (TTL Buffer)" "$SCRIPT_DIR/demo_working_memory.py"
run_demo "Linkage Table (Cross-DB)" "$SCRIPT_DIR/demo_linkage_table.py"
run_demo "MemoryStore (Unified Interface)" "$SCRIPT_DIR/demo_memory_store.py"
run_demo "Consolidator (Memory Synthesis)" "$SCRIPT_DIR/demo_consolidator.py"
run_demo "UnifiedDatasetStore (Phase 3)" "$SCRIPT_DIR/demo_unified_dataset.py"

# Summary
echo -e "${CYAN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Summary                                               ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Passed: $PASSED${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}Failed: 0${NC}"
    echo ""
    echo -e "${GREEN}All demos completed successfully!${NC}"
fi
