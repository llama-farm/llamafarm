#!/bin/bash
# Run All New RAG Demos
#
# This script runs all the demos for the new RAG parsing system.
# It can be run with or without a running server.
#
# Usage:
#   bash run_all_demos.sh           # Run local demos only
#   bash run_all_demos.sh --api     # Run API demos (requires server)
#   bash run_all_demos.sh --all     # Run all demos

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAG_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")/rag"
SERVER_URL="${SERVER_URL:-http://localhost:8005}"

# Parse arguments
RUN_API_DEMOS=false
RUN_ALL=false

for arg in "$@"; do
    case $arg in
        --api)
            RUN_API_DEMOS=true
            ;;
        --all)
            RUN_ALL=true
            RUN_API_DEMOS=true
            ;;
    esac
done

echo "============================================================"
echo "LLAMAFARM RAG - Running All Demos"
echo "============================================================"
echo ""
echo "Script directory: ${SCRIPT_DIR}"
echo "RAG directory: ${RAG_DIR}"
echo ""

# Track results
PASSED=0
FAILED=0
SKIPPED=0

run_demo() {
    local name="$1"
    local command="$2"
    local requires_server="${3:-false}"

    echo ""
    echo -e "${BLUE}>>> Running: ${name}${NC}"
    echo "Command: ${command}"
    echo "-"

    if [ "$requires_server" = true ]; then
        # Check if server is running
        if ! curl -s -o /dev/null -w "%{http_code}" "${SERVER_URL}/health" 2>/dev/null | grep -q "200"; then
            echo -e "${YELLOW}SKIPPED${NC} (server not running)"
            ((SKIPPED++))
            return 0
        fi
    fi

    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}PASSED${NC}"
        ((PASSED++))
    else
        echo -e "${RED}FAILED${NC}"
        ((FAILED++))
        # Show error output
        eval "$command" 2>&1 | tail -20 || true
    fi
}

# Demo 1: Parser Basics
echo ""
echo "============================================================"
echo "LOCAL DEMOS (No Server Required)"
echo "============================================================"

run_demo "Demo 01: Parser Basics" \
    "cd ${RAG_DIR} && uv run python ${SCRIPT_DIR}/demo_01_parser_basics.py"

run_demo "Demo 02: Chunking Strategies" \
    "cd ${RAG_DIR} && uv run python ${SCRIPT_DIR}/demo_02_chunking_strategies.py"

run_demo "Demo 03: Docling Full Pipeline" \
    "cd ${RAG_DIR} && uv run python ${SCRIPT_DIR}/demo_03_docling_full_pipeline.py"

run_demo "Demo 04: Simple Configuration" \
    "cd ${RAG_DIR} && uv run python ${SCRIPT_DIR}/demo_04_simple_config.py"

run_demo "Demo 06: Full Pipeline Integration" \
    "cd ${RAG_DIR} && uv run python ${SCRIPT_DIR}/demo_06_full_pipeline.py"

# API Demos (require server)
if [ "$RUN_API_DEMOS" = true ]; then
    echo ""
    echo "============================================================"
    echo "API DEMOS (Require Running Server)"
    echo "============================================================"

    run_demo "Demo 05: Parse API" \
        "bash ${SCRIPT_DIR}/demo_05_parse_api.sh" \
        true
fi

# Run tests
echo ""
echo "============================================================"
echo "RUNNING UNIT TESTS"
echo "============================================================"

# RAG component tests
echo ""
echo -e "${BLUE}>>> Running RAG Unit Tests${NC}"
if cd "${RAG_DIR}" && uv run pytest tests/components/parsers/test_docling_parser.py tests/components/parsers/test_markitdown_parser.py tests/components/chunkers/ -v --tb=short 2>&1 | tail -30; then
    echo -e "${GREEN}RAG Tests: PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}RAG Tests: FAILED${NC}"
    ((FAILED++))
fi

# Integration tests
echo ""
echo -e "${BLUE}>>> Running Integration Tests${NC}"
if cd "${RAG_DIR}" && uv run pytest tests/test_full_pipeline_integration.py -v --tb=short 2>&1 | tail -20; then
    echo -e "${GREEN}Integration Tests: PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}Integration Tests: FAILED${NC}"
    ((FAILED++))
fi

# Schema validation tests
echo ""
echo -e "${BLUE}>>> Running Schema Validation Tests${NC}"
if cd "${RAG_DIR}" && uv run pytest tests/test_schema_validation.py -v --tb=short 2>&1 | tail -20; then
    echo -e "${GREEN}Schema Tests: PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}Schema Tests: FAILED${NC}"
    ((FAILED++))
fi

# API tests (if server running)
if [ "$RUN_API_DEMOS" = true ]; then
    echo ""
    echo -e "${BLUE}>>> Running API Integration Tests${NC}"

    # Check server
    if curl -s -o /dev/null -w "%{http_code}" "${SERVER_URL}/health" 2>/dev/null | grep -q "200"; then
        # Install requests if needed
        cd "${SCRIPT_DIR}" && pip install requests -q 2>/dev/null || true

        if cd "${SCRIPT_DIR}" && python -m pytest tests/test_api_integration.py -v --tb=short 2>&1 | tail -20; then
            echo -e "${GREEN}API Tests: PASSED${NC}"
            ((PASSED++))
        else
            echo -e "${RED}API Tests: FAILED${NC}"
            ((FAILED++))
        fi
    else
        echo -e "${YELLOW}API Tests: SKIPPED (server not running)${NC}"
        ((SKIPPED++))
    fi
fi

# Summary
echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo ""
echo -e "  ${GREEN}Passed:${NC}  ${PASSED}"
echo -e "  ${RED}Failed:${NC}  ${FAILED}"
echo -e "  ${YELLOW}Skipped:${NC} ${SKIPPED}"
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
