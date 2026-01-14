#!/bin/bash
# Run All ML Example Scripts
#
# This script runs all ML example scripts sequentially and reports results.
# Requires the Universal Runtime to be running (uses LF_RUNTIME_PORT from .env).
#
# Usage: ./run_all_ml_examples.sh [--skip-api]
#   --skip-api: Skip LlamaFarm API tests (port 8000)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Load port from .env file
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    source "$SCRIPT_DIR/../../.env"
fi
RUNTIME_PORT="${LF_RUNTIME_PORT:-11540}"
API_PORT="${PORT:-8000}"

SKIP_API=false

if [ "$1" = "--skip-api" ]; then
    SKIP_API=true
fi

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Run All ML Example Scripts${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if Universal Runtime is running
echo -e "${YELLOW}Checking Universal Runtime (port $RUNTIME_PORT)...${NC}"
if ! curl -s "http://localhost:$RUNTIME_PORT/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Universal Runtime not running on port $RUNTIME_PORT${NC}"
    echo "Start it with: cd runtimes/universal && uv run python server.py"
    exit 1
fi
echo -e "${GREEN}✓ Universal Runtime is healthy${NC}"
echo ""

# Check if LlamaFarm API is running (if not skipping)
if [ "$SKIP_API" = false ]; then
    echo -e "${YELLOW}Checking LlamaFarm API (port $API_PORT)...${NC}"
    if ! curl -s "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠ LlamaFarm API not running - skipping API tests${NC}"
        SKIP_API=true
    else
        echo -e "${GREEN}✓ LlamaFarm API is healthy${NC}"
    fi
    echo ""
fi

# Direct runtime tests
DIRECT_TESTS=(
    "test_anomaly.sh"
    "test_anomaly_explain.sh"
    "test_clip.sh"
    "test_object_detection.sh"
    "test_background_removal.sh"
    "test_language_detection.sh"
    "test_keywords.sh"
    "test_pii_redaction.sh"
    "test_time_series.sh"
    "test_drift_detection.sh"
    "test_dataset_audit.sh"
)

# API proxy tests
API_TESTS=(
    "test_anomaly_api.sh"
    "test_anomaly_explain_api.sh"
    "test_clip_api.sh"
    "test_object_detection_api.sh"
    "test_background_removal_api.sh"
    "test_language_detection_api.sh"
    "test_keywords_api.sh"
    "test_pii_redaction_api.sh"
    "test_time_series_api.sh"
    "test_drift_detection_api.sh"
    "test_dataset_audit_api.sh"
)

PASS=0
FAIL=0
SKIP=0

run_test() {
    local SCRIPT=$1
    local SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT}"

    if [ ! -f "$SCRIPT_PATH" ]; then
        echo -e "${YELLOW}⚠ SKIP: ${SCRIPT} (not found)${NC}"
        SKIP=$((SKIP + 1))
        return
    fi

    echo -e "${CYAN}Running: ${SCRIPT}${NC}"

    if bash "$SCRIPT_PATH" > /tmp/ml_test_output.txt 2>&1; then
        echo -e "${GREEN}✓ PASS: ${SCRIPT}${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}✗ FAIL: ${SCRIPT}${NC}"
        echo "  Output:"
        tail -5 /tmp/ml_test_output.txt | sed 's/^/    /'
        FAIL=$((FAIL + 1))
    fi
    echo ""
}

# Run direct tests
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Direct Universal Runtime Tests${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

for TEST in "${DIRECT_TESTS[@]}"; do
    run_test "$TEST"
done

# Run API tests if not skipped
if [ "$SKIP_API" = false ]; then
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  LlamaFarm API Proxy Tests${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""

    for TEST in "${API_TESTS[@]}"; do
        run_test "$TEST"
    done
else
    echo -e "${YELLOW}Skipping API tests${NC}"
    echo ""
fi

# Summary
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Test Results Summary${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo -e "  ${GREEN}Passed:${NC}  ${PASS}"
echo -e "  ${RED}Failed:${NC}  ${FAIL}"
echo -e "  ${YELLOW}Skipped:${NC} ${SKIP}"
echo ""

if [ $FAIL -gt 0 ]; then
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi

echo -e "${GREEN}All tests passed!${NC}"
echo ""
