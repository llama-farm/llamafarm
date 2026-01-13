#!/bin/bash
# Validate ML Example Scripts
#
# This script checks that all ML example scripts are:
# 1. Present and executable
# 2. Include health checks
# 3. Have proper error handling
#
# Usage: ./validate_ml_examples.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLES_DIR="${SCRIPT_DIR}/ocr_and_document"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  ML Example Scripts Validation${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Expected ML example scripts
EXPECTED_SCRIPTS=(
    # Vision
    "test_clip.sh"
    "test_clip_api.sh"
    "test_object_detection.sh"
    "test_object_detection_api.sh"
    "test_background_removal.sh"
    "test_background_removal_api.sh"
    # NLP
    "test_language_detection.sh"
    "test_language_detection_api.sh"
    "test_keywords.sh"
    "test_keywords_api.sh"
    "test_pii_redaction.sh"
    "test_pii_redaction_api.sh"
    # Time-Series
    "test_time_series.sh"
    "test_time_series_api.sh"
    "test_drift_detection.sh"
    "test_drift_detection_api.sh"
    # Anomaly
    "test_anomaly.sh"
    "test_anomaly_api.sh"
    "test_anomaly_explain.sh"
    "test_anomaly_explain_api.sh"
    # Data Quality
    "test_dataset_audit.sh"
    "test_dataset_audit_api.sh"
)

PASS=0
FAIL=0
WARN=0

echo -e "${YELLOW}Checking scripts in ${EXAMPLES_DIR}...${NC}"
echo ""

for SCRIPT in "${EXPECTED_SCRIPTS[@]}"; do
    SCRIPT_PATH="${EXAMPLES_DIR}/${SCRIPT}"

    if [ ! -f "$SCRIPT_PATH" ]; then
        echo -e "${RED}✗ MISSING: ${SCRIPT}${NC}"
        FAIL=$((FAIL + 1))
        continue
    fi

    # Check if executable
    if [ ! -x "$SCRIPT_PATH" ]; then
        echo -e "${YELLOW}⚠ NOT EXECUTABLE: ${SCRIPT}${NC}"
        chmod +x "$SCRIPT_PATH"
        WARN=$((WARN + 1))
    fi

    # Check for health check
    if ! grep -q "health" "$SCRIPT_PATH"; then
        echo -e "${YELLOW}⚠ NO HEALTH CHECK: ${SCRIPT}${NC}"
        WARN=$((WARN + 1))
    fi

    # Check for error handling (set -e)
    if ! grep -q "set -e" "$SCRIPT_PATH"; then
        echo -e "${YELLOW}⚠ NO ERROR HANDLING: ${SCRIPT}${NC}"
        WARN=$((WARN + 1))
    fi

    echo -e "${GREEN}✓ ${SCRIPT}${NC}"
    PASS=$((PASS + 1))
done

echo ""
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Validation Summary${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo -e "  ${GREEN}Passed:${NC} ${PASS}"
echo -e "  ${YELLOW}Warnings:${NC} ${WARN}"
echo -e "  ${RED}Failed:${NC} ${FAIL}"
echo ""

if [ $FAIL -gt 0 ]; then
    echo -e "${RED}Some scripts are missing!${NC}"
    exit 1
fi

echo -e "${GREEN}All scripts validated successfully!${NC}"
echo ""
