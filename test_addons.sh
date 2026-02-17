#!/bin/bash
set -e

echo "=== LlamaFarm Addon System Integration Tests ==="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
TESTS_PASSED=0
TESTS_FAILED=0

test_passed() {
    echo -e "${GREEN}✓ $1${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

test_failed() {
    echo -e "${RED}✗ $1${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

test_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Helper function to check if a package is installed
check_package() {
    local package=$1
    local runtime_dir
    runtime_dir="$(cd "$(dirname "$0")" && pwd)/runtimes/universal"
    if uv pip list --python "${runtime_dir}/.venv/bin/python" 2>/dev/null | grep -q "^${package} "; then
        return 0
    else
        return 1
    fi
}

# Helper function to check addon state
check_addon_state() {
    local addon=$1
    local expected_state=$2  # "installed" or "not_installed"

    if [ -f ~/.llamafarm/addons.json ]; then
        if jq -e ".installed_addons.${addon}" ~/.llamafarm/addons.json > /dev/null 2>&1; then
            if [ "$expected_state" = "installed" ]; then
                test_passed "Addon '${addon}' is marked as installed in state file"
                return 0
            else
                test_failed "Addon '${addon}' should NOT be in state file but is"
                return 1
            fi
        else
            if [ "$expected_state" = "not_installed" ]; then
                test_passed "Addon '${addon}' is correctly not in state file"
                return 0
            else
                test_failed "Addon '${addon}' should be in state file but isn't"
                return 1
            fi
        fi
    else
        if [ "$expected_state" = "not_installed" ]; then
            test_passed "No addons.json file (fresh state)"
            return 0
        else
            test_failed "Expected addon state file but it doesn't exist"
            return 1
        fi
    fi
}

echo "📋 Pre-test Setup"
echo "=================="
echo ""

# Check current addon state
test_info "Current addon state:"
if [ -f ~/.llamafarm/addons.json ]; then
    cat ~/.llamafarm/addons.json | jq '.installed_addons // {}' || echo "Failed to parse addons.json"
else
    echo "  No addons currently installed"
fi
echo ""

# Check if services are running
test_info "Checking if services are running..."
if lsof -ti:11540 > /dev/null 2>&1; then
    echo "  Universal runtime is running on port 11540"
    RUNTIME_WAS_RUNNING=true
else
    echo "  Universal runtime is NOT running"
    RUNTIME_WAS_RUNNING=false
fi

if lsof -ti:8000 > /dev/null 2>&1; then
    echo "  Server is running on port 8000"
    SERVER_WAS_RUNNING=true
else
    echo "  Server is NOT running"
    SERVER_WAS_RUNNING=false
fi
echo ""

echo "🧪 Test 1: Addon Installation (STT)"
echo "===================================="
test_info "Installing STT addon via API..."

# Install STT via API
INSTALL_RESPONSE=$(curl -s -X POST http://localhost:8000/v1/addons/install \
    -H "Content-Type: application/json" \
    -d '{"name": "stt", "restart_service": true}' 2>&1)

if echo "$INSTALL_RESPONSE" | grep -q "task_id"; then
    TASK_ID=$(echo "$INSTALL_RESPONSE" | jq -r '.task_id')
    test_passed "Installation initiated (task_id: $TASK_ID)"

    # Wait for installation to complete
    test_info "Waiting for installation to complete..."
    MAX_WAIT=180  # 3 minutes max
    ELAPSED=0
    while [ $ELAPSED -lt $MAX_WAIT ]; do
        STATUS_RESPONSE=$(curl -s http://localhost:8000/v1/addons/tasks/$TASK_ID 2>&1)
        STATUS=$(echo "$STATUS_RESPONSE" | jq -r '.status')

        if [ "$STATUS" = "completed" ]; then
            test_passed "Installation completed successfully"
            break
        elif [ "$STATUS" = "failed" ]; then
            ERROR=$(echo "$STATUS_RESPONSE" | jq -r '.error')
            test_failed "Installation failed: $ERROR"
            break
        fi

        sleep 2
        ELAPSED=$((ELAPSED + 2))
    done

    if [ $ELAPSED -ge $MAX_WAIT ]; then
        test_failed "Installation timed out after ${MAX_WAIT}s"
    fi
else
    test_failed "Failed to initiate installation: $INSTALL_RESPONSE"
fi
echo ""

echo "🔍 Test 2: Verify STT Installation"
echo "==================================="

# Check addon state file
check_addon_state "stt" "installed"

# Check if STT packages are installed
test_info "Checking if STT packages are installed..."
if check_package "faster-whisper"; then
    test_passed "faster-whisper package is installed"
else
    test_failed "faster-whisper package NOT installed"
fi

if check_package "av"; then
    test_passed "av package is installed"
else
    test_failed "av package NOT installed"
fi

# Check if onnxruntime dependency was installed
if check_package "onnxruntime"; then
    test_passed "onnxruntime dependency is installed"
else
    test_failed "onnxruntime dependency NOT installed"
fi
echo ""

echo "🔄 Test 3: Server Restart Verification"
echo "======================================="
test_info "Checking if services restarted after installation..."

# Give services time to restart
sleep 5

if lsof -ti:11540 > /dev/null 2>&1; then
    test_passed "Universal runtime is running after installation"
else
    test_failed "Universal runtime is NOT running after installation"
fi

if lsof -ti:8000 > /dev/null 2>&1; then
    test_passed "Server is running after installation"
else
    test_failed "Server is NOT running after installation"
fi
echo ""

echo "🧪 Test 4: STT Functionality Test"
echo "=================================="
test_info "Testing if STT endpoint is available..."

# Test if /v1/speech/transcribe endpoint exists
STT_ENDPOINT_CHECK=$(curl -s -X POST http://localhost:11540/v1/speech/transcribe \
    -H "Content-Type: application/json" \
    -d '{}' 2>&1)

if echo "$STT_ENDPOINT_CHECK" | grep -qE "model|error|detail"; then
    test_passed "STT endpoint is responding"
else
    test_failed "STT endpoint not responding correctly"
fi
echo ""

echo "🗑️  Test 5: Addon Removal (STT)"
echo "==============================="
test_info "Removing STT addon via API..."

UNINSTALL_RESPONSE=$(curl -s -X POST http://localhost:8000/v1/addons/uninstall \
    -H "Content-Type: application/json" \
    -d '{"name": "stt"}' 2>&1)

if echo "$UNINSTALL_RESPONSE" | grep -q "success\|removed"; then
    test_passed "Uninstallation initiated successfully"

    # Wait a bit for uninstallation
    sleep 5
else
    test_failed "Failed to initiate uninstallation: $UNINSTALL_RESPONSE"
fi
echo ""

echo "🔍 Test 6: Verify STT Removal"
echo "============================="

# Check addon state file
check_addon_state "stt" "not_installed"

# Check if STT packages are removed
test_info "Checking if STT packages were removed..."
if check_package "faster-whisper"; then
    test_failed "faster-whisper package is STILL installed (should be removed)"
else
    test_passed "faster-whisper package was successfully removed"
fi

if check_package "av"; then
    test_failed "av package is STILL installed (should be removed)"
else
    test_passed "av package was successfully removed"
fi

# Check if onnxruntime was also removed (since nothing depends on it now)
if check_package "onnxruntime"; then
    test_info "onnxruntime is still installed (this may be expected if other addons need it)"
else
    test_passed "onnxruntime was removed (no dependencies remain)"
fi
echo ""

echo ""
echo "============================================"
echo "           TEST SUMMARY"
echo "============================================"
echo -e "${GREEN}Passed: ${TESTS_PASSED}${NC}"
echo -e "${RED}Failed: ${TESTS_FAILED}${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    exit 1
fi
