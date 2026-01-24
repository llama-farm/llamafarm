#!/usr/bin/env bash
#
# wt-test.sh - Test script for wt (LlamaFarm Worktree Manager)
#
# Tests the core lifecycle commands: create, list, switch, status, delete
# Does NOT test start/stop (would require full service setup)
#
# Usage: ./wt-test.sh

set -uo pipefail
# Note: Not using -e because we want to continue on test failures

# Use a test-specific root to avoid interfering with real worktrees
export WT_ROOT="${WT_ROOT_TEST:-/tmp/wt-test-worktrees}"
export WT_DATA_ROOT="${WT_DATA_ROOT_TEST:-/tmp/wt-test-data}"

# Get the wt script path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT_SCRIPT="$SCRIPT_DIR/wt"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

# Test worktree name
TEST_WT_NAME="test-wt-feature"
TEST_BRANCH_NAME="test/wt-feature"

log_test() {
    echo -e "${YELLOW}[TEST]${NC} $*"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $*"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $*"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

cleanup() {
    echo ""
    echo "Cleaning up test artifacts..."

    # Remove test worktree if it exists
    if [[ -d "$WT_ROOT/$TEST_WT_NAME" ]]; then
        cd "$SCRIPT_DIR/../.."
        git worktree remove "$WT_ROOT/$TEST_WT_NAME" --force 2>/dev/null || true
    fi

    # Remove test branch if it exists
    cd "$SCRIPT_DIR/../.."
    git branch -D "$TEST_BRANCH_NAME" 2>/dev/null || true

    # Remove test directories
    rm -rf "$WT_ROOT" 2>/dev/null || true
    rm -rf "$WT_DATA_ROOT" 2>/dev/null || true

    echo "Cleanup complete."
}

# Cleanup on exit
trap cleanup EXIT

# =============================================================================
# Tests
# =============================================================================

test_help() {
    log_test "wt help"

    if "$WT_SCRIPT" help | grep -q "LlamaFarm Worktree Manager"; then
        log_pass "help command works"
    else
        log_fail "help command failed"
    fi
}

test_init_bash() {
    log_test "wt init bash"

    local output
    output=$("$WT_SCRIPT" init bash 2>&1)

    if echo "$output" | grep -q "# wt - LlamaFarm Worktree Manager"; then
        log_pass "init bash outputs wt comment header"
    else
        log_fail "init bash missing header"
    fi

    if echo "$output" | grep -q "wt()"; then
        log_pass "init bash outputs wt function"
    else
        log_fail "init bash missing wt function"
    fi

    if echo "$output" | grep -q "_wt_completions()"; then
        log_pass "init bash outputs completions"
    else
        log_fail "init bash missing completions"
    fi

    if echo "$output" | grep -q "complete -F _wt_completions wt"; then
        log_pass "init bash registers completions"
    else
        log_fail "init bash missing completion registration"
    fi
}

test_init_zsh() {
    log_test "wt init zsh"

    local output
    output=$("$WT_SCRIPT" init zsh 2>&1)

    if echo "$output" | grep -q "# wt - LlamaFarm Worktree Manager"; then
        log_pass "init zsh outputs wt comment header"
    else
        log_fail "init zsh missing header"
    fi

    if echo "$output" | grep -q "wt()"; then
        log_pass "init zsh outputs wt function"
    else
        log_fail "init zsh missing wt function"
    fi

    if echo "$output" | grep -q "_wt()"; then
        log_pass "init zsh outputs completions"
    else
        log_fail "init zsh missing completions"
    fi

    if echo "$output" | grep -q "compdef _wt wt"; then
        log_pass "init zsh registers completions"
    else
        log_fail "init zsh missing completion registration"
    fi
}

test_init_fish() {
    log_test "wt init fish"

    local output
    output=$("$WT_SCRIPT" init fish 2>&1)

    if echo "$output" | grep -q "# wt - LlamaFarm Worktree Manager"; then
        log_pass "init fish outputs wt comment header"
    else
        log_fail "init fish missing header"
    fi

    if echo "$output" | grep -q "function wt"; then
        log_pass "init fish outputs wt function"
    else
        log_fail "init fish missing wt function"
    fi

    if echo "$output" | grep -q "complete -c wt"; then
        log_pass "init fish outputs completions"
    else
        log_fail "init fish missing completions"
    fi
}

test_init_auto_detect() {
    log_test "wt init (auto-detect)"

    # Should auto-detect based on $SHELL
    local output
    output=$("$WT_SCRIPT" init 2>&1)

    if echo "$output" | grep -q "wt()"; then
        log_pass "init auto-detect outputs shell function"
    else
        log_fail "init auto-detect failed to output function"
    fi
}

test_init_env_override() {
    log_test "wt init with WT_SHELL override"

    local output
    output=$(WT_SHELL=zsh "$WT_SCRIPT" init 2>&1)

    if echo "$output" | grep -q "compdef _wt wt"; then
        log_pass "init respects WT_SHELL env var"
    else
        log_fail "init ignores WT_SHELL env var"
    fi
}

test_list_empty() {
    log_test "wt list (empty)"

    # Clean slate
    rm -rf "$WT_ROOT" 2>/dev/null || true

    local output
    output=$("$WT_SCRIPT" list 2>&1) || true

    if echo "$output" | grep -q "WORKTREE"; then
        log_pass "list command works with no worktrees"
    else
        log_fail "list command failed. Output: $output"
    fi
}

test_create() {
    log_test "wt create $TEST_BRANCH_NAME"

    # Clean slate
    rm -rf "$WT_ROOT" 2>/dev/null || true
    rm -rf "$WT_DATA_ROOT" 2>/dev/null || true
    cd "$SCRIPT_DIR/../.."
    git branch -D "$TEST_BRANCH_NAME" 2>/dev/null || true

    # Create worktree - may fail on build step (nx not installed in fresh worktree)
    # but worktree/env/data creation should succeed
    "$WT_SCRIPT" create "$TEST_BRANCH_NAME" 2>&1 || true

    # Check worktree directory was created
    if [[ -d "$WT_ROOT/$TEST_WT_NAME" ]]; then
        log_pass "create command created worktree directory"
    else
        log_fail "create command did not create worktree directory"
    fi

    # Check env file was generated
    if [[ -f "$WT_ROOT/$TEST_WT_NAME/.env.wt" ]]; then
        log_pass "create command generated .env.wt"
    else
        log_fail "create command did not generate .env.wt"
    fi

    # Check data directory was created
    if [[ -d "$WT_DATA_ROOT/$TEST_WT_NAME" ]]; then
        log_pass "create command created data directory"
    else
        log_fail "create command did not create data directory"
    fi
}

test_env_file() {
    log_test "Checking .env.wt contents"

    local env_file="$WT_ROOT/$TEST_WT_NAME/.env.wt"

    if [[ ! -f "$env_file" ]]; then
        log_fail ".env.wt file does not exist"
        return
    fi

    # Check for required variables
    if grep -q "WT_NAME=$TEST_WT_NAME" "$env_file"; then
        log_pass ".env.wt contains WT_NAME"
    else
        log_fail ".env.wt missing WT_NAME"
    fi

    if grep -q "LF_SERVER_PORT=" "$env_file"; then
        log_pass ".env.wt contains LF_SERVER_PORT"
    else
        log_fail ".env.wt missing LF_SERVER_PORT"
    fi

    if grep -q "LF_DESIGNER_PORT=" "$env_file"; then
        log_pass ".env.wt contains LF_DESIGNER_PORT"
    else
        log_fail ".env.wt missing LF_DESIGNER_PORT"
    fi

    if grep -q "LF_RUNTIME_PORT=" "$env_file"; then
        log_pass ".env.wt contains LF_RUNTIME_PORT"
    else
        log_fail ".env.wt missing LF_RUNTIME_PORT"
    fi

    if grep -q "LF_DATA_DIR=$WT_DATA_ROOT/$TEST_WT_NAME" "$env_file"; then
        log_pass ".env.wt contains correct LF_DATA_DIR"
    else
        log_fail ".env.wt has incorrect LF_DATA_DIR"
    fi
}

test_list_with_worktree() {
    log_test "wt list (with worktree)"

    # Capture output first to avoid pipe buffering issues with grep -q
    local list_output
    list_output=$("$WT_SCRIPT" list 2>&1)

    if echo "$list_output" | grep -q "$TEST_WT_NAME"; then
        log_pass "list shows created worktree"
    else
        log_fail "list does not show created worktree"
    fi
}

test_switch() {
    log_test "wt switch $TEST_WT_NAME"

    local output
    output=$("$WT_SCRIPT" switch "$TEST_WT_NAME" --print-dir 2>&1)

    if echo "$output" | grep -q "$WT_ROOT/$TEST_WT_NAME"; then
        log_pass "switch returns correct directory"
    else
        log_fail "switch did not return correct directory: $output"
    fi
}

test_status() {
    log_test "wt status $TEST_WT_NAME"

    local output
    output=$("$WT_SCRIPT" status "$TEST_WT_NAME" 2>&1)

    if echo "$output" | grep -q "Worktree: $TEST_WT_NAME"; then
        log_pass "status shows worktree name"
    else
        log_fail "status missing worktree name"
    fi

    if echo "$output" | grep -q "server"; then
        log_pass "status shows server service"
    else
        log_fail "status missing server service"
    fi
}

test_url() {
    log_test "wt url $TEST_WT_NAME"

    local output
    output=$("$WT_SCRIPT" url "$TEST_WT_NAME" 2>&1)

    if echo "$output" | grep -q "Server:"; then
        log_pass "url shows server URL"
    else
        log_fail "url missing server URL"
    fi

    if echo "$output" | grep -q "Designer:"; then
        log_pass "url shows designer URL"
    else
        log_fail "url missing designer URL"
    fi
}

test_port_uniqueness() {
    log_test "Port offset calculation"

    # Source the env file and check ports are in valid range
    source "$WT_ROOT/$TEST_WT_NAME/.env.wt"

    if [[ $LF_SERVER_PORT -ge 8100 && $LF_SERVER_PORT -le 8999 ]]; then
        log_pass "Server port in valid range: $LF_SERVER_PORT"
    else
        log_fail "Server port out of range: $LF_SERVER_PORT"
    fi

    if [[ $LF_DESIGNER_PORT -ge 5100 && $LF_DESIGNER_PORT -le 5999 ]]; then
        log_pass "Designer port in valid range: $LF_DESIGNER_PORT"
    else
        log_fail "Designer port out of range: $LF_DESIGNER_PORT"
    fi

    if [[ $LF_RUNTIME_PORT -ge 11100 && $LF_RUNTIME_PORT -le 11999 ]]; then
        log_pass "Runtime port in valid range: $LF_RUNTIME_PORT"
    else
        log_fail "Runtime port out of range: $LF_RUNTIME_PORT"
    fi
}

test_data_directories() {
    log_test "Data directory structure"

    local data_dir="$WT_DATA_ROOT/$TEST_WT_NAME"

    if [[ -d "$data_dir/logs" ]]; then
        log_pass "logs directory exists"
    else
        log_fail "logs directory missing"
    fi

    if [[ -d "$data_dir/pids" ]]; then
        log_pass "pids directory exists"
    else
        log_fail "pids directory missing"
    fi

    if [[ -d "$data_dir/broker/in" ]]; then
        log_pass "broker/in directory exists"
    else
        log_fail "broker/in directory missing"
    fi

    if [[ -d "$data_dir/broker/processed" ]]; then
        log_pass "broker/processed directory exists"
    else
        log_fail "broker/processed directory missing"
    fi

    if [[ -d "$data_dir/broker/results" ]]; then
        log_pass "broker/results directory exists"
    else
        log_fail "broker/results directory missing"
    fi
}

test_delete() {
    log_test "wt delete $TEST_WT_NAME --force"

    if "$WT_SCRIPT" delete "$TEST_WT_NAME" --force 2>&1; then
        if [[ ! -d "$WT_ROOT/$TEST_WT_NAME" ]]; then
            log_pass "delete removed worktree directory"
        else
            log_fail "delete did not remove worktree directory"
        fi

        if [[ ! -d "$WT_DATA_ROOT/$TEST_WT_NAME" ]]; then
            log_pass "delete removed data directory"
        else
            log_fail "delete did not remove data directory"
        fi
    else
        log_fail "delete command failed"
    fi
}

test_list_after_delete() {
    log_test "wt list (after delete)"

    if ! "$WT_SCRIPT" list 2>&1 | grep -q "$TEST_WT_NAME"; then
        log_pass "list no longer shows deleted worktree"
    else
        log_fail "list still shows deleted worktree"
    fi
}

# =============================================================================
# Main
# =============================================================================

main() {
    echo "========================================"
    echo "wt Test Suite"
    echo "========================================"
    echo ""
    echo "WT_ROOT: $WT_ROOT"
    echo "WT_DATA_ROOT: $WT_DATA_ROOT"
    echo "WT_SCRIPT: $WT_SCRIPT"
    echo ""

    # Verify wt script exists
    if [[ ! -x "$WT_SCRIPT" ]]; then
        echo "Error: wt script not found at $WT_SCRIPT"
        exit 1
    fi

    # Run tests
    test_help
    test_init_bash
    test_init_zsh
    test_init_fish
    test_init_auto_detect
    test_init_env_override
    test_list_empty
    test_create
    test_env_file
    test_list_with_worktree
    test_switch
    test_status
    test_url
    test_port_uniqueness
    test_data_directories
    test_delete
    test_list_after_delete

    # Summary
    echo ""
    echo "========================================"
    echo "Test Summary"
    echo "========================================"
    echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
    echo -e "${RED}Failed: $TESTS_FAILED${NC}"
    echo ""

    if [[ $TESTS_FAILED -gt 0 ]]; then
        exit 1
    fi
}

main "$@"
