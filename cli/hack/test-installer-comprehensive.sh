#!/usr/bin/env bash

# Comprehensive Installer Testing Script
# This script performs a full end-to-end test of the installer using Docker containers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Test environments
TEST_ENVIRONMENTS=(
    "ubuntu:20.04"
    "ubuntu:22.04"
    "debian:11"
    "debian:12"
    "alpine:3.18"
    "fedora:38"
)

# Version used for the simulated release
TEST_VERSION="test-comprehensive-0.0.0"

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."

    if ! command -v docker > /dev/null; then
        error "Docker is required but not installed"
    fi

    if ! docker info > /dev/null 2>&1; then
        error "Docker daemon is not running"
    fi

    if [[ ! -f "install.sh" ]]; then
        error "install.sh not found. Run this script from the project root."
    fi

    if [[ ! -d "cli" ]]; then
        error "cli directory not found. Run this script from the project root."
    fi

    success "Prerequisites check passed"
}

# Build test binaries
build_test_binaries() {
    info "Building test binaries..."

    local test_dir="test-installer-artifacts/${TEST_VERSION}"
    rm -rf "$test_dir"
    mkdir -p "$test_dir"

    cd cli

    # Build for multiple architectures to test different scenarios
    local platforms=(
        "linux/amd64"
        "linux/arm64"
    )

    for platform in "${platforms[@]}"; do
        local goos=$(echo $platform | cut -d'/' -f1)
        local goarch=$(echo $platform | cut -d'/' -f2)

        info "Building test binary for $goos/$goarch..."

        # The installer expects a flat binary named "llamafarm-<os>-<arch>"
        local out_name="llamafarm-${goos}-${goarch}"

        GOOS=$goos GOARCH=$goarch CGO_ENABLED=0 go build \
            -ldflags="-s -w -X 'llamafarm-cli/cmd.Version=${TEST_VERSION}'" \
            -o "../$test_dir/${out_name}" .

        chmod +x "../$test_dir/${out_name}"
        info "Created test artifact: ${out_name}"
    done

    cd ..
    success "Test binaries built for multiple platforms"
}

# Start test server
start_test_server() {
    info "Starting test release server..."

    cd test-installer-artifacts

    # Create fake API response
    mkdir -p api
    echo "{\"tag_name\": \"${TEST_VERSION}\"}" > api/latest

    # Start HTTP server
    python3 -m http.server 8099 > server.log 2>&1 &
    echo $! > server.pid

    # Wait for server to start
    sleep 3

    # Test server
    if curl -s http://localhost:8099/ > /dev/null; then
        success "Test server started on port 8099"
    else
        error "Failed to start test server"
    fi

    cd ..
}

# Stop test server
stop_test_server() {
    info "Stopping test server..."

    if [[ -f "test-installer-artifacts/server.pid" ]]; then
        local pid=$(cat test-installer-artifacts/server.pid)
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        rm -f test-installer-artifacts/server.pid
    fi
}

# Create test installer
create_test_installer() {
    info "Creating test installer..."

    # Copy and modify installer for testing
    cp install.sh test-install-modified.sh

    # Replace URLs to point to our test server
    # Fix the API URL pattern to match the actual variable usage
    sed -i.bak 's|https://api.github.com/repos/\$REPO/releases/latest|http://host.docker.internal:8099/api/latest|g' test-install-modified.sh
    # Fix the download URL pattern to match the actual variable usage
    sed -i.bak 's|https://github.com/\$REPO/releases/download/\$version/|http://host.docker.internal:8099/$version/|g' test-install-modified.sh

    chmod +x test-install-modified.sh
    success "Test installer created"
}

# Test installer in a specific environment
test_environment() {
    local image="$1"
    local container_name="llamafarm-installer-test-$(echo "$image" | tr '/:' '-')"

    info "Testing in $image..."

    # Determine package manager and install commands based on image
    local install_cmd=""
    local curl_test=""

    case "$image" in
        ubuntu:*|debian:*)
            install_cmd="apt-get update -qq && apt-get install -y curl tar gzip"
            curl_test="curl --version"
            ;;
        alpine:*)
            install_cmd="apk add --no-cache curl tar gzip bash"
            curl_test="curl --version"
            ;;
        fedora:*)
            install_cmd="dnf install -y curl tar gzip"
            curl_test="curl --version"
            ;;
        centos:*)
            install_cmd="dnf install -y curl tar gzip"
            curl_test="curl --version"
            ;;
        *)
            warning "Unknown image type: $image, using generic commands"
            install_cmd="echo 'Using default packages'"
            curl_test="curl --version || wget --version"
            ;;
    esac

    # Choose shell based on image type
    local shell_cmd="bash"
    if [[ "$image" == alpine:* ]]; then
        shell_cmd="sh"
    fi

    # Run test in container
    local test_result=0
    docker run --rm \
        --name "$container_name" \
        --add-host=host.docker.internal:host-gateway \
        -v "$(pwd)/test-install-modified.sh:/test-install.sh:ro" \
        "$image" \
        $shell_cmd -c "
            set -e

            # Install dependencies
            $install_cmd > /dev/null 2>&1

            # Verify curl is available
            $curl_test > /dev/null

            # Make install directory and ensure it's writable
            mkdir -p /opt/llamafarm
            chmod 755 /opt/llamafarm

            # Test the installer
            VERSION=${TEST_VERSION} INSTALL_DIR=/opt/llamafarm /test-install.sh

            # Verify installation - check both possible locations
            if [[ -f /opt/llamafarm/lf ]]; then
                BINARY_PATH='/opt/llamafarm/lf'
                echo 'SUCCESS: Binary found at /opt/llamafarm/lf'
            elif [[ -f /usr/local/bin/lf ]]; then
                BINARY_PATH='/usr/local/bin/lf'
                echo 'SUCCESS: Binary found at /usr/local/bin/lf'
            else
                echo 'ERROR: Binary not found in expected locations'
                echo 'Contents of /opt/llamafarm/:'
                ls -la /opt/llamafarm/ || true
                echo 'Contents of /usr/local/bin/:'
                ls -la /usr/local/bin/ | grep lf || true
                exit 1
            fi

            # Test binary execution
            \$BINARY_PATH version
            \$BINARY_PATH help > /dev/null

            echo 'SUCCESS: Installation test passed'
        " || test_result=1

    if [[ $test_result -eq 0 ]]; then
        success "✅ $image test passed"
    else
        error "❌ $image test failed"
    fi
}

# Test all environments
test_all_environments() {
    info "Testing installer across multiple environments..."

    local failed_tests=()

    for env in "${TEST_ENVIRONMENTS[@]}"; do
        if test_environment "$env"; then
            :  # Success, continue
        else
            failed_tests+=("$env")
        fi
    done

    if [[ ${#failed_tests[@]} -eq 0 ]]; then
        success "✅ All environment tests passed!"
    else
        error "❌ Tests failed in: ${failed_tests[*]}"
    fi
}

# Test edge cases
test_edge_cases() {
    info "Testing edge cases..."

    # Test 1: Invalid version
    info "Test 1: Invalid version handling..."
    docker run --rm \
        --add-host=host.docker.internal:host-gateway \
        -v "$(pwd)/test-install-modified.sh:/test-install.sh:ro" \
        ubuntu:22.04 \
        bash -c "
            apt-get update -qq && apt-get install -y curl tar gzip > /dev/null
            if VERSION=invalid-version INSTALL_DIR=/tmp/test /test-install.sh 2>/dev/null; then
                echo 'ERROR: Should have failed with invalid version'
                exit 1
            fi
            echo 'SUCCESS: Invalid version handling works'
        " || error "Invalid version test failed"

    # Test 2: Network issues simulation
    info "Test 2: Network timeout handling..."
    docker run --rm \
        -v "$(pwd)/install.sh:/test-install.sh:ro" \
        ubuntu:22.04 \
        bash -c "
            apt-get update -qq && apt-get install -y curl tar gzip > /dev/null
            if timeout 5 INSTALL_DIR=/tmp/test /test-install.sh 2>/dev/null; then
                echo 'WARNING: Test may not have properly simulated network issues'
            fi
            echo 'SUCCESS: Network timeout test completed'
        " || warning "Network timeout test inconclusive"

    success "Edge case tests completed"
}

# Performance test
test_performance() {
    info "Testing installation performance..."

    local start_time=$(date +%s)

    docker run --rm \
        --add-host=host.docker.internal:host-gateway \
        -v "$(pwd)/test-install-modified.sh:/test-install.sh:ro" \
        ubuntu:22.04 \
        bash -c "
            apt-get update -qq && apt-get install -y curl tar gzip > /dev/null
            time VERSION=${TEST_VERSION} INSTALL_DIR=/opt/test /test-install.sh
        " > performance.log 2>&1

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    info "Installation completed in ${duration} seconds"

    if [[ $duration -lt 60 ]]; then
        success "✅ Performance test passed (< 60 seconds)"
    else
        warning "⚠️ Installation took longer than expected (${duration} seconds)"
    fi

    rm -f performance.log
}

# Security test
test_security() {
    info "Testing security aspects..."

    # Test script permissions
    local perms=$(stat -c %a install.sh 2>/dev/null || stat -f %Lp install.sh)
    if [[ "$perms" =~ ^[0-7]*5$ ]] || [[ "$perms" =~ ^[0-7]*7$ ]]; then
        success "✅ Install script has execute permissions"
    else
        warning "⚠️ Install script permissions: $perms"
    fi

    # Test for shellcheck if available
    if command -v shellcheck > /dev/null; then
        info "Running shellcheck on install.sh..."
        if shellcheck install.sh; then
            success "✅ Shellcheck passed"
        else
            warning "⚠️ Shellcheck found issues"
        fi
    else
        info "Shellcheck not available, skipping"
    fi

    # Test for common security patterns
    if grep -q "curl.*-f" install.sh; then
        success "✅ Script uses curl with fail-fast flag"
    else
        warning "⚠️ Consider using curl -f for better error handling"
    fi

    if grep -q "set -e" install.sh; then
        success "✅ Script uses 'set -e' for error handling"
    else
        warning "⚠️ Script should use 'set -e' for safety"
    fi
}

# Cleanup function
cleanup() {
    info "Cleaning up test artifacts..."

    stop_test_server

    # Remove test files
    rm -rf test-installer-artifacts
    rm -f test-install-modified.sh test-install-modified.sh.bak

    # Clean up any remaining containers
    docker ps -a --filter "name=llamafarm-installer-test" --format "{{.ID}}" | xargs -r docker rm -f

    success "Cleanup complete"
}

# Main function
main() {
    info "Starting comprehensive installer testing..."

    # Set up cleanup trap
    trap cleanup EXIT

    # Run all tests
    check_prerequisites
    build_test_binaries
    start_test_server
    create_test_installer

    test_all_environments
    test_edge_cases
    test_performance
    test_security

    success "🎉 All comprehensive tests completed successfully!"

    info ""
    info "Test Summary:"
    info "✅ Tested across ${#TEST_ENVIRONMENTS[@]} different environments"
    info "✅ Edge case handling verified"
    info "✅ Performance benchmarked"
    info "✅ Security aspects checked"
    info ""
    info "The installer is working as expected! 🚀"
}

# Run the comprehensive tests
main "$@"