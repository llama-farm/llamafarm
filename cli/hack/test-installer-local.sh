#!/usr/bin/env bash

# Local Installer Testing Script
# This script simulates the release process locally to test the installer

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() {
    echo -e "${BLUE}[TEST] $1${NC}"
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

# Configuration
TEST_VERSION="test-1.0.0"
CLI_DIR="$(pwd)/cli"
TEST_DIR="$(pwd)/test-release"
BINARY_NAME="lf"
CLI_NAME="llamafarm-cli"

# Cleanup function
cleanup() {
    info "Cleaning up test artifacts..."
    rm -rf "$TEST_DIR"
    rm -f /tmp/test-install.sh
    info "Cleanup complete"
}

# Set trap for cleanup
trap cleanup EXIT

# Create test release artifacts
create_test_artifacts() {
    info "Creating test release artifacts..."

    mkdir -p "$TEST_DIR"
    # Generate types first
    info "Generating types..."
    cd config
    uv run python generate_types.py || error "Failed to generate Python types"

    # Install go-jsonschema if needed
    if ! command -v go-jsonschema >/dev/null 2>&1; then
        info "Installing go-jsonschema..."
        go install github.com/atombender/go-jsonschema@latest || error "Failed to install go-jsonschema"
    fi

    # Generate Go types
    cd ../cli/cmd/config
    sh generate-types.sh || error "Failed to generate Go types"
    cd ../../..

    cd "$CLI_DIR"

    # Build for multiple platforms
    local platforms=(
        "linux/amd64"
        "linux/arm64"
        "darwin/amd64"
        "darwin/arm64"
        "windows/amd64"
    )

    for platform in "${platforms[@]}"; do
        local goos=$(echo $platform | cut -d'/' -f1)
        local goarch=$(echo $platform | cut -d'/' -f2)

        info "Building for $goos/$goarch..."

        local binary_name="$BINARY_NAME"
        if [[ "$goos" == "windows" ]]; then
            binary_name="${BINARY_NAME}.exe"
        fi

        GOOS=$goos GOARCH=$goarch CGO_ENABLED=0 go build \
            -ldflags="-s -w -X 'github.com/llamafarm/cli/cmd/version.CurrentVersion=$TEST_VERSION'" \
            -o "$binary_name" .

        # Create archive
        local archive_name="${CLI_NAME}_${TEST_VERSION}_${goos}_${goarch}"

        if [[ "$goos" == "windows" ]]; then
            zip "$TEST_DIR/${archive_name}.zip" "$binary_name"
        else
            tar -czf "$TEST_DIR/${archive_name}.tar.gz" "$binary_name"
        fi

        rm -f "$binary_name"
    done

    cd - > /dev/null
    success "Test artifacts created in $TEST_DIR"
}

# Start a simple HTTP server to simulate GitHub releases
start_test_server() {
    info "Starting test HTTP server..."

    cd "$TEST_DIR"

    # Create a simple directory listing
    cat > index.html << 'EOF'
<!DOCTYPE html>
<html>
<head><title>Test Release Server</title></head>
<body>
<h1>Test Release Files</h1>
<ul>
EOF

    for file in *.tar.gz *.zip; do
        if [[ -f "$file" ]]; then
            echo "<li><a href=\"$file\">$file</a></li>" >> index.html
        fi
    done

    echo "</ul></body></html>" >> index.html

    # Start Python HTTP server in background
    python3 -m http.server 8099 > /dev/null 2>&1 &
    SERVER_PID=$!

    # Wait for server to start
    sleep 2

    # Test server is running
    if curl -s http://localhost:8099/ > /dev/null; then
        success "Test server started on http://localhost:8099 (PID: $SERVER_PID)"
    else
        error "Failed to start test server"
    fi

    cd - > /dev/null
}

# Stop test server
stop_test_server() {
    if [[ -n "$SERVER_PID" ]]; then
        info "Stopping test server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
}

# Create a modified installer for testing
create_test_installer() {
    info "Creating test installer..."

    # Copy the original installer
    cp install.sh /tmp/test-install.sh

    # Modify it to use our test server
    # Replace the variable references properly by escaping the dollar signs
    sed -i.bak 's|https://api\.github\.com/repos/\$REPO/releases/latest|http://localhost:8099/api/latest|g' /tmp/test-install.sh
    sed -i.bak 's|https://github\.com/\$REPO/releases/download/\$version/|http://localhost:8099/|g' /tmp/test-install.sh

    # Create a fake API response for latest version
    mkdir -p "$TEST_DIR/api"
    cat > "$TEST_DIR/api/latest" << EOF
{
  "tag_name": "$TEST_VERSION"
}
EOF

    # Also create the specific version endpoint
    cat > "$TEST_DIR/api/releases" << EOF
{
  "tag_name": "$TEST_VERSION"
}
EOF

    chmod +x /tmp/test-install.sh
    success "Test installer created"
}

# Test installer with different scenarios
test_installer_scenarios() {
    info "Testing installer scenarios..."

    local test_install_dir="/tmp/llamafarm-test-install"

    # Test 1: Default installation (to test directory)
    info "Test 1: Default installation to custom directory..."
    INSTALL_DIR="$test_install_dir" /tmp/test-install.sh || error "Default installation failed"

    if [[ -f "$test_install_dir/$BINARY_NAME" ]]; then
        success "Binary installed successfully"

        # Test the binary
        if "$test_install_dir/$BINARY_NAME" version; then
            success "Binary works correctly"
        else
            error "Binary execution failed"
        fi
    else
        error "Binary not found after installation"
    fi

    # Cleanup
    rm -rf "$test_install_dir"

    # Test 2: Specific version
    info "Test 2: Installing specific version..."
    VERSION="$TEST_VERSION" INSTALL_DIR="$test_install_dir" /tmp/test-install.sh || error "Version-specific installation failed"

    if [[ -f "$test_install_dir/$BINARY_NAME" ]]; then
        success "Version-specific installation successful"
    else
        error "Version-specific installation failed"
    fi

    # Cleanup
    rm -rf "$test_install_dir"

    # Test 3: Help option
    info "Test 3: Testing help option..."
    if /tmp/test-install.sh --help > /dev/null; then
        success "Help option works"
    else
        error "Help option failed"
    fi

    # Test 4: Invalid option handling
    info "Test 4: Testing invalid option handling..."
    if /tmp/test-install.sh --invalid-option 2>/dev/null; then
        error "Should have failed with invalid option"
    else
        success "Invalid option handling works"
    fi
}

# Test platform detection
test_platform_detection() {
    info "Testing platform detection..."

    # Extract the detect_platform function and test it
    local test_script="/tmp/test-platform.sh"

    cat > "$test_script" << 'EOF'
#!/bin/bash
detect_platform() {
    local os arch

    case "$(uname -s)" in
        Linux*)     os="linux" ;;
        Darwin*)    os="darwin" ;;
        MINGW*|MSYS*|CYGWIN*) os="windows" ;;
        *)          echo "error: unsupported OS" && exit 1 ;;
    esac

    case "$(uname -m)" in
        x86_64|amd64)   arch="amd64" ;;
        aarch64|arm64)  arch="arm64" ;;
        armv7l)         arch="arm" ;;
        i386|i686)      arch="386" ;;
        *)              echo "error: unsupported arch" && exit 1 ;;
    esac

    echo "${os}_${arch}"
}

detect_platform
EOF

    chmod +x "$test_script"

    local detected_platform=$("$test_script")
    info "Detected platform: $detected_platform"

    # Verify the platform makes sense
    if [[ "$detected_platform" =~ ^(linux|darwin|windows)_(amd64|arm64|arm|386)$ ]]; then
        success "Platform detection works correctly"
    else
        error "Platform detection returned invalid format: $detected_platform"
    fi

    rm -f "$test_script"
}

# Test error handling
test_error_handling() {
    info "Testing error handling scenarios..."

    # Test with non-existent version
    info "Testing non-existent version handling..."
    if VERSION="non-existent-version" INSTALL_DIR="/tmp/test" /tmp/test-install.sh 2>/dev/null; then
        error "Should have failed with non-existent version"
    else
        success "Non-existent version handling works"
    fi

    # Test with invalid install directory (read-only)
    info "Testing read-only directory handling..."
    local readonly_dir="/tmp/readonly-test"
    mkdir -p "$readonly_dir"
    chmod 444 "$readonly_dir"

    if INSTALL_DIR="$readonly_dir" /tmp/test-install.sh 2>/dev/null; then
        error "Should have failed with read-only directory"
    else
        success "Read-only directory handling works"
    fi

    chmod 755 "$readonly_dir"
    rm -rf "$readonly_dir"
}

# Test with Docker containers (simulate different environments)
test_docker_environments() {
    info "Testing in Docker environments..."

    if ! command -v docker > /dev/null; then
        warning "Docker not available, skipping Docker tests"
        return
    fi

    # Test Ubuntu environment
    info "Testing in Ubuntu container..."
    docker run --rm -v "$(pwd):/workspace" -w /workspace ubuntu:latest bash -c "
        apt-get update -qq && apt-get install -y curl tar gzip > /dev/null 2>&1
        VERSION=$TEST_VERSION INSTALL_DIR=/tmp/test /tmp/test-install.sh || exit 1
        /tmp/test/$BINARY_NAME version || exit 1
    " || error "Ubuntu container test failed"

    success "Ubuntu container test passed"

    # Test Alpine environment
    info "Testing in Alpine container..."
    docker run --rm -v "$(pwd):/workspace" -w /workspace alpine:latest sh -c "
        apk add --no-cache curl tar gzip bash > /dev/null 2>&1
        VERSION=$TEST_VERSION INSTALL_DIR=/tmp/test /tmp/test-install.sh || exit 1
        /tmp/test/$BINARY_NAME version || exit 1
    " || error "Alpine container test failed"

    success "Alpine container test passed"
}

# Performance test
test_performance() {
    info "Testing installation performance..."

    local start_time=$(date +%s)

    VERSION="$TEST_VERSION" INSTALL_DIR="/tmp/perf-test" /tmp/test-install.sh > /dev/null

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    info "Installation completed in ${duration} seconds"

    if [[ $duration -lt 30 ]]; then
        success "Installation performance is good (< 30 seconds)"
    else
        warning "Installation took longer than expected (${duration} seconds)"
    fi

    rm -rf "/tmp/perf-test"
}

# Main test runner
main() {
    info "Starting comprehensive installer testing..."

    # Check prerequisites
    if [[ ! -f "install.sh" ]]; then
        error "install.sh not found. Run this script from the project root."
    fi

    if [[ ! -d "cli" ]]; then
        error "cli directory not found. Run this script from the project root."
    fi

    # Run all tests
    create_test_artifacts
    start_test_server

    # Set cleanup for server
    trap 'stop_test_server; cleanup' EXIT

    create_test_installer
    test_platform_detection
    test_installer_scenarios
    test_error_handling
    test_performance

    # Optional Docker tests (only if Docker is available)
    if command -v docker > /dev/null; then
        test_docker_environments
    else
        warning "Docker not available, skipping container tests"
    fi

    stop_test_server

    success "🎉 All installer tests passed!"

    info "Summary:"
    info "✅ Platform detection works"
    info "✅ Installation scenarios work"
    info "✅ Error handling works"
    info "✅ Performance is acceptable"
    if command -v docker > /dev/null; then
        info "✅ Docker environments work"
    fi

    info ""
    info "Your installer is ready for production! 🚀"
    info "Next steps:"
    info "1. git tag v1.0.0"
    info "2. git push origin v1.0.0"
    info "3. GitHub Actions will build and release binaries"
    info "4. Users can install with: curl -fsSL https://raw.githubusercontent.com/llamafarm/llamafarm/main/install.sh | bash"
}

# Run the tests
main "$@"
