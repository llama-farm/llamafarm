#!/usr/bin/env bash

# Test script for the LlamaFarm CLI installation
# This script tests the installation process locally

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
    echo -e "${GREEN}[TEST] $1${NC}"
}

error() {
    echo -e "${RED}[TEST] $1${NC}"
    exit 1
}

# Test help functionality
test_help() {
    info "Testing help functionality..."
    ./install.sh --help >/dev/null || error "Help option failed"
    success "Help functionality works"
}

# Test CLI build (local development test)
test_local_build() {
    info "Testing local CLI build..."

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

    # Navigate to CLI directory
    cd cli

    # Build the CLI locally
    info "Building CLI locally..."
    go build -ldflags="-X 'github.com/llamafarm/cli/cmd/version.CurrentVersion=test-1.0.0'" -o lf . || error "Failed to build CLI"

    # Test the binary
    info "Testing built binary..."
    ./lf version || error "Binary version command failed"
    ./lf help >/dev/null || error "Binary help command failed"

    success "Local build test passed"

    # Cleanup
    rm -f lf
    cd ..
}

# Test cross-compilation
test_cross_compile() {
    info "Testing cross-compilation..."

    # Types should already be generated from test_local_build
    # But ensure they exist
    if [[ ! -f "cli/cmd/config/types.go" ]]; then
        error "Go types not generated. Run test_local_build first."
    fi

    cd cli

    # Test Linux build
    info "Testing Linux amd64 build..."
    GOOS=linux GOARCH=amd64 go build -ldflags="-s -w -X 'github.com/llamafarm/cli/cmd/version.CurrentVersion=test-1.0.0'" -o lf-linux . || error "Linux build failed"

    # Test macOS build
    info "Testing macOS amd64 build..."
    GOOS=darwin GOARCH=amd64 go build -ldflags="-s -w -X 'github.com/llamafarm/cli/cmd/version.CurrentVersion=test-1.0.0'" -o lf-macos . || error "macOS build failed"

    # Test Windows build
    info "Testing Windows amd64 build..."
    GOOS=windows GOARCH=amd64 go build -ldflags="-s -w -X 'github.com/llamafarm/cli/cmd/version.CurrentVersion=test-1.0.0'" -o lf-windows.exe . || error "Windows build failed"

    success "Cross-compilation test passed"

    # Show file sizes
    info "Build artifact sizes:"
    ls -lh lf-* | awk '{print $5 " " $9}'

    # Cleanup
    rm -f lf-*
    cd ..
}

# Test install script validation
test_install_script() {
    info "Testing install script validation..."

    # Test with invalid arguments
    if ./install.sh --invalid-option 2>/dev/null; then
        error "Install script should reject invalid options"
    fi

    success "Install script validation works"
}

# Main test runner
main() {
    info "Starting LlamaFarm CLI installation tests..."

    # Check if we're in the right directory
    if [[ ! -f "install.sh" || ! -d "cli" ]]; then
        error "Please run this script from the project root directory"
    fi

    # Run tests
    test_help
    test_install_script
    test_local_build
    test_cross_compile

    success "All tests passed! 🎉"

    info "Next steps:"
    echo "1. Create a git tag: git tag v1.0.0"
    echo "2. Push the tag: git push origin v1.0.0"
    echo "3. The GitHub Actions workflow will build and release binaries"
    echo "4. Users can then install with: curl -fsSL https://raw.githubusercontent.com/llamafarm/llamafarm/main/install.sh | bash"
}

main "$@"
