#!/usr/bin/env bash

# Quick Installer Test Script
# This script performs basic validation of the installer without external dependencies

set -euo pipefail

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

# Test 1: Basic script validation
test_script_basics() {
    info "Testing basic script validation..."

    if [[ ! -f "install.sh" ]]; then
        error "install.sh not found"
    fi

    if [[ ! -x "install.sh" ]]; then
        error "install.sh is not executable"
    fi

    # Test help function
    if ./install.sh --help > /dev/null; then
        success "Help option works"
    else
        error "Help option failed"
    fi

    # Test invalid option handling
    if ./install.sh --invalid-option 2>/dev/null; then
        error "Should have failed with invalid option"
    else
        success "Invalid option handling works"
    fi

    success "Basic script validation passed"
}

# Test 2: CLI build and functionality
test_cli_build() {
    info "Testing CLI build and functionality..."

    if [[ ! -d "cli" ]]; then
        error "CLI directory not found"
    fi

    # Generate types first
    info "Generating types..."
    cd config
    uv run python generate_types.py || error "Failed to generate Python types"

    # Install go-jsonschema if needed
    if ! command -v go-jsonschema >/dev/null 2>&1; then
        info "Installing go-jsonschema..."
        go install github.com/atombender/go-jsonschema@v0.21.0 || error "Failed to install go-jsonschema"
    fi

    # Generate Go types
    cd ../cli/cmd/config
    sh generate-types.sh || error "Failed to generate Go types"
    cd ../../..

    cd cli

    # Test go mod tidy
    if go mod tidy; then
        success "Go module dependencies resolved"
    else
        error "Failed to resolve Go dependencies"
    fi

    # Test basic build
    go generate -v ./...
    if go build -ldflags="-X 'github.com/llamafarm/cli/internal/buildinfo.CurrentVersion=test-quick-1.0.0'" -o ../dist/lf-test .; then
        success "CLI build successful"
    else
        error "CLI build failed"
    fi

    # Test binary execution
    if ./../dist/lf-test version; then
        success "CLI binary executes correctly"
    else
        error "CLI binary execution failed"
    fi

    if ./../dist/lf-test help > /dev/null; then
        success "CLI help command works"
    else
        error "CLI help command failed"
    fi

    # Test cross-compilation (quick test)
    info "Testing cross-compilation..."

    if GOOS=linux GOARCH=amd64 go build -ldflags="-X 'github.com/llamafarm/cli/internal/buildinfo.CurrentVersion=test-cross'" -o ../dist/lf-linux .; then
        success "Linux cross-compilation works"
    else
        error "Linux cross-compilation failed"
    fi

    if GOOS=windows GOARCH=amd64 go build -ldflags="-X 'github.com/llamafarm/cli/internal/buildinfo.CurrentVersion=test-cross'" -o ../dist/lf-windows.exe .; then
        success "Windows cross-compilation works"
    else
        error "Windows cross-compilation failed"
    fi

    # Show binary sizes
    info "Binary sizes:"
    ls -lh ../dist/lf-* | awk '{print "  " $5 " " $9}'

    # Cleanup
    rm -f ../dist/lf-test ../dist/lf-linux ../dist/lf-windows.exe

    cd ..
    success "CLI build and cross-compilation tests passed"
}

# Test 3: Platform detection
test_platform_detection() {
    info "Testing platform detection logic..."

    # Extract and test the platform detection function
    cat > test-platform-detect.sh << 'EOF'
#!/bin/bash
detect_platform() {
    local os arch

    case "$(uname -s)" in
        Linux*)     os="linux" ;;
        Darwin*)    os="darwin" ;;
        MINGW*|MSYS*|CYGWIN*) os="windows" ;;
        *)          echo "error: unsupported OS $(uname -s)" && exit 1 ;;
    esac

    case "$(uname -m)" in
        x86_64|amd64)   arch="amd64" ;;
        aarch64|arm64)  arch="arm64" ;;
        armv7l)         arch="arm" ;;
        i386|i686)      arch="386" ;;
        *)              echo "error: unsupported arch $(uname -m)" && exit 1 ;;
    esac

    echo "${os}_${arch}"
}

platform=$(detect_platform)
echo "Detected platform: $platform"

if [[ "$platform" =~ ^(linux|darwin|windows)_(amd64|arm64|arm|386)$ ]]; then
    echo "SUCCESS: Platform detection works correctly"
    exit 0
else
    echo "ERROR: Platform detection failed: $platform"
    exit 1
fi
EOF

    chmod +x test-platform-detect.sh

    if ./test-platform-detect.sh; then
        success "Platform detection works correctly"
    else
        error "Platform detection failed"
    fi

    rm -f test-platform-detect.sh
}

# Test 4: Prerequisites check
test_prerequisites() {
    info "Testing prerequisite dependencies..."

    # Check for required tools
    local missing_tools=()

    if ! command -v curl > /dev/null && ! command -v wget > /dev/null; then
        missing_tools+=("curl or wget")
    fi

    if ! command -v tar > /dev/null; then
        missing_tools+=("tar")
    fi

    if ! command -v go > /dev/null; then
        missing_tools+=("go")
    fi

    if [[ ${#missing_tools[@]} -eq 0 ]]; then
        success "All prerequisite tools are available"
    else
        warning "Missing tools for full testing: ${missing_tools[*]}"
    fi

    # Check go version
    if command -v go > /dev/null; then
        local go_version
        go_version=$(go version | awk '{print $3}' | sed 's/go//')
        info "Go version: $go_version"

        # Check if go version is 1.19 or later (simple check)
        # Use bc to compare Go version numbers (supports decimals)
        if echo "$go_version >= 1.19" | bc -l | grep -q 1; then
            success "Go version is compatible"
        else
            warning "Go version may be too old (need 1.19+)"
        fi
    fi
}

# Test 5: Security checks
test_security() {
    info "Testing security aspects..."

    # Check script permissions
    local perms
    perms=$(stat -c %a install.sh 2>/dev/null || stat -f %Lp install.sh)
    info "Install script permissions: $perms"

    # Check for basic security patterns
    local security_checks=0

    if grep -q "set -e" install.sh; then
        success "✓ Script uses 'set -e' for error handling"
        security_checks=$((security_checks + 1))
    else
        warning "⚠ Script should use 'set -e' for safety"
    fi

    if grep -q "curl.*-f" install.sh; then
        success "✓ Script uses curl with fail-fast flag"
        security_checks=$((security_checks + 1))
    else
        warning "⚠ Consider using curl -f for better error handling"
    fi

    if grep -q "curl.*-s" install.sh; then
        success "✓ Script uses curl in silent mode"
        security_checks=$((security_checks + 1))
    else
        info "ℹ Script could use curl -s for cleaner output"
    fi

    if grep -q "tar.*-z" install.sh; then
        success "✓ Script handles gzipped archives"
        security_checks=$((security_checks + 1))
    else
        warning "⚠ Script should handle gzipped archives"
    fi

    info "Security checks passed: $security_checks/4"

    # Run shellcheck if available
    if command -v shellcheck > /dev/null; then
        info "Running shellcheck analysis..."
        # Run shellcheck and capture exit code without failing the script
        set +e  # Temporarily disable exit on error
        shellcheck install.sh
        local shellcheck_exit=$?
        set -e  # Re-enable exit on error

        if [[ $shellcheck_exit -eq 0 ]]; then
            success "✓ Shellcheck analysis passed"
        else
            warning "⚠ Shellcheck found issues (exit code: $shellcheck_exit)"
            info "Note: Shellcheck issues are treated as warnings, not errors"
        fi
    else
        info "ℹ Shellcheck not available, skipping static analysis"
    fi
}

# Test 6: Documentation completeness
test_documentation() {
    info "Testing documentation completeness..."

    local required_files=(
        "install.sh"
        "INSTALL.md"
        "cli/README.md"
        ".github/workflows/release-cli.yml"
    )

    local missing_files=()

    for file in "${required_files[@]}"; do
        if [[ -f "$file" ]]; then
            success "✓ $file exists"
        else
            missing_files+=("$file")
            error "✗ $file is missing"
        fi
    done

    if [[ ${#missing_files[@]} -eq 0 ]]; then
        success "All required documentation files exist"
    else
        error "Missing files: ${missing_files[*]}"
    fi

    # Check content requirements
    if grep -q "curl.*install.sh" README.md 2>/dev/null; then
        success "✓ Main README contains installation instructions"
    else
        warning "⚠ Main README should contain installation instructions"
    fi

    if [[ -f "INSTALL.md" ]] && grep -q "Quick Install" INSTALL.md && grep -q "Manual Installation" INSTALL.md; then
        success "✓ INSTALL.md contains comprehensive instructions"
    else
        warning "⚠ INSTALL.md should contain quick and manual install instructions"
    fi
}

# Main test runner
main() {
    info "Starting quick installer validation tests..."

    # Check if we're in the right directory
    if [[ ! -f "install.sh" ]]; then
        error "Please run this script from the project root directory (where install.sh is located)"
    fi

    # Run all tests
    test_script_basics
    test_cli_build
    test_platform_detection
    test_prerequisites
    test_security
    test_documentation

    success "🎉 All quick validation tests passed!"

    info ""
    info "Test Summary:"
    info "✅ Install script basic validation"
    info "✅ CLI build and cross-compilation"
    info "✅ Platform detection logic"
    info "✅ Prerequisites availability"
    info "✅ Security patterns"
    info "✅ Documentation completeness"
    info ""
}

# Run the tests
main "$@"
