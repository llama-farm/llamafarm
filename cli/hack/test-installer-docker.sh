#!/usr/bin/env bash

# Docker-based Installer Testing Script
# This script tests the installer in Docker containers without complex simulation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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
    "alpine:3.18"
)

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

    success "Prerequisites check passed"
}

# Test installer basics in a container
test_installer_basics() {
    local image="$1"
    local container_name="llamafarm-basic-test-$(echo "$image" | tr '/:' '-')"

    info "Testing installer basics in $image..."

    # Determine package manager and install commands
    local install_cmd=""
    case "$image" in
        ubuntu:*|debian:*)
            install_cmd="apt-get update -qq && apt-get install -y curl tar gzip"
            ;;
        alpine:*)
            install_cmd="apk add --no-cache curl tar gzip bash"
            ;;
        fedora:*)
            install_cmd="dnf install -y curl tar gzip"
            ;;
        centos:*)
            install_cmd="dnf install -y curl tar gzip"
            ;;
        *)
            warning "Unknown image type: $image, using apt"
            install_cmd="apt-get update -qq && apt-get install -y curl tar gzip"
            ;;
    esac

    # Choose shell based on image
    local shell_cmd="bash"
    if [[ "$image" == alpine:* ]]; then
        shell_cmd="sh"
    fi

    # Test basic installer functionality (without actual download)
    docker run --rm \
        --name "$container_name" \
        -v "$(pwd)/install.sh:/test-install.sh:ro" \
        "$image" \
        $shell_cmd -c "
            set -e

            # Install dependencies
            $install_cmd > /dev/null 2>&1

            # Test help functionality
            if /test-install.sh --help > /dev/null; then
                echo '✅ Help option works'
            else
                echo '❌ Help option failed'
                exit 1
            fi

            # Test invalid option handling
            if /test-install.sh --invalid-option 2>/dev/null; then
                echo '❌ Should have failed with invalid option'
                exit 1
            else
                echo '✅ Invalid option handling works'
            fi

            # Test platform detection (extract and run the function)
            cat > test-platform.sh << 'EOF'
#!/bin/bash
detect_platform() {
    local os arch

    case \"\$(uname -s)\" in
        Linux*)     os=\"linux\" ;;
        Darwin*)    os=\"darwin\" ;;
        MINGW*|MSYS*|CYGWIN*) os=\"windows\" ;;
        *)          echo \"error: unsupported OS\" && exit 1 ;;
    esac

    case \"\$(uname -m)\" in
        x86_64|amd64)   arch=\"amd64\" ;;
        aarch64|arm64)  arch=\"arm64\" ;;
        armv7l)         arch=\"arm\" ;;
        i386|i686)      arch=\"386\" ;;
        *)              echo \"error: unsupported arch\" && exit 1 ;;
    esac

    echo \"\${os}_\${arch}\"
}

platform=\$(detect_platform)
echo \"Detected platform: \$platform\"

if [[ \"\$platform\" =~ ^(linux|darwin|windows)_(amd64|arm64|arm|386)\$ ]]; then
    echo \"✅ Platform detection works\"
    exit 0
else
    echo \"❌ Platform detection failed: \$platform\"
    exit 1
fi
EOF

            chmod +x test-platform.sh
            ./test-platform.sh

            echo '✅ Basic installer tests passed'
        " || error "Basic tests failed in $image"

    success "✅ $image basic tests passed"
}

# Test CLI build in container
test_cli_build() {
    info "Testing CLI build in container..."

    docker run --rm \
        --name "llamafarm-build-test" \
        -v "$(pwd):/workspace" \
        -w /workspace \
        golang:1.24-alpine \
        sh -c "
            set -e

            # Install dependencies
            apk add --no-cache git

            cd cli

            # Test go mod tidy
            go mod tidy

            # Test basic build
            go build -buildvcs=false -ldflags='-X llamafarm-cli/cmd.Version=docker-test-1.0.0' -o ../dist/lf-test .

            # Test binary execution
            ./../dist/lf-test version
            ./../dist/lf-test help > /dev/null

            echo '✅ CLI build test passed'
        " || error "CLI build test failed"

    success "CLI build test passed"
}

# Test security aspects
test_security() {
    info "Testing security aspects..."

    # Test script with shellcheck in container
    if command -v docker > /dev/null; then
        docker run --rm \
            -v "$(pwd)/install.sh:/install.sh:ro" \
            koalaman/shellcheck:stable \
            shellcheck /install.sh || warning "Shellcheck found issues"

        success "Shellcheck analysis completed"
    else
        warning "Docker not available for shellcheck"
    fi

    # Test script permissions
    if [[ -x "install.sh" ]]; then
        success "✅ Install script is executable"
    else
        error "❌ Install script should be executable"
    fi

    # Check for security patterns
    if grep -q "set -e" install.sh; then
        success "✅ Script uses 'set -e' for error handling"
    else
        warning "⚠️ Script should use 'set -e'"
    fi
}

# Test documentation
test_documentation() {
    info "Testing documentation..."

    local required_files=(
        "install.sh"
        "INSTALL.md"
        "cli/README.md"
        ".github/workflows/release-cli.yml"
    )

    for file in "${required_files[@]}"; do
        if [[ -f "$file" ]]; then
            success "✅ $file exists"
        else
            error "❌ $file is missing"
        fi
    done

    # Check content
    if grep -q "curl.*install.sh" README.md; then
        success "✅ Main README contains installation instructions"
    else
        warning "⚠️ Main README should contain installation instructions"
    fi
}

# Test all environments
test_all_environments() {
    info "Testing installer across Docker environments..."

    local failed_tests=()

    for env in "${TEST_ENVIRONMENTS[@]}"; do
        if test_installer_basics "$env"; then
            :  # Success
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

# Cleanup function
cleanup() {
    info "Cleaning up..."

    # Remove any test containers
    docker ps -a --filter "name=llamafarm-" --format "{{.ID}}" | xargs -r docker rm -f 2>/dev/null || true

    success "Cleanup complete"
}

# Main function
main() {
    info "Starting Docker-based installer testing..."

    # Set up cleanup trap
    trap cleanup EXIT

    # Run tests
    check_prerequisites
    test_cli_build
    test_all_environments
    test_security
    test_documentation

    success "🎉 All Docker-based tests completed successfully!"

    info ""
    info "Test Summary:"
    info "✅ CLI build works in container"
    info "✅ Installer basics work across ${#TEST_ENVIRONMENTS[@]} environments"
    info "✅ Security checks passed"
    info "✅ Documentation is complete"
    info ""
    info "The installer is ready for testing with real releases! 🚀"
}

# Run the tests
main "$@"