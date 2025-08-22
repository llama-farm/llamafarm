#!/usr/bin/env bash

# LlamaFarm CLI Installation Script
# This script installs the LlamaFarm CLI (lf) binary for your platform

set -e

# Configuration
REPO="llama-farm/llamafarm"
BINARY_NAME="lf"
INSTALL_DIR="/usr/local/bin"
CLI_NAME="llamafarm"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
error() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

info() {
    echo -e "${BLUE}Info: $1${NC}"
}

success() {
    echo -e "${GREEN}Success: $1${NC}"
}

warning() {
    echo -e "${YELLOW}Warning: $1${NC}"
}

# Detect OS and architecture
detect_platform() {
    local os arch

    case "$(uname -s)" in
        Linux*)     os="linux" ;;
        Darwin*)    os="darwin" ;;
        MINGW*|MSYS*|CYGWIN*) os="windows" ;;
        *)          error "Unsupported operating system: $(uname -s)" ;;
    esac

    case "$(uname -m)" in
        x86_64|amd64)   arch="amd64" ;;
        aarch64|arm64)  arch="arm64" ;;
        armv7l)         arch="arm" ;;
        i386|i686)      arch="386" ;;
        *)              error "Unsupported architecture: $(uname -m)" ;;
    esac

    echo "${os}-${arch}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
check_dependencies() {
    if ! command_exists curl && ! command_exists wget; then
        error "Neither curl nor wget found. Please install one of them."
    fi

    if ! command_exists tar; then
        error "tar command not found. Please install tar."
    fi
}

# Download file
download_file() {
    local url="$1"
    local output="$2"

    info "Downloading from: $url"

    if command_exists curl; then
        curl -f -L -o "$output" "$url" || error "Failed to download with curl"
    elif command_exists wget; then
        wget -O "$output" "$url" || error "Failed to download with wget"
    fi
}

# Get latest release version
get_latest_version() {
    local api_url="https://api.github.com/repos/$REPO/releases/latest"

    if command_exists curl; then
        curl -f -s "$api_url" | grep '"tag_name"' | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/' | head -1
    elif command_exists wget; then
        wget -qO- "$api_url" | grep '"tag_name"' | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/' | head -1
    fi
}

# Check if running as root for system-wide installation
check_permissions() {
    if [[ "$INSTALL_DIR" == "/usr/local/bin" && ! -w "$INSTALL_DIR" ]]; then
        if [[ $EUID -ne 0 ]]; then
            warning "Installing to $INSTALL_DIR requires sudo privileges"
            SUDO_CMD="sudo"
        fi
    fi
}

# Create install directory if it doesn't exist
ensure_install_dir() {
    if [[ ! -d "$INSTALL_DIR" ]]; then
        info "Creating install directory: $INSTALL_DIR"
        $SUDO_CMD mkdir -p "$INSTALL_DIR" || error "Failed to create install directory"
    fi
}

# Main installation function
install_cli() {
    local platform version download_url temp_dir

    info "Starting LlamaFarm CLI installation..."

    # Check dependencies
    check_dependencies

    # Detect platform
    platform=$(detect_platform)
    info "Detected platform: $platform"

    # Get version (allow override with VERSION env var)
    if [[ -n "$VERSION" ]]; then
        version="$VERSION"
        info "Using specified version: $version"
    else
        info "Fetching latest release version..."
        version=$(get_latest_version)
        if [[ -z "$version" ]]; then
            error "Failed to get latest version. You can specify a version with: VERSION=v1.0.0 $0"
        fi
        info "Latest version: $version"
    fi

    # Construct download URL
    local filename="${CLI_NAME}-${platform}"
    if [[ "$platform" == *"windows"* ]]; then
        filename="${filename}.exe"
    fi
    download_url="https://github.com/$REPO/releases/download/$version/${filename}"

    # Create temporary directory
    temp_dir=$(mktemp -d)
    trap 'rm -rf $temp_dir' EXIT

    # Download and extract
    local download_path="$temp_dir/${filename}"
    download_file "$download_url" "$download_path"

    # Check permissions and prepare for installation
    check_permissions
    ensure_install_dir

    # Install binary
    info "Installing binary to $INSTALL_DIR/$BINARY_NAME"
    $SUDO_CMD cp "$download_path" "$INSTALL_DIR/$BINARY_NAME" || error "Failed to copy binary"
    $SUDO_CMD chmod +x "$INSTALL_DIR/$BINARY_NAME" || error "Failed to make binary executable"

    success "LlamaFarm CLI installed successfully!"

    # Verify installation
    if command_exists "$BINARY_NAME"; then
        info "Verifying installation..."
        "$BINARY_NAME" version
    else
        warning "Binary installed but not found in PATH. You may need to add $INSTALL_DIR to your PATH"
        echo "Add this to your shell profile (.bashrc, .zshrc, etc.):"
        echo "export PATH=\"$INSTALL_DIR:\$PATH\""
    fi
}

# Handle command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --help|-h)
            echo "LlamaFarm CLI Installation Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --install-dir DIR    Install binary to DIR (default: /usr/local/bin)"
            echo "  --version VERSION    Install specific version (default: latest)"
            echo "  --help, -h           Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  VERSION              Specify version to install"
            echo "  INSTALL_DIR          Specify installation directory"
            echo ""
            echo "Examples:"
            echo "  $0                              # Install latest version"
            echo "  $0 --version v1.0.0             # Install specific version"
            echo "  $0 --install-dir ~/.local/bin   # Install to user directory"
            echo "  VERSION=v1.0.0 $0               # Install using environment variable"
            exit 0
            ;;
        *)
            error "Unknown option: $1. Use --help for usage information."
            ;;
    esac
done

# Run installation
install_cli