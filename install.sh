#!/usr/bin/env bash

# LlamaFarm CLI Installation Script
# This script installs the LlamaFarm CLI (lf) binary for your platform

set -e

# Configuration
REPO="llama-farm/llamafarm"
BINARY_NAME="lf"
INSTALL_DIR="/usr/local/bin"
CLI_NAME="lf"
LF_DATA_DIR="${LF_DATA_DIR:-$HOME/.llamafarm}"
BUNDLE_ARCHIVE=""
SUDO_CMD=""

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

# Download file (fatal on failure)
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

# Download file (non-fatal, returns 1 on failure)
try_download_file() {
    local url="$1"
    local output="$2"

    if command_exists curl; then
        curl -f -sL -o "$output" "$url" 2>/dev/null && return 0
    elif command_exists wget; then
        wget -q -O "$output" "$url" 2>/dev/null && return 0
    fi
    return 1
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

    success "CLI binary installed!"

    # Download and install PyApp service binaries
    info "Installing service binaries..."
    local pyapp_os="${platform%%-*}"
    local pyapp_goarch="${platform##*-}"
    local pyapp_arch="$pyapp_goarch"
    case "$pyapp_goarch" in
        amd64) pyapp_arch="x86_64" ;;
    esac
    # PyApp release binaries use "macos" instead of "darwin"
    case "$pyapp_os" in
        darwin) pyapp_os="macos" ;;
    esac
    local pyapp_suffix="${pyapp_os}-${pyapp_arch}"

    local bin_dir="$LF_DATA_DIR/bin"
    mkdir -p "$bin_dir"

    for component in server rag runtime; do
        local pyapp_name="llamafarm-${component}-${pyapp_suffix}"
        if [[ "$pyapp_os" == "windows" ]]; then
            pyapp_name="${pyapp_name}.exe"
        fi
        local pyapp_url="https://github.com/$REPO/releases/download/$version/${pyapp_name}"
        local pyapp_dest="$bin_dir/llamafarm-${component}"

        info "  Downloading $component..."
        if try_download_file "$pyapp_url" "$temp_dir/$pyapp_name"; then
            cp "$temp_dir/$pyapp_name" "$pyapp_dest"
            chmod +x "$pyapp_dest"
        else
            warning "  Failed to download $component (may not be available for this platform)"
        fi
    done

    # Bootstrap PyApp binaries
    bootstrap_services

    success "LlamaFarm installed successfully!"
    verify_installation
}

# Bootstrap PyApp binaries (trigger first-run extraction)
bootstrap_services() {
    info "Bootstrapping service binaries (this may take a moment)..."
    if command_exists "$BINARY_NAME"; then
        "$BINARY_NAME" bundle bootstrap || {
            warning "Bootstrap failed. Services will extract on first run instead."
        }
    else
        # CLI not in PATH yet, use the install dir directly
        "$INSTALL_DIR/$BINARY_NAME" bundle bootstrap || {
            warning "Bootstrap failed. Services will extract on first run instead."
        }
    fi
}

# Verify installation
verify_installation() {
    if command_exists "$BINARY_NAME"; then
        info "Verifying installation..."
        if "$BINARY_NAME" version; then
            success "Installation verified!"
        else
            warning "CLI binary installed but 'lf version' returned an error"
            exit 1
        fi
    else
        warning "Binary installed but not found in PATH. You may need to add $INSTALL_DIR to your PATH"
        echo "Add this to your shell profile (.bashrc, .zshrc, etc.):"
        echo "export PATH=\"$INSTALL_DIR:\$PATH\""
    fi
}

# Install PyApp service binaries to ~/.llamafarm/bin/
install_pyapp_binaries() {
    local source_dir="$1"
    local bin_dir="$LF_DATA_DIR/bin"

    mkdir -p "$bin_dir"

    for component in server rag runtime; do
        local binary
        binary=$(find "$source_dir" -maxdepth 1 -name "llamafarm-${component}*" -not -name "*.sha256" | head -1)
        if [[ -n "$binary" ]]; then
            local dest_name="llamafarm-${component}"
            info "  Installing $dest_name to $bin_dir/"
            cp "$binary" "$bin_dir/$dest_name"
            chmod +x "$bin_dir/$dest_name"
        else
            warning "  PyApp binary for $component not found in bundle"
        fi
    done
}

# Install torch wheels from bundle (offline torch upgrade)
install_torch_wheels() {
    local torch_dir="$1"
    local bin_dir="$LF_DATA_DIR/bin"

    if [[ ! -d "$torch_dir" ]] || [[ -z "$(ls -A "$torch_dir" 2>/dev/null)" ]]; then
        info "No torch wheels in bundle, using built-in CPU torch"
        return 0
    fi

    info "Installing accelerator-specific torch from bundle..."

    if [[ -x "$bin_dir/llamafarm-runtime" ]]; then
        "$bin_dir/llamafarm-runtime" self pip install \
            --find-links "$torch_dir" --force-reinstall torch 2>&1 || {
            warning "Failed to install torch wheels. The runtime will use CPU-only torch."
            return 0
        }
        success "Accelerator-specific torch installed"
    else
        warning "Runtime binary not found, skipping torch upgrade"
    fi
}

# Install addon wheels from bundle
install_bundle_addons() {
    local addons_dir="$1"
    local registry_dir="$2"
    local addon_install_dir="$LF_DATA_DIR/addons"

    if [[ ! -d "$addons_dir" ]] || [[ -z "$(ls -A "$addons_dir" 2>/dev/null)" ]]; then
        return 0
    fi

    info "Installing addons from bundle..."
    mkdir -p "$addon_install_dir"

    # Initialize state file if missing
    if [[ ! -f "$LF_DATA_DIR/addons.json" ]]; then
        echo '{"installed":{}}' > "$LF_DATA_DIR/addons.json"
    fi

    for wheel_archive in "$addons_dir"/*-wheels-*.tar.gz; do
        [[ -f "$wheel_archive" ]] || continue
        local addon_name
        addon_name=$(basename "$wheel_archive" | sed 's/-wheels-.*//')
        info "  Installing addon: $addon_name"

        local addon_dir="$addon_install_dir/$addon_name"
        mkdir -p "$addon_dir"
        tar xzf "$wheel_archive" -C "$addon_dir"

        success "  Addon $addon_name installed"
    done

    # Copy registry files if present
    if [[ -d "$registry_dir" ]] && [[ -n "$(ls -A "$registry_dir" 2>/dev/null)" ]]; then
        local dest_registry="$LF_DATA_DIR/src/addons/registry"
        mkdir -p "$dest_registry"
        cp "$registry_dir"/*.yaml "$dest_registry/" 2>/dev/null || true
    fi
}

# Offline installation from bundle archive
install_from_bundle() {
    local archive="$1"

    if [[ ! -f "$archive" ]]; then
        error "Bundle archive not found: $archive"
    fi

    info "Installing LlamaFarm from bundle: $archive"

    # Create temporary directory for extraction
    local temp_dir
    temp_dir=$(mktemp -d)
    trap 'rm -rf $temp_dir' EXIT

    # Extract archive
    info "Extracting bundle..."
    tar xzf "$archive" -C "$temp_dir"

    # Read manifest
    local manifest="$temp_dir/manifest.json"
    if [[ ! -f "$manifest" ]]; then
        error "Bundle is missing manifest.json — this may not be a valid LlamaFarm bundle"
    fi

    local bundle_version bundle_platform bundle_accel
    bundle_version=$(grep '"version"' "$manifest" | sed 's/.*: *"\([^"]*\)".*/\1/')
    bundle_platform=$(grep '"platform"' "$manifest" | sed 's/.*: *"\([^"]*\)".*/\1/')
    bundle_accel=$(grep '"accelerator"' "$manifest" | sed 's/.*: *"\([^"]*\)".*/\1/')

    info "Bundle version: $bundle_version"
    info "Bundle platform: $bundle_platform ($bundle_accel)"

    # Check permissions and prepare install directory
    check_permissions
    ensure_install_dir

    # 1. Install CLI binary
    local cli_binary
    cli_binary=$(find "$temp_dir" -maxdepth 1 -name "lf-*" -not -name "*.sha256" | head -1)
    if [[ -n "$cli_binary" ]]; then
        info "Installing CLI binary to $INSTALL_DIR/$BINARY_NAME"
        $SUDO_CMD cp "$cli_binary" "$INSTALL_DIR/$BINARY_NAME"
        $SUDO_CMD chmod +x "$INSTALL_DIR/$BINARY_NAME"
    else
        error "CLI binary not found in bundle"
    fi

    # 2. Install PyApp service binaries
    install_pyapp_binaries "$temp_dir"

    # 3. Install accelerator-specific torch (if present)
    install_torch_wheels "$temp_dir/torch"

    # 4. Install addons (if present)
    install_bundle_addons "$temp_dir/addons" "$temp_dir/addons-registry"

    # 5. Bootstrap PyApp binaries
    bootstrap_services

    success "LlamaFarm installed from bundle!"

    # 6. Verify installation
    verify_installation
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
            echo "Usage: $0 [OPTIONS] [BUNDLE_ARCHIVE]"
            echo ""
            echo "Online mode (no archive):"
            echo "  $0                              # Install latest version from GitHub"
            echo "  $0 --version v1.0.0             # Install specific version"
            echo ""
            echo "Offline mode (with bundle archive):"
            echo "  $0 bundle.tar.gz                # Install from bundle archive"
            echo ""
            echo "Options:"
            echo "  --install-dir DIR    Install CLI binary to DIR (default: /usr/local/bin)"
            echo "  --version VERSION    Install specific version (default: latest)"
            echo "  --help, -h           Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  VERSION              Specify version to install"
            echo "  INSTALL_DIR          Specify installation directory"
            echo "  LF_DATA_DIR          LlamaFarm data directory (default: ~/.llamafarm)"
            echo ""
            echo "Create a bundle with: lf bundle --platform linux --arch x86_64 --accelerator cuda -o bundle.tar.gz"
            exit 0
            ;;
        *.tar.gz)
            BUNDLE_ARCHIVE="$1"
            shift
            ;;
        *)
            error "Unknown option: $1. Use --help for usage information."
            ;;
    esac
done

# Run installation
if [[ -n "$BUNDLE_ARCHIVE" ]]; then
    install_from_bundle "$BUNDLE_ARCHIVE"
else
    install_cli
fi