#!/usr/bin/env bash

# LlamaFarm Jetson Setup Script
# Sets up Universal Runtime with CUDA PyTorch on NVIDIA Jetson devices
#
# Prerequisites:
#   - dustynv/l4t-pytorch container (or similar with CUDA PyTorch)
#   - UV package manager installed
#
# Usage:
#   cd /path/to/llamafarm
#   ./deployment/jetson/setup.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

error() { echo -e "${RED}Error: $1${NC}" >&2; exit 1; }
info() { echo -e "${BLUE}Info: $1${NC}"; }
success() { echo -e "${GREEN}$1${NC}"; }
warning() { echo -e "${YELLOW}Warning: $1${NC}"; }

# Detect if running on Jetson
detect_jetson() {
    if [[ -f /proc/device-tree/model ]]; then
        if grep -qi "jetson\|tegra" /proc/device-tree/model 2>/dev/null; then
            return 0
        fi
    fi

    if uname -r | grep -qi "tegra"; then
        return 0
    fi

    return 1
}

# Find system Python with CUDA PyTorch
find_cuda_python() {
    local candidates=("/usr/local/bin/python3" "/usr/bin/python3.10" "/usr/bin/python3")

    for py in "${candidates[@]}"; do
        if [[ -x "$py" ]]; then
            if "$py" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
                echo "$py"
                return 0
            fi
        fi
    done

    return 1
}

# Check UV is installed
check_uv() {
    if ! command -v uv &>/dev/null; then
        error "UV package manager not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    fi
}

# Main setup
main() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local repo_root
    repo_root="$(cd "$script_dir/../.." && pwd)"
    local runtime_dir="$repo_root/runtimes/universal"

    info "LlamaFarm Jetson Setup"
    echo ""

    # Detect Jetson
    if ! detect_jetson; then
        warning "Not running on Jetson hardware. This script is designed for NVIDIA Jetson devices."
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        info "Detected Jetson/Tegra hardware"
    fi

    # Check UV
    check_uv
    info "UV package manager found"

    # Find system Python with CUDA
    info "Looking for system Python with CUDA PyTorch..."
    local cuda_python
    if ! cuda_python=$(find_cuda_python); then
        warning "No system Python with CUDA PyTorch found."
        warning "This script works best inside dustynv/l4t-pytorch container."
        warning "Run: docker run -it --rm --runtime=nvidia --network host \\"
        warning "       -v \$HOME/.cache:/root/.cache \\"
        warning "       -v /path/to/llamafarm:/data/llamafarm \\"
        warning "       dustynv/l4t-pytorch:r36.2.0 bash"
        exit 1
    fi

    local py_version
    py_version=$("$cuda_python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    local torch_version
    torch_version=$("$cuda_python" -c "import torch; print(torch.__version__)")

    success "Found: $cuda_python (Python $py_version, PyTorch $torch_version with CUDA)"

    # Setup Universal Runtime
    info "Setting up Universal Runtime..."
    cd "$runtime_dir"

    # Create venv with system-site-packages to inherit CUDA PyTorch
    if [[ -d ".venv" ]]; then
        warning "Existing .venv found. Removing..."
        rm -rf .venv
    fi

    info "Creating venv with --system-site-packages..."
    uv venv .venv --python "$cuda_python" --system-site-packages

    # Install dependencies
    info "Installing dependencies..."
    uv pip install -r pyproject.toml --python .venv/bin/python

    # Remove PyPI torch (CPU-only) to fall back to system CUDA torch
    info "Removing PyPI torch (using system CUDA torch instead)..."
    uv pip uninstall torch torchvision torchaudio --python .venv/bin/python 2>/dev/null || true

    # Downgrade numpy for system PyTorch compatibility
    info "Installing numpy<2 (required by system PyTorch)..."
    uv pip install "numpy<2" --python .venv/bin/python

    # Verify CUDA PyTorch
    info "Verifying CUDA PyTorch..."
    if .venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
        success "CUDA PyTorch verified!"
    else
        warning "CUDA PyTorch not available in venv. GPU acceleration for transformers may be limited."
    fi

    # Check for llama.cpp libraries
    info "Checking llama.cpp libraries..."
    local lib_dir="${LLAMAFARM_LLAMA_LIB_DIR:-$HOME/.cache/llamafarm-llama/jetson}"
    if [[ -f "$lib_dir/libllama.so" ]]; then
        success "Found llama.cpp libraries at: $lib_dir"
    else
        warning "llama.cpp libraries not found at: $lib_dir"
        warning "You need to build llama.cpp from source for Jetson:"
        echo ""
        echo "  git clone https://github.com/ggml-org/llama.cpp.git"
        echo "  cd llama.cpp"
        echo "  cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=OFF \\"
        echo "        -DCMAKE_CUDA_ARCHITECTURES=\"87\" -DCMAKE_BUILD_TYPE=Release"
        echo "  cmake --build build --config Release -j\$(nproc)"
        echo ""
        echo "  # Copy libraries:"
        echo "  mkdir -p $lib_dir"
        echo "  cp build/src/libllama.so $lib_dir/"
        echo "  cp build/ggml/src/libggml*.so* $lib_dir/"
        echo ""
    fi

    # Create env file
    local env_file="$runtime_dir/.env.jetson"
    info "Creating $env_file..."
    cat > "$env_file" <<EOF
# Jetson-specific environment variables
# Source this file before running the server:
#   source .env.jetson && .venv/bin/python server.py

# Path to Jetson-compiled llama.cpp libraries
export LLAMAFARM_LLAMA_LIB_DIR="${lib_dir}"

# Force synchronous inference (recommended for Jetson stability)
export LLAMAFARM_SYNC_INFERENCE=1

# Optional: Disable CUDA cache to reduce memory fragmentation
# export CUDA_CACHE_DISABLE=1
EOF

    success "Jetson environment file created: $env_file"

    echo ""
    success "=========================================="
    success "  LlamaFarm Jetson Setup Complete!"
    success "=========================================="
    echo ""
    echo "To start the server:"
    echo ""
    echo "  cd $runtime_dir"
    echo "  source .env.jetson"
    echo "  .venv/bin/python server.py --model 'unsloth/Qwen3-1.7B-GGUF:Q4_K_M' --ctx-len 2048"
    echo ""
    echo "Or use the helper script:"
    echo ""
    echo "  ./deployment/jetson/start.sh"
    echo ""
}

main "$@"
