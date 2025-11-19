#!/usr/bin/env bash
# Package LlamaFarm source code with only necessary directories for CLI installation
# This creates a minimal archive containing:
# - server/ (FastAPI server)
# - rag/ (Celery worker and RAG pipeline)
# - common/ (shared Python utilities)
# - config/ (configuration schema and types)
# - runtimes/universal/ (universal runtime)
# - designer/dist/ (built designer static files)
# - ruff.toml (code formatting configuration)

set -e

VERSION="${1:-unknown}"
OUTPUT_DIR="${2:-dist}"

if [ -z "$VERSION" ] || [ "$VERSION" == "unknown" ]; then
    echo "Error: Version is required" >&2
    echo "Usage: $0 <version> [output-dir]" >&2
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Archive name
ARCHIVE_NAME="llamafarm-dist-${VERSION}.tar.gz"
ARCHIVE_PATH="${OUTPUT_DIR}/${ARCHIVE_NAME}"

echo "Packaging LlamaFarm source code (version: $VERSION)..."

# Verify required directories exist
REQUIRED_DIRS=("server" "rag" "common" "config" "runtimes/universal" "designer/dist")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "Error: Required directory not found: $dir" >&2
        exit 1
    fi
done

# Verify required files exist
REQUIRED_FILES=("ruff.toml")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Error: Required file not found: $file" >&2
        exit 1
    fi
done

# Verify designer/dist/index.html exists
if [ ! -f "designer/dist/index.html" ]; then
    echo "Error: designer/dist/index.html not found. Designer must be built first." >&2
    echo "Run: cd designer && npm run build" >&2
    exit 1
fi

# Create a temporary directory for packaging
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Create the source directory structure
SOURCE_DIR="${TEMP_DIR}/llamafarm-${VERSION}"
mkdir -p "$SOURCE_DIR"

# Copy required directories with exclusions
# Exclude: dotfiles/folders, __pycache__, test(s) folders
RSYNC_EXCLUDE=(
    "--exclude=.*"
    "--exclude=__pycache__"
    "--exclude=test"
    "--exclude=tests"
    "--exclude=*_test.py"
)

echo "Copying server/..."
rsync -a "${RSYNC_EXCLUDE[@]}" server/ "$SOURCE_DIR/server/"

echo "Copying rag/..."
rsync -a "${RSYNC_EXCLUDE[@]}" rag/ "$SOURCE_DIR/rag/"

echo "Copying common/..."
rsync -a "${RSYNC_EXCLUDE[@]}" common/ "$SOURCE_DIR/common/"

echo "Copying config/..."
rsync -a "${RSYNC_EXCLUDE[@]}" config/ "$SOURCE_DIR/config/"

echo "Copying runtimes/universal/..."
mkdir -p "$SOURCE_DIR/runtimes"
rsync -a "${RSYNC_EXCLUDE[@]}" runtimes/universal/ "$SOURCE_DIR/runtimes/universal/"

echo "Copying designer/dist/..."
mkdir -p "$SOURCE_DIR/designer"
rsync -a "${RSYNC_EXCLUDE[@]}" designer/dist/ "$SOURCE_DIR/designer/dist/"

# Copy root-level configuration files needed for builds
echo "Copying ruff.toml..."
cp ruff.toml "$SOURCE_DIR/ruff.toml"

# Create the archive
echo "Creating archive: $ARCHIVE_PATH"
# Compute absolute path before changing directories
# realpath doesn't work on non-existent files, so we'll compute it manually
ARCHIVE_ABS_PATH="$(cd "$(dirname "$ARCHIVE_PATH")" && pwd)/$(basename "$ARCHIVE_PATH")"
cd "$TEMP_DIR"
# Create archive with absolute path, suppress all output
tar -czf "$ARCHIVE_ABS_PATH" "llamafarm-${VERSION}" > /dev/null 2>&1

if [ ! -f "$ARCHIVE_ABS_PATH" ]; then
    echo "Error: Failed to create archive at $ARCHIVE_ABS_PATH" >&2
    exit 1
fi

# Generate SHA256 checksum
echo "Generating SHA256 checksum..."
sha256sum "$ARCHIVE_ABS_PATH" > "${ARCHIVE_ABS_PATH}.sha256"

echo "✓ Source archive created: $ARCHIVE_PATH"
echo "✓ SHA256 checksum: ${ARCHIVE_PATH}.sha256"
echo ""
echo "Archive summary (first 20 entries):"
tar -tzf "$ARCHIVE_ABS_PATH" 2>/dev/null | head -20
echo "..."
echo "Total files: $(tar -tzf "$ARCHIVE_ABS_PATH" 2>/dev/null | wc -l)"
