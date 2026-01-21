#!/usr/bin/env bash
# Universal RAG Demo
#
# This script demonstrates the zero-config RAG capability using universal_rag.
# By not defining any data_processing_strategies, LlamaFarm automatically uses
# the built-in universal_rag strategy which handles 90%+ of document formats.
#
# Prerequisites:
#   - LlamaFarm server running on port 8000
#   - Universal runtime running (for embeddings and inference)
#   - This example project registered
#
# Usage:
#   bash examples/universal_rag/demo.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="${1:-${PROJECT_ROOT_DEFAULT}}"
LF_BIN="${LF_BIN:-${PROJECT_ROOT}/dist/lf}"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/llamafarm.yaml}"
FILES_DIR="${SCRIPT_DIR}/files"
RUN_ID=$(date +%Y%m%d%H%M%S)
DATABASE_NAME="universal_db_${RUN_ID}"
DATASET_NAME="universal_docs_${RUN_ID}"
NO_PAUSE=${NO_PAUSE:-0}

# Load port from .env if available
ENV_FILE="$PROJECT_ROOT/.env"
if [ -f "$ENV_FILE" ]; then
  PORT=$(grep -E "^PORT=" "$ENV_FILE" | cut -d= -f2 || echo "8000")
fi
PORT="${PORT:-8000}"
API_BASE="${API_BASE:-http://localhost:$PORT}"

lf() {
  "${LF_BIN}" --cwd "${SCRIPT_DIR}" "$@"
}

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
info() { printf '\033[0;34m  %s\033[0m\n' "$1"; }
success() { printf '\033[0;32m  %s\033[0m\n' "$1"; }
warn() { printf '\033[0;33m  %s\033[0m\n' "$1"; }
error() { printf '\033[0;31m  %s\033[0m\n' "$1"; }

pause() {
  if [[ "$NO_PAUSE" != "1" ]]; then
    read -rp $'\nPress Enter to continue...'
  fi
}

ensure_file() {
  local path="$1" msg="$2"
  [[ -f "$path" ]] || { error "$msg"; exit 1; }
}

ensure_dir() {
  local path="$1" msg="$2"
  [[ -d "$path" ]] || { error "$msg"; exit 1; }
}

any_files() {
  compgen -G "$1" >/dev/null
}

# Duplicate database configuration for this run
duplicate_database() {
  PYTHON_CONFIG_PATH="$CONFIG_PATH" \
  PYTHON_DATABASE_NAME="$DATABASE_NAME" \
  PYTHON_BASE_DATABASE_NAME="universal_db" \
  uv run --directory "${PROJECT_ROOT}/rag" python3 <<'PY'
import copy
import os
import sys
from pathlib import Path
import yaml

cfg_path = Path(os.environ['PYTHON_CONFIG_PATH'])
if not cfg_path.exists():
    print(f"Config file {cfg_path} does not exist", file=sys.stderr)
    sys.exit(1)

cfg = yaml.safe_load(cfg_path.read_text()) or {}
rag = cfg.setdefault('rag', {})
databases = rag.setdefault('databases', [])
new_name = os.environ['PYTHON_DATABASE_NAME']
base_name = os.environ['PYTHON_BASE_DATABASE_NAME']

if any(db.get('name') == new_name for db in databases):
    print(f"Database {new_name} already present; skipping duplication")
else:
    base = next((db for db in databases if db.get('name') == base_name), None)
    if base is None:
        if databases:
            base = databases[0]
            print(f"Base database {base_name} not found; cloning first entry.")
        else:
            print("No databases defined in config; aborting.", file=sys.stderr)
            sys.exit(1)
    new_db = copy.deepcopy(base)
    new_db['name'] = new_name
    cfg_dir = new_db.setdefault('config', {})
    persist = cfg_dir.get('persist_directory') or f"./data/{base_name}"
    # Use pathlib for robust cross-platform path manipulation
    from pathlib import Path
    new_persist_dir = Path(persist).parent / new_name
    cfg_dir['persist_directory'] = str(new_persist_dir)
    databases.append(new_db)
    cfg_path.write_text(yaml.dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"Added database {new_name} to configuration.")
PY
}

# Header
echo ""
bold "============================================================"
bold "         Universal RAG Demo - Zero Config Document Processing"
bold "============================================================"
echo ""
info "This demo shows how universal_rag enables document ingestion"
info "WITHOUT any parser configuration. Just point at files and go!"
echo ""
info "Project root: ${PROJECT_ROOT}"
info "API Base: ${API_BASE}"
info "Using database '${DATABASE_NAME}' and dataset '${DATASET_NAME}'"
echo ""

# Validate prerequisites
ensure_file "$LF_BIN" "LlamaFarm CLI not found at ${LF_BIN}. Build with 'nx build cli'."
ensure_file "$CONFIG_PATH" "Config not found at ${CONFIG_PATH}."
ensure_dir "$FILES_DIR" "Sample files directory missing: ${FILES_DIR}."
if ! any_files "${FILES_DIR}/*"; then
  error "No sample files found in ${FILES_DIR}."
  exit 1
fi

duplicate_database
pause

pushd "$PROJECT_ROOT" >/dev/null
trap 'warn "Demo interrupted."' INT TERM

# Step 1: Verify CLI
bold "Step 1: Verify CLI connectivity"
lf version
success "CLI connected."
pause

# Step 2: Create dataset (no explicit strategy - uses universal_rag)
bold "Step 2: Create dataset '${DATASET_NAME}'"
info "NOTE: No data_processing_strategy specified - using universal_rag by default!"
lf datasets create -b "${DATABASE_NAME}" "${DATASET_NAME}"
success "Dataset created with universal_rag strategy."
pause

# Step 3: Upload documents
bold "Step 3: Upload documents (mixed formats supported)"
for file in "${FILES_DIR}"/*; do
  info "Uploading $(basename "$file")..."
  lf datasets upload "${DATASET_NAME}" "$file"
done
success "Documents uploaded."
pause

# Step 4: Process documents
bold "Step 4: Process documents with universal_rag"
info "UniversalParser handles: MD, TXT, PDF, DOCX, HTML, CSV, JSON, etc."
info "UniversalExtractor adds: keywords, entities, language detection, etc."
lf datasets process "${DATASET_NAME}"
success "Processing complete!"
pause

# Step 5: Query the knowledge base
bold "Step 5: Query the knowledge base"
echo ""

info "Question 1: What are the types of machine learning?"
lf chat --database "${DATABASE_NAME}" \
  "What are the three main types of machine learning? Cite the source."
pause

info "Question 2: What is tokenization in NLP?"
lf chat --database "${DATABASE_NAME}" \
  "Explain tokenization in NLP. What tokenizers are mentioned?"
pause

info "Question 3: API best practices"
lf chat --database "${DATABASE_NAME}" \
  "What HTTP status codes should I use for errors? Cite the source file."
pause

# Step 6: Raw RAG query with metadata
bold "Step 6: RAG query with metadata (showing universal_rag metadata)"
lf rag query --database "${DATABASE_NAME}" --top-k 3 --include-metadata --include-score \
  "transformer architecture attention mechanism"
pause

# Step 7: Compare with no RAG
bold "Step 7: Compare with baseline (no RAG)"
lf chat --no-rag "What tokenizers are commonly used in NLP? List three."
info "Note: Without RAG, the model can't cite specific documents."
pause

# Summary
echo ""
bold "============================================================"
bold "                      Demo Complete!"
bold "============================================================"
echo ""
success "Universal RAG processed multiple file formats with ZERO parser config!"
echo ""
info "Key takeaways:"
info "  1. No data_processing_strategies defined = universal_rag is used"
info "  2. UniversalParser handles PDF, DOCX, MD, TXT, HTML, CSV, JSON, etc."
info "  3. UniversalExtractor adds rich metadata automatically"
info "  4. Semantic chunking produces better retrieval results"
echo ""
info "Continue exploring:"
info "  lf chat --database ${DATABASE_NAME} \"YOUR QUESTION\""
info "  lf rag query --database ${DATABASE_NAME} --include-metadata \"QUERY\""
echo ""
warn "Database '${DATABASE_NAME}' and dataset '${DATASET_NAME}' remain available."
warn "Clean up when finished."

popd >/dev/null
