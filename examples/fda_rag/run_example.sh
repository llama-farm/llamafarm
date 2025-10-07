#!/usr/bin/env bash
# FDA Correspondence RAG example workflow

set -euo pipefail

# ----------------------------------------
# Helpers
# ----------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${1:-${PROJECT_ROOT_DEFAULT}}"
LF_BIN="${PROJECT_ROOT}/lf"
CONFIG_PATH="${PROJECT_ROOT}/llamafarm.yaml"
DATASET_NAME="fda_letters"
DATABASE_NAME="fda_letters_db"
PROCESSOR_NAME="fda_pdf_processor"
SAMPLE_DIR="${SCRIPT_DIR}/files"

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
info() { printf '\033[0;34mℹ %s\033[0m\n' "$1"; }
success() { printf '\033[0;32m✓ %s\033[0m\n' "$1"; }
warn() { printf '\033[0;33m⚠ %s\033[0m\n' "$1"; }
error() { printf '\033[0;31m✗ %s\033[0m\n' "$1"; }

fail() {
  error "$1"
  exit 1
}

ensure_file() {
  local path="$1" msg="$2"
  [[ -f "$path" ]] || fail "$msg"
}

ensure_dir() {
  local path="$1" msg="$2"
  [[ -d "$path" ]] || fail "$msg"
}

run_cmd() {
  local description="$1"; shift
  info "$description"
  if ! "$@"; then
    warn "Command failed; rerun manually if needed:\n     $*"
  fi
}

# ----------------------------------------
# Pre-flight checks
# ----------------------------------------

bold "FDA Correspondence RAG Example"
info "Project root: ${PROJECT_ROOT}"
info "Config path: ${CONFIG_PATH}"
info "Using dataset '${DATASET_NAME}' with database '${DATABASE_NAME}'"

ensure_file "$LF_BIN" "LlamaFarm CLI not found at ${LF_BIN}. Build it with 'go build -o lf cli/main.go'."
ensure_file "$CONFIG_PATH" "No llamafarm.yaml found at ${CONFIG_PATH}. Run './examples/fda_rag/update_config.sh' first."
ensure_dir "$SAMPLE_DIR" "Sample FDA PDFs not found at ${SAMPLE_DIR}."
if ! compgen -G "${SAMPLE_DIR}"/*.pdf > /dev/null; then
  fail "No PDF files found in ${SAMPLE_DIR}."
fi

# Ensure we operate from project root for CLI commands
pushd "$PROJECT_ROOT" > /dev/null

trap 'warn "Example script interrupted."' INT TERM

# ----------------------------------------
# Workflow
# ----------------------------------------

bold "Step 1: Confirm CLI connectivity"
"${LF_BIN}" version || fail "Unable to execute LlamaFarm CLI."

bold "Step 2: Prepare dataset '${DATASET_NAME}'"
if ! grep -q "${PROCESSOR_NAME}" "$CONFIG_PATH"; then
  fail "Data processing strategy '${PROCESSOR_NAME}' not found. Run './examples/fda_rag/update_config.sh' to install the example config."
fi
if "${LF_BIN}" datasets show "${DATASET_NAME}" >/dev/null 2>&1; then
  warn "Dataset ${DATASET_NAME} already exists; removing so the demo can start clean."
  "${LF_BIN}" datasets remove "${DATASET_NAME}" || warn "Could not remove dataset; continuing."
fi
"${LF_BIN}" datasets add "${DATASET_NAME}" -s "${PROCESSOR_NAME}" -b "${DATABASE_NAME}"
success "Dataset ${DATASET_NAME} created."

bold "Step 3: Ingest FDA correspondence PDFs"
for pdf in "${SAMPLE_DIR}"/*.pdf; do
  info "Adding $(basename "$pdf")"
  "${LF_BIN}" datasets ingest "${DATASET_NAME}" "$pdf"
done
success "All FDA documents ingested."

bold "Step 4: Review dataset metadata"
"${LF_BIN}" datasets list

bold "Step 5: Process dataset into the vector store"
info "Processing may take a couple of minutes due to PDF size."
if "${LF_BIN}" datasets process "${DATASET_NAME}"; then
  success "Processing complete."
else
  fail "Processing failed; check RAG worker logs and retry."
fi

bold "Step 6: Inspect retrieval results"
sleep 10
run_cmd "RAG query: highlight clinical trial data requirements." \
  "${LF_BIN}" rag query --database "${DATABASE_NAME}" --top-k 3 --include-metadata --include-score \
  "Which FDA letters mention additional clinical trial data requirements?"

bold "Step 7: Ask targeted questions with and without RAG"
run_cmd "RAG-enabled response summarising 2024 deficiencies." \
  "${LF_BIN}" run --database "${DATABASE_NAME}" \
  "Summarize the key deficiencies highlighted across the 2024 FDA correspondence letters. Focus on clinical data requests and timeline impacts."

run_cmd "RAG-enabled response targeting letter 761240." \
  "${LF_BIN}" run --database "${DATABASE_NAME}" \
  "According to FDA correspondence 761240, what follow-up actions were requested from the sponsor? Provide a concise list."

run_cmd "RAG-enabled: follow-up timelines." \
  "${LF_BIN}" run --database "${DATABASE_NAME}" \
  "What timeline-related requests are mentioned for sponsors in the 2024 FDA letters? Cite the relevant letter identifiers."

run_cmd "Baseline response without RAG (LLM only)." \
  "${LF_BIN}" run --no-rag \
  "According to FDA correspondence 761240, what follow-up actions were requested from the sponsor?"

bold "Example complete"
success "You can now explore additional questions with '${LF_BIN} run --database ${DATABASE_NAME} \"YOUR QUESTION\"'."

popd > /dev/null
