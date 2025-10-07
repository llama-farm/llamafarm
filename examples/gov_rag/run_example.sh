#!/usr/bin/env bash
# Raleigh UDO RAG example workflow

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${1:-${PROJECT_ROOT_DEFAULT}}"
LF_BIN="${PROJECT_ROOT}/lf"
CONFIG_PATH="${PROJECT_ROOT}/llamafarm.yaml"
DATASET_NAME="raleigh_udo_dataset"
DATABASE_NAME="raleigh_udo_db"
PROCESSOR_NAME="udo_pdf_processor"
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

bold "Raleigh UDO RAG Example"
info "Project root: ${PROJECT_ROOT}"
info "Using dataset '${DATASET_NAME}' with database '${DATABASE_NAME}'"

ensure_file "$LF_BIN" "LlamaFarm CLI not found at ${LF_BIN}. Build it with 'go build -o lf cli/main.go'."
ensure_file "$CONFIG_PATH" "No llamafarm.yaml found at ${CONFIG_PATH}. Run './examples/gov_rag/update_config.sh' first."
ensure_dir "$SAMPLE_DIR" "Sample UDO PDF not found at ${SAMPLE_DIR}."
if ! compgen -G "${SAMPLE_DIR}"/*.pdf > /dev/null; then
  fail "No PDF files located in ${SAMPLE_DIR}."
fi

run_cmd() {
  local description="$1"; shift
  info "$description"
  if ! "$@"; then
    warn "Command failed; rerun manually if needed:\n     $*"
  fi
}

pushd "$PROJECT_ROOT" > /dev/null
trap 'warn "Example interrupted."' INT TERM

bold "Step 1: Confirm CLI connectivity"
"${LF_BIN}" version || fail "Unable to execute LlamaFarm CLI."

bold "Step 1a: Check RAG worker health"
health_output="$(${LF_BIN} rag health || true)"
echo "$health_output"
# Use POSIX character class to match whitespace reliably
if grep -Eq 'celery[[:space:]]+degraded' <<<"$health_output"; then
  warn "RAG worker appears offline. Start it with './start-local.sh' or 'nx start rag' and rerun this script."
  exit 1
fi

bold "Step 2: Prepare dataset '${DATASET_NAME}'"
if "${LF_BIN}" datasets list | grep -q "${DATASET_NAME}"; then
  warn "Dataset ${DATASET_NAME} already exists; removing so the demo can start clean."
  "${LF_BIN}" datasets remove "${DATASET_NAME}" || warn "Could not remove dataset; continuing."
fi
"${LF_BIN}" datasets add "${DATASET_NAME}" -s "${PROCESSOR_NAME}" -b "${DATABASE_NAME}"
success "Dataset ${DATASET_NAME} created."

bold "Step 3: Ingest UDO PDF"
for pdf in "${SAMPLE_DIR}"/*.pdf; do
  info "Adding $(basename "$pdf")"
  "${LF_BIN}" datasets ingest "${DATASET_NAME}" "$pdf"
done
success "UDO documents uploaded."

bold "Step 4: Review dataset metadata"
"${LF_BIN}" datasets list

bold "Step 5: Process dataset into the vector store"
info "Processing may take several minutes due to the ordinance size."
if "${LF_BIN}" datasets process "${DATASET_NAME}"; then
  success "Processing complete."
else
  warn "Processing reported an error; verify the RAG worker logs if ingestion appears incomplete."
fi

bold "Step 6: Inspect retrieval results"
sleep 20
run_cmd "Query: Which section discusses neighborhood transition requirements?" \
  "${LF_BIN}" rag query --database "${DATABASE_NAME}" --top-k 3 --include-metadata --include-score \
  "Which section of the UDO covers neighborhood transition requirements?"

run_cmd "Query: Identify requirements for neighborhood conservation overlay districts." \
  "${LF_BIN}" rag query --database "${DATABASE_NAME}" --top-k 5 --include-metadata --include-score \
  "Where does the ordinance describe Neighborhood Conservation Overlay District standards?"

bold "Step 7: Ask targeted questions"
run_cmd "RAG-enabled: summarize village mixed use building height limits." \
  "${LF_BIN}" run --database "${DATABASE_NAME}" \
  "Summarize the maximum building height allowances for Village Mixed Use districts, including section citations."

run_cmd "RAG-enabled: list buffer requirements adjacent to residential zoning." \
  "${LF_BIN}" run --database "${DATABASE_NAME}" \
  "What buffering or screening does the UDO require when non-residential development abuts residential lots? Cite the relevant sections."

run_cmd "RAG-enabled: parking reductions in transit overlay districts." \
  "${LF_BIN}" run --database "${DATABASE_NAME}" \
  "Detail the parking reductions permitted in Transit Overlay Districts with citations to the relevant subsections."

run_cmd "Baseline without RAG (LLM only for comparison)." \
  "${LF_BIN}" run --no-rag \
  "How tall can buildings be in Village Mixed Use zoning?"

bold "Step 8: Additional exploration"
success "Explore further with '${LF_BIN} run --database ${DATABASE_NAME} \"YOUR QUESTION\"' or 'lf rag query --database ${DATABASE_NAME} "topic"'."

popd > /dev/null
