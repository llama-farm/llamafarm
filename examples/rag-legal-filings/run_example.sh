#!/usr/bin/env bash
# Legal filings PDF RAG demo (007 legal filings)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${1:-${PROJECT_ROOT_DEFAULT}}"
LF_BIN="${LF_BIN:-${PROJECT_ROOT}/lf}"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/llamafarm.yaml}"
EXAMPLE_NAME="007"
BASE_DATABASE_NAME="legal_filings_db_007"
PROCESSOR_NAME="legal_filings_pdf_processor_007"
FILES_DIR="${SCRIPT_DIR}/files"
RUN_ID=$(date +%Y%m%d%H%M%S)
DATABASE_NAME="${EXAMPLE_NAME}_db_${RUN_ID}"
DATASET_NAME="${EXAMPLE_NAME}_dataset_${RUN_ID}"
NO_PAUSE=${NO_PAUSE:-0}

lf() {
  "${LF_BIN}" --cwd "${SCRIPT_DIR}" "$@"
}

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
info() { printf '\033[0;34mℹ %s\033[0m\n' "$1"; }
success() { printf '\033[0;32m✓ %s\033[0m\n' "$1"; }
warn() { printf '\033[0;33m⚠ %s\033[0m\n' "$1"; }
error() { printf '\033[0;31m✗ %s\033[0m\n' "$1"; }

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

any_pdf() {
  compgen -G "$1"/*.pdf >/dev/null
}

duplicate_database() {
  PYTHON_CONFIG_PATH="$CONFIG_PATH" \
  PYTHON_DATABASE_NAME="$DATABASE_NAME" \
  PYTHON_BASE_DATABASE_NAME="$BASE_DATABASE_NAME" \
  python3 <<'PY'
import copy
import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML is required for this example. Install with 'uv pip install pyyaml'.", file=sys.stderr)
    sys.exit(1)

cfg_path = Path(os.environ['PYTHON_CONFIG_PATH'])
if not cfg_path.exists():
    print(f"Config file {cfg_path} does not exist", file=sys.stderr)
    sys.exit(1)

cfg = yaml.safe_load(cfg_path.read_text()) or {}
rag = cfg.setdefault('rag', {})
databases = rag.setdefault('databases', [])
target_name = os.environ['PYTHON_DATABASE_NAME']
base_name = os.environ['PYTHON_BASE_DATABASE_NAME']

if any(db.get('name') == target_name for db in databases):
    print(f"Database {target_name} already present; skipping duplication")
else:
    base = next((db for db in databases if db.get('name') == base_name), None)
    if base is None:
        if databases:
            base = databases[0]
            print(f"Base database {base_name} not found; cloning the first entry {base.get('name')} instead.")
        else:
            print("No databases defined in config; aborting.", file=sys.stderr)
            sys.exit(1)
    new_db = copy.deepcopy(base)
    new_db['name'] = target_name
    cfg_dir = new_db.setdefault('config', {})
    persist = cfg_dir.get('persist_directory') or f"./data/{base_name}"
    cfg_dir['persist_directory'] = persist.rsplit('/', 1)[0] + f"/{target_name}"
    databases.append(new_db)
    cfg_path.write_text(yaml.dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"Added database {target_name} to configuration.")
PY
}

bold "007 Legal Filings RAG Demo (James Bond franchise litigation)"
info "Project root: ${PROJECT_ROOT}"
info "Using temporary database '${DATABASE_NAME}' and dataset '${DATASET_NAME}'"

ensure_file "$LF_BIN" "LlamaFarm CLI not found at ${LF_BIN}. Build it with 'go build -o lf cli/main.go'."
ensure_file "$CONFIG_PATH" "No example config found at ${CONFIG_PATH}."
ensure_dir "$FILES_DIR" "Legal filings directory missing: ${FILES_DIR}."
if ! any_pdf "$FILES_DIR"; then
  error "No PDF files found in ${FILES_DIR}."
  exit 1
fi

duplicate_database
pause

pushd "$PROJECT_ROOT" >/dev/null
trap 'warn "Example interrupted."' INT TERM

bold "Step 1: Verify CLI connectivity"
lf version
pause

bold "Step 2: Create dataset '${DATASET_NAME}'"
lf datasets create -s "${PROCESSOR_NAME}" -b "${DATABASE_NAME}" "${DATASET_NAME}"
success "Dataset created."
pause

bold "Step 3: Upload legal filing PDFs"
for pdf in "${FILES_DIR}"/*.pdf; do
  info "Uploading $(basename "$pdf")"
  lf datasets upload "${DATASET_NAME}" "$pdf"
done
success "Legal documents uploaded."
pause

bold "Step 4: Review datasets"
lf datasets list
pause

bold "Step 5: Process dataset"
info "Processing may take several minutes due to the legal document complexity."
lf datasets process "${DATASET_NAME}"
success "Processing complete."
pause

bold "Step 6: Explore retrieval"
lf rag query --database "${DATABASE_NAME}" --top-k 3 --include-metadata --include-score \
  "What was the main dispute in MGM v. American Honda Motor?"
pause

bold "Step 7: Ask questions with RAG context"
lf chat --database "${DATABASE_NAME}" \
  "Summarize the key legal issues in the Danjaq LLC v. McClory case with citations."

pause
lf chat --database "${DATABASE_NAME}" \
  "What trademark disputes involved the James Bond franchise? Include case names and outcomes."

pause
lf chat --database "${DATABASE_NAME}" \
  "List the parties involved in the Universal City Studios litigation with relevant details."

pause

bold "Step 8: Compare with no RAG"
lf chat --no-rag \
  "What legal issues has the James Bond franchise faced?"

pause
success "Demo complete. Explore further with 'lf chat --database ${DATABASE_NAME} "YOUR QUESTION"'."
warn "Database '${DATABASE_NAME}' and dataset '${DATASET_NAME}' remain available. Remove them when finished."

popd >/dev/null
