#!/usr/bin/env bash
# Replace the current llamafarm.yaml with the Raleigh UDO example configuration.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${1:-${PROJECT_ROOT_DEFAULT}}"
TARGET_CONFIG="${PROJECT_ROOT}/llamafarm.yaml"
EXAMPLE_CONFIG="${SCRIPT_DIR}/llamafarm-example-gov.yaml"
BACKUP_SUFFIX="backup_$(date +%s)"

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
success() { printf '\033[0;32m✓ %s\033[0m\n' "$1"; }
info() { printf '\033[0;34mℹ %s\033[0m\n' "$1"; }
warn() { printf '\033[0;33m⚠ %s\033[0m\n' "$1"; }
fail() { printf '\033[0;31m✗ %s\033[0m\n' "$1"; exit 1; }

bold "Applying Raleigh UDO example configuration"
info "Project root: ${PROJECT_ROOT}"
info "Target config: ${TARGET_CONFIG}"

[[ -f "$EXAMPLE_CONFIG" ]] || fail "Example config not found at ${EXAMPLE_CONFIG}."

if [[ -f "$TARGET_CONFIG" ]]; then
  BACKUP_PATH="${TARGET_CONFIG}.${BACKUP_SUFFIX}"
  cp "$TARGET_CONFIG" "$BACKUP_PATH"
  success "Backed up existing config to ${BACKUP_PATH}."
else
  warn "No existing llamafarm.yaml found; a new one will be created."
fi

cp "$EXAMPLE_CONFIG" "$TARGET_CONFIG"
success "Copied Raleigh UDO example config to ${TARGET_CONFIG}."

info "Next: run './examples/gov_rag/run_example.sh' (optionally pass the project root)."
