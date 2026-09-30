#!/usr/bin/env bash

# Exercise online installer failures without downloads or a system-wide install.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TEST_DIR=$(mktemp -d)
trap 'rm -rf "$TEST_DIR"' EXIT

mkdir -p "$TEST_DIR/tools"
cat > "$TEST_DIR/tools/curl" <<'CURL'
#!/usr/bin/env bash
set -euo pipefail
output=""
url=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) output="$2"; shift 2 ;;
    http*) url="$1"; shift ;;
    *) shift ;;
  esac
done
name="${url##*/}"
if [[ "$name" == lf-* ]]; then
  cat > "$output" <<'CLI'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_CLI_LOG"
exit 0
CLI
  exit 0
fi
for component in $FAILED_COMPONENTS; do
  if [[ "$name" == "llamafarm-${component}-"* ]]; then
    exit 22
  fi
done
printf '#!/usr/bin/env bash\nexit 0\n' > "$output"
CURL
chmod +x "$TEST_DIR/tools/curl"

run_case() {
  local name="$1" failed="$2" expected="$3" stale="${4:-false}"
  local case_dir="$TEST_DIR/$name" status=0
  mkdir -p "$case_dir/install" "$case_dir/data/bin"
  if [[ "$stale" == true ]]; then
    for component in server rag runtime; do
      printf 'old service\n' > "$case_dir/data/bin/llamafarm-$component"
    done
  fi
  PATH="$TEST_DIR/tools:$case_dir/install:$PATH" \
    FAILED_COMPONENTS="$failed" FAKE_CLI_LOG="$case_dir/cli.log" \
    LF_DATA_DIR="$case_dir/data" \
    bash "$REPO_ROOT/install.sh" --version v-test --install-dir "$case_dir/install" \
    > "$case_dir/output" 2>&1 || status=$?

  if [[ "$expected" == success ]]; then
    [[ "$status" == 0 ]] || { cat "$case_dir/output"; exit 1; }
    grep -q 'LlamaFarm installed successfully!' "$case_dir/output"
    grep -q '^bundle bootstrap$' "$case_dir/cli.log"
    grep -q '^version$' "$case_dir/cli.log"
    for component in server rag runtime; do
      local binary="$case_dir/data/bin/llamafarm-$component"
      [[ -s "$binary" && -x "$binary" ]] || {
        echo "FAIL: $name did not install a non-empty executable $component binary"
        cat "$case_dir/output"
        exit 1
      }
    done
  else
    [[ "$status" != 0 ]] || { echo "FAIL: $name returned success"; cat "$case_dir/output"; exit 1; }
    if grep -q 'LlamaFarm installed successfully!' "$case_dir/output"; then
      echo "FAIL: $name reported success"; exit 1
    fi
    [[ ! -e "$case_dir/cli.log" ]] || { echo "FAIL: $name bootstrapped incomplete services"; exit 1; }
    grep -q 'Installation incomplete: failed to download required services' "$case_dir/output" || {
      echo "FAIL: $name did not explain the incomplete installation"
      cat "$case_dir/output"
      exit 1
    }
    for component in $failed; do
      grep -q "Failed to download $component" "$case_dir/output" || {
        echo "FAIL: $name did not identify the failed $component download"
        cat "$case_dir/output"
        exit 1
      }
    done
  fi
  echo "PASS: $name"
}

run_case all-downloads-succeed '' success
run_case one-download-fails 'rag' failure
run_case all-downloads-fail 'server rag runtime' failure
run_case stale-binaries-do-not-mask-failure 'server' failure true
