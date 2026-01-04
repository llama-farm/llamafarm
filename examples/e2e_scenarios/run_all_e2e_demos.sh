#!/bin/bash
#
# Run All End-to-End LlamaFarm Demos
#
# This script demonstrates the full power of the Embedded Trinity Memory System
# by running comprehensive military and medical scenarios that showcase:
#
# - Time-Series Store (DuckDB): Biometrics, vitals, telemetry
# - Graph Store (DuckDB): Personnel, patients, relationships
# - Working Memory (DuckDB): Communications, clinical notes
# - Linkage Table: Cross-store UUID tracking
# - ML Operations: Classifiers and anomaly detection
# - Memory Consolidation: The "hippocampus" process
#
# Usage:
#   ./run_all_e2e_demos.sh           # Run all demos
#   ./run_all_e2e_demos.sh --quick   # Skip slow operations
#   ./run_all_e2e_demos.sh military  # Run only military demo
#   ./run_all_e2e_demos.sh medical   # Run only medical demo
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
MAGENTA='\033[0;35m'
NC='\033[0m'
BOLD='\033[1m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RAG_DIR="$PROJECT_DIR/rag"

# Track results
PASSED=0
FAILED=0
SKIPPED=0

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

print_banner() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  ${BOLD}$1${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_section() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${CYAN}→${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# ─────────────────────────────────────────────────────────────────────────────
# Demo Execution
# ─────────────────────────────────────────────────────────────────────────────

run_demo() {
    local name="$1"
    local script="$2"
    local description="$3"

    print_section "$name"
    echo -e "${MAGENTA}$description${NC}"
    echo ""

    if [ ! -f "$script" ]; then
        print_error "Demo script not found: $script"
        FAILED=$((FAILED + 1))
        return 1
    fi

    print_info "Running: $script"
    echo ""

    if uv run python "$script"; then
        echo ""
        print_success "$name - PASSED"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo ""
        print_error "$name - FAILED"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight Checks
# ─────────────────────────────────────────────────────────────────────────────

preflight_checks() {
    print_section "Pre-flight Checks"

    # Check we're in the right place
    if [ ! -d "$RAG_DIR" ]; then
        print_error "RAG directory not found: $RAG_DIR"
        exit 1
    fi
    print_success "RAG directory found"

    # Check Python/uv is available
    if ! command -v uv &> /dev/null; then
        print_error "uv not found. Please install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    print_success "uv package manager found"

    # Check for required dependencies
    cd "$RAG_DIR"
    print_info "Checking dependencies..."
    if uv run python -c "from components.stores.duckdb_store import DuckDBStore" 2>/dev/null; then
        print_success "DuckDB store components available"
    else
        print_warning "Installing dependencies..."
        uv sync
    fi

    # Check core modules
    if uv run python -c "from core.memory import MemoryStore" 2>/dev/null; then
        print_success "MemoryStore component available"
    else
        print_error "MemoryStore component not found"
        exit 1
    fi

    if uv run python -c "from core.consolidator import Consolidator" 2>/dev/null; then
        print_success "Consolidator component available"
    else
        print_error "Consolidator component not found"
        exit 1
    fi

    echo ""
    print_success "All pre-flight checks passed!"
}

# ─────────────────────────────────────────────────────────────────────────────
# Print Summary
# ─────────────────────────────────────────────────────────────────────────────

print_summary() {
    print_section "Demo Summary"

    echo -e "  ${GREEN}Passed:${NC}  $PASSED"
    echo -e "  ${RED}Failed:${NC}  $FAILED"
    echo -e "  ${YELLOW}Skipped:${NC} $SKIPPED"
    echo ""

    if [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║${NC}  ${BOLD}All demos completed successfully!${NC}"
        echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
        return 0
    else
        echo -e "${RED}╔══════════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║${NC}  ${BOLD}Some demos failed. See output above for details.${NC}"
        echo -e "${RED}╚══════════════════════════════════════════════════════════════════════╝${NC}"
        return 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

main() {
    local run_military=true
    local run_medical=true

    # Parse arguments
    for arg in "$@"; do
        case $arg in
            military)
                run_medical=false
                ;;
            medical)
                run_military=false
                ;;
            --quick)
                print_warning "Quick mode: Some operations may be simplified"
                export QUICK_MODE=1
                ;;
            --help|-h)
                echo "Usage: $0 [options] [demo]"
                echo ""
                echo "Options:"
                echo "  --quick    Skip slow operations"
                echo "  --help     Show this help"
                echo ""
                echo "Demos:"
                echo "  military   Run only military rescue demo"
                echo "  medical    Run only medical patient demo"
                echo ""
                echo "Examples:"
                echo "  $0                    # Run all demos"
                echo "  $0 military           # Run only military demo"
                echo "  $0 --quick medical    # Run medical demo quickly"
                exit 0
                ;;
        esac
    done

    print_banner "LlamaFarm End-to-End Demos"
    echo -e "  ${BOLD}Embedded Trinity Memory System${NC}"
    echo -e "  Vector + Time-Series + Graph + Working Memory"
    echo ""

    # Change to RAG directory
    cd "$RAG_DIR"

    # Run pre-flight checks
    preflight_checks

    # Run demos
    if [ "$run_military" = true ]; then
        run_demo \
            "Military Rescue Scenario" \
            "$SCRIPT_DIR/demo_military_rescue.py" \
            "Demonstrates biometric telemetry, radio communications, command structure,
  distress detection, and rescue coordination."
    else
        SKIPPED=$((SKIPPED + 1))
    fi

    if [ "$run_medical" = true ]; then
        run_demo \
            "Medical Patient Scenario" \
            "$SCRIPT_DIR/demo_medical_patient.py" \
            "Demonstrates patient monitoring, clinical documentation, medication tracking,
  vital sign anomaly detection, and clinical decision support."
    else
        SKIPPED=$((SKIPPED + 1))
    fi

    # Print summary
    print_summary
}

# Run main with all arguments
main "$@"
