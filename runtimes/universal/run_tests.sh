#!/bin/bash
# Test runner for Universal Runtime

set -e

cd "$(dirname "$0")"

echo "================================================"
echo "Universal Runtime Test Suite"
echo "================================================"
echo ""

# Parse arguments
RUN_SLOW=false
COVERAGE=false
VERBOSE=false
TEST_PATH="tests/"

while [[ $# -gt 0 ]]; do
    case $1 in
        --slow)
            RUN_SLOW=true
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            TEST_PATH="$1"
            shift
            ;;
    esac
done

# Build pytest command (use python -m pytest to ensure uv environment)
PYTEST_CMD=(uv run python -m pytest "$TEST_PATH")

# Add verbose flag
if [ "$VERBOSE" = true ]; then
    PYTEST_CMD+=(-v)
fi

# Add coverage
if [ "$COVERAGE" = true ]; then
    PYTEST_CMD+=(--cov=models --cov-report=term-missing --cov-report=html)
fi

# Skip slow tests unless explicitly requested
if [ "$RUN_SLOW" = false ]; then
    PYTEST_CMD+=(-m "not slow")
    echo "Running fast tests only (use --slow to include all tests)"
else
    echo "Running all tests including slow model downloads"
fi

echo ""
echo "Command: ${PYTEST_CMD[*]}"
echo ""

# Run tests
"${PYTEST_CMD[@]}"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ All tests passed!"
else
    echo "❌ Some tests failed (exit code: $EXIT_CODE)"
fi

if [ "$COVERAGE" = true ]; then
    echo ""
    echo "📊 Coverage report generated in htmlcov/index.html"
fi

exit $EXIT_CODE
