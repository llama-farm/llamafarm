#!/bin/bash
# Demo: Table Question Answering (TAPAS)
# Phase 13 of Universal Runtime ML Enhancements
#
# Answers natural language questions about tabular data.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Load port from .env file
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    source "$SCRIPT_DIR/../../.env"
fi
RUNTIME_URL="${RUNTIME_URL:-http://127.0.0.1:${LF_RUNTIME_PORT:-11540}}"

echo "=========================================="
echo "Table Question Answering Demo (TAPAS)"
echo "=========================================="
echo ""

# Check if server is running
echo "1. Checking server health..."
if ! curl -s "$RUNTIME_URL/health" > /dev/null 2>&1; then
    echo "   ERROR: Server not running at $RUNTIME_URL"
    echo "   Start with: cd runtimes/universal && uv run python server.py"
    exit 1
fi
echo "   Server is healthy!"
echo ""

# Basic question answering
echo "2. Basic Question Answering"
echo "   Table: Employee information"
echo ""
RESPONSE=$(curl -s -X POST "$RUNTIME_URL/v1/analysis/table-qa" \
    -H "Content-Type: application/json" \
    -d '{
        "table": {
            "columns": ["Name", "Age", "Department", "Salary"],
            "rows": [
                ["Alice", "30", "Engineering", "85000"],
                ["Bob", "25", "Marketing", "65000"],
                ["Charlie", "35", "Engineering", "95000"],
                ["Diana", "28", "Sales", "70000"]
            ]
        },
        "question": "Who works in Engineering?"
    }')

echo "   Question: Who works in Engineering?"
echo "   Response:"
echo "$RESPONSE" | python3 -m json.tool
echo ""

# Age-related query
echo "3. Selection Query"
echo "   Question: Who is the oldest employee?"
echo ""
RESPONSE=$(curl -s -X POST "$RUNTIME_URL/v1/analysis/table-qa" \
    -H "Content-Type: application/json" \
    -d '{
        "table": {
            "columns": ["Name", "Age", "Department", "Salary"],
            "rows": [
                ["Alice", "30", "Engineering", "85000"],
                ["Bob", "25", "Marketing", "65000"],
                ["Charlie", "35", "Engineering", "95000"],
                ["Diana", "28", "Sales", "70000"]
            ]
        },
        "question": "Who is the oldest employee?"
    }')

echo "   Response:"
echo "$RESPONSE" | python3 -m json.tool
echo ""

# Server metrics example
echo "4. Server Metrics Analysis"
echo "   Table: Server health metrics"
echo ""
RESPONSE=$(curl -s -X POST "$RUNTIME_URL/v1/analysis/table-qa" \
    -H "Content-Type: application/json" \
    -d '{
        "table": {
            "columns": ["Server", "CPU%", "Memory%", "Status", "Region"],
            "rows": [
                ["web-01", "45", "60", "healthy", "us-east"],
                ["web-02", "85", "70", "warning", "us-east"],
                ["db-01", "30", "80", "healthy", "us-west"],
                ["cache-01", "15", "40", "healthy", "eu-west"],
                ["api-01", "92", "85", "critical", "us-east"]
            ]
        },
        "question": "Which server has the highest CPU usage?"
    }')

echo "   Question: Which server has the highest CPU usage?"
echo "   Response:"
echo "$RESPONSE" | python3 -m json.tool
echo ""

# Multiple questions (batch)
echo "5. Batch Questions (Same Table)"
echo "   Asking multiple questions about server metrics..."
echo ""
RESPONSE=$(curl -s -X POST "$RUNTIME_URL/v1/analysis/table-qa/batch" \
    -H "Content-Type: application/json" \
    -d '{
        "table": {
            "columns": ["Server", "CPU%", "Memory%", "Status"],
            "rows": [
                ["web-01", "45", "60", "healthy"],
                ["web-02", "85", "70", "warning"],
                ["db-01", "30", "80", "healthy"],
                ["cache-01", "15", "40", "healthy"]
            ]
        },
        "questions": [
            "Which server is in warning status?",
            "Which servers are healthy?",
            "What is the CPU of db-01?"
        ]
    }')

echo "   Questions:"
echo "   1. Which server is in warning status?"
echo "   2. Which servers are healthy?"
echo "   3. What is the CPU of db-01?"
echo ""
echo "   Response:"
echo "$RESPONSE" | python3 -m json.tool
echo ""

# Product inventory example
echo "6. Product Inventory Query"
echo "   Table: Product inventory"
echo ""
RESPONSE=$(curl -s -X POST "$RUNTIME_URL/v1/analysis/table-qa" \
    -H "Content-Type: application/json" \
    -d '{
        "table": {
            "columns": ["Product", "Category", "Price", "Stock", "Supplier"],
            "rows": [
                ["Laptop", "Electronics", "999", "50", "TechCorp"],
                ["Mouse", "Electronics", "29", "200", "TechCorp"],
                ["Desk", "Furniture", "349", "30", "OfficePro"],
                ["Chair", "Furniture", "199", "45", "OfficePro"],
                ["Monitor", "Electronics", "449", "75", "TechCorp"]
            ]
        },
        "question": "What is the most expensive product?"
    }')

echo "   Question: What is the most expensive product?"
echo "   Response:"
echo "$RESPONSE" | python3 -m json.tool
echo ""

# Status filtering
echo "7. Status Query"
echo "   Finding items by status..."
echo ""
RESPONSE=$(curl -s -X POST "$RUNTIME_URL/v1/analysis/table-qa" \
    -H "Content-Type: application/json" \
    -d '{
        "table": {
            "columns": ["Task", "Assignee", "Priority", "Status"],
            "rows": [
                ["Fix login bug", "Alice", "High", "In Progress"],
                ["Update docs", "Bob", "Low", "Complete"],
                ["Add caching", "Charlie", "High", "Pending"],
                ["Write tests", "Alice", "Medium", "In Progress"]
            ]
        },
        "question": "Which tasks are in progress?"
    }')

echo "   Question: Which tasks are in progress?"
echo "   Response:"
echo "$RESPONSE" | python3 -m json.tool
echo ""

echo "=========================================="
echo "Demo Complete!"
echo "=========================================="
echo ""
echo "Key features demonstrated:"
echo "  - Natural language questions about tables"
echo "  - Selection queries (find specific rows)"
echo "  - Filtering by column values"
echo "  - Batch question answering"
echo "  - Various table types (employees, servers, products, tasks)"
echo ""
echo "Supported aggregations:"
echo "  - NONE: Direct cell selection"
echo "  - SUM: Sum of numeric values"
echo "  - AVERAGE: Average of numeric values"
echo "  - COUNT: Count of matching cells"
