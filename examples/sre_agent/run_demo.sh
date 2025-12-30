#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_PATH="$DIR/llamafarm.yaml"

# Point to our config
export LF_CONFIG_PATH="$CONFIG_PATH"

echo "=============================================="
echo "SRE AGENT DEMO: Incident Response"
echo "=============================================="
echo "Simulation: CPU is at 85% (Critical) and 'payment-gateway' is degraded."
echo "Agent Instructions: Automatically fix high CPU/degraded services."
echo ""

# Query 1: The Trigger
# We simulate an alert coming in.
echo "[Alert System]: Triggering Agent check..."
# Note: --auto-start is default true, so we can omit it or be explicit.
# We expect the agent to:
# 1. Call get_system_metrics() -> see 85% CPU
# 2. Call fetch_recent_logs('payment-gateway') -> see errors
# 3. Call restart_service('payment-gateway') -> fix it
lf chat --cwd "$DIR" "Alert: System usage is high and payment success rate is dropping. Please triage and fix."

echo ""
echo "=============================================="
echo "Verification"
echo "=============================================="
echo "Checking system status again..."

lf chat --cwd "$DIR" "Report current system status."
