#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_PATH="$DIR/llamafarm.yaml"

# Point to our config
export LF_CONFIG_PATH="$CONFIG_PATH"

echo "=============================================="
echo "IDP COMPLETE APP: From Doc to Action"
echo "=============================================="

# 1. Invoice Scenario
echo ""
echo "--- Scenario 1: Invoice Processing ---"
DOC_INVOICE="INVOICE #9923
Date: 2025-02-14
Vendor: Nvidia Corp
Bill To: LlamaFarm Inc.
Items:
- H100 GPU Cluster (x8) ... $250,000.00
Total: $250,000.00
"
echo "Document: Invoice from Nvidia"
lf chat --cwd "$DIR" "Process this document: $DOC_INVOICE"

# 2. NDA Scenario
echo ""
echo "--- Scenario 2: Legal Archival ---"
DOC_NDA="MUTUAL NON-DISCLOSURE AGREEMENT
This agreement is between LlamaFarm Inc and OpenAI (the 'Parties').
Effective Date: 2025-01-01.
Term: This agreement shall expire 3 years from effective date.
CONFIDENTIALITY: All model weights are secret.
"
echo "Document: NDA with OpenAI"
lf chat --cwd "$DIR" "Process this document: $DOC_NDA"

# 3. Anomaly Scenario
echo ""
echo "--- Scenario 3: Anomaly Detection ---"
DOC_BAD="sd8s7d87s8d7s8d7s8d7s
UNKNOWN DATA STREAM
x98989898
"
echo "Document: Corrupted Data"
lf chat --cwd "$DIR" "Process this document: $DOC_BAD"
