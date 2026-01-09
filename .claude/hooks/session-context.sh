#!/bin/bash
# Session start hook to inject project context

echo "LlamaFarm Project Context:"
echo "  Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
echo "  Last commit: $(git log -1 --format='%h %s' 2>/dev/null || echo 'unknown')"
