#!/usr/bin/env bash
# ROW-79: Monitor edge-runtime container memory and CPU during load test.
#
# Polls docker stats every 30s, writes CSV to stdout or a file.
#
# Usage:
#   ./monitor_memory.sh                          # stdout
#   ./monitor_memory.sh -o memory_log.csv        # file
#   ./monitor_memory.sh -c edge-runtime -i 15    # custom container & interval
#   ./monitor_memory.sh -d 1800                  # run for 30 min then stop

set -euo pipefail

CONTAINER="${CONTAINER:-edge-runtime}"
INTERVAL=30
DURATION=0  # 0 = run until killed
OUTPUT=""

usage() {
    echo "Usage: $0 [-c container] [-i interval_sec] [-d duration_sec] [-o output.csv]"
    exit 1
}

while getopts "c:i:d:o:h" opt; do
    case $opt in
        c) CONTAINER="$OPTARG" ;;
        i) INTERVAL="$OPTARG" ;;
        d) DURATION="$OPTARG" ;;
        o) OUTPUT="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

header="timestamp,memory_rss_mb,cpu_percent,memory_percent,pids"

if [[ -n "$OUTPUT" ]]; then
    echo "$header" > "$OUTPUT"
    exec >> "$OUTPUT"
else
    echo "$header"
fi

start=$(date +%s)
count=0

while true; do
    # Check duration limit
    if [[ "$DURATION" -gt 0 ]]; then
        elapsed=$(( $(date +%s) - start ))
        if [[ "$elapsed" -ge "$DURATION" ]]; then
            echo "# Duration limit reached (${DURATION}s)" >&2
            break
        fi
    fi

    # Poll docker stats (single snapshot, no stream)
    stats=$(docker stats "$CONTAINER" --no-stream --format \
        '{{.MemUsage}}|{{.CPUPerc}}|{{.MemPerc}}|{{.PIDs}}' 2>/dev/null) || {
        echo "# WARNING: docker stats failed for $CONTAINER" >&2
        sleep "$INTERVAL"
        continue
    }

    # Parse fields: "123.4MiB / 1.94GiB|0.50%|6.35%|12"
    mem_usage=$(echo "$stats" | cut -d'|' -f1 | cut -d'/' -f1 | xargs)
    cpu_pct=$(echo "$stats" | cut -d'|' -f2 | tr -d '%')
    mem_pct=$(echo "$stats" | cut -d'|' -f3 | tr -d '%')
    pids=$(echo "$stats" | cut -d'|' -f4)

    # Normalize memory to MB
    mem_value=$(echo "$mem_usage" | grep -oE '[0-9]+\.?[0-9]*')
    mem_unit=$(echo "$mem_usage" | grep -oiE '[KMGT]i?B')

    case "$mem_unit" in
        KiB|kB)  mem_mb=$(echo "$mem_value / 1024" | bc -l) ;;
        MiB|MB)  mem_mb="$mem_value" ;;
        GiB|GB)  mem_mb=$(echo "$mem_value * 1024" | bc -l) ;;
        *)       mem_mb="$mem_value" ;;
    esac

    mem_mb=$(printf "%.1f" "$mem_mb")
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    echo "${ts},${mem_mb},${cpu_pct},${mem_pct},${pids}"

    count=$((count + 1))
    if [[ $((count % 10)) -eq 0 ]]; then
        echo "# Samples collected: $count" >&2
    fi

    sleep "$INTERVAL"
done
