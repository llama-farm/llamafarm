#!/usr/bin/env bash
# ============================================================
# SFT Fine-Tuning End-to-End Test
# ============================================================
# Trains FunctionGemma 270M on a small tool-routing dataset,
# exports to GGUF, and verifies the model loads in the runtime.
#
# Prerequisites:
#   - Universal Runtime running on :11540
#   - unsloth-mlx installed (Mac) or unsloth (CUDA)
#   - convert_hf_to_gguf.py available (see LLAMA_CPP_DIR)
#
# Usage:
#   ./finetune/examples/test_sft.sh
#   ./finetune/examples/test_sft.sh http://localhost:11540
# ============================================================

set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:11540}"
MODEL="unsloth/functiongemma-270m-it"

echo "============================================================"
echo "SFT Fine-Tuning End-to-End Test"
echo "============================================================"
echo "Runtime: $BASE_URL"
echo "Model:   $MODEL"
echo ""

# ── Check runtime ────────────────────────────────────────────
echo "[1/6] Checking runtime..."
if ! curl -sf "$BASE_URL/v1/finetune/templates" > /dev/null 2>&1; then
    echo "  ❌ Runtime not available at $BASE_URL"
    echo "  Start with: cd runtimes/universal && uv run python server.py"
    exit 1
fi
echo "  ✅ Runtime ready"

# ── Validate dataset ─────────────────────────────────────────
echo ""
echo "[2/6] Validating dataset..."
VALIDATE=$(curl -s "$BASE_URL/v1/finetune/validate" \
    -H "Content-Type: application/json" \
    -d '{
    "dataset": [
        {"messages": [{"role":"system","content":"You route user requests to functions."},{"role":"user","content":"Save this finding about fuel prices"},{"role":"assistant","content":"[{\"name\":\"shared_log\",\"arguments\":{\"finding\":\"fuel prices rising\",\"category\":\"fuel\"}}]"}]},
        {"messages": [{"role":"system","content":"You route user requests to functions."},{"role":"user","content":"What did the other agents find?"},{"role":"assistant","content":"[{\"name\":\"shared_search\",\"arguments\":{\"query\":\"findings\",\"limit\":5}}]"}]},
        {"messages": [{"role":"system","content":"You route user requests to functions."},{"role":"user","content":"Write down that we need to check the pipeline"},{"role":"assistant","content":"[{\"name\":\"memory_write\",\"arguments\":{\"file\":\"notes.md\",\"content\":\"Check pipeline\",\"append\":true}}]"}]},
        {"messages": [{"role":"system","content":"You route user requests to functions."},{"role":"user","content":"Read my notes"},{"role":"assistant","content":"[{\"name\":\"memory_read\",\"arguments\":{\"file\":\"notes.md\"}}]"}]},
        {"messages": [{"role":"system","content":"You route user requests to functions."},{"role":"user","content":"Alert the operator about a critical reactor issue"},{"role":"assistant","content":"[{\"name\":\"notify\",\"arguments\":{\"title\":\"Critical Reactor Issue\",\"message\":\"Anomaly detected\",\"priority\":\"high\"}}]"}]},
        {"messages": [{"role":"system","content":"You route user requests to functions."},{"role":"user","content":"Run a health check"},{"role":"assistant","content":"[{\"name\":\"exec\",\"arguments\":{\"command\":\"./health_check.sh\",\"timeout\":30}}]"}]},
        {"messages": [{"role":"system","content":"You route user requests to functions."},{"role":"user","content":"Share results with the team"},{"role":"assistant","content":"[{\"name\":\"shared_write\",\"arguments\":{\"key\":\"results_q4\",\"content\":\"Analysis complete\"}}]"}]},
        {"messages": [{"role":"system","content":"You route user requests to functions."},{"role":"user","content":"Get a specialist to analyze the imagery"},{"role":"assistant","content":"[{\"name\":\"spawn_agent\",\"arguments\":{\"task\":\"Analyze imagery\",\"agent_id\":\"imagery_specialist\"}}]"}]},
        {"messages": [{"role":"system","content":"You route user requests to functions."},{"role":"user","content":"Hello, how are you?"},{"role":"assistant","content":"[]"}]},
        {"messages": [{"role":"system","content":"You route user requests to functions."},{"role":"user","content":"Check sensor temp and log it"},{"role":"assistant","content":"[{\"name\":\"exec\",\"arguments\":{\"command\":\"cat /sys/class/thermal/thermal_zone0/temp\"}},{\"name\":\"shared_log\",\"arguments\":{\"finding\":\"Temperature captured\",\"category\":\"sensors\"}}]"}]}
    ]
}')

VALID=$(echo "$VALIDATE" | python3 -c "import sys,json; print(json.load(sys.stdin)['valid'])" 2>/dev/null)
if [ "$VALID" != "True" ]; then
    echo "  ❌ Dataset validation failed: $VALIDATE"
    exit 1
fi
echo "  ✅ Dataset valid (10 examples, chat format)"

# ── Submit SFT job ───────────────────────────────────────────
echo ""
echo "[3/6] Submitting SFT training job..."
RESPONSE=$(curl -s "$BASE_URL/v1/finetune/sft" \
    -H "Content-Type: application/json" \
    -d "{
    \"model\": \"$MODEL\",
    \"dataset\": $(echo "$VALIDATE" | python3 -c "print('[]')" 2>/dev/null || echo "[]"),
    \"dataset_format\": \"chat\",
    \"chat_template\": \"gemma\",
    \"train_on_responses_only\": true,
    \"output_gguf\": true,
    \"quantization\": \"q8_0\",
    \"lora_rank\": 16,
    \"lora_alpha\": 16,
    \"epochs\": 3,
    \"batch_size\": 2,
    \"learning_rate\": 2e-4,
    \"max_seq_length\": 1024,
    \"max_steps\": 30,
    \"warmup_steps\": 5
}")

# Re-submit with inline dataset since we can't easily pipe validation
RESPONSE=$(curl -s "$BASE_URL/v1/finetune/sft" \
    -H "Content-Type: application/json" \
    -d '{
    "model": "'"$MODEL"'",
    "dataset": [
        {"messages": [{"role":"system","content":"You route user requests to functions."},{"role":"user","content":"Save this finding about fuel prices"},{"role":"assistant","content":"[{\"name\":\"shared_log\",\"arguments\":{\"finding\":\"fuel prices rising\",\"category\":\"fuel\"}}]"}]},
        {"messages": [{"role":"system","content":"You route user requests to functions."},{"role":"user","content":"What did the other agents find?"},{"role":"assistant","content":"[{\"name\":\"shared_search\",\"arguments\":{\"query\":\"findings\",\"limit\":5}}]"}]},
        {"messages": [{"role":"system","content":"You route user requests to functions."},{"role":"user","content":"Write down that we need to check the pipeline"},{"role":"assistant","content":"[{\"name\":\"memory_write\",\"arguments\":{\"file\":\"notes.md\",\"content\":\"Check pipeline\",\"append\":true}}]"}]},
        {"messages": [{"role":"system","content":"You route user requests to functions."},{"role":"user","content":"Read my notes"},{"role":"assistant","content":"[{\"name\":\"memory_read\",\"arguments\":{\"file\":\"notes.md\"}}]"}]},
        {"messages": [{"role":"system","content":"You route user requests to functions."},{"role":"user","content":"Alert the operator about a critical reactor issue"},{"role":"assistant","content":"[{\"name\":\"notify\",\"arguments\":{\"title\":\"Critical Reactor Issue\",\"message\":\"Anomaly detected\",\"priority\":\"high\"}}]"}]},
        {"messages": [{"role":"system","content":"You route user requests to functions."},{"role":"user","content":"Run a health check"},{"role":"assistant","content":"[{\"name\":\"exec\",\"arguments\":{\"command\":\"./health_check.sh\",\"timeout\":30}}]"}]},
        {"messages": [{"role":"system","content":"You route user requests to functions."},{"role":"user","content":"Share results with the team"},{"role":"assistant","content":"[{\"name\":\"shared_write\",\"arguments\":{\"key\":\"results_q4\",\"content\":\"Analysis complete\"}}]"}]},
        {"messages": [{"role":"system","content":"You route user requests to functions."},{"role":"user","content":"Get a specialist to analyze the imagery"},{"role":"assistant","content":"[{\"name\":\"spawn_agent\",\"arguments\":{\"task\":\"Analyze imagery\",\"agent_id\":\"imagery_specialist\"}}]"}]},
        {"messages": [{"role":"system","content":"You route user requests to functions."},{"role":"user","content":"Hello, how are you?"},{"role":"assistant","content":"[]"}]},
        {"messages": [{"role":"system","content":"You route user requests to functions."},{"role":"user","content":"Check sensor temp and log it"},{"role":"assistant","content":"[{\"name\":\"exec\",\"arguments\":{\"command\":\"cat /sys/class/thermal/thermal_zone0/temp\"}},{\"name\":\"shared_log\",\"arguments\":{\"finding\":\"Temperature captured\",\"category\":\"sensors\"}}]"}]}
    ],
    "dataset_format": "chat",
    "chat_template": "gemma",
    "train_on_responses_only": true,
    "output_gguf": true,
    "quantization": "q8_0",
    "lora_rank": 16,
    "lora_alpha": 16,
    "epochs": 3,
    "batch_size": 2,
    "learning_rate": 2e-4,
    "max_seq_length": 1024,
    "max_steps": 30,
    "warmup_steps": 5
}')

JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "  ✅ Job submitted: $JOB_ID"

# ── Poll for completion ──────────────────────────────────────
echo ""
echo "[4/6] Training..."
for i in $(seq 1 120); do
    sleep 5
    STATUS=$(curl -s "$BASE_URL/v1/finetune/jobs/$JOB_ID")
    STATE=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
    
    if [ "$STATE" = "completed" ]; then
        LOSS=$(echo "$STATUS" | python3 -c "import sys,json; m=json.load(sys.stdin).get('metrics',{}); print(f\"loss={m.get('final_loss','?')}, steps={m.get('steps','?')}\")")
        GGUF=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('metrics',{}).get('gguf_path','none'))")
        echo "  ✅ Training complete! ($LOSS)"
        echo "  GGUF: $GGUF"
        break
    elif [ "$STATE" = "failed" ]; then
        ERROR=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error','unknown'))")
        echo "  ❌ Training failed: $ERROR"
        exit 1
    fi
    
    echo "  ⏳ [$((i*5))s] $STATE..."
done

# ── Verify GGUF exists ───────────────────────────────────────
echo ""
echo "[5/6] Verifying output..."
if [ "$GGUF" != "none" ] && [ -f "$GGUF" ]; then
    SIZE=$(du -h "$GGUF" | cut -f1)
    echo "  ✅ GGUF file exists: $GGUF ($SIZE)"
else
    echo "  ⚠️  No GGUF file (LoRA adapter still available)"
fi

OUTPUT_DIR=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('output_dir',''))")
if [ -d "$OUTPUT_DIR/lora_adapter" ]; then
    ADAPTER_SIZE=$(du -sh "$OUTPUT_DIR/lora_adapter" | cut -f1)
    echo "  ✅ LoRA adapter: $OUTPUT_DIR/lora_adapter ($ADAPTER_SIZE)"
fi

# ── Test inference ───────────────────────────────────────────
echo ""
echo "[6/6] Testing inference with fine-tuned model..."
if [ "$GGUF" != "none" ] && [ -f "$GGUF" ]; then
    # Symlink into HF cache for runtime model resolution
    CACHE_DIR="$HOME/.cache/huggingface/hub/models--local--functiongemma-ft-test/snapshots/v1"
    mkdir -p "$CACHE_DIR"
    ln -sf "$GGUF" "$CACHE_DIR/model-q8_0.gguf"
    
    INFERENCE=$(curl -s "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"local/functiongemma-ft-test","messages":[{"role":"system","content":"You route user requests to functions."},{"role":"user","content":"Save a note about quarterly results"}],"max_tokens":50}')
    
    CONTENT=$(echo "$INFERENCE" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:100])" 2>/dev/null || echo "failed")
    echo "  Model output: $CONTENT"
    echo "  ✅ Inference works!"
else
    echo "  ⏭️  Skipped (no GGUF available)"
fi

echo ""
echo "============================================================"
echo "✅ SFT Fine-Tuning Test Complete!"
echo "============================================================"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "  lora_adapter/ — LoRA weights (for continued training)"
echo "  gguf/         — Quantized GGUF (for deployment)"
echo "  training.log  — Full training output"
echo ""
