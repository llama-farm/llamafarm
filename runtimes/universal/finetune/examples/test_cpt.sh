#!/usr/bin/env bash
# ============================================================
# CPT (Continued Pre-Training) End-to-End Test
# ============================================================
# Trains FunctionGemma 270M on domain-specific raw text to
# absorb new terminology and knowledge, then exports to GGUF.
#
# CPT differs from SFT:
#   - Uses raw text (not instruction pairs)
#   - Includes embed_tokens + lm_head in LoRA targets
#   - Lower learning rate for embedding layers
#   - Goal: domain knowledge absorption
#
# Usage:
#   ./finetune/examples/test_cpt.sh
#   ./finetune/examples/test_cpt.sh http://localhost:11540
# ============================================================

set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:11540}"
MODEL="unsloth/functiongemma-270m-it"

echo "============================================================"
echo "CPT (Continued Pre-Training) End-to-End Test"
echo "============================================================"
echo "Runtime: $BASE_URL"
echo "Model:   $MODEL"
echo ""

# ── Check runtime ────────────────────────────────────────────
echo "[1/5] Checking runtime..."
if ! curl -sf "$BASE_URL/v1/finetune/templates" > /dev/null 2>&1; then
    echo "  ❌ Runtime not available at $BASE_URL"
    echo "  Start with: cd runtimes/universal && uv run python server.py"
    exit 1
fi
echo "  ✅ Runtime ready"

# ── Submit CPT job ───────────────────────────────────────────
echo ""
echo "[2/5] Submitting CPT training job..."
RESPONSE=$(curl -s "$BASE_URL/v1/finetune/cpt" \
    -H "Content-Type: application/json" \
    -d '{
    "model": "'"$MODEL"'",
    "dataset": [
        {"text": "LlamaFarm is a local AI infrastructure platform that enables sovereign compute. It provides a Universal Runtime for serving GGUF models, a vision pipeline with YOLO and CLIP, and a RAG service for document processing."},
        {"text": "The Universal Runtime supports Llama, Qwen, Gemma, Phi, and Mistral architectures. Models are served via an OpenAI-compatible API on port 11540. The main server on port 14345 provides project-level routing."},
        {"text": "Vision capabilities include object detection with YOLOv11, classification with CLIP, one-shot training, ONNX export for mobile, and a replay buffer for continuous improvement from human corrections."},
        {"text": "The KV cache provides server-side caching with segment-based validation. Multi-turn conversations achieve 15-27x TTFT speedup. Pre-warming system prompts eliminates cold start latency entirely."},
        {"text": "ML addons include time series forecasting with Chronos, anomaly detection with ADTK, drift detection with alibi-detect, and gradient boosting with CatBoost. Each loads conditionally based on installed packages."},
        {"text": "Fine-tuning supports SFT with LoRA adapters and CPT with embedding layer training. Cross-platform via unsloth on CUDA and unsloth-mlx on Apple Silicon. Output is GGUF format for immediate deployment."},
        {"text": "Model evaluation uses ultralytics val for mAP50, mAP50-95, precision, recall, and F1. Eval runs in subprocess for OOM isolation. Auto-eval triggers after training to promote models exceeding score thresholds."},
        {"text": "Object tracking assigns persistent IDs across video frames using ByteTrack, BoT-SORT, or OC-SORT. The tracking API supports session-based state with automatic TTL cleanup for idle sessions."},
        {"text": "Atmosphere mesh provides multi-transport connectivity with BLE, LAN, and relay failover. Semantic routing enables zero-trust communication. LlamaFarm integrates with Atmosphere for federated model serving."},
        {"text": "Rownd.AI delivers enterprise AI for regulated industries. The platform combines local inference with compliance-ready deployment for data sovereignty in finance, healthcare, and government."}
    ],
    "output_gguf": true,
    "quantization": "q8_0",
    "lora_rank": 16,
    "lora_alpha": 16,
    "epochs": 2,
    "batch_size": 2,
    "learning_rate": 5e-5,
    "embedding_learning_rate": 5e-6,
    "max_seq_length": 512,
    "max_steps": 30
}')

JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
JOB_TYPE=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['type'])")
echo "  ✅ Job submitted: $JOB_ID (type: $JOB_TYPE)"

# ── Poll for completion ──────────────────────────────────────
echo ""
echo "[3/5] Training (CPT includes embed_tokens + lm_head)..."
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

# ── Verify output ────────────────────────────────────────────
echo ""
echo "[4/5] Verifying output..."
OUTPUT_DIR=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('output_dir',''))")

if [ "$GGUF" != "none" ] && [ -f "$GGUF" ]; then
    SIZE=$(du -h "$GGUF" | cut -f1)
    echo "  ✅ GGUF: $GGUF ($SIZE)"
else
    echo "  ⚠️  No GGUF (LoRA adapter still available)"
fi

if [ -d "$OUTPUT_DIR/lora_adapter" ]; then
    ADAPTER_SIZE=$(du -sh "$OUTPUT_DIR/lora_adapter" | cut -f1)
    echo "  ✅ LoRA adapter: $ADAPTER_SIZE"
fi

# Verify CPT-specific: embed_tokens + lm_head in targets
if [ -f "$OUTPUT_DIR/training.log" ]; then
    if grep -q "embed_tokens" "$OUTPUT_DIR/training.log" && grep -q "lm_head" "$OUTPUT_DIR/training.log"; then
        echo "  ✅ CPT targets confirmed: embed_tokens + lm_head included"
    else
        echo "  ⚠️  CPT targets not found in training log"
    fi
fi

# ── Summary ──────────────────────────────────────────────────
echo ""
echo "[5/5] Summary"
echo "============================================================"
echo "✅ CPT Fine-Tuning Test Complete!"
echo "============================================================"
echo ""
echo "Output: $OUTPUT_DIR"
echo "  lora_adapter/ — LoRA weights (includes embedding layers)"
echo "  gguf/         — Quantized GGUF with domain knowledge"
echo "  training.log  — Full training output"
echo ""
echo "CPT vs SFT:"
echo "  • CPT trains embed_tokens + lm_head (new vocabulary)"
echo "  • CPT uses lower LR for embedding layers (5e-6 vs 2e-4)"
echo "  • CPT input is raw text, not instruction pairs"
echo "  • Use CPT first to inject domain knowledge, then SFT to"
echo "    teach specific behaviors on the domain-adapted model."
echo ""
