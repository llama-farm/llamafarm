# Phase 2: Multi-Instance Lemonade - Analysis & Design

**Status:** NOT STARTED (Phase 1 Complete)
**Date:** 2025-10-02
**Branch:** feat/multi-model

---

## Executive Summary

**Do we need Phase 2?** Probably **NOT** for most use cases.

Phase 1 already supports multiple Lemonade models - they just need different ports. The current implementation works well for:
- Switching between different models on-demand
- Running multiple Ollama models (share same port/backend)
- Running one Lemonade model at a time per port

**Phase 2 would only be needed if:**
- You want multiple Lemonade models running simultaneously on different ports
- You need automatic orchestration to start/stop all Lemonade instances
- You want health checks across all Lemonade instances

---

## What Phase 1 Already Supports

✅ **Multiple models of any provider:**
```yaml
runtime:
  models:
    fast-ollama: {provider: ollama, model: gemma3:1b, ...}
    smart-ollama: {provider: ollama, model: qwen3:8b, ...}
    lemon-small: {provider: lemonade, model: Qwen3-0.6B-GGUF, lemonade: {port: 11534}}
    lemon-large: {provider: lemonade, model: Qwen3-3B-GGUF, lemonade: {port: 11535}}
```

✅ **Model selection via API/CLI:**
- `lf chat --model lemon-large "question"`
- `POST /chat/completions` with `{"model": "lemon-large"}`

✅ **Each model can have its own config:**
- Different ports for different Lemonade instances
- Different backends (llamacpp, transformers, onnx)
- Different context sizes

---

## What Phase 1 DOESN'T Support (Phase 2 Scope)

❌ **Automatic multi-instance orchestration:**
- Currently: `nx start lemonade` starts ONE Lemonade instance (first found in config)
- Phase 2: `nx start lemonade` would start ALL Lemonade instances automatically

❌ **Multi-instance health checks:**
- Currently: Health check assumes one Lemonade instance
- Phase 2: Would check all Lemonade ports and report status for each

❌ **Per-model instance management:**
- Currently: Manual start of each instance on different ports
- Phase 2: Automatic start/stop/restart of each model instance

---

## Phase 2 Design (IF NEEDED)

### Architecture Changes

**1. Multi-Instance Start Script:**
```bash
# runtimes/lemonade/start-multi.sh
# Parse config, find all lemonade models, start each on its own port
# Use process management to run multiple instances in parallel
```

**2. NX Configuration Update:**
```json
// runtimes/lemonade/project.json
{
  "targets": {
    "start": {
      "command": "bash start-multi.sh",  // Changed from start.sh
      "options": {
        "cwd": "runtimes/lemonade"
      }
    }
  }
}
```

**3. Health Service Updates:**
```python
# server/services/health_service.py
# Check all Lemonade ports from config
# Return status for each instance
```

### Implementation Approach

**Step 1: Config Parsing**
```python
# In start-multi.sh (Python helper script)
import yaml

config = yaml.safe_load(open('llamafarm.yaml'))
lemonade_models = [
    (name, model_config)
    for name, model_config in config['runtime']['models'].items()
    if model_config['provider'] == 'lemonade'
]

# Start each in background with its own port
for name, model_config in lemonade_models:
    port = model_config['lemonade']['port']
    model = model_config['model']
    # Start lemonade-server-dev serve --port {port} --model {model} &
```

**Step 2: Process Management**
- Use bash background processes with PID tracking
- Create lock files to prevent duplicate starts
- Handle graceful shutdown on SIGTERM

**Step 3: Health Checks**
```python
# Iterate through all lemonade models in config
# Check HTTP health on each port
# Aggregate status into overall health report
```

---

## Lessons Learned from Phase 1

### 1. **Package/Command Naming is Critical**
- ✅ Correct: `lemonade-sdk` package → `lemonade-server-dev` command
- ❌ Wrong: `lemonade-server-dev` package (doesn't exist)
- ❌ Wrong: `lemonade-server` command (standalone installer only)

**Action for Phase 2:** Document this clearly, add validation

### 2. **Base URL Formats Matter**
- ✅ Ollama: `http://localhost:11434/v1` (needs `/v1` suffix)
- ✅ Lemonade: `http://localhost:11534/api/v1` (auto-constructed in provider)

**Action for Phase 2:** Ensure each instance constructs correct URLs

### 3. **Config Normalization is Essential**
- RAG worker broke because it loaded config directly without normalization
- Fixed by calling `ModelService.normalize_config()` in RAG api.py

**Action for Phase 2:** Any new components must normalize configs on load

### 4. **Multi-Model Config Parsing Works Well**
- Python YAML parsing in bash scripts works reliably
- Fallback from legacy → multi-model format is robust
- Config validation happens after normalization

**Action for Phase 2:** Reuse same parsing patterns for multi-instance

### 5. **Port Management is Already Flexible**
- Each model can specify its own port
- No conflicts between Ollama (11434) and Lemonade (11534+)

**Action for Phase 2:** Just ensure unique ports in validation

---

## Complexity Assessment

### Phase 2 Complexity: **MEDIUM-HIGH**

**Why it's complex:**
1. Process management in bash (multiple background processes)
2. PID tracking and cleanup on shutdown
3. Race conditions between instances starting
4. Health check aggregation logic
5. Error handling when one instance fails
6. Log aggregation from multiple processes

**Estimated effort:** 4-6 hours

**Alternatives to Phase 2:**
1. **Manual multi-instance (current):** Users start each Lemonade model manually
   - Pro: Simple, works today
   - Con: Manual process

2. **Docker Compose orchestration:**
   - Pro: Better process management
   - Con: Requires Docker knowledge, more setup

3. **Systemd services (Linux only):**
   - Pro: Robust process management
   - Con: Platform-specific

---

## Recommendation

### ⚠️ **HOLD on Phase 2 until user need is confirmed**

**Reasons:**
1. Phase 1 supports the core use case (multi-model selection)
2. Most users will use 1-2 models max, not 5+ simultaneously
3. Multiple Ollama models share same backend (no orchestration needed)
4. Lemonade multi-instance is edge case (high memory/GPU usage)
5. Complexity doesn't justify benefit for most users

**Better approach:**
1. Document how to manually run multiple Lemonade instances (if needed)
2. Wait for user feedback/requests for auto-orchestration
3. Consider Docker Compose for multi-instance if demand exists

### If Phase 2 is needed later:

**Prerequisites:**
- [ ] User stories showing need for 3+ simultaneous Lemonade models
- [ ] Decision on process management approach (bash/docker/systemd)
- [ ] Health check UI design for multi-instance status

**Implementation order:**
1. start-multi.sh with Python helper for config parsing
2. Process management with PID files
3. Health check updates
4. Integration tests
5. Documentation

---

## Phase 2 Config Example (Future)

```yaml
runtime:
  default_model: fast

  models:
    # Ollama models (share same backend, no multi-instance needed)
    fast:
      provider: ollama
      model: gemma3:1b
      base_url: http://localhost:11434/v1

    smart:
      provider: ollama
      model: qwen3:8b
      base_url: http://localhost:11434/v1

    # Lemonade models (each needs own instance/port)
    lemon-tiny:
      provider: lemonade
      model: Qwen3-0.6B-GGUF
      lemonade: {backend: llamacpp, port: 11534, context_size: 16384}

    lemon-small:
      provider: lemonade
      model: Qwen3-1.7B-GGUF
      lemonade: {backend: llamacpp, port: 11535, context_size: 32768}

    lemon-medium:
      provider: lemonade
      model: Qwen3-3B-GGUF
      lemonade: {backend: llamacpp, port: 11536, context_size: 65536}
```

**With Phase 2:**
- `nx start lemonade` → Starts 3 Lemonade instances on ports 11534, 11535, 11536
- Health check shows status of all 3 instances
- `lf chat --model lemon-medium` → Routes to port 11536

**Without Phase 2 (current):**
- User manually starts each:
  - `LEMONADE_MODEL=Qwen3-0.6B-GGUF LEMONADE_PORT=11534 nx start lemonade &`
  - `LEMONADE_MODEL=Qwen3-1.7B-GGUF LEMONADE_PORT=11535 nx start lemonade &`
  - `LEMONADE_MODEL=Qwen3-3B-GGUF LEMONADE_PORT=11536 nx start lemonade &`
- Same model selection works via API/CLI

---

## Conclusion

**Phase 1 is COMPLETE and sufficient for most use cases.**

Phase 2 adds convenience (auto-orchestration) but is NOT required for functionality.

Recommend documenting manual multi-instance setup and waiting for user demand before implementing Phase 2.
