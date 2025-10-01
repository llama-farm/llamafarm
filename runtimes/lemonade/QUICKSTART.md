# Lemonade Runtime - Quick Start

Get up and running with Lemonade in under 5 minutes.

## Prerequisites

1. **Install Lemonade SDK**:
   ```bash
   pip install lemonade-sdk
   ```

## Step 1: Start Lemonade Server

```bash
nx start lemonade
```

This starts Lemonade on port 11534 with ONNX backend (works on all systems).

## Step 2: Configure Your Project

Create or update `llamafarm.yaml`:

```yaml
version: v1
name: my-lemonade-project
namespace: default

runtime:
  provider: lemonade
  model: "Phi-3-mini-4k-instruct-onnx"

prompts:
  - role: system
    content: "You are a helpful assistant."
```

## Step 3: Chat!

```bash
lf chat "What is the capital of France?"
```

## Custom Backends

### Using llama.cpp (for GGUF models)

```bash
LEMONADE_BACKEND=llamacpp nx start lemonade
```

Or in `llamafarm.yaml`:

```yaml
runtime:
  provider: lemonade
  model: "llama-3.1-8b.gguf"

  lemonade:
    backend: llamacpp
```

### Using Transformers (PyTorch)

```bash
LEMONADE_BACKEND=transformers nx start lemonade
```

## Troubleshooting

### Port Already in Use
```bash
LEMONADE_PORT=11535 nx start lemonade
```

### Lemonade Not Installed
```bash
pip install lemonade-sdk
# or
uv pip install lemonade-sdk
```

### Check Health
Visit: http://localhost:8000/health

Look for the "lemonade" component status.

## Next Steps

- Read the full [README.md](./README.md) for advanced configuration
- Check the [example config](./example.llamafarm.yaml)
- See available models at https://lemonade-server.ai/models/
