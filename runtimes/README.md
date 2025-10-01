# LlamaFarm Runtimes

This directory contains runtime providers for LlamaFarm. Each runtime is an optional, isolated service that can be started independently.

## Available Runtimes

### Lemonade
**Location**: `runtimes/lemonade/`
**Command**: `nx start lemonade`
**Port**: 11534 (default)

Lemonade is an SDK for running local LLMs with optimized performance across different hardware configurations. It provides:

- Multiple inference backends (ONNX, llama.cpp, Transformers)
- Hardware-aware optimization (NPUs, GPUs, CPUs)
- OpenAI-compatible API
- Support for GGUF and ONNX models

See `runtimes/lemonade/README.md` for detailed documentation.

## Runtime Architecture

### Design Principles

1. **Isolated Services**: Each runtime runs as a separate nx service
2. **Optional**: Runtimes do NOT start with `nx dev` - they must be started explicitly
3. **Health Checked**: Main server monitors runtime availability if configured
4. **OpenAI-Compatible**: All runtimes expose standard OpenAI API endpoints
5. **Configurable**: Environment variables + schema configuration support
6. **Config-Driven**: Runtime settings can be defined in `llamafarm.yaml`

### Port Allocation

- **8000**: LlamaFarm main server
- **11434**: Ollama (default)
- **11534**: Lemonade
- **Future runtimes**: Will use similar high-numbered ports

### Adding New Runtimes

To add a new runtime (e.g., vLLM, TGI):

1. **Create directory**: `runtimes/{runtime-name}/`
2. **Add NX config**: Create `project.json` with start target
3. **Write startup script**: Create executable script (e.g., `start.sh`)
4. **Update schema**: Add provider enum in `config/schema.yaml`
5. **Add health check**: Update `server/services/health_service.py`
6. **Update client factory**: Add provider case in `server/agents/project_chat_orchestrator.py` (_get_client)
7. **Document**: Create README.md with setup and usage instructions

### Configuration Flow

```yaml
# llamafarm.yaml
runtime:
  provider: lemonade  # or ollama, openai, etc.
  model: "model-name"

  # Provider-specific config
  lemonade:
    backend: onnx
    port: 11534
```

The startup script reads from the project config if available, allowing centralized configuration.

### Health Checks

The LlamaFarm server performs health checks on configured runtimes:

- **Server availability**: Checks if runtime is reachable
- **Model availability**: Verifies model is loaded (provider-specific)
- **Latency tracking**: Monitors response times

Health check results are available via `/health` endpoint.

## Usage

### Starting a Runtime

```bash
# Start Lemonade
nx start lemonade

# Or with custom config
LEMONADE_PORT=11535 nx start lemonade
```

### Using in Project Config

```yaml
version: v1
name: my-project
namespace: default

runtime:
  provider: lemonade
  model: "Phi-3-mini-4k-instruct-onnx"

  lemonade:
    backend: onnx
    port: 11534
```

Then use the CLI:

```bash
lf chat "Hello, how are you?"
```

The chat service will automatically connect to the configured runtime.

## Future Runtimes

Planned additions:
- **vLLM**: High-throughput serving for production
- **TGI** (Text Generation Inference): HuggingFace's serving solution
- **LocalAI**: Multi-model serving platform
- **Triton**: NVIDIA's inference server

Each will follow the same architectural pattern established by Lemonade.
