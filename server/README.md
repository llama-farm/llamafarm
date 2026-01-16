# LlamaFarm Server

FastAPI application that powers project chat, dataset APIs, health checks, and real-time voice chat. The server provides a REST API and WebSocket endpoints consumed by the `lf` CLI, the Designer web UI, and custom integrations.

## Features

- **Project Management**: Create, configure, and manage LlamaFarm projects
- **Chat Completions**: OpenAI-compatible chat API with RAG integration
- **Dataset Management**: Upload, process, and manage document datasets
- **RAG Operations**: Vector search, document retrieval, and knowledge base queries
- **Vision/OCR**: Document extraction and OCR via Universal Runtime
- **Voice Chat**: Real-time voice assistant via WebSocket (`/v1/{namespace}/{project}/voice/chat`)

## Running Locally

The CLI (`lf start`) will launch the server and RAG worker for you, but you can run it manually while developing inside `server/`.

```bash
uv sync
uv run uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```

To execute Celery ingestion jobs alongside it, start the worker from `rag/` (see that README) or run `lf datasets process …` which will auto-start the worker via Docker.

### API Clients

The server API is consumed by:

- **CLI**: Command-line interface (`lf`) for automation and scripting
- **Designer**: Web-based visual interface at `http://localhost:8000` (see [Designer docs](../docs/website/docs/designer/index.md))
- **Custom integrations**: Any OpenAI-compatible client

Interactive API docs are available at [http://localhost:8000/docs](http://localhost:8000/docs).

### Voice Chat

The server provides a full-duplex WebSocket endpoint for real-time voice conversations:

```javascript
// Connect to voice chat (uses voice config from llamafarm.yaml)
const ws = new WebSocket(
  'ws://localhost:8000/v1/default/my-project/voice/chat'
);

// Or override settings via query params
const ws2 = new WebSocket(
  'ws://localhost:8000/v1/default/my-project/voice/chat?tts_voice=am_adam'
);
```

Features:
- Config-driven: Voice settings in `llamafarm.yaml`
- Speech-to-Text via Universal Runtime (faster-whisper)
- LLM inference with conversation history
- Text-to-Speech via Universal Runtime (Kokoro TTS)
- Barge-in support (interrupt TTS when user speaks)

See the [API docs](../docs/website/docs/api/index.md) for complete WebSocket protocol details.

### Running via Nx from the Repository Root

If you prefer to use the same orchestration as the CLI without Docker auto-start:

```bash
git clone https://github.com/llama-farm/llamafarm.git
cd llamafarm

npm install -g nx
nx init --useDotNxInstallation --interactive=false

# Option A: single command
nx dev

# Option B: separate terminals
nx start rag    # Terminal 1
nx start server # Terminal 2
```

## Tests

```bash
uv run --group test python -m pytest
```

## Configuration

The server reads `llamafarm.yaml` via the config package. Ensure your project config includes:

- `runtime` with provider/model/base_url/api_key as required.
- `rag` strategies/databases that match datasets you plan to ingest.

For a complete schema reference and instructions on extending endpoints, see the main documentation at `docs/website/docs`.
