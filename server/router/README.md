# Semantic Router - Needle Integration

This module integrates the Needle semantic routing protocol into LlamaFarm, enabling distributed AI workload routing across a mesh of devices.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Semantic Router                        │
├──────────────────────────────────────────────────────────┤
│  embeddings.py  - Generate 768-dim embeddings           │
│  matcher.py     - Match intents to capabilities         │
│  gradient.py    - Gradient routing tables with TTL      │
│  gossip.py      - Capability announcement protocol      │
│  discovery.py   - mDNS/UDP peer discovery               │
│  service.py     - Lifecycle management                  │
│  api.py         - FastAPI endpoints                     │
└──────────────────────────────────────────────────────────┘
```

## Components

### 1. Embeddings (`embeddings.py`)
- **Purpose**: Generate semantic embeddings for capability and intent matching
- **Backend**: Ollama (local) or LlamaFarm API (cloud)
- **Model**: `nomic-embed-text` (768-dimensional vectors)
- **Features**: 
  - Automatic backend selection
  - Embedding cache (LRU, 1000 entries)
  - Batch processing support
  - L2 normalization

### 2. Capability Matcher (`matcher.py`)
- **Purpose**: Match intents to capabilities using cosine similarity
- **Thresholds**:
  - `MATCH_THRESHOLD`: 0.75 (process locally if above)
  - `MIN_ROUTE_THRESHOLD`: 0.50 (drop if below)
- **Features**:
  - Local capability matching
  - Remote routing via gradient table
  - Hop penalty (0.95^hops)
  - Multi-intent decomposition

### 3. Gradient Table (`gradient.py`)
- **Purpose**: Store routing information (capability → next_hop)
- **Implementation**: 
  - Thread-safe with RLock
  - TTL expiration (5 minutes default)
  - Vector index for fast similarity search
  - Eviction policy (confidence * recency)
- **Size**: Up to 1000 entries

### 4. Gossip Protocol (`gossip.py`)
- **Purpose**: Propagate capability announcements through mesh
- **Mechanism**: 
  - Periodic announcements (30s interval)
  - TTL-based flooding (max 10 hops)
  - Nonce-based replay protection
  - Resource information exchange
- **Message Format**: JSON over UDP/WebSocket

### 5. Peer Discovery (`discovery.py`)
- **Purpose**: Discover other LlamaFarm nodes on local network
- **Methods**:
  - UDP broadcast (port 47471)
  - mDNS/Bonjour (optional, via zeroconf)
  - Static peer registry
- **Interval**: 30 seconds

### 6. Router Service (`service.py`)
- **Purpose**: Initialize and manage all router components
- **Lifecycle**:
  - Startup: Initialize embedding engine, gradient table, matcher, gossip
  - Runtime: Periodic maintenance (prune expired entries)
  - Shutdown: Clean shutdown of all components
- **Capabilities**: Register local capabilities with semantic descriptions

### 7. FastAPI Router (`api.py`)
- **Endpoints**:
  - `GET /v1/router/health` - Health check
  - `POST /v1/router/route` - Route an intent
  - `GET /v1/router/capabilities` - List all capabilities
  - `POST /v1/router/capabilities/register` - Register capability
  - `GET /v1/router/gradient` - Inspect gradient table
  - `GET /v1/router/mesh/status` - Mesh network status
  - `POST /v1/router/mesh/announce` - Trigger announcement

## Integration with LlamaFarm

The router is initialized during FastAPI app startup in `api/main.py`:

```python
# In lifespan context manager
router_service = await get_router_service()

# Register default capabilities
await router_service.register_capability(
    label="llm",
    description="Large language model text generation and chat"
)
await router_service.register_capability(
    label="embeddings",  
    description="Generate semantic embeddings for text"
)
```

## Usage Examples

### Route an Intent

```bash
curl -X POST http://localhost:8000/v1/router/route \
  -H "Content-Type: application/json" \
  -d '{
    "text": "analyze this image for objects",
    "min_score": 0.5
  }'
```

Response:
```json
{
  "action": "process_local",
  "capability": {
    "id": "llamafarm-abc123:vision",
    "label": "vision",
    "description": "Analyze images and detect objects",
    "score": 0.87,
    "hops": 0,
    "local": true
  },
  "score": 0.87
}
```

### Register a Capability

```bash
curl -X POST http://localhost:8000/v1/router/capabilities/register \
  -H "Content-Type: application/json" \
  -d '{
    "label": "tts",
    "description": "Convert text to natural speech audio",
    "handler": "tts_handler",
    "models": ["xtts-v2"]
  }'
```

### Check Mesh Status

```bash
curl http://localhost:8000/v1/router/mesh/status
```

Response:
```json
{
  "node_id": "llamafarm-abc123",
  "known_nodes": 3,
  "gradient_table_size": 12,
  "local_capabilities": 2,
  "announcements_sent": 45,
  "announcements_received": 38,
  "avg_hops": 1.5,
  "avg_latency_ms": 15.2
}
```

## Configuration

Environment variables (TODO - add to settings.py):

```bash
# Embedding backend
ROUTER_EMBEDDING_BACKEND=ollama  # or llamafarm
ROUTER_EMBEDDING_MODEL=nomic-embed-text

# Ollama connection
ROUTER_OLLAMA_HOST=localhost
ROUTER_OLLAMA_PORT=11434

# LlamaFarm API (if using remote backend)
LLAMAFARM_API_KEY=your-api-key
LLAMAFARM_URL=https://llamafarm.dev/api

# Discovery
ROUTER_DISCOVERY_ENABLED=true
ROUTER_DISCOVERY_PORT=47471

# Gossip
ROUTER_GOSSIP_ENABLED=true
ROUTER_GOSSIP_INTERVAL=30
```

## Next Steps

### Phase 2: Agent Framework (from INTEGRATION.md)
- [ ] Agent loop with tool calling (`server/agents/loop.py`)
- [ ] Agent memory (short-term + long-term) (`server/agents/memory/`)
- [ ] Session management (`server/agents/sessions.py`)
- [ ] Cron/scheduler (`server/agents/scheduler.py`)
- [ ] Skills system (`server/agents/skills/`)

### Phase 3: Channels Layer
- [ ] Channel abstraction (`server/channels/base.py`)
- [ ] Telegram integration (`server/channels/telegram.py`)
- [ ] Slack integration (`server/channels/slack.py`)
- [ ] Discord integration (`server/channels/discord.py`)
- [ ] WhatsApp integration (`server/channels/whatsapp.py`)

### Phase 4: Nodes Layer
- [ ] Node registry (`server/nodes/registry.py`)
- [ ] Task distribution (`server/nodes/dispatch.py`)
- [ ] Health monitoring (`server/nodes/health.py`)

## Testing

Basic integration test:

```bash
cd ~/clawd/projects/llamafarm-core
uv run pytest server/tests/test_router.py -v
```

(TODO: Create test file)

## Dependencies

Added to `pyproject.toml`:

```toml
[project.dependencies]
numpy = "^1.24.0"
aiohttp = "^3.9.0"
# Optional: zeroconf for mDNS discovery
# zeroconf = "^0.131.0"
```

## Implementation Notes

1. **Embedding Backend Selection**: The system tries Ollama first (local, no API key), then falls back to LlamaFarm API if available.

2. **Thread Safety**: GradientTable uses threading.RLock since it may be accessed from both async event loop and sync contexts.

3. **Gossip Transport**: Currently placeholder - needs actual UDP/WebSocket transport implementation.

4. **Peer Discovery**: UDP broadcast works on local network; mDNS support requires zeroconf package.

5. **Capability Handlers**: Registered capabilities reference handler functions by name - actual handler dispatch not yet implemented.

## Integration Points

The router integrates with LlamaFarm at:

1. **API Layer** (`api/main.py`): Lifecycle management, endpoint registration
2. **Model Service**: Could use LlamaFarm's model management for embeddings
3. **Task Queue**: Future integration for distributed task execution
4. **Storage**: Could persist gradient table to disk for faster startup

## References

- [INTEGRATION.md](../../INTEGRATION.md) - Full integration design doc
- [Needle Router](~/clawd/projects/needle-router/) - Original protocol implementation
- [Needle ARCHITECTURE](~/clawd/projects/needle-router/ARCHITECTURE.md) - Protocol design
- [Needle PROTOCOL](~/clawd/projects/needle-router/PROTOCOL.md) - Wire protocol spec
