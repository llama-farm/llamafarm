# Semantic Router Architecture

## Overview

The semantic router enables capability-based routing across distributed LlamaFarm nodes. Instead of routing by address, we route by **intent** — finding the best node to handle a request based on semantic matching.

```
┌─────────────────────────────────────────────────────────────┐
│                        REQUEST FLOW                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Intent                                                    │
│     │                                                       │
│     ▼                                                       │
│   ┌─────────────────┐                                       │
│   │ Embedding Engine │  ←─ nomic-embed-text / ollama       │
│   └────────┬────────┘                                       │
│            │ 768-dim vector                                 │
│            ▼                                                │
│   ┌─────────────────┐                                       │
│   │ Capability      │  ←─ Local capabilities + gradient    │
│   │ Matcher         │     table entries                     │
│   └────────┬────────┘                                       │
│            │                                                │
│            ▼                                                │
│   ┌─────────────────┐                                       │
│   │ Route Decision  │  ←─ PROCESS_LOCAL / ROUTE_FORWARD   │
│   └────────┬────────┘                                       │
│            │                                                │
│     ┌──────┴──────┐                                        │
│     ▼             ▼                                         │
│   Local        Forward to                                   │
│   Handler      Next Hop                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Embedding Engine (`embeddings.py`)

Generates 768-dimensional semantic embeddings for text.

**Backends:**
- Ollama (default): Uses `nomic-embed-text` model
- OpenAI: Uses `text-embedding-3-small`
- Local: HuggingFace transformers

**Usage:**
```python
engine = EmbeddingEngine(EmbeddingConfig())
await engine.initialize()
vector = await engine.embed("Generate a summary of this document")
```

### 2. Capability Matcher (`matcher.py`)

Matches intent vectors against known capabilities.

**Algorithm:**
1. Compute cosine similarity between intent and all capabilities
2. Apply hop penalty for remote capabilities (0.95^hops)
3. Select highest-scoring match above threshold

**Route Decisions:**
- `PROCESS_LOCAL`: Handle locally (local capability matches)
- `ROUTE_FORWARD`: Forward to specific peer
- `ROUTE_BROADCAST`: Broadcast to mesh (no good match)
- `ROUTE_REJECT`: Reject request (security/policy)

### 3. Gradient Table (`gradient.py`)

Stores routing information to remote capabilities.

**Entry Structure:**
```python
GradientEntry(
    capability_id="peer-1:vision:abc123",
    capability_label="vision",
    capability_vector=np.array([...]),  # 768-dim
    hops=2,
    next_hop="peer-1",       # Immediate neighbor
    via_node="peer-2",       # Final destination
    estimated_latency_ms=30,
    confidence=0.90          # 0.95^hops
)
```

**Key Operations:**
- `update()`: Add or update route (only if better)
- `find_best_route()`: Find best route for intent
- `prune_expired()`: Remove stale entries
- `invalidate_node()`: Remove routes through disconnected peer

### 4. Gossip Protocol (`gossip.py`)

Propagates capability information across the mesh.

**Announcement Structure:**
```json
{
  "from_node": "llamafarm-abc123",
  "timestamp": 1706745600,
  "capabilities": [
    {
      "id": "llamafarm-abc123:llm:xyz",
      "label": "llm",
      "description": "Large language model",
      "vector": [0.1, -0.2, ...],
      "hops": 0
    }
  ]
}
```

**Protocol:**
1. Nodes announce capabilities every 30 seconds
2. Receivers update gradient table
3. Receivers re-announce with hops+1 (up to max_hops)
4. Convergence in O(log N) rounds

### 5. Peer Discovery (`discovery.py`)

Discovers other nodes on the local network.

**Method:** UDP broadcast on port 47471

**Discovery Packet:**
```json
{
  "type": "discovery",
  "node_id": "llamafarm-abc123",
  "capabilities": ["llm", "vision"],
  "port": 8000
}
```

### 6. Route Learner (`learning.py`)

Learns from routing feedback to improve decisions.

**Metrics Tracked:**
- Success rate per route
- Average latency
- P95 latency
- Recency of failures

**Quality Score:**
```
quality = 0.6 * success_rate 
        + 0.2 * (1 - latency/1000)
        + 0.2 * recency_factor
```

**API:**
```python
learner.record_success("route-1", latency_ms=50)
learner.record_failure("route-1", error_type="timeout")
adjusted_score = learner.adjust_score("route-1", base_score=0.9)
```

## API Endpoints

### Routing

```
POST /v1/router/route
{
  "text": "Generate a summary of this document",
  "min_score": 0.5
}

Response:
{
  "action": "process_local",
  "capability": {
    "id": "llamafarm-abc:llm:xyz",
    "label": "llm",
    "score": 0.87
  }
}
```

### Capabilities

```
GET /v1/router/capabilities

Response:
[
  {"id": "...", "label": "llm", "local": true, "hops": 0},
  {"id": "...", "label": "vision", "local": false, "hops": 1}
]
```

### Feedback

```
POST /v1/router/feedback
{
  "route_id": "llamafarm-abc:llm:xyz",
  "success": true,
  "latency_ms": 150
}
```

### Quality

```
GET /v1/router/quality/llamafarm-abc:llm:xyz

Response:
{
  "route_id": "llamafarm-abc:llm:xyz",
  "success_rate": 0.95,
  "avg_latency_ms": 120,
  "quality_score": 0.89
}
```

## Configuration

```yaml
router:
  node_id: "auto"  # Auto-generate or specify
  enable_discovery: true
  enable_gossip: true
  
  embedding:
    backend: "ollama"
    model: "nomic-embed-text"
    
  matcher:
    match_threshold: 0.75
    min_route_threshold: 0.50
    hop_penalty: 0.95
    
  gradient:
    max_size: 1000
    expire_sec: 300
    
  gossip:
    announce_interval: 30
    max_hops: 5
```

## Multi-Node Deployment

### Scenario: 3-Node Mesh

```
   Node A (llm, embeddings)
        │
        │ gossip
        ▼
   Node B (vision, rag)
        │
        │ gossip
        ▼
   Node C (code, tool-calling)
```

**Routing Example:**

1. Request arrives at Node A: "Analyze this image"
2. Node A computes intent embedding
3. Matcher checks local capabilities (llm, embeddings) - no match
4. Matcher checks gradient table - finds "vision" at Node B (1 hop)
5. Request forwarded to Node B
6. Node B processes locally
7. Response returned through Node A

### Startup Sequence

1. Each node starts RouterService
2. PeerDiscovery broadcasts presence
3. GossipProtocol exchanges capabilities
4. Gradient tables converge (~10-30 seconds)
5. Routing operational

## Integration with LlamaFarm

The router integrates with:

- **Chat API**: Route chat requests to appropriate models
- **Vision API**: Route image analysis to vision-capable nodes
- **RAG API**: Route document queries to RAG-enabled nodes
- **Agents**: Agents use router for task delegation

## Performance

| Metric | Target | Actual |
|--------|--------|--------|
| Embedding latency | <50ms | ~30ms (Ollama) |
| Route decision | <5ms | ~2ms |
| Gossip convergence | <30s | ~15s (3 nodes) |
| Gradient table lookup | O(1) | ~0.1ms |

## Future Work

1. **WebSocket transport**: Real-time gossip over WebSocket
2. **TLS/mTLS**: Secure node-to-node communication
3. **Capability versioning**: Handle capability upgrades
4. **Load balancing**: Route based on node capacity
5. **Geographic routing**: Consider network topology
