# LlamaFarm Router API Reference

## Base URL
```
http://localhost:8000/v1/router
```

## Endpoints

### GET /health
Check router subsystem health.

**Response:**
```json
{
  "status": "healthy",
  "embedding_backend": "ollama",
  "gradient_table_size": 0,
  "local_capabilities": 2
}
```

---

### GET /capabilities
List all known capabilities (local + discovered).

**Response:**
```json
[
  {
    "id": "llamafarm-071cc04f:llm:a61a000e",
    "label": "llm",
    "description": "Large language model text generation and chat",
    "score": 1.0,
    "hops": 0,
    "local": true
  },
  {
    "id": "llamafarm-071cc04f:embeddings:1e317ec7",
    "label": "embeddings",
    "description": "Generate semantic embeddings for text",
    "score": 1.0,
    "hops": 0,
    "local": true
  }
]
```

---

### POST /route
Route an intent to the best matching capability.

**Request:**
```json
{
  "text": "Generate embeddings for this document",
  "min_score": 0.5
}
```

**Response (match found):**
```json
{
  "action": "process_local",
  "capability": {
    "id": "llamafarm-071cc04f:embeddings:1e317ec7",
    "label": "embeddings",
    "description": "Generate semantic embeddings for text",
    "score": 0.8778419494628906,
    "hops": 0,
    "local": true
  },
  "score": 0.8778419494628906
}
```

**Response (no match):**
```json
{
  "action": "no_match",
  "score": 0.4760782718658447,
  "reason": "No capability match above threshold"
}
```

**Response (forward to peer):**
```json
{
  "action": "forward",
  "capability": {
    "id": "remote-node:vision:abc123",
    "label": "vision",
    "description": "Computer vision and object detection",
    "score": 0.92,
    "hops": 1,
    "via_node": "remote-node",
    "local": false
  },
  "score": 0.92,
  "next_hop": "192.168.1.100:47471"
}
```

---

## Request Models

### IntentRequest
```typescript
{
  text: string;        // Intent text to route
  min_score?: number;  // Minimum match score (0.0-1.0, default: 0.5)
}
```

## Response Models

### CapabilityResponse
```typescript
{
  id: string;          // Unique capability ID
  label: string;       // Capability name
  description: string; // What this capability does
  score: number;       // Match score (0.0-1.0)
  hops: number;        // Distance from this node
  via_node?: string;   // Next-hop node ID (if remote)
  local: boolean;      // True if on this node
}
```

### RouteResponse
```typescript
{
  action: "process_local" | "forward" | "no_match";
  capability?: CapabilityResponse;  // Best match (if found)
  score: number;                    // Match score
  next_hop?: string;                // IP:port for forwarding
  reason?: string;                  // Explanation (especially for no_match)
}
```

---

## Examples

### Example 1: Route an embedding request
```bash
curl -X POST http://localhost:8000/v1/router/route \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Generate embeddings for this text",
    "min_score": 0.5
  }'
```

### Example 2: Check router health
```bash
curl http://localhost:8000/v1/router/health
```

### Example 3: List all capabilities
```bash
curl http://localhost:8000/v1/router/capabilities
```

### Example 4: Route with custom threshold
```bash
curl -X POST http://localhost:8000/v1/router/route \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Summarize this document",
    "min_score": 0.7
  }'
```

---

## Match Actions

| Action | Meaning | Next Step |
|--------|---------|-----------|
| `process_local` | Capability found on this node | Execute locally |
| `forward` | Capability found on peer node | Forward to `next_hop` |
| `no_match` | No capability above threshold | Reject or fallback |

---

## Capability Registration

Capabilities are registered at server startup in `server/api/main.py`:

```python
await router_service.register_capability(
    label="llm",
    description="Large language model text generation and chat"
)

await router_service.register_capability(
    label="embeddings",
    description="Generate semantic embeddings for text"
)
```

To add a new capability:
1. Call `router_service.register_capability()` during startup
2. Provide semantic description (used for matching)
3. Capability ID auto-generated as `{node_id}:{label}:{hash}`

---

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ POST /route {"text": "..."}
       ↓
┌─────────────────────────┐
│  Embedding Engine       │ ← Ollama (nomic-embed-text)
│  intent → vector        │
└──────────┬──────────────┘
           │
           ↓
┌─────────────────────────┐
│  Capability Matcher     │
│  Local capabilities     │
└──────────┬──────────────┘
           │
           ↓
┌─────────────────────────┐
│  Gradient Table         │ ← Learned from gossip
│  Remote capabilities    │
└──────────┬──────────────┘
           │
           ↓
┌─────────────────────────┐
│  Routing Decision       │
│  action + next_hop      │
└─────────────────────────┘
```

---

## Gossip Protocol

Router nodes automatically discover each other via UDP gossip on port **47471**.

**Announcement Format:**
```json
{
  "node_id": "llamafarm-071cc04f",
  "capabilities": [
    {
      "id": "llamafarm-071cc04f:llm:a61a000e",
      "label": "llm",
      "vector": [0.1, 0.2, ...],  // 768-dim embedding
      "metadata": {}
    }
  ],
  "timestamp": 1769917825
}
```

**Discovery:**
- Periodic broadcasts every 10s
- Peer table updated on receipt
- Gradient table rebuilt from peer announcements

---

## Performance

| Metric | Value |
|--------|-------|
| Embedding latency | ~65ms |
| Match latency | ~50ms |
| Total /route latency | ~100-150ms |
| Capability limit | ~1000 per node |
| Vector dimension | 768 |

---

## Future Enhancements

1. **Multi-hop Routing**
   - Route through intermediate nodes
   - Gradient descent optimization

2. **Load Balancing**
   - Track node capacity
   - Round-robin across peers

3. **Capability Learning**
   - Update gradient weights from usage
   - Prefer low-latency paths

4. **Security**
   - Signed capability announcements
   - Peer authentication

---

**Last Updated:** January 31, 2026  
**Version:** 1.0.0  
**Status:** Production Ready
