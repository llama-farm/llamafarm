# Semantic Router - How It Works

> **Intent-based routing using embeddings and semantic similarity**

---

## 🎯 The Problem

Traditional routing uses keywords, regex, or hardcoded rules:

```python
# Brittle and limited
if "weather" in query or "temperature" in query:
    return weather_service
elif "email" in query or "send" in query:
    return email_service
```

**Issues**:
- Misses synonyms ("forecast", "climate")
- Can't handle variations ("what's it like outside?")
- No confidence scoring
- Breaks on typos
- Doesn't scale to complex intents

---

## 💡 The Solution

**Semantic routing**: Understand the *meaning* of queries, not just keywords.

```python
# Semantic understanding
query_vector = embed("what's it like outside?")
weather_vector = embed(weather_capability)

score = cosine_similarity(query_vector, weather_vector)
# score = 0.82 → High confidence, route to weather
```

**Wins**:
- ✅ Understands intent, not just words
- ✅ Handles synonyms naturally
- ✅ Quantified confidence (0-1 scale)
- ✅ Typo-resistant (fuzzy matching)
- ✅ Scales to complex queries

---

## 🏗️ Architecture

### High-Level Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    Semantic Router                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Query                                                    │
│     "What's the weather?"                                    │
│            ↓                                                 │
│  2. Embedding Engine                                         │
│     Ollama (localhost:11434) + nomic-embed-text              │
│     → [0.23, -0.15, 0.42, ..., 0.08]  (768 dimensions)      │
│            ↓                                                 │
│  3. Capability Matcher                                       │
│     Compare query vector to capability vectors               │
│     Cosine similarity: 0.82 (weather), 0.51 (email), ...    │
│            ↓                                                 │
│  4. Route Decision                                           │
│     Best match: weather (82% confidence)                     │
│     Threshold: 50% (configurable)                            │
│     Decision: ROUTE TO WEATHER SERVICE ✅                    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Components

#### **1. Embedding Engine** (`embeddings.py`)
Converts text to 768-dimensional vectors using `nomic-embed-text` model.

```python
from router import EmbeddingEngine

engine = EmbeddingEngine()
await engine.initialize()

# Generate embedding
vec = await engine.embed("What's the weather?")
# → np.ndarray([0.23, -0.15, ..., 0.08], shape=(768,))

# Normalized (L2 norm = 1.0)
assert abs(np.linalg.norm(vec) - 1.0) < 0.01
```

**Features**:
- Async/await for performance
- Automatic backend selection (Ollama → LlamaFarm fallback)
- Caching for repeated queries
- Batch processing support
- Health checks

#### **2. Capability Matcher** (`matcher.py`)
Matches query vectors to capability vectors via cosine similarity.

```python
from router import CapabilityMatcher, Capability

# Define capability with examples
weather = Capability(
    id="weather-001",
    label="weather",
    description="Weather information and forecasts",
    vector=average_embedding(examples),
    handler="weather_handler"
)

# Match query
matcher = CapabilityMatcher()
result = matcher.match_local(query_vec, [weather, calculator, email])

# result.capability = weather
# result.score = 0.82
# result.action = PROCESS_LOCAL
```

**Features**:
- Cosine similarity matching
- Configurable confidence thresholds
- Multi-capability ranking
- Hop-distance penalty (for mesh routing)

#### **3. Route Decision** (`matcher.py`)
Decides whether to process locally, forward, or reject based on scores.

```python
if result.score >= MATCH_THRESHOLD:        # Default: 0.75
    process_locally(result.capability)
elif result.score >= MIN_ROUTE_THRESHOLD:  # Default: 0.50
    forward_to_next_hop(result)
else:
    fallback_handler()
```

---

## 🔬 How Semantic Matching Works

### 1. **Embedding Generation**

Text → Vector transformation using neural networks.

```
Input:  "What's the weather like today?"

Model:  nomic-embed-text (768 dimensions)

Output: [0.23, -0.15, 0.42, 0.08, ..., -0.12]
         └─ 768 floating-point numbers ─┘
```

**Why 768 dimensions?**
- High-dimensional space captures nuanced meaning
- Similar concepts cluster together in this space
- Distance between vectors = semantic similarity

### 2. **Capability Representation**

Each capability is represented by example queries:

```python
weather_examples = [
    "What's the weather?",
    "Will it rain tomorrow?",
    "Temperature forecast",
    "Is it sunny today?"
]

# Embed each example
example_vectors = [embed(ex) for ex in weather_examples]

# Average = capability vector
weather_vector = mean(example_vectors)
```

**Why averaging?**
- Captures the semantic "center" of the capability
- Robust to individual example variations
- New examples easily added

### 3. **Similarity Calculation**

Cosine similarity measures angle between vectors:

```
similarity = (A · B) / (||A|| × ||B||)
           = cos(θ)

Where:
  A · B     = dot product
  ||A||     = magnitude (L2 norm)
  θ         = angle between vectors
  
Result: -1.0 to 1.0
  1.0  = identical direction (same meaning)
  0.0  = orthogonal (unrelated)
 -1.0  = opposite direction (antonyms)
```

**For normalized vectors** (||A|| = ||B|| = 1.0):
```python
similarity = np.dot(query_vec, capability_vec)
```

### 4. **Routing Decision**

```python
scores = {
    "weather": 0.82,    # High confidence ✅
    "email": 0.51,      # Medium
    "calculator": 0.48  # Below threshold ❌
}

# Apply threshold
MATCH_THRESHOLD = 0.75

best = max(scores, key=scores.get)  # "weather"

if scores[best] >= MATCH_THRESHOLD:
    route_to(best)  # Route to weather ✅
else:
    fallback()      # Confidence too low
```

---

## 📊 Real-World Performance

### Typical Similarity Scores

| Query 1 | Query 2 | Similarity | Interpretation |
|---------|---------|------------|----------------|
| "weather today?" | "temperature?" | 0.67-0.83 | ✅ Same intent |
| "weather" | "email" | 0.49-0.51 | ❌ Different |
| "calculate 2+2" | "what is 5*3" | 0.60-0.75 | ✅ Same intent |
| "search python" | "find tutorials" | 0.50-0.70 | ✅ Same intent |

### Confidence by Domain

| Domain | Avg Confidence | Status |
|--------|---------------|--------|
| Weather | 82-84% | ✅ Excellent |
| Calculator | 60-65% | ✅ Good |
| Email | 70-75% | ✅ Good |
| Search | 65-70% | ✅ Good |

**Threshold Configuration**:
- `MATCH_THRESHOLD = 0.75`: Process locally
- `MIN_ROUTE_THRESHOLD = 0.50`: Forward to another node
- Below 0.50: Fallback handler

---

## 🚀 Usage Examples

### Basic Usage

```python
from router import EmbeddingEngine, CapabilityMatcher, Capability
import numpy as np

# 1. Initialize engine
engine = EmbeddingEngine()
await engine.initialize()

# 2. Define capabilities
weather = Capability(
    id="weather-001",
    label="weather",
    description="Weather information",
    vector=await create_capability_vector(engine, [
        "What's the weather?",
        "Will it rain?",
        "Temperature today?"
    ]),
    handler="weather_handler"
)

calculator = Capability(
    id="calc-001",
    label="calculator",
    description="Math calculations",
    vector=await create_capability_vector(engine, [
        "What is 2 + 2?",
        "Calculate 15 * 23",
        "Square root of 144"
    ]),
    handler="calculator_handler"
)

# 3. Route query
query = "What's the temperature going to be tomorrow?"
query_vec = await engine.embed(query)

matcher = CapabilityMatcher()
result = matcher.match_local(query_vec, [weather, calculator])

print(f"Best match: {result.capability.label}")  # "weather"
print(f"Confidence: {result.score:.2%}")         # "82%"

# 4. Route
if result.matched:
    await result.capability.handler(query)
```

### Helper: Create Capability Vector

```python
async def create_capability_vector(
    engine: EmbeddingEngine,
    examples: list[str]
) -> np.ndarray:
    """Average embeddings of example queries."""
    vectors = []
    for example in examples:
        vec = await engine.embed(example)
        vectors.append(vec)
    
    # Average and normalize
    cap_vec = np.mean(vectors, axis=0)
    cap_vec = cap_vec / np.linalg.norm(cap_vec)
    
    return cap_vec
```

---

## 🎛️ Configuration

### Embedding Engine

```python
from router import EmbeddingConfig, EmbeddingBackend

config = EmbeddingConfig(
    backend=EmbeddingBackend.OLLAMA,     # or LLAMAFARM
    model="nomic-embed-text",             # 768 dimensions
    ollama_host="localhost",
    ollama_port=11434,
    timeout=30.0,
    max_retries=3
)

engine = EmbeddingEngine(config=config)
```

### Capability Matcher

```python
matcher = CapabilityMatcher(
    match_threshold=0.75,      # Local processing threshold
    min_route_threshold=0.50,  # Minimum for forwarding
    hop_penalty=0.95           # Per-hop score reduction
)
```

---

## 🧪 Testing

### Run Tests

```bash
# Test embedding engine
uv run pytest tests/test_router_embeddings.py -v

# Test capability matching
uv run pytest tests/test_router_matching.py -v

# Integration tests
uv run pytest router/tests/test_integration.py -v
```

### Test Coverage

```
test_router_embeddings.py:
  ✅ Engine initialization
  ✅ Single text embedding
  ✅ Batch embedding
  ✅ Semantic similarity
  ✅ Batch similarity
  ✅ Caching
  ✅ Normalization
  ✅ Backend detection
  ✅ Special characters
  ✅ Dimension validation

Current: 10/11 passing (91%)
```

---

## 🔧 Troubleshooting

### Ollama Not Available

```python
# Engine will auto-fallback to LlamaFarm if configured
config = EmbeddingConfig(
    llamafarm_url="https://llamafarm.dev/api",
    llamafarm_api_key="your-api-key"
)
```

### Low Similarity Scores

**Problem**: All queries score low (< 0.5)

**Solutions**:
1. Add more diverse examples to capabilities
2. Check examples are relevant to capability
3. Ensure model is loaded: `ollama list | grep nomic`

### Slow Performance

**Solutions**:
1. Enable caching (on by default)
2. Use batch embedding for multiple queries
3. Pre-compute capability vectors at startup

---

## 🌐 Mesh Routing (Advanced)

### Multi-Node Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Node A    │────▶│   Node B    │────▶│   Node C    │
│ weather     │     │ calculator  │     │ email       │
│ (local)     │     │ (1 hop)     │     │ (2 hops)    │
└─────────────┘     └─────────────┘     └─────────────┘
       │
       ├─ Local capabilities: Process immediately
       ├─ 1-hop capabilities: Forward with 95% confidence
       └─ 2-hop capabilities: Forward with 90% confidence
```

### Gradient Table

Stores learned routes to remote capabilities:

```python
gradient_entry = (
    "calculator",              # Capability label
    calculator_vector,         # 768-dim vector
    1,                         # Hops away
    "node-b",                  # Next hop
    "node-c"                   # Origin node
)

# Adjusted score with hop penalty
adjusted_score = raw_score * (0.95 ** hops)
```

### Gossip Protocol

Nodes advertise capabilities to neighbors:

```python
announcement = Announcement(
    node_id="node-a",
    capabilities=[
        CapabilityInfo(
            label="weather",
            vector=weather_vector,
            hops=0
        )
    ],
    timestamp=time.time()
)

# Broadcast to neighbors
gossip.broadcast(announcement)
```

---

## 📚 Files Reference

### Core Files

| File | Purpose |
|------|---------|
| `embeddings.py` | Embedding engine (Ollama integration) |
| `matcher.py` | Capability matching and routing |
| `gradient.py` | Gradient table for mesh routing |
| `gossip.py` | Capability discovery protocol |
| `learning.py` | Route quality learning |
| `service.py` | FastAPI service integration |

### Demos

| File | Purpose |
|------|---------|
| `demos/simple_routing_demo.py` | Keyword-based baseline |
| `demos/semantic_routing_demo.py` | Semantic routing showcase |
| `demos/session_demo.py` | Multi-turn sessions |

### Tests

| File | Purpose |
|------|---------|
| `tests/test_router_embeddings.py` | Embedding engine tests |
| `tests/test_router_matching.py` | Matcher tests |
| `router/tests/test_integration.py` | Integration tests |

---

## 🎓 Learn More

### Concepts

- **Embeddings**: Vector representations of text
- **Cosine Similarity**: Angle-based similarity metric
- **Semantic Space**: High-dimensional space where similar concepts cluster
- **Gradient Descent**: Learning optimal routes over time

### Models

- **nomic-embed-text**: 768-dim multilingual embedding model
- **Alternative**: `all-MiniLM-L6-v2` (384-dim, faster)
- **Comparison**: Higher dimensions = more nuance, slower

### Papers

- "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
- "Semantic Textual Similarity: A Survey"
- "Dense Passage Retrieval for Open-Domain Question Answering"

---

## 🚀 Quick Start

```bash
# 1. Ensure Ollama is running
ollama serve

# 2. Pull nomic-embed-text model
ollama pull nomic-embed-text

# 3. Run semantic routing demo
cd ~/clawd/projects/llamafarm-core/server
uv run python demos/semantic_routing_demo.py

# 4. Run tests
uv run pytest tests/test_router_embeddings.py -v
```

---

## 💬 Questions?

- Architecture: See `ARCHITECTURE.md`
- Full demo guide: See `../../DEMO_GUIDE.md`
- Technical report: See `demos/FINAL_REPORT.md`

**Built with ❤️ for LlamaFarm**
