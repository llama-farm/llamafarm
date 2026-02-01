# LlamaFarm Demo Scripts

## Quick Start

Each demo is standalone and showcases a specific aspect of LlamaFarm's semantic routing and agent capabilities.

### Running Demos

**From the server directory:**
```bash
cd ~/clawd/projects/llamafarm-core/server

# Quick intro demo (no dependencies)
./demos/quick_start.py

# Full semantic routing demo (requires uv)
uv run demos/semantic_routing_demo.py

# Agent basics demo
uv run demos/agent_basics_demo.py

# Session management demo
uv run demos/session_demo.py
```

**All demos work best with:**
- Python 3.12+
- `uv` package manager (for dependency management)
- Run from `server/` directory

---

## Demo Overview

### 1. `quick_start.py` ⚡ (30 seconds)

**Purpose:** Instant visual demo of semantic routing in action.

**What it shows:**
- How user intents match to capabilities
- Confidence scores for routing decisions
- Real-world query examples

**Expected output:**
```
🦙 LlamaFarm Semantic Router - Quick Demo

📦 Available Capabilities
  ✓ vision - Analyze images, detect objects...
  ✓ code_generation - Write, debug, explain code...
  ✓ data_analysis - Analyze datasets, visualizations...
  
🎬 Routing Queries
  💬 "Can you detect faces in this photo?"
  🎯 ━━━▶ vision (94.0% match)
```

**Why it's impressive:**
- Zero config required
- Beautiful terminal output
- Shows ML-powered routing instantly

---

### 2. `semantic_routing_demo.py` 🧠 (2 minutes)

**Purpose:** Deep dive into semantic matching engine.

**What it shows:**
- Embedding generation (768-dim vectors)
- Semantic similarity scoring
- Capability registration and matching
- Route decision logic

**Expected output:**
```
=== Embedding Engine Demo ===
  'What's the weather like today?' → 768-dim vector
  'Tell me about the forecast' → 768-dim vector
  
 Semantic similarity:
  Weather queries: 0.673
  
=== Capability Matching Demo ===
  Query: 'What's the temperature tomorrow?'
  Top matches:
    1. weather (82.9%) → ROUTE TO THIS
    2. search (56.8%)
```

**Why it's impressive:**
- Shows the math behind routing
- Real similarity scores
- Multi-node routing simulation

---

### 3. `agent_basics_demo.py` 🤖 (2 minutes)

**Purpose:** Agent lifecycle and capability discovery.

**What it shows:**
- Agent registration
- Capability announcement
- Network gossip protocol
- Discovery mechanisms

**Run:**
```bash
cd ~/clawd/projects/llamafarm-core/server
uv run demos/agent_basics_demo.py
```

---

### 4. `session_demo.py` 💬 (3 minutes)

**Purpose:** Session management and context handling.

**What it shows:**
- Multi-turn conversations
- Context preservation
- Session routing
- State management

**Run:**
```bash
cd ~/clawd/projects/llamafarm-core/server
uv run demos/session_demo.py
```

---

## Architecture Highlights

**What makes this special:**

1. **Semantic Routing** - Not keyword matching. True semantic understanding via embeddings.
2. **Zero-config Discovery** - Agents announce capabilities automatically via gossip protocol.
3. **Confidence Scoring** - Every routing decision includes confidence metrics.
4. **Multi-node** - Designed for distributed deployment from day one.

**Key tech:**
- SentenceTransformers for embeddings
- Cosine similarity for matching
- Gossip protocol for discovery
- Graph-based routing for complex queries

---

## Troubleshooting

**`ModuleNotFoundError: numpy`**
→ Use `uv run` instead of direct execution:
```bash
uv run demos/semantic_routing_demo.py
```

**`ImportError: DatabaseEmbeddingType`**
→ This is a test suite issue, not demo-related. Demos work independently.

**Permission denied**
→ Make scripts executable:
```bash
chmod +x demos/*.py
```

---

## What to Demo to Engineers

**5-minute walkthrough:**
1. Run `quick_start.py` - show instant routing (30s)
2. Run `semantic_routing_demo.py` - show embedding math (2m)
3. Show `server/router/README.md` - explain architecture (2m)

**15-minute deep dive:**
- Add `agent_basics_demo.py` - discovery protocol
- Add `session_demo.py` - stateful conversations
- Walk through router code architecture

**Impressive talking points:**
- "No keyword lists - pure semantic understanding"
- "Check out these confidence scores - 94% match"
- "Agents discover each other automatically via gossip"
- "This scales to hundreds of nodes with zero central config"

---

## Next Steps

After demos, show:
- `server/router/ARCHITECTURE.md` - Technical deep-dive
- `server/router/README.md` - API and integration guide
- `DEMO_GUIDE.md` (root) - Full product walkthrough

## Testing

To verify all demos work:
```bash
cd ~/clawd/projects/llamafarm-core/server

# Quick check (no dependencies)
./demos/quick_start.py | head -20

# Full suite
uv run demos/semantic_routing_demo.py | grep "ROUTE TO THIS"
uv run demos/agent_basics_demo.py | grep "registered"
```

All demos should complete without errors and show colorful terminal output.
