# LlamaFarm Demos - Semantic Routing & Agent Framework

> **Quick demos** showing semantic routing with real embeddings and autonomous agents

---

## 🚀 Quick Start

```bash
cd ~/clawd/projects/llamafarm-core/server

# Ensure Ollama is running
curl http://localhost:11434/api/tags

# Run any demo
uv run python demos/<demo_name>.py
```

**Recommended order**: simple → semantic → session

---

## 📂 Available Demos

### ⭐ 1. Semantic Routing Demo (MAIN SHOWCASE)
**File:** `semantic_routing_demo.py`

**What it shows:**
Full semantic routing with real Ollama embeddings, confidence scoring, and multi-capability matching.

**Run:**
```bash
uv run python demos/semantic_routing_demo.py
```

**Duration:** ~30 seconds

**Expected Output:**
```
=== Embedding Engine Demo ===
Embedding texts...
  'What's the weather like today?' → 768-dim vector
  ...

Semantic similarity:
  Weather queries: 0.673    ← Similar queries
  Weather vs Email: 0.497   ← Different topics

=== Capability Matching Demo ===
Query: 'What's the temperature going to be tomorrow?'
Top matches:
  1. weather (82.9%) ← ROUTE TO THIS ✅
  2. search (56.8%)
  3. email (55.3%)
```

**Why it's cool:**
- Real embeddings from Ollama (nomic-embed-text)
- 768-dimensional semantic vectors
- 60-85% confidence scores on real queries
- Multi-capability routing for complex intents
- Threshold-based routing decisions

**Use this for demos!** 🎯

---

### 2. Simple Routing Demo (BASELINE)
**File:** `simple_routing_demo.py`

**What it shows:**
Keyword-based routing to demonstrate the basic concept before showing semantic version.

**Run:**
```bash
uv run python demos/simple_routing_demo.py
```

**Duration:** ~5 seconds

**Expected Output:**
```
Query: 'What's the weather forecast for tomorrow?'
Matches:
  • weather (score: 2) → weather-service-001 ✓ ROUTE TO THIS
```

**Why start here:**
- Simple to understand (keywords, not embeddings)
- Shows core routing concept
- Good baseline for comparison
- Highlights limitations (misses synonyms, typos)

---

### 3. Session Demo (MULTI-TURN CONVERSATIONS)
**File:** `session_demo.py`

**What it shows:**
Multi-turn conversation management, context retention, and session lifecycle.

**Run:**
```bash
uv run python demos/session_demo.py
```

**Duration:** ~10 seconds

**Expected Output:**
```
=== Simple Conversation Demo ===
Created session: simple-chat-001

Conversation:
  user      : Hi! My name is Alice.
  assistant : Hello Alice! Nice to meet you.
  user      : What's the weather like in New York?
  assistant : The current weather in New York is 68°F...

Total messages in session: 8

=== Context Retention Demo ===
Notice how later messages reference:
  • The destination (Paris)
  • The duration (a week)
  • The time frame (March)
```

**Why it matters:**
- Shows session management
- Context retention across turns
- Metadata usage (user preferences, channel info)
- Concurrent session handling
- Session lifecycle (active → paused → completed)

---

### 4. Agent Basics Demo
**File:** `agent_basics_demo.py`

**What it shows:**
Autonomous agent framework with memory, tasks, and sessions.

**Run:**
```bash
uv run python demos/agent_basics_demo.py
```

**Expected Output:**
- Agent memory creation and persistence
- Session management
- Task hierarchies with priorities
- Agent initialization and binding

**Key Concepts:**
- Agents maintain memory (short-term + long-term)
- Sessions track conversations
- Tasks can be delegated and prioritized

---

## 🎬 Demo Script for Engineers (5 Minutes)

**1. Start with Simple** (1 min)
```bash
uv run python demos/simple_routing_demo.py
```
"Here's keyword-based routing. It works but misses synonyms and typos."

**2. Show Semantic** (3 min) ⭐
```bash
uv run python demos/semantic_routing_demo.py
```
"Now with embeddings. Watch the confidence scores: 82% for weather queries, 60-70% for others."

**Key points:**
- Similarity scores: 0.67 for related queries, 0.49 for unrelated
- No keywords needed - pure semantic understanding
- Threshold prevents bad routing

**3. Sessions (1 min)**
```bash
uv run python demos/session_demo.py
```
"Multi-turn conversations with context retention."

---

## 📊 Expected Performance

| Demo | Runtime | Key Metric |
|------|---------|-----------|
| Simple | ~5 sec | 100% keyword match |
| Semantic | ~30 sec | 82% weather confidence |
| Session | ~10 sec | 8 messages with context |
| Agent | ~5 sec | Memory + tasks demo |

---

## 🔧 Customization Tips

### Adding Custom Capabilities

```python
from dataclasses import dataclass

@dataclass
class DemoCapability:
    name: str
    description: str
    examples: list[str]
    node_id: str

# Define your capability
my_capability = DemoCapability(
    name="custom_task",
    description="What your capability does",
    examples=[
        "Example query 1",
        "Example query 2",
        "Example query 3"
    ],
    node_id="your-service-id"
)

# Add to capabilities dict
capabilities = {
    "custom_task": my_capability,
    # ... other capabilities
}
```

### Configuring Session Metadata

```python
from agents import SessionManager

manager = SessionManager()

session = await manager.create_session(
    metadata={
        "user_id": "user123",
        "channel": "discord",
        "language": "en",
        "timezone": "America/New_York"
    }
)
```

---

## 🧪 Testing

### Run Tests
```bash
cd ~/clawd/projects/llamafarm-core/server

# Test embedding engine (91% pass rate)
uv run pytest tests/test_router_embeddings.py -v

# Test all router components
uv run pytest tests/test_router_*.py -v

# Full test suite
uv run pytest tests/ -v --tb=short
```

### Verify Ollama
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Verify nomic-embed-text model
ollama list | grep nomic-embed-text

# Pull model if needed
ollama pull nomic-embed-text
```

### Run All Demos
```bash
# Simple baseline
uv run python demos/simple_routing_demo.py

# Semantic routing ⭐
uv run python demos/semantic_routing_demo.py

# Sessions
uv run python demos/session_demo.py

# Agents
uv run python demos/agent_basics_demo.py
```

---

## 🚨 Troubleshooting

### "No embedding backend available"
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# If not running
ollama serve

# Pull model
ollama pull nomic-embed-text
```

### Import Errors
```bash
# Ensure you're in server directory
cd ~/clawd/projects/llamafarm-core/server

# Install dependencies
uv sync
```

### Slow Performance
```bash
# First run is slower (model loading)
# Subsequent runs use caching

# Check Ollama status
curl http://localhost:11434/api/tags
```

### Tests Failing
```bash
# Some tests need API alignment (known issue)
# Embedding tests should pass: 10/11 (91%)

uv run pytest tests/test_router_embeddings.py -v
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────┐
│         LlamaFarm Framework                 │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────────┐      ┌────────────────┐   │
│  │  Semantic   │──────│  Embedding     │   │
│  │  Router     │      │  Engine        │   │
│  └─────────────┘      └────────────────┘   │
│         │                     │             │
│  ┌─────────────┐      ┌────────────────┐   │
│  │  Capability │      │  Ollama        │   │
│  │  Matcher    │      │  (nomic-768)   │   │
│  └─────────────┘      └────────────────┘   │
│         │                     │             │
│  ┌─────────────┐      ┌────────────────┐   │
│  │  Autonomous │      │  Session       │   │
│  │  Agent      │      │  Manager       │   │
│  └─────────────┘      └────────────────┘   │
│                                             │
└─────────────────────────────────────────────┘

Flow: Query → Embed → Match → Route → Execute
```

---

## 📚 Additional Resources

### Documentation
- **`DEMO_GUIDE.md`** - 5-minute demo walkthrough for engineers
- **`server/router/README.md`** - How semantic routing works
- **`demos/FINAL_REPORT.md`** - Full technical report with metrics
- **`server/router/embeddings.py`** - Embedding engine source code

### Performance Metrics
- **Weather queries**: 82-84% confidence ✅
- **Calculator**: 60-65% confidence ✅
- **Email**: 70-75% confidence ✅
- **Search**: 65-70% confidence ✅

### Next Steps
1. Try all demos to understand the flow
2. Read `DEMO_GUIDE.md` for talking points
3. Check `server/router/README.md` for technical details
4. Run tests to verify integration

---

## 🎯 Quick Reference

| Want to... | Run this... |
|------------|-------------|
| Show semantic routing | `uv run python demos/semantic_routing_demo.py` ⭐ |
| Explain concepts | `uv run python demos/simple_routing_demo.py` |
| Show sessions | `uv run python demos/session_demo.py` |
| Verify embeddings | `uv run pytest tests/test_router_embeddings.py -v` |
| Check Ollama | `curl http://localhost:11434/api/tags` |

---

**Built for LlamaFarm | Semantic Routing + Agent Framework**

Questions? See `DEMO_GUIDE.md` or `server/router/README.md`
