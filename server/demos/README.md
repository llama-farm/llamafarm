# OpenClaw Lite Framework Demos

Demonstration scripts showcasing the integrated OpenClaw Lite framework in LlamaFarm.

## Quick Start

```bash
cd ~/clawd/projects/llamafarm-core/server
uv run python demos/<demo_name>.py
```

## Available Demos

### 1. Simple Routing Demo
**File:** `simple_routing_demo.py`

Demonstrates basic capability-based routing with keyword matching.

**Shows:**
- Capability registration and discovery
- Query-to-capability matching
- Routing decisions
- Node targeting

**Run:**
```bash
uv run python demos/simple_routing_demo.py
```

**Expected Output:**
- Registered capabilities with keywords
- Test queries routed to best-matching capabilities
- Routing decisions with scores

**Key Concepts:**
- Capabilities advertise what they can do
- Queries are matched to capabilities semantically
- Best match gets routed
- Foundation for distributed mesh routing

---

### 2. Agent Basics Demo
**File:** `agent_basics_demo.py`

Comprehensive demonstration of the autonomous agent framework.

**Shows:**
- Agent memory (short-term + long-term)
- Session management
- Task tracking and delegation
- Agent lifecycle

**Run:**
```bash
uv run python demos/agent_basics_demo.py
```

**Expected Output:**
- Memory creation and persistence
- Session creation and message history
- Task hierarchies with priorities
- Agent initialization and binding

**Key Concepts:**
- Agents maintain persistent memory
- Sessions manage conversation state
- Tasks can be delegated and tracked
- Components integrate seamlessly

---

### 3. Semantic Routing Demo (Advanced)
**File:** `semantic_routing_demo.py`

Full semantic routing with embeddings and gradient learning.

**Shows:**
- Text embedding generation
- Semantic similarity calculation
- Multi-capability matching
- Confidence thresholding

**Requirements:**
- Ollama running locally with `nomic-embed-text` model
- Or LlamaFarm cloud embedding endpoint

**Run:**
```bash
# Ensure Ollama is running
ollama pull nomic-embed-text

# Run demo
uv run python demos/semantic_routing_demo.py
```

**Note:** This demo uses async embedding APIs and requires proper backend configuration.

---

## Customization Tips

### Adding Custom Capabilities

```python
from router import Capability, CapabilityMatcher

# Define your capability
my_capability = Capability(
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

# Create matcher
matcher = CapabilityMatcher(capabilities)
```

### Configuring Agent Memory

```python
from pathlib import Path
from agents import AgentMemory

memory = AgentMemory(
    storage_path=Path("/path/to/memory.json"),
    max_short_term=50,      # Recent observations
    max_long_term=500       # Permanent learnings
)
```

### Custom Session Metadata

```python
from agents import SessionManager

manager = SessionManager()

session = await manager.create_session(
    session_id="custom-session",
    metadata={
        "user_id": "user123",
        "channel": "discord",
        "language": "en",
        "timezone": "America/New_York"
    }
)
```

---

## Testing Your Changes

After modifying the framework:

```bash
# Run all demos
./run_all_demos.sh

# Run specific demo
uv run python demos/simple_routing_demo.py

# Run with verbose output
uv run python -u demos/agent_basics_demo.py
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│         OpenClaw Lite Framework             │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────────┐      ┌────────────────┐   │
│  │  Autonomous │──────│  Session       │   │
│  │  Agent      │      │  Manager       │   │
│  └─────────────┘      └────────────────┘   │
│         │                     │             │
│  ┌─────────────┐      ┌────────────────┐   │
│  │  Memory     │      │  Scheduler     │   │
│  │  (ST + LT)  │      │  (Cron Jobs)   │   │
│  └─────────────┘      └────────────────┘   │
│         │                     │             │
│  ┌──────────────────────────────────────┐  │
│  │         Semantic Router              │  │
│  │  ┌────────────┐  ┌────────────────┐  │  │
│  │  │ Embeddings │  │ Capability     │  │  │
│  │  │ Engine     │  │ Matcher        │  │  │
│  │  └────────────┘  └────────────────┘  │  │
│  │  ┌────────────┐  ┌────────────────┐  │  │
│  │  │ Gradient   │  │ Gossip         │  │  │
│  │  │ Learning   │  │ Protocol       │  │  │
│  │  └────────────┘  └────────────────┘  │  │
│  └──────────────────────────────────────┘  │
│         │                     │             │
│  ┌─────────────┐      ┌────────────────┐   │
│  │  Skills     │      │  Channels      │   │
│  │  Registry   │      │  Registry      │   │
│  └─────────────┘      └────────────────┘   │
│                                             │
└─────────────────────────────────────────────┘
```

---

## Next Steps

1. **Extend capabilities** - Add domain-specific capabilities
2. **Build skills** - Create reusable agent skills
3. **Connect channels** - Integrate messaging platforms
4. **Deploy nodes** - Set up distributed routing mesh
5. **Enable learning** - Activate gradient-based optimization

---

## Troubleshooting

### Import Errors
Make sure you're in the server directory:
```bash
cd ~/clawd/projects/llamafarm-core/server
```

### Missing Dependencies
Install via uv:
```bash
uv sync
```

### Ollama Not Running
Start Ollama for embedding demos:
```bash
ollama serve
```

---

## Contributing

To add a new demo:

1. Create `demos/your_demo_name.py`
2. Follow the structure of existing demos
3. Add documentation to this README
4. Test with `uv run python demos/your_demo_name.py`
5. Submit PR

---

**Built with ❤️ for LlamaFarm | OpenClaw Lite Integration**
