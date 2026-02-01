# LlamaFarm + Needle + OpenClaw Lite Integration

## Vision
Transform LlamaFarm from a standalone AI development tool into a **distributed, agentic AI platform** that can:
1. **Route AI tasks semantically** across a mesh of devices (Needle protocol)
2. **Run autonomous agents** with memory, tool use, and scheduling (OpenClaw Lite)
3. **Scale from single laptop to enterprise fleet** of edge devices

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        LlamaFarm Core                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Models     │  │    RAG       │  │   Tools      │          │
│  │  (existing)  │  │  (existing)  │  │  (existing)  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────┐          │
│  │              Semantic Router (NEW)                │          │
│  │  • Capability matching via embeddings             │          │
│  │  • Gradient routing tables                        │          │
│  │  • Gossip protocol for mesh discovery             │          │
│  └──────────────────────────────────────────────────┘          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────┐          │
│  │              Agent Framework (NEW)                │          │
│  │  • Agent loops with tool calling                  │          │
│  │  • Short-term + long-term memory                  │          │
│  │  • Sessions & instances                           │          │
│  │  • Cron/scheduled tasks                           │          │
│  │  • Skills (modular capabilities)                  │          │
│  └──────────────────────────────────────────────────┘          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────┐          │
│  │              Channels Layer (NEW)                 │          │
│  │  • Messaging surface abstraction                  │          │
│  │  • WhatsApp, Telegram, Slack, Discord, etc.       │          │
│  │  • Webhook/WebSocket handlers                     │          │
│  └──────────────────────────────────────────────────┘          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────┐          │
│  │              Nodes Layer (NEW)                    │          │
│  │  • Device registry & health monitoring            │          │
│  │  • Capability announcements                       │          │
│  │  • Task distribution                              │          │
│  └──────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Phase 1: Semantic Router Integration (Week 1-2)

### Source
Copy from `~/clawd/projects/needle-router/router/`:
- `embeddings.py` - Async embedding engine (Ollama/LlamaFarm models)
- `matcher.py` - Semantic capability matching
- `gradient.py` - Routing tables with TTL and decay
- `gossip.py` - UDP/mDNS mesh protocol

### Target
Create `server/router/`:
```
server/router/
├── __init__.py
├── embeddings.py      # Adapted for LlamaFarm model loading
├── matcher.py         # Capability semantic matching
├── gradient.py        # Gradient routing tables
├── gossip.py          # Gossip protocol for mesh
├── discovery.py       # mDNS + UDP peer discovery
└── api.py             # FastAPI endpoints for routing
```

### Integration Points
1. **Model Loading**: Use LlamaFarm's existing model management for embeddings
2. **API Integration**: Add `/router/` endpoints to FastAPI server
3. **Configuration**: Add router config to `server/core/settings.py`

## Phase 2: Agent Framework (Week 2-3)

### Components (OpenClaw Lite)

#### 1. Agent Loop (`server/agents/loop.py`)
```python
class AgentLoop:
    """Core agent execution loop with tool calling."""
    
    async def run(self, message: str, context: AgentContext) -> AgentResponse:
        """Execute agent turn with reasoning and tool use."""
        pass
    
    async def call_tool(self, tool_name: str, args: dict) -> ToolResult:
        """Execute a tool and return results."""
        pass
```

#### 2. Agent Memory (`server/agents/memory/`)
```python
class AgentMemory:
    """Short-term and long-term memory management."""
    
    def __init__(self, short_term_limit: int = 50, long_term_path: str = None):
        self.short_term = []  # Recent messages
        self.long_term = MemoryStore(long_term_path)  # Persistent
    
    async def add(self, message: Message) -> None:
        """Add message to memory, promoting to long-term if significant."""
        pass
    
    async def recall(self, query: str, k: int = 5) -> List[Message]:
        """Semantic recall from memory."""
        pass
```

#### 3. Sessions (`server/agents/sessions.py`)
```python
class SessionManager:
    """Manage agent sessions and instances."""
    
    def create_session(self, session_id: str, config: SessionConfig) -> Session:
        pass
    
    def get_session(self, session_id: str) -> Optional[Session]:
        pass
    
    async def send_to_session(self, session_id: str, message: str) -> Response:
        pass
```

#### 4. Cron/Scheduler (`server/agents/scheduler.py`)
```python
class AgentScheduler:
    """Schedule recurring agent tasks."""
    
    async def add_job(self, job: CronJob) -> str:
        """Add a scheduled job."""
        pass
    
    async def run_job(self, job_id: str) -> JobResult:
        """Execute a scheduled job."""
        pass
```

#### 5. Skills (`server/agents/skills/`)
```python
class Skill:
    """Modular agent capability."""
    
    name: str
    description: str
    tools: List[Tool]
    
    async def execute(self, context: SkillContext) -> SkillResult:
        pass
```

## Phase 3: Channels Layer (Week 3-4)

### Abstraction
```python
class Channel(ABC):
    """Abstract messaging surface."""
    
    @abstractmethod
    async def send(self, target: str, message: Message) -> bool:
        pass
    
    @abstractmethod
    async def receive(self) -> AsyncIterator[IncomingMessage]:
        pass
```

### Implementations
- `channels/telegram.py` - grammY-style Telegram
- `channels/slack.py` - Bolt-style Slack
- `channels/discord.py` - discord.py wrapper
- `channels/whatsapp.py` - Baileys adapter
- `channels/webhook.py` - Generic webhook ingestion

## Phase 4: Nodes Layer (Week 4-5)

### Device Registry
```python
class NodeRegistry:
    """Track available compute nodes."""
    
    def register(self, node: NodeInfo) -> None:
        """Register a new node."""
        pass
    
    def get_capable_nodes(self, capability: str) -> List[NodeInfo]:
        """Find nodes with specific capability."""
        pass
    
    async def dispatch(self, task: Task, node_id: str) -> TaskResult:
        """Send task to specific node."""
        pass
```

### Integration with Semantic Router
The Nodes layer uses the Semantic Router to:
1. Match task intents to node capabilities
2. Route tasks to optimal nodes based on:
   - Capability match score
   - Node health/availability
   - Network proximity (gradient tables)
   - Load balancing

## Deployment Scenarios

### 1. Single Laptop (Development)
- All components on one machine
- Semantic router in "local" mode
- Memory stored in local files
- Channels configured for dev webhooks

### 2. Small Team (5-20 nodes)
- Central LlamaFarm server
- Nodes on team devices (laptops, phones, tablets)
- Gossip protocol for mesh discovery
- Shared memory via central server

### 3. Enterprise (100+ nodes)
- Multiple LlamaFarm servers (regional)
- Nodes across data centers, edge devices, mobile
- Federated memory with sync
- Full mesh networking with Needle protocol

## Implementation Priority

1. **Week 1**: Semantic Router integration (core mesh routing)
2. **Week 2**: Agent Loop + Memory (basic agent framework)
3. **Week 3**: Sessions + Scheduler (multi-agent orchestration)
4. **Week 4**: Channels (messaging integration)
5. **Week 5**: Nodes (distributed compute)

## Key Design Decisions

### 1. Python-First
LlamaFarm server is Python/FastAPI. All new components will be Python to maintain consistency.

### 2. Async Throughout
All new code uses `asyncio` for non-blocking operations.

### 3. Minimal Dependencies
Avoid heavy frameworks. Use:
- `aiohttp` for HTTP
- `websockets` for WS
- `zeroconf` for mDNS (optional)
- `msgpack` for serialization

### 4. Configuration via Environment
All settings via env vars or config files, matching LlamaFarm patterns.

### 5. Backward Compatible
Existing LlamaFarm APIs remain unchanged. New features are additive.

## Success Metrics

1. **Routing Latency**: < 100ms for semantic capability matching
2. **Agent Response Time**: < 2s for simple tool-using responses
3. **Node Discovery**: < 5s for new node to join mesh
4. **Memory Recall**: < 500ms for semantic memory search
5. **Channel Latency**: < 1s from message receipt to agent response

## Next Steps

1. Create `server/router/` with Needle integration
2. Extend `server/agents/` with memory and sessions
3. Add `server/channels/` abstraction
4. Add `server/nodes/` registry
5. Write integration tests for end-to-end flows
6. Document API changes in docs/

---

*This document tracks the integration of Needle semantic routing and OpenClaw Lite agent framework into LlamaFarm. Updated: 2026-01-31*
