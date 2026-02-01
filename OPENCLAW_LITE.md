# OpenClaw Lite - Agent Framework for LlamaFarm

OpenClaw Lite is a lightweight agent framework integrated into LlamaFarm, providing autonomous AI agents with memory, sessions, scheduling, and multi-channel messaging.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     OpenClaw Lite Framework                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │  Autonomous      │  │   Session        │  │   Agent       │  │
│  │  Agent Loop      │  │   Manager        │  │   Scheduler   │  │
│  │                  │  │                  │  │               │  │
│  │  • Task Queue    │  │  • Create/Get    │  │  • Cron Jobs  │  │
│  │  • Observe       │  │  • Send/Receive  │  │  • Intervals  │  │
│  │  • Think         │  │  • History       │  │  • One-shot   │  │
│  │  • Act           │  │  • Spawn Sub     │  │  • Persistence│  │
│  └──────────────────┘  └──────────────────┘  └───────────────┘  │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │  Agent Memory    │  │   Skill System   │  │   Channels    │  │
│  │                  │  │                  │  │               │  │
│  │  • Short-term    │  │  • Tool Defs     │  │  • Webhook    │  │
│  │  • Long-term     │  │  • Execution     │  │  • WebSocket  │  │
│  │  • Facts         │  │  • Discovery     │  │  • Extensible │  │
│  │  • Persistence   │  │  • Hot Reload    │  │               │  │
│  └──────────────────┘  └──────────────────┘  └───────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  Semantic Router                          │   │
│  │  • Capability Matching  • Gradient Tables  • Gossip Mesh │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Autonomous Agent (`server/agents/autonomous.py`)

The core agent loop with task execution and memory.

```python
from server.agents import AutonomousAgent, get_or_create_agent

# Create an agent
agent = get_or_create_agent(
    agent_id="my-agent",
    storage_path=Path("./agent-memory.json")
)

# Add a task
task = agent.add_task(
    description="Research the latest AI developments",
    priority=8
)

# Start the agent loop
await agent.run()

# Stop gracefully
agent.stop()
```

**Features:**
- Priority task queue
- Observe-Think-Act cycle
- Memory persistence
- Router integration for delegation

### 2. Sessions (`server/agents/sessions.py`)

Manage conversation contexts and cross-session messaging.

```python
from server.agents import SessionManager, SessionConfig, get_session_manager

manager = get_session_manager()

# Create a session
session = manager.create_session(
    agent_id="my-agent",
    config=SessionConfig(
        model="llama3.1:8b",
        system_prompt="You are a helpful assistant.",
        channel="telegram"
    ),
    label="main-chat"
)

# Add messages
session.add_message(role="user", content="Hello!")
session.add_message(role="assistant", content="Hi there!")

# Send to another session
response = await manager.send_to_session(
    target_key="other-session",
    message="Please analyze this data"
)

# Spawn a sub-agent session
child = await manager.spawn_session(
    parent_key=session.session_key,
    agent_id="analyzer",
    task="Analyze the uploaded document"
)
```

### 3. Scheduler (`server/agents/scheduler.py`)

Cron-like scheduling for agent tasks.

```python
from server.agents import AgentScheduler, JobType, get_scheduler
from datetime import datetime, timedelta

scheduler = get_scheduler()

# One-shot reminder
job = scheduler.add_job(
    name="reminder",
    task="Remind me to check emails",
    job_type=JobType.ONCE,
    delay_sec=3600  # In 1 hour
)

# Recurring cron job
job = scheduler.add_job(
    name="daily-summary",
    task="Generate a daily summary of activities",
    job_type=JobType.CRON,
    cron_expr="0 9 * * *",  # Every day at 9 AM
    channel="telegram"
)

# Interval job
job = scheduler.add_job(
    name="health-check",
    task="Check system health",
    job_type=JobType.INTERVAL,
    interval_sec=300  # Every 5 minutes
)

# Start scheduler
await scheduler.start()

# Run a job manually
run = await scheduler.run_job(job.job_id)
```

### 4. Skills (`server/agents/skills/`)

Modular capabilities with tools for function calling.

```python
from server.agents.skills import (
    Skill, SkillConfig, SkillContext, SkillResult,
    Tool, ToolParameter, ToolParameterType,
    get_skill_registry
)

class WeatherSkill(Skill):
    name = "weather"
    description = "Get weather forecasts and conditions"
    
    def __init__(self, config=None):
        super().__init__(config)
        self.tools = [
            Tool(
                name="get_weather",
                description="Get current weather for a location",
                parameters=[
                    ToolParameter(
                        name="location",
                        param_type=ToolParameterType.STRING,
                        description="City name or coordinates"
                    )
                ],
                handler=self._get_weather
            )
        ]
    
    async def _get_weather(self, location: str) -> dict:
        # Implementation
        return {"temp": 72, "conditions": "sunny"}
    
    async def execute(self, context: SkillContext) -> SkillResult:
        # Main execution logic
        return SkillResult(
            success=True,
            output="Weather retrieved successfully"
        )

# Register the skill
registry = get_skill_registry()
await registry.register_skill(WeatherSkill())

# Get all tools for OpenAI-style function calling
tools = registry.get_tools_for_openai()

# Execute a tool
result = await registry.execute_tool("get_weather", location="San Francisco")
```

### 5. Channels (`server/agents/channels/`)

Messaging surface abstraction for multi-platform support.

```python
from server.agents.channels import (
    Channel, ChannelConfig, ChannelMessage,
    WebhookChannel, WebSocketChannel,
    get_channel_registry
)

# Configure a webhook channel
config = ChannelConfig(
    channel_type="webhook",
    channel_id="my-webhook",
    webhook_url="https://api.example.com/messages",
    api_key="secret"
)

channel = WebhookChannel(config)
await channel.connect()

# Send a message
msg = await channel.send(
    target="channel-123",
    content="Hello from LlamaFarm!"
)

# Receive messages (async generator)
async for message in channel.receive():
    print(f"Received: {message.content}")
    await channel.reply(message, "Got it!")

# Use the registry for multiple channels
registry = get_channel_registry()
registry.add_channel(channel)

# Broadcast to all channels
await registry.broadcast(
    content="System update complete",
    channel_type="webhook"
)
```

### 6. Agent Memory (`server/agents/autonomous.py`)

Persistent memory with short-term and long-term storage.

```python
from server.agents import AgentMemory

memory = AgentMemory(
    storage_path=Path("./memory.json"),
    max_short_term=100,
    max_long_term=1000
)

# Add observations
memory.add_observation("User asked about weather")
memory.add_action("Called weather API")
memory.add_result("Temperature: 72°F, Sunny")

# Store facts
memory.set_fact("user_location", "San Francisco")
memory.set_fact("preferred_units", "fahrenheit")

# Memorize important info (long-term)
memory.memorize("User prefers detailed explanations")

# Recall relevant memories
memories = memory.get_relevant_memories("weather preferences")

# Get context for prompts
context = memory.get_context(max_entries=20)
```

## API Endpoints

All endpoints are mounted under `/agents`:

### Agents
- `GET /agents/` - List all agents
- `POST /agents/` - Create an agent
- `GET /agents/{agent_id}` - Get agent status
- `POST /agents/{agent_id}/tasks` - Add a task
- `POST /agents/{agent_id}/start` - Start agent loop
- `POST /agents/{agent_id}/stop` - Stop agent loop

### Sessions
- `GET /agents/sessions` - List sessions
- `POST /agents/sessions` - Create session
- `GET /agents/sessions/{key}` - Get session
- `GET /agents/sessions/{key}/history` - Get history
- `POST /agents/sessions/{key}/messages` - Add message
- `POST /agents/sessions/{key}/send` - Send & get response
- `DELETE /agents/sessions/{key}` - Delete session

### Scheduler
- `GET /agents/cron` - List jobs
- `POST /agents/cron` - Create job
- `GET /agents/cron/{job_id}` - Get job
- `PATCH /agents/cron/{job_id}` - Update job
- `DELETE /agents/cron/{job_id}` - Remove job
- `POST /agents/cron/{job_id}/run` - Run job now
- `GET /agents/cron/{job_id}/runs` - Get run history

### Skills
- `GET /agents/skills` - List skills
- `GET /agents/skills/{name}` - Get skill
- `GET /agents/skills/{name}/tools` - Get skill tools
- `GET /agents/tools` - List all tools

### Health
- `GET /agents/health` - Framework health status

## Integration with Semantic Router

The agent framework integrates with the Semantic Router for intelligent task routing:

```python
from server.router import RouterService

# In your agent
class MyAgent(AutonomousAgent):
    async def _think(self, context, memories):
        # Route intent to find best capability
        match = await self.router_service.route_intent(
            "I need to analyze this image"
        )
        
        if match.action == "route_forward":
            # Delegate to another node
            return f"Delegating to {match.peer}"
        elif match.action == "process_local":
            # Handle locally
            return f"Using local capability: {match.capability}"
```

## Configuration

Add to your `server/core/settings.py`:

```python
class Settings(BaseSettings):
    # ... existing settings ...
    
    # Agent Framework
    agent_storage_dir: str = "~/.llamafarm/agents"
    agent_max_sessions: int = 1000
    agent_session_idle_timeout: int = 3600
    
    # Scheduler
    scheduler_storage_path: str = "~/.llamafarm/scheduler.json"
    scheduler_max_runs_history: int = 100
```

## Getting Started

1. **Start the LlamaFarm server:**
   ```bash
   cd ~/clawd/projects/llamafarm-core
   python -m server.api.main
   ```

2. **Create an agent via API:**
   ```bash
   curl -X POST http://localhost:8000/agents/ \
     -H "Content-Type: application/json" \
     -d '{"agent_id": "my-agent"}'
   ```

3. **Create a session:**
   ```bash
   curl -X POST http://localhost:8000/agents/sessions \
     -H "Content-Type: application/json" \
     -d '{"agent_id": "my-agent", "label": "main"}'
   ```

4. **Add a scheduled task:**
   ```bash
   curl -X POST http://localhost:8000/agents/cron \
     -H "Content-Type: application/json" \
     -d '{
       "name": "daily-check",
       "task": "Check for updates and summarize",
       "job_type": "cron",
       "cron_expr": "0 9 * * *"
     }'
   ```

## Philosophy

OpenClaw Lite follows LlamaFarm's principles:
- **Minimal dependencies** - Pure Python with asyncio
- **Local-first** - All data stored locally
- **Composable** - Mix and match components
- **Extensible** - Easy to add new skills and channels

## Files

```
server/agents/
├── __init__.py           # Main exports
├── api.py                # FastAPI endpoints
├── autonomous.py         # Agent loop + memory
├── sessions.py           # Session management
├── scheduler.py          # Cron/scheduling
├── base/                 # Base agent classes
│   ├── agent.py
│   ├── history.py
│   └── types.py
├── channels/             # Messaging channels
│   ├── __init__.py
│   ├── base.py
│   ├── registry.py
│   ├── webhook.py
│   └── websocket.py
└── skills/               # Skill system
    ├── __init__.py
    ├── base.py
    ├── registry.py
    └── loader.py
```

---

*OpenClaw Lite - Lightweight agents for LlamaFarm*
*February 2026*
