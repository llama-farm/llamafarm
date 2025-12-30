# Building Active Agents

Active Agents are autonomous processes that run in the background alongside your tools. They are perfect for monitoring, polling, or scheduled tasks.

## The Agent Class

Subclass `llamafarm.sdk.Agent` to define an agent.

```python
from llamafarm.sdk import Agent
import asyncio

class SystemMonitor(Agent):
    """Monitors system status every 5 seconds."""
    
    interval = 5.0  # Run on_tick every 5 seconds
    
    async def on_start(self):
        print(f"[{self.name}] Monitor starting up...")
        
    async def on_tick(self):
        # Your logic here
        print(f"[{self.name}] Checking system health...")
        # e.g., if cpu > 90: self.alert()
        
    async def on_stop(self):
        print(f"[{self.name}] Monitor shutting down.")
```

## Accessing LlamaFarm Intelligence

Agents come with a built-in authenticated client (`self.client`) to talk to the LlamaFarm Universal Runtime. This allows you to run Inference, Training, or RAG queries from within your agent loop.

```python
    async def on_tick(self):
        # Example: Detect Anomaly
        response = await self.client.post("/v1/anomaly/detect", json={
            "model": "my_model",
            "data": [[0.1, 0.5, 0.9]]
        })
        if response["is_anomaly"]:
            print("Anomaly Detected!")
```

## Running Agents

When you run your script with `llamafarm.runtime`, it automatically discovers all `Agent` subclasses and manages their lifecycle:

```bash
python -m llamafarm.runtime my_agent.py
```

The runtime will:
1. Instantiate your Agents.
2. Inject the `LlamaFarmClient`.
3. Start their loops concurrently.
4. Manage graceful shutdown.
