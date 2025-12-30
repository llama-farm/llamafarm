"""Universal Agent Example.

Demonstrates how to use the LlamaFarm 'Agent' class to create
generic, autonomous background tasks that run alongside your tools.
"""

import asyncio
import random
from llamafarm.sdk import Agent, tool

# Inline Tools can coexist with Agents
@tool
def get_system_status() -> str:
    return "All systems operational."

class SystemMonitor(Agent):
    """Monitors system metric (simulated) every 2 seconds."""
    
    interval = 2.0
    
    async def on_start(self):
        print(f"[{self.name}] Initializing sensors...")
        
    async def on_tick(self):
        cpu_usage = random.randint(10, 30)
        print(f"[{self.name}] CPU Usage: {cpu_usage}%")
        
        if cpu_usage > 25:
            print(f"[{self.name}] ALERT: High CPU usage detected!")

class NewsTicker(Agent):
    """Simulates checking for news updates."""
    
    interval = 5.0
    
    async def on_tick(self):
        headlines = ["Market up", "Weather sunny", "LlamaFarm v2 released"]
        print(f"[{self.name}] Breaking News: {random.choice(headlines)}")
