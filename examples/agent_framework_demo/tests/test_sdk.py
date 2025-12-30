"""Unit tests for LlamaFarm Agent Framework."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
import sys
from pathlib import Path

# Add the example directory to sys.path so we can import llamafarm
sys.path.insert(0, str(Path(__file__).parent.parent))

from llamafarm import sdk
from llamafarm import runtime

# Reset registry for each test
@pytest.fixture(autouse=True)
def clean_registry():
    old_registry = sdk._TOOLS_REGISTRY.copy()
    sdk._TOOLS_REGISTRY.clear()
    yield
    sdk._TOOLS_REGISTRY = old_registry

def test_tool_decorator():
    """Verify @tool registers functions correctly."""
    
    @sdk.tool
    def my_func(x: int):
        return x * 2
        
    assert "my_func" in sdk._TOOLS_REGISTRY
    registered_func = sdk._TOOLS_REGISTRY["my_func"]
    assert registered_func.__name__ == "my_func"
    assert getattr(registered_func, "_is_tool", False) is True
    
    # Test custom name
    @sdk.tool("custom_name")
    def another_func():
        pass
        
    assert "custom_name" in sdk._TOOLS_REGISTRY
    assert "custom_name" in sdk._TOOLS_REGISTRY
    reg_custom = sdk._TOOLS_REGISTRY["custom_name"]
    assert reg_custom.__name__ == "another_func"
    assert getattr(reg_custom, "_is_tool", False) is True

@pytest.mark.asyncio
async def test_agent_lifecycle():
    """Verify Agent start/stop/tick lifecycle."""
    
    class MockAgent(sdk.Agent):
        interval = 0.01
        
        def __init__(self):
            super().__init__("TestAgent")
            self.tick_count = 0
            self.started = False
            self.stopped = False
            
        async def on_start(self):
            self.started = True
            
        async def on_tick(self):
            self.tick_count += 1
            if self.tick_count >= 3:
                self._is_running = False # Stop self
                
        async def on_stop(self):
            self.stopped = True
            
    agent = MockAgent()
    
    # Run agent - it should stop itself after 3 ticks
    # We use explicit timeout to prevents hanging if it fails
    try:
        await asyncio.wait_for(agent.start(), timeout=1.0)
    except asyncio.TimeoutError:
        await agent.stop()
        pytest.fail("Agent did not stop itself")
        
    assert agent.started
    assert agent.tick_count >= 3
    assert agent.stopped

def test_runtime_agent_discovery():
    """Verify runtime can discover agents in a module."""
    
    # create a dummy module object
    class DummyModule:
        pass
        
    mod = DummyModule()
    
    class MyAgent(sdk.Agent):
        pass
        
    class NotAgent:
        pass
        
    mod.MyAgent = MyAgent
    mod.NotAgent = NotAgent
    mod.SomeOtherThing = 123
    
    agents = runtime.discover_agents(mod)
    
    assert len(agents) == 1
    assert isinstance(agents[0], MyAgent)
    assert agents[0].name == "MyAgent"
