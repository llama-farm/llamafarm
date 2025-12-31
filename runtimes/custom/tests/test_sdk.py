"""Unit tests for LlamaFarm Agent Framework."""

import pytest
import asyncio
import sys
from pathlib import Path

# Add the runtime directory to sys.path so we can import sdk and server
RUNTIME_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(RUNTIME_DIR))

import sdk
import server

# Reset registry for each test
@pytest.fixture(autouse=True)
def clean_registry():
    old_registry = sdk._TOOLS_REGISTRY.copy()
    sdk._TOOLS_REGISTRY.clear()
    yield
    sdk._TOOLS_REGISTRY = old_registry

def test_tool_decorator_strict():
    """Verify @tool registers functions correctly with strict API (docstrings)."""
    
    @sdk.tool
    def my_func(x: int):
        """This is a test tool."""
        return x * 2
        
    assert "my_func" in sdk._TOOLS_REGISTRY
    registered_func = sdk._TOOLS_REGISTRY["my_func"]
    assert registered_func.__name__ == "my_func"
    assert getattr(registered_func, "_is_tool", False) is True
    assert getattr(registered_func, "_tool_description") == "This is a test tool."
    
    # Test custom name
    @sdk.tool(name="custom_name")
    def another_func():
        """Another desc."""
        pass
        
    assert "custom_name" in sdk._TOOLS_REGISTRY
    reg_custom = sdk._TOOLS_REGISTRY["custom_name"]
    assert reg_custom.__name__ == "another_func"
    assert getattr(reg_custom, "_is_tool", False) is True
    assert getattr(reg_custom, "_tool_description") == "Another desc."

def test_tool_decorator_no_description_arg():
    """Verify that using 'description' arg raises error or is ignored (API contract)."""
    # Since we removed the argument from the signature, this should raise TypeError
    with pytest.raises(TypeError):
        @sdk.tool(description="Should fail")
        def fail_func():
            pass

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
    try:
        await asyncio.wait_for(agent.start(), timeout=1.0)
    except asyncio.TimeoutError:
        await agent.stop()
        pytest.fail("Agent did not stop itself")
        
    assert agent.started
    assert agent.tick_count >= 3
    assert agent.stopped

def test_server_discovery():
    """Verify server.py discovery logic."""
    
    # Create a dummy module
    class DummyModule:
        pass
        
    mod = DummyModule()
    
    # Add tools
    @sdk.tool
    def tool1(): pass
    
    @sdk.tool(name="tool2")
    def original_tool2(): pass
    
    mod.tool1 = tool1
    mod.tool2 = original_tool2
    mod.random_func = lambda: None
    
    # Add Agents
    class MyAgent(sdk.Agent): pass
    class NotAgent: pass
    
    mod.MyAgent = MyAgent
    mod.NotAgent = NotAgent
    mod.AgentBase = sdk.Agent # Should be ignored
    
    # Test Tool Discovery
    found_tools = server.discover_tools(mod)
    assert "tool1" in found_tools
    assert "tool2" in found_tools
    assert len(found_tools) == 2
    
    # Test Agent Discovery
    found_agents = server.discover_agents(mod)
    assert len(found_agents) == 1
    assert found_agents[0] == MyAgent
