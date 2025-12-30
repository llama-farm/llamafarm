# Quickstart: Inline Tools

This guide will show you how to create a simple set of tools using the Agent Framework.

## 1. Create a Python File

Create a file named `my_tools.py`:

```python
from llamafarm.sdk import tool

@tool
def calculate_tax(amount: float, rate: float = 0.2) -> float:
    """Calculates tax for a given amount."""
    return amount * rate

@tool
def greet(name: str) -> str:
    """Greets the user."""
    return f"Hello, {name}!"
```

## 2. Run with Magic Runtime

Use the LlamaFarm runtime command (or the python module directly):

```bash
# If installed as a package
python -m llamafarm.runtime my_tools.py

# Expected Output:
# Loading user code from my_tools.py...
# Registering tool: calculate_tax
# Registering tool: greet
# Starting Magic Runtime...
```

That's it! Your tools are now being served as an MCP Server. You can connect LlamaFarm Orchestrator or any MCP Client to this process to use the tools.

## 3. Configuration (Optional)

You can customize tool names:

```python
@tool("custom_tax_calc")
def calculate_tax(...): ...
```
