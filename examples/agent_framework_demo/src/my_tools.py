"""User-defined tools for the Agent Framework Demo.

This file demonstrates how a user can define tools using the
new LlamaFarm @tool decorator. No server code is required.
"""

from sdk import tool

@tool
def calculate_tax(amount: float, rate: float = 0.2) -> float:
    """Calculate tax for a given amount.
    
    Args:
        amount: The base amount.
        rate: The tax rate (default 0.2).
    """
    return amount * rate

@tool
def greet_user(name: str) -> str:
    """Greet the user warmly."""
    return f"Hello, {name}! Welcome to the LlamaFarm Agent Framework."
