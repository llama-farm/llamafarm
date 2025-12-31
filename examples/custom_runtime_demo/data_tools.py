from sdk import tool
import json

@tool
def process_data(data: str) -> str:
    """Process data by reversing strings."""
    return json.dumps({
        "original": data,
        "reversed": data[::-1],
        "length": len(data)
    })

@tool
def factorial(n: int) -> int:
    """Calculate factorial."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0:
        return 1
    return n * factorial(n-1)
