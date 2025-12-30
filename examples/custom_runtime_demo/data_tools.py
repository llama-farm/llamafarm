from sdk import tool
import json

@tool(description="Process data by reversing strings")
def process_data(data: str) -> str:
    """
    Takes a string input, reverses it, and returns a JSON summary.
    """
    return json.dumps({
        "original": data,
        "reversed": data[::-1],
        "length": len(data)
    })

@tool(description="Calculate factorial")
def factorial(n: int) -> int:
    """
    Calculates the factorial of a number containing logic.
    """
    if n < 0:
        return -1
    if n == 0:
        return 1
    return n * factorial(n-1)
