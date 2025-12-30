from sdk import tool

@tool(description="A test tool that adds two numbers")
def add_numbers(a: int, b: int) -> int:
    return a + b
