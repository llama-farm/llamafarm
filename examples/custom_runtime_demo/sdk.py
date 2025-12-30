from functools import wraps

def tool(func=None, *, name=None, description=None):
    """
    Decorator to mark a function as a tool usable by LlamaFarm agents.
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        
        wrapper._is_tool = True
        wrapper._tool_name = name or f.__name__
        wrapper._tool_description = description or f.__doc__ or "No description provided"
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)
