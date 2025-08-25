"""Framework integrations for the prompts system.

This module provides adapters for integrating with various LLM frameworks:
- LangChain
- LangGraph
- Native APIs (OpenAI, Anthropic, etc.)
- LlamaIndex
- Custom frameworks

Each adapter provides a consistent interface while leveraging framework-specific features.
"""

from .langgraph_integration import LangGraphWorkflowManager

# Future adapters can be imported here
# from .langchain_adapter import LangChainAdapter
# from .llamaindex_adapter import LlamaIndexAdapter
# from .native_adapter import NativeAdapter

__all__ = [
    'LangGraphWorkflowManager'
]