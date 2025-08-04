"""
LlamaFarm Prompts Management System

A comprehensive, configuration-driven prompt management framework that integrates
seamlessly with the LlamaFarm RAG ecosystem. Built with best practices for
production use, extensibility, and developer experience.

Key Features:
- Configuration-driven prompt templates
- LangGraph integration for advanced workflows  
- CLI-first management interface
- High-level global prompts system
- 20+ built-in prompt templates
- Validation and testing framework
- RAG system integration
"""

__version__ = "0.1.0"
__author__ = "LlamaFarm Team"

from .core.prompt_system import PromptSystem
from .core.template_engine import TemplateEngine
from .core.strategy_engine import StrategyEngine
from .core.template_registry import TemplateRegistry
from .models.template import PromptTemplate
from .models.strategy import PromptStrategy
from .models.config import PromptConfig

__all__ = [
    "PromptSystem",
    "TemplateEngine", 
    "StrategyEngine",
    "TemplateRegistry",
    "PromptTemplate",
    "PromptStrategy", 
    "PromptConfig",
]