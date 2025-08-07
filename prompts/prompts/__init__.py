"""
LlamaFarm Prompts System

A strategy-based prompt management system for LLMs, providing dynamic template 
selection, multi-framework support, and extensible architecture.

Key Features:
- Strategy-based configuration
- Dynamic template selection
- Multi-framework support (LangChain, LangGraph, Native APIs)
- Performance optimization
- Extensible architecture
- Comprehensive CLI
- 20+ built-in templates across 6 categories

Directory Structure:
- core/        Internal components (engines, models, loaders, CLI)
- frameworks/  Framework integrations (extensible)
- strategies/  Strategy definitions (extensible)
- templates/   Template definitions (extensible)
"""

__version__ = "0.2.0"
__author__ = "LlamaFarm Team"

# Core components
from .core.engines import (
    PromptSystem,
    TemplateEngine,
    StrategyEngine,
    TemplateRegistry
)

# Models
from .core.models import (
    PromptTemplate,
    PromptStrategy,
    PromptConfig,
    PromptContext
)

__all__ = [
    # Core engines
    "PromptSystem",
    "TemplateEngine", 
    "StrategyEngine",
    "TemplateRegistry",
    
    # Models
    "PromptTemplate",
    "PromptStrategy", 
    "PromptConfig",
    "PromptContext",
]