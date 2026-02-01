"""
Skills Layer for OpenClaw Lite.

Provides modular agent capabilities that can be:
- Discovered and loaded dynamically
- Configured per-agent
- Shared across agents
"""

from .base import Skill, SkillConfig, SkillContext, SkillResult, Tool
from .registry import SkillRegistry, get_skill_registry
from .loader import SkillLoader

__all__ = [
    "Skill",
    "SkillConfig",
    "SkillContext",
    "SkillResult",
    "Tool",
    "SkillRegistry",
    "get_skill_registry",
    "SkillLoader",
]
