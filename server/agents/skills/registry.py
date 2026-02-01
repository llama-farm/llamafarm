"""
Skill Registry for managing loaded skills.
"""

import logging
from typing import Dict, List, Optional, Type

from .base import Skill, SkillConfig, SkillContext, SkillResult, Tool

logger = logging.getLogger(__name__)


class SkillRegistry:
    """
    Registry for managing loaded skills.
    
    Features:
    - Register/unregister skills
    - Skill lookup by name
    - Intent matching across skills
    - Tool aggregation
    """
    
    def __init__(self):
        self._skills: Dict[str, Skill] = {}
        self._skill_types: Dict[str, Type[Skill]] = {}
    
    def register_skill_type(
        self,
        name: str,
        skill_class: Type[Skill]
    ) -> None:
        """Register a skill class for factory creation."""
        self._skill_types[name] = skill_class
        logger.info(f"Registered skill type: {name}")
    
    def create_skill(
        self,
        name: str,
        config: Optional[SkillConfig] = None
    ) -> Skill:
        """Create a skill instance from registered type."""
        skill_class = self._skill_types.get(name)
        if not skill_class:
            raise ValueError(f"Unknown skill type: {name}")
        
        return skill_class(config)
    
    async def register_skill(self, skill: Skill) -> None:
        """Register a skill instance."""
        if skill.name in self._skills:
            logger.warning(f"Replacing existing skill: {skill.name}")
            await self._skills[skill.name].cleanup()
        
        if not skill._initialized:
            await skill.initialize()
        
        self._skills[skill.name] = skill
        logger.info(f"Registered skill: {skill.skill_id}")
    
    async def unregister_skill(self, name: str) -> bool:
        """Unregister a skill."""
        skill = self._skills.get(name)
        if skill:
            await skill.cleanup()
            del self._skills[name]
            logger.info(f"Unregistered skill: {name}")
            return True
        return False
    
    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return self._skills.get(name)
    
    def list_skills(
        self,
        enabled_only: bool = True
    ) -> List[Skill]:
        """List all registered skills."""
        skills = list(self._skills.values())
        if enabled_only:
            skills = [s for s in skills if s.config.enabled]
        return skills
    
    def get_all_tools(self) -> List[Tool]:
        """Get all tools from all enabled skills."""
        tools = []
        for skill in self.list_skills(enabled_only=True):
            tools.extend(skill.tools)
        return tools
    
    def get_tools_for_openai(self) -> List[Dict]:
        """Get all tools in OpenAI format."""
        tools = []
        for skill in self.list_skills(enabled_only=True):
            tools.extend(skill.get_tools_for_openai())
        return tools
    
    def find_tool(self, tool_name: str) -> Optional[tuple[Skill, Tool]]:
        """Find a tool by name across all skills."""
        for skill in self.list_skills(enabled_only=True):
            tool = skill.get_tool(tool_name)
            if tool:
                return (skill, tool)
        return None
    
    async def execute_tool(
        self,
        tool_name: str,
        context: Optional[SkillContext] = None,
        **kwargs
    ) -> Optional[any]:
        """Execute a tool by name."""
        result = self.find_tool(tool_name)
        if not result:
            logger.warning(f"Tool not found: {tool_name}")
            return None
        
        skill, tool = result
        
        # Update skill context if provided
        if context and hasattr(skill, '_context'):
            skill._context = context
        
        return await tool.execute(**kwargs)
    
    def match_intent(
        self,
        intent: str,
        threshold: float = 0.5
    ) -> List[tuple[Skill, float]]:
        """
        Match an intent to skills.
        
        Returns list of (skill, score) tuples above threshold,
        sorted by score descending.
        """
        matches = []
        
        for skill in self.list_skills(enabled_only=True):
            score = skill.matches_intent(intent)
            if score >= threshold:
                matches.append((skill, score))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    async def execute_best_match(
        self,
        intent: str,
        context: SkillContext,
        threshold: float = 0.5
    ) -> Optional[SkillResult]:
        """
        Find and execute the best matching skill for an intent.
        """
        matches = self.match_intent(intent, threshold)
        
        if not matches:
            logger.debug(f"No skill match for intent: {intent[:50]}...")
            return None
        
        skill, score = matches[0]
        logger.info(f"Matched skill '{skill.name}' with score {score:.2f}")
        
        return await skill.execute(context)
    
    def get_skill_summaries(self) -> List[Dict]:
        """Get summaries of all registered skills."""
        return [skill.to_dict() for skill in self.list_skills()]
    
    async def cleanup_all(self) -> None:
        """Clean up all skills."""
        for skill in list(self._skills.values()):
            try:
                await skill.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up skill {skill.name}: {e}")
        self._skills.clear()


# Global registry instance
_registry: Optional[SkillRegistry] = None


def get_skill_registry() -> SkillRegistry:
    """Get or create the global skill registry."""
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
        
        # Register built-in skill types
        from .base import EchoSkill, MemorySkill
        _registry.register_skill_type("echo", EchoSkill)
        _registry.register_skill_type("memory", MemorySkill)
    
    return _registry
