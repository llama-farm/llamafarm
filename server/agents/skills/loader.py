"""
Skill Loader for loading skills from the filesystem.

Supports:
- Loading from Python modules
- Loading from skill directories (with SKILL.md)
- Hot reloading
"""

import importlib
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Type

from .base import Skill, SkillConfig
from .registry import SkillRegistry, get_skill_registry

logger = logging.getLogger(__name__)


class SkillLoader:
    """
    Loads skills from the filesystem.
    
    Skill Directory Structure:
        my_skill/
        ├── SKILL.md          # Description (optional but recommended)
        ├── skill.py          # Main skill implementation
        ├── config.json       # Default config (optional)
        └── requirements.txt  # Dependencies (optional)
    
    Or as a single Python file:
        my_skill.py  # Must define a Skill subclass
    """
    
    def __init__(
        self,
        registry: Optional[SkillRegistry] = None,
        skills_dirs: Optional[List[Path]] = None
    ):
        self.registry = registry or get_skill_registry()
        self.skills_dirs = skills_dirs or []
        self._loaded_modules: Dict[str, str] = {}  # skill_name -> module_path
    
    def add_skills_dir(self, path: Path) -> None:
        """Add a directory to search for skills."""
        if path.exists() and path.is_dir():
            self.skills_dirs.append(path)
            logger.info(f"Added skills directory: {path}")
    
    async def load_skill_from_path(
        self,
        path: Path,
        config: Optional[SkillConfig] = None
    ) -> Optional[Skill]:
        """
        Load a skill from a file or directory path.
        """
        if path.is_file() and path.suffix == ".py":
            return await self._load_from_file(path, config)
        elif path.is_dir():
            return await self._load_from_directory(path, config)
        else:
            logger.warning(f"Invalid skill path: {path}")
            return None
    
    async def _load_from_file(
        self,
        file_path: Path,
        config: Optional[SkillConfig] = None
    ) -> Optional[Skill]:
        """Load a skill from a Python file."""
        try:
            # Generate unique module name
            module_name = f"skill_{file_path.stem}_{id(file_path)}"
            
            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if not spec or not spec.loader:
                logger.error(f"Failed to load spec for: {file_path}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Find Skill subclass
            skill_class = None
            for name in dir(module):
                obj = getattr(module, name)
                if (
                    isinstance(obj, type) and
                    issubclass(obj, Skill) and
                    obj is not Skill
                ):
                    skill_class = obj
                    break
            
            if not skill_class:
                logger.warning(f"No Skill class found in: {file_path}")
                return None
            
            # Create instance
            skill = skill_class(config)
            await skill.initialize()
            
            # Register
            await self.registry.register_skill(skill)
            self._loaded_modules[skill.name] = str(file_path)
            
            logger.info(f"Loaded skill '{skill.name}' from {file_path}")
            return skill
            
        except Exception as e:
            logger.error(f"Failed to load skill from {file_path}: {e}")
            return None
    
    async def _load_from_directory(
        self,
        dir_path: Path,
        config: Optional[SkillConfig] = None
    ) -> Optional[Skill]:
        """Load a skill from a directory."""
        skill_py = dir_path / "skill.py"
        if not skill_py.exists():
            # Try __init__.py
            skill_py = dir_path / "__init__.py"
        
        if not skill_py.exists():
            logger.warning(f"No skill.py found in: {dir_path}")
            return None
        
        # Load config.json if exists and no config provided
        if not config:
            config_file = dir_path / "config.json"
            if config_file.exists():
                try:
                    config_data = json.loads(config_file.read_text())
                    config = SkillConfig(
                        enabled=config_data.get("enabled", True),
                        priority=config_data.get("priority", 5),
                        require_confirmation=config_data.get("require_confirmation", False),
                        allowed_agents=config_data.get("allowed_agents"),
                        metadata=config_data.get("metadata", {})
                    )
                except Exception as e:
                    logger.warning(f"Failed to load config.json: {e}")
        
        return await self._load_from_file(skill_py, config)
    
    async def discover_skills(self) -> List[Skill]:
        """
        Discover and load all skills from registered directories.
        """
        loaded = []
        
        for skills_dir in self.skills_dirs:
            if not skills_dir.exists():
                continue
            
            # Check for skill directories
            for item in skills_dir.iterdir():
                if item.is_dir() and not item.name.startswith((".", "_")):
                    skill = await self.load_skill_from_path(item)
                    if skill:
                        loaded.append(skill)
                
                elif item.is_file() and item.suffix == ".py" and not item.name.startswith("_"):
                    skill = await self.load_skill_from_path(item)
                    if skill:
                        loaded.append(skill)
        
        logger.info(f"Discovered {len(loaded)} skills from {len(self.skills_dirs)} directories")
        return loaded
    
    async def reload_skill(self, skill_name: str) -> Optional[Skill]:
        """
        Reload a skill from its original path.
        
        Useful for hot reloading during development.
        """
        module_path = self._loaded_modules.get(skill_name)
        if not module_path:
            logger.warning(f"No module path for skill: {skill_name}")
            return None
        
        # Unregister existing
        await self.registry.unregister_skill(skill_name)
        
        # Reload
        return await self.load_skill_from_path(Path(module_path))
    
    def get_skill_path(self, skill_name: str) -> Optional[Path]:
        """Get the file path for a loaded skill."""
        path_str = self._loaded_modules.get(skill_name)
        return Path(path_str) if path_str else None
    
    @staticmethod
    def parse_skill_md(path: Path) -> Dict[str, str]:
        """
        Parse SKILL.md for metadata.
        
        Expected format:
        ```
        # Skill Name
        
        Description of the skill.
        
        ## Configuration
        
        ...
        ```
        """
        if not path.exists():
            return {}
        
        content = path.read_text()
        result = {}
        
        lines = content.split("\n")
        
        # First # heading is the name
        for line in lines:
            if line.startswith("# "):
                result["name"] = line[2:].strip()
                break
        
        # First paragraph after name is description
        in_description = False
        description_lines = []
        for line in lines:
            if line.startswith("# "):
                in_description = True
                continue
            if in_description:
                if line.startswith("##"):
                    break
                if line.strip():
                    description_lines.append(line.strip())
        
        if description_lines:
            result["description"] = " ".join(description_lines)
        
        return result


# Convenience function for loading skills
async def load_skills_from_directory(
    path: Path,
    registry: Optional[SkillRegistry] = None
) -> List[Skill]:
    """
    Load all skills from a directory.
    
    Usage:
        skills = await load_skills_from_directory(Path("./skills"))
    """
    loader = SkillLoader(registry=registry)
    loader.add_skills_dir(path)
    return await loader.discover_skills()
