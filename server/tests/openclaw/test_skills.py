"""Tests for OpenClaw Lite skill system."""
import pytest
from agents.skills import SkillRegistry, Skill


class TestSkill:
    """Test individual Skill objects."""
    
    def test_skill_creation(self):
        """Test creating a skill."""
        def handler(x):
            return x * 2
        
        skill = Skill(
            name="double",
            handler=handler,
            description="Doubles a number"
        )
        
        assert skill.name == "double"
        assert skill.description == "Doubles a number"
        assert callable(skill.handler)
    
    def test_skill_execution(self):
        """Test executing a skill."""
        skill = Skill(
            name="greet",
            handler=lambda name: f"Hello, {name}!",
            description="Greets a person"
        )
        
        result = skill.execute("Alice")
        assert result == "Hello, Alice!"
    
    def test_skill_with_metadata(self):
        """Test skill with additional metadata."""
        skill = Skill(
            name="test",
            handler=lambda: None,
            description="Test skill",
            metadata={
                "category": "utility",
                "version": "1.0",
                "author": "test"
            }
        )
        
        assert skill.metadata["category"] == "utility"
        assert skill.metadata["version"] == "1.0"


class TestSkillRegistry:
    """Test SkillRegistry for managing skills."""
    
    def test_registry_initialization(self):
        """Test registry starts empty."""
        registry = SkillRegistry()
        assert len(registry.list_skills()) == 0
    
    def test_register_skill(self):
        """Test registering a skill."""
        registry = SkillRegistry()
        
        def search(query):
            return f"Searching for: {query}"
        
        registry.register("search", search, description="Web search")
        
        assert "search" in registry.list_skills()
        assert registry.has_skill("search")
    
    def test_get_skill(self):
        """Test retrieving a registered skill."""
        registry = SkillRegistry()
        registry.register("calc", lambda x, y: x + y)
        
        skill = registry.get_skill("calc")
        assert skill is not None
        assert skill.execute(5, 3) == 8
    
    def test_unregister_skill(self):
        """Test removing a skill."""
        registry = SkillRegistry()
        registry.register("temp", lambda: None)
        
        assert registry.has_skill("temp")
        registry.unregister("temp")
        assert not registry.has_skill("temp")
    
    def test_execute_skill(self):
        """Test executing skill via registry."""
        registry = SkillRegistry()
        registry.register("multiply", lambda a, b: a * b)
        
        result = registry.execute("multiply", 4, 5)
        assert result == 20
    
    def test_skill_not_found(self):
        """Test accessing non-existent skill."""
        registry = SkillRegistry()
        
        with pytest.raises(KeyError):
            registry.get_skill("nonexistent")
    
    def test_list_skills_with_descriptions(self):
        """Test listing skills with metadata."""
        registry = SkillRegistry()
        registry.register("a", lambda: 1, description="First")
        registry.register("b", lambda: 2, description="Second")
        
        skills = registry.list_skills_with_info()
        assert len(skills) == 2
        assert any(s["name"] == "a" and s["description"] == "First" for s in skills)


class TestSkillLoading:
    """Test loading skills from external sources."""
    
    def test_load_from_dict(self):
        """Test loading skills from dict config."""
        registry = SkillRegistry()
        
        config = {
            "weather": {
                "handler": lambda city: f"Weather in {city}",
                "description": "Get weather"
            },
            "news": {
                "handler": lambda: "Latest news",
                "description": "Get news"
            }
        }
        
        registry.load_from_dict(config)
        
        assert registry.has_skill("weather")
        assert registry.has_skill("news")
        assert registry.execute("weather", "NYC") == "Weather in NYC"
    
    def test_skill_categories(self):
        """Test organizing skills by category."""
        registry = SkillRegistry()
        
        registry.register("search", lambda q: q, category="web")
        registry.register("calc", lambda x: x, category="math")
        registry.register("weather", lambda: None, category="web")
        
        web_skills = registry.get_by_category("web")
        assert len(web_skills) == 2
        assert "search" in [s.name for s in web_skills]
