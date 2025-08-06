"""
Strategy Configuration Classes

Defines the data structures for strategy configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class PerformancePriority(Enum):
    """Performance priority levels for strategy optimization."""
    SPEED = "speed"
    ACCURACY = "accuracy" 
    BALANCED = "balanced"


class ResourceUsage(Enum):
    """Resource usage levels for strategy selection."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Complexity(Enum):
    """Strategy complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class ComponentConfig:
    """Configuration for a single RAG component."""
    type: str
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyComponents:
    """All components that make up a strategy."""
    parser: ComponentConfig
    extractors: List[ComponentConfig] = field(default_factory=list)
    embedder: ComponentConfig = field(default_factory=lambda: ComponentConfig("OllamaEmbedder"))
    vector_store: ComponentConfig = field(default_factory=lambda: ComponentConfig("ChromaStore"))
    retrieval_strategy: ComponentConfig = field(default_factory=lambda: ComponentConfig("BasicSimilarityStrategy"))


@dataclass
class StrategyConfig:
    """Complete strategy configuration."""
    name: str
    description: str
    use_cases: List[str]
    performance_priority: PerformancePriority
    resource_usage: ResourceUsage
    complexity: Complexity
    components: StrategyComponents
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy config to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "use_cases": self.use_cases,
            "performance_priority": self.performance_priority.value,
            "resource_usage": self.resource_usage.value,
            "complexity": self.complexity.value,
            "components": {
                "parser": {
                    "type": self.components.parser.type,
                    "config": self.components.parser.config
                },
                "extractors": [
                    {
                        "type": extractor.type,
                        "config": extractor.config
                    }
                    for extractor in self.components.extractors
                ],
                "embedder": {
                    "type": self.components.embedder.type,
                    "config": self.components.embedder.config
                },
                "vector_store": {
                    "type": self.components.vector_store.type,
                    "config": self.components.vector_store.config
                },
                "retrieval_strategy": {
                    "type": self.components.retrieval_strategy.type,
                    "config": self.components.retrieval_strategy.config
                }
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyConfig':
        """Create strategy config from dictionary."""
        components_data = data["components"]
        
        # Parse extractors
        extractors = []
        for extractor_data in components_data.get("extractors", []):
            extractors.append(ComponentConfig(
                type=extractor_data["type"],
                config=extractor_data.get("config", {})
            ))
        
        # Create components
        components = StrategyComponents(
            parser=ComponentConfig(
                type=components_data["parser"]["type"],
                config=components_data["parser"].get("config", {})
            ),
            extractors=extractors,
            embedder=ComponentConfig(
                type=components_data["embedder"]["type"],
                config=components_data["embedder"].get("config", {})
            ),
            vector_store=ComponentConfig(
                type=components_data["vector_store"]["type"],
                config=components_data["vector_store"].get("config", {})
            ),
            retrieval_strategy=ComponentConfig(
                type=components_data["retrieval_strategy"]["type"],
                config=components_data["retrieval_strategy"].get("config", {})
            )
        )
        
        return cls(
            name=data["name"],
            description=data["description"],
            use_cases=data["use_cases"],
            performance_priority=PerformancePriority(data["performance_priority"]),
            resource_usage=ResourceUsage(data["resource_usage"]),
            complexity=Complexity(data["complexity"]),
            components=components
        )
    
    def apply_overrides(self, overrides: Dict[str, Any]) -> 'StrategyConfig':
        """Apply configuration overrides to create a new strategy config."""
        # Deep copy the current config
        config_dict = self.to_dict()
        
        # Apply overrides recursively
        def deep_update(base_dict, override_dict):
            for key, value in override_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        if "components" in overrides:
            deep_update(config_dict["components"], overrides["components"])
        
        # Apply top-level overrides
        for key, value in overrides.items():
            if key != "components":
                config_dict[key] = value
        
        return StrategyConfig.from_dict(config_dict)