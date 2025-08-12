"""
Strategy Configuration Classes

Defines the data structures for strategy configurations compatible with
the unified strategy schema format.
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
    # New fields for extractors
    priority: Optional[int] = None
    enabled: bool = True


@dataclass
class OptimizationConfig:
    """Optimization settings for strategy."""
    performance_priority: str = "balanced"
    resource_constraints: Dict[str, Any] = field(default_factory=dict)
    batch_settings: Dict[str, Any] = field(default_factory=dict)
    caching: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationConfig:
    """Validation rules for documents."""
    max_document_length: Optional[int] = None
    min_document_length: Optional[int] = None
    required_metadata: List[str] = field(default_factory=list)
    content_filters: List[str] = field(default_factory=list)


@dataclass
class MonitoringConfig:
    """Monitoring and observability settings."""
    metrics_enabled: bool = False
    tracing_enabled: bool = False
    log_level: str = "INFO"
    export_metrics: bool = False


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
    """Complete strategy configuration compatible with unified schema."""
    name: str
    description: str
    tags: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)
    components: StrategyComponents = field(default_factory=StrategyComponents)
    
    # Optional fields from new schema
    optimization: Optional[OptimizationConfig] = None
    validation: Optional[ValidationConfig] = None
    monitoring: Optional[MonitoringConfig] = None
    
    # Legacy fields for backward compatibility
    performance_priority: Optional[PerformancePriority] = None
    resource_usage: Optional[ResourceUsage] = None
    complexity: Optional[Complexity] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy config to dictionary format."""
        result = {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "use_cases": self.use_cases,
            "components": {
                "parser": {
                    "type": self.components.parser.type,
                    "config": self.components.parser.config
                }
            }
        }
        
        # Add extractors if present
        if self.components.extractors:
            extractors_list = []
            for extractor in self.components.extractors:
                ext_dict = {
                    "type": extractor.type,
                    "config": extractor.config
                }
                if extractor.priority is not None:
                    ext_dict["priority"] = extractor.priority
                if not extractor.enabled:
                    ext_dict["enabled"] = extractor.enabled
                extractors_list.append(ext_dict)
            result["components"]["extractors"] = extractors_list
        
        # Add other components
        result["components"]["embedder"] = {
            "type": self.components.embedder.type,
            "config": self.components.embedder.config
        }
        result["components"]["vector_store"] = {
            "type": self.components.vector_store.type,
            "config": self.components.vector_store.config
        }
        result["components"]["retrieval_strategy"] = {
            "type": self.components.retrieval_strategy.type,
            "config": self.components.retrieval_strategy.config
        }
        
        # Add optional sections if present
        if self.optimization:
            result["optimization"] = {
                "performance_priority": self.optimization.performance_priority,
                "resource_constraints": self.optimization.resource_constraints,
                "batch_settings": self.optimization.batch_settings,
                "caching": self.optimization.caching
            }
        
        if self.validation:
            result["validation"] = {
                "max_document_length": self.validation.max_document_length,
                "min_document_length": self.validation.min_document_length,
                "required_metadata": self.validation.required_metadata,
                "content_filters": self.validation.content_filters
            }
        
        if self.monitoring:
            result["monitoring"] = {
                "metrics_enabled": self.monitoring.metrics_enabled,
                "tracing_enabled": self.monitoring.tracing_enabled,
                "log_level": self.monitoring.log_level,
                "export_metrics": self.monitoring.export_metrics
            }
        
        # Add legacy fields if present (for backward compatibility)
        if self.performance_priority:
            result["performance_priority"] = self.performance_priority.value
        if self.resource_usage:
            result["resource_usage"] = self.resource_usage.value
        if self.complexity:
            result["complexity"] = self.complexity.value
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyConfig':
        """Create strategy config from dictionary (supports both old and new formats)."""
        # Handle components
        components_data = data.get("components", {})
        
        # Parse parser
        parser_data = components_data.get("parser", {})
        parser = ComponentConfig(
            type=parser_data.get("type", "PlainTextParser"),
            config=parser_data.get("config", {})
        )
        
        # Parse extractors
        extractors = []
        for extractor_data in components_data.get("extractors", []):
            extractors.append(ComponentConfig(
                type=extractor_data["type"],
                config=extractor_data.get("config", {}),
                priority=extractor_data.get("priority"),
                enabled=extractor_data.get("enabled", True)
            ))
        
        # Parse embedder
        embedder_data = components_data.get("embedder", {})
        embedder = ComponentConfig(
            type=embedder_data.get("type", "OllamaEmbedder"),
            config=embedder_data.get("config", {})
        )
        
        # Parse vector_store
        store_data = components_data.get("vector_store", {})
        vector_store = ComponentConfig(
            type=store_data.get("type", "ChromaStore"),
            config=store_data.get("config", {})
        )
        
        # Parse retrieval_strategy
        retrieval_data = components_data.get("retrieval_strategy", {})
        retrieval_strategy = ComponentConfig(
            type=retrieval_data.get("type", "BasicSimilarityStrategy"),
            config=retrieval_data.get("config", {})
        )
        
        # Create components
        components = StrategyComponents(
            parser=parser,
            extractors=extractors,
            embedder=embedder,
            vector_store=vector_store,
            retrieval_strategy=retrieval_strategy
        )
        
        # Parse optional sections
        optimization = None
        if "optimization" in data:
            opt_data = data["optimization"]
            optimization = OptimizationConfig(
                performance_priority=opt_data.get("performance_priority", "balanced"),
                resource_constraints=opt_data.get("resource_constraints", {}),
                batch_settings=opt_data.get("batch_settings", {}),
                caching=opt_data.get("caching", {})
            )
        
        validation = None
        if "validation" in data:
            val_data = data["validation"]
            validation = ValidationConfig(
                max_document_length=val_data.get("max_document_length"),
                min_document_length=val_data.get("min_document_length"),
                required_metadata=val_data.get("required_metadata", []),
                content_filters=val_data.get("content_filters", [])
            )
        
        monitoring = None
        if "monitoring" in data:
            mon_data = data["monitoring"]
            monitoring = MonitoringConfig(
                metrics_enabled=mon_data.get("metrics_enabled", False),
                tracing_enabled=mon_data.get("tracing_enabled", False),
                log_level=mon_data.get("log_level", "INFO"),
                export_metrics=mon_data.get("export_metrics", False)
            )
        
        # Handle legacy fields for backward compatibility
        performance_priority = None
        if "performance_priority" in data:
            try:
                performance_priority = PerformancePriority(data["performance_priority"])
            except (ValueError, KeyError):
                pass
        
        resource_usage = None
        if "resource_usage" in data:
            try:
                resource_usage = ResourceUsage(data["resource_usage"])
            except (ValueError, KeyError):
                pass
        
        complexity = None
        if "complexity" in data:
            try:
                complexity = Complexity(data["complexity"])
            except (ValueError, KeyError):
                pass
        
        return cls(
            name=data.get("name", "unnamed_strategy"),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            use_cases=data.get("use_cases", []),
            components=components,
            optimization=optimization,
            validation=validation,
            monitoring=monitoring,
            performance_priority=performance_priority,
            resource_usage=resource_usage,
            complexity=complexity
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
        
        # Apply all overrides
        deep_update(config_dict, overrides)
        
        return StrategyConfig.from_dict(config_dict)