"""Configuration models for the prompts system."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator

from .template import PromptTemplate
from .strategy import PromptStrategy


class GlobalPromptConfig(BaseModel):
    """Configuration for high-level global prompts."""
    
    global_id: str = Field(..., description="Unique global prompt identifier")
    name: str = Field(..., description="Human-readable name")
    description: Optional[str] = Field(None, description="Global prompt description")
    
    # The global prompt content
    system_prompt: Optional[str] = Field(
        None,
        description="System-level prompt applied to all templates"
    )
    prefix_prompt: Optional[str] = Field(
        None,
        description="Prompt added before template content"
    )
    suffix_prompt: Optional[str] = Field(
        None,
        description="Prompt added after template content"
    )
    
    # Application rules
    applies_to: List[str] = Field(
        default_factory=lambda: ["*"],
        description="Template IDs or patterns this applies to (* for all)"
    )
    excludes: List[str] = Field(
        default_factory=list,
        description="Template IDs to exclude from this global prompt"
    )
    
    # Conditional application
    conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Conditions for when to apply this global prompt"
    )
    
    # Priority and ordering
    priority: int = Field(
        default=100,
        description="Application priority (lower = applied first)"
    )
    enabled: bool = Field(default=True, description="Whether this global prompt is active")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Tags for organization")
    created_by: Optional[str] = Field(None, description="Creator of this global prompt")
    
    def applies_to_template(self, template_id: str, context: Dict[str, Any] = None) -> bool:
        """Check if this global prompt applies to a specific template."""
        if not self.enabled:
            return False
        
        # Check exclusions first
        if template_id in self.excludes:
            return False
        
        # Check if explicitly included
        if template_id in self.applies_to:
            return True
        
        # Check for wildcard patterns
        import fnmatch
        for pattern in self.applies_to:
            if fnmatch.fnmatch(template_id, pattern):
                # Check conditions if specified
                if self.conditions and context:
                    return self._evaluate_conditions(context)
                return True
        
        return False
    
    def _evaluate_conditions(self, context: Dict[str, Any]) -> bool:
        """Evaluate conditions for applying this global prompt."""
        for key, expected_value in self.conditions.items():
            context_value = context.get(key)
            
            if isinstance(expected_value, list):
                if context_value not in expected_value:
                    return False
            elif context_value != expected_value:
                return False
        
        return True


class PromptConfig(BaseModel):
    """Main configuration for the prompts system."""
    
    name: str = Field(..., description="Configuration name")
    version: str = Field(default="1.0.0", description="Configuration version")
    description: Optional[str] = Field(None, description="Configuration description")
    
    # Core system settings
    enabled: bool = Field(default=True, description="Whether prompts system is enabled")
    default_strategy: str = Field(
        default="default_strategy",
        description="Default strategy ID to use"
    )
    
    # Global prompts (high-level prompts applied to all templates)
    global_prompts: List[GlobalPromptConfig] = Field(
        default_factory=list,
        description="Global prompts applied across templates"
    )
    
    # Template definitions
    templates: Dict[str, PromptTemplate] = Field(
        default_factory=dict,
        description="Template definitions by ID"
    )
    
    # Strategy definitions
    strategies: Dict[str, PromptStrategy] = Field(
        default_factory=dict,
        description="Strategy definitions by ID"
    )
    
    # System behavior
    fallback_behavior: Dict[str, Any] = Field(
        default_factory=dict,
        description="System-wide fallback behavior"
    )
    
    # Integration settings
    integrations: Dict[str, Any] = Field(
        default_factory=dict,
        description="Integration configurations (RAG, LangGraph, etc.)"
    )
    
    # Performance and monitoring
    monitoring: Dict[str, Any] = Field(
        default_factory=dict,
        description="Monitoring and performance settings"
    )
    
    # Environment-specific settings
    environments: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Environment-specific overrides"
    )
    
    @validator("templates", pre=True)
    def parse_templates(cls, v):
        """Parse templates from dict format."""
        if isinstance(v, dict):
            parsed_templates = {}
            for template_id, template_data in v.items():
                if isinstance(template_data, PromptTemplate):
                    parsed_templates[template_id] = template_data
                elif isinstance(template_data, dict):
                    # Ensure template_id is set
                    if "template_id" not in template_data:
                        template_data["template_id"] = template_id
                    parsed_templates[template_id] = PromptTemplate(**template_data)
                else:
                    raise ValueError(f"Invalid template data for {template_id}")
            return parsed_templates
        return v
    
    @validator("strategies", pre=True)
    def parse_strategies(cls, v):
        """Parse strategies from dict format."""
        if isinstance(v, dict):
            parsed_strategies = {}
            for strategy_id, strategy_data in v.items():
                if isinstance(strategy_data, PromptStrategy):
                    parsed_strategies[strategy_id] = strategy_data
                elif isinstance(strategy_data, dict):
                    # Ensure strategy_id is set
                    if "strategy_id" not in strategy_data:
                        strategy_data["strategy_id"] = strategy_id
                    parsed_strategies[strategy_id] = PromptStrategy(**strategy_data)
                else:
                    raise ValueError(f"Invalid strategy data for {strategy_id}")
            return parsed_strategies
        return v
    
    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get a template by ID."""
        return self.templates.get(template_id)
    
    def get_strategy(self, strategy_id: str) -> Optional[PromptStrategy]:
        """Get a strategy by ID."""
        return self.strategies.get(strategy_id)
    
    def get_global_prompts_for_template(
        self, 
        template_id: str, 
        context: Dict[str, Any] = None
    ) -> List[GlobalPromptConfig]:
        """Get all global prompts that apply to a specific template."""
        applicable_globals = []
        
        for global_prompt in self.global_prompts:
            if global_prompt.applies_to_template(template_id, context):
                applicable_globals.append(global_prompt)
        
        # Sort by priority (lower = higher priority)
        return sorted(applicable_globals, key=lambda g: g.priority)
    
    def add_template(self, template: PromptTemplate) -> None:
        """Add a template to the configuration."""
        self.templates[template.template_id] = template
    
    def remove_template(self, template_id: str) -> bool:
        """Remove a template from the configuration."""
        if template_id in self.templates:
            del self.templates[template_id]
            return True
        return False
    
    def add_strategy(self, strategy: PromptStrategy) -> None:
        """Add a strategy to the configuration."""
        self.strategies[strategy.strategy_id] = strategy
    
    def remove_strategy(self, strategy_id: str) -> bool:
        """Remove a strategy from the configuration."""
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
            return True
        return False
    
    def add_global_prompt(self, global_prompt: GlobalPromptConfig) -> None:
        """Add a global prompt to the configuration."""
        self.global_prompts.append(global_prompt)
    
    def remove_global_prompt(self, global_id: str) -> bool:
        """Remove a global prompt from the configuration."""
        original_count = len(self.global_prompts)
        self.global_prompts = [g for g in self.global_prompts if g.global_id != global_id]
        return len(self.global_prompts) < original_count
    
    def validate_config(self) -> List[str]:
        """Validate the entire configuration and return any errors."""
        errors = []
        
        # Check that default strategy exists
        if self.default_strategy not in self.strategies:
            errors.append(f"Default strategy '{self.default_strategy}' not found")
        
        # Validate each strategy
        for strategy_id, strategy in self.strategies.items():
            strategy_errors = strategy.validate_config()
            for error in strategy_errors:
                errors.append(f"Strategy '{strategy_id}': {error}")
        
        # Check for template references in strategies
        for strategy_id, strategy in self.strategies.items():
            # Check fallback templates
            if strategy.fallback_template and strategy.fallback_template not in self.templates:
                errors.append(
                    f"Strategy '{strategy_id}' references unknown fallback template "
                    f"'{strategy.fallback_template}'"
                )
            
            # Check rule templates
            for rule in strategy.rules:
                if rule.template_id not in self.templates:
                    errors.append(
                        f"Strategy '{strategy_id}' rule '{rule.rule_id}' references "
                        f"unknown template '{rule.template_id}'"
                    )
        
        # Validate global prompt IDs are unique
        global_ids = [g.global_id for g in self.global_prompts]
        if len(global_ids) != len(set(global_ids)):
            errors.append("Global prompt IDs must be unique")
        
        return errors
    
    def apply_environment_overrides(self, environment: str) -> None:
        """Apply environment-specific overrides."""
        if environment in self.environments:
            overrides = self.environments[environment]
            
            # Apply overrides to configuration
            for key, value in overrides.items():
                if hasattr(self, key):
                    if isinstance(getattr(self, key), dict) and isinstance(value, dict):
                        getattr(self, key).update(value)
                    else:
                        setattr(self, key, value)
    
    @classmethod
    def from_file(cls, file_path: str, environment: Optional[str] = None) -> "PromptConfig":
        """Load configuration from file."""
        import json
        import yaml
        from pathlib import Path
        
        path = Path(file_path)
        
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        config = cls(**data)
        
        if environment:
            config.apply_environment_overrides(environment)
        
        return config
    
    def save_to_file(self, file_path: str) -> None:
        """Save configuration to file."""
        import json
        import yaml
        from pathlib import Path
        
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and handle serialization
        data = self.dict()
        
        with open(path, 'w', encoding='utf-8') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            else:
                json.dump(data, f, indent=2, default=str)