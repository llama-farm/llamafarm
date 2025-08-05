"""
Strategy Configuration Models

Defines the configuration structures for prompt strategies.
"""

from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator


class PerformanceProfile(str, Enum):
    """Performance optimization profiles."""
    SPEED = "speed"
    ACCURACY = "accuracy"
    BALANCED = "balanced"
    COST_OPTIMIZED = "cost_optimized"


class Complexity(str, Enum):
    """Strategy complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class TransformType(str, Enum):
    """Input/output transformation types."""
    # Input transforms
    LOWERCASE = "lowercase"
    UPPERCASE = "uppercase"
    TRIM = "trim"
    TRUNCATE = "truncate"
    CLEAN_WHITESPACE = "clean_whitespace"
    ESCAPE_HTML = "escape_html"
    LIMIT_LAST_N = "limit_last_n"
    SORT_BY_RELEVANCE = "sort_by_relevance"
    LIMIT_TOKENS = "limit_tokens"
    
    # Output transforms
    PARSE_JSON = "parse_json"
    EXTRACT_CODE = "extract_code"
    CLEAN_MARKDOWN = "clean_markdown"
    VALIDATE_FORMAT = "validate_format"
    HIGHLIGHT_ISSUES = "highlight_issues"


class Transform(BaseModel):
    """Transform configuration."""
    transform: TransformType
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)


class InputTransform(Transform):
    """Input-specific transform with target variable."""
    input: str = Field(..., description="Input variable name to transform")


class TemplateConfig(BaseModel):
    """Configuration for a template within a strategy."""
    template: str = Field(..., description="Template ID to use")
    config: Dict[str, Any] = Field(default_factory=dict, description="Template-specific configuration")
    input_transforms: List[InputTransform] = Field(default_factory=list)
    output_transforms: List[Transform] = Field(default_factory=list)


class ConditionConfig(BaseModel):
    """Condition configuration for template selection."""
    query_type: Optional[str] = None
    context_size: Optional[Dict[str, int]] = None  # min/max
    has_context: Optional[bool] = None
    input_length: Optional[Dict[str, int]] = None  # min/max
    custom: Optional[Dict[str, Any]] = None


class SpecializedTemplate(TemplateConfig):
    """Specialized template with conditions."""
    condition: ConditionConfig
    priority: int = Field(default=0, description="Priority when multiple conditions match")


class SelectionRule(BaseModel):
    """Rule for automatic template selection."""
    name: str = Field(..., description="Rule name for debugging")
    condition: Dict[str, Any] = Field(..., description="Conditions to evaluate")
    template: str = Field(..., description="Template ID to use when rule matches")
    priority: int = Field(default=0, description="Rule priority (higher wins)")
    stop_on_match: bool = Field(default=False, description="Stop evaluating further rules")


class SystemPrompt(BaseModel):
    """System prompt configuration."""
    content: str
    position: str = Field(default="system", pattern="^(prefix|suffix|system)$")
    priority: int = Field(default=0)


class GlobalConfig(BaseModel):
    """Global configuration for all templates in a strategy."""
    system_prompts: List[SystemPrompt] = Field(default_factory=list)
    temperature: Optional[float] = Field(None, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1)
    model_preferences: List[str] = Field(default_factory=list)


class OptimizationConfig(BaseModel):
    """Performance optimization settings."""
    caching: bool = Field(default=False)
    compression: bool = Field(default=False)
    token_optimization: bool = Field(default=True)
    parallel_processing: bool = Field(default=False)


class TemplatesConfig(BaseModel):
    """Templates configuration within a strategy."""
    default: TemplateConfig
    fallback: Optional[TemplateConfig] = None
    specialized: List[SpecializedTemplate] = Field(default_factory=list)


class StrategyMetadata(BaseModel):
    """Strategy metadata."""
    version: str = Field(default="1.0.0")
    author: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class StrategyConfig(BaseModel):
    """Complete strategy configuration."""
    name: str
    description: str
    use_cases: List[str] = Field(default_factory=list)
    templates: TemplatesConfig
    selection_rules: List[SelectionRule] = Field(default_factory=list)
    input_transforms: List[InputTransform] = Field(default_factory=list)
    output_transforms: List[Transform] = Field(default_factory=list)
    global_config: GlobalConfig = Field(default_factory=GlobalConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    performance_profile: PerformanceProfile = Field(default=PerformanceProfile.BALANCED)
    complexity: Complexity = Field(default=Complexity.MODERATE)
    metadata: StrategyMetadata = Field(default_factory=StrategyMetadata)
    
    @validator('selection_rules')
    def validate_selection_rules(cls, v, values):
        """Ensure selection rules reference valid templates."""
        if 'templates' in values:
            template_ids = {values['templates'].default.template}
            if values['templates'].fallback:
                template_ids.add(values['templates'].fallback.template)
            for spec in values['templates'].specialized:
                template_ids.add(spec.template)
            
            for rule in v:
                if rule.template not in template_ids:
                    # Allow external template references
                    pass
        return v
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyConfig':
        """Create StrategyConfig from dictionary."""
        # Handle templates section specially
        if 'templates' in data and isinstance(data['templates'], dict):
            if 'default' not in data['templates']:
                raise ValueError("Strategy must have a default template")
            
            # Convert templates to proper structure
            templates_data = data['templates']
            if isinstance(templates_data['default'], dict) and 'template' not in templates_data['default']:
                # Simple format: just template ID
                templates_data['default'] = {'template': templates_data['default']}
            
            data['templates'] = TemplatesConfig(**templates_data)
        
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.dict(exclude_none=True)
    
    def get_template_for_context(self, context: Dict[str, Any]) -> Optional[TemplateConfig]:
        """Get the best template for the given context."""
        # First check selection rules
        for rule in sorted(self.selection_rules, key=lambda r: r.priority, reverse=True):
            if self._evaluate_condition(rule.condition, context):
                # Find template config by ID
                if rule.template == self.templates.default.template:
                    return self.templates.default
                if self.templates.fallback and rule.template == self.templates.fallback.template:
                    return self.templates.fallback
                for spec in self.templates.specialized:
                    if rule.template == spec.template:
                        return spec
                
                if rule.stop_on_match:
                    break
        
        # Check specialized templates
        matching_specialized = []
        for spec in self.templates.specialized:
            if self._evaluate_condition(spec.condition.dict(exclude_none=True), context):
                matching_specialized.append(spec)
        
        if matching_specialized:
            # Return highest priority match
            return max(matching_specialized, key=lambda s: s.priority)
        
        # Return default
        return self.templates.default
    
    def _evaluate_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate if a condition matches the context."""
        if 'expression' in condition:
            # Simple expression evaluation (can be extended)
            return eval(condition['expression'], {'context': context})
        
        if 'match_all' in condition:
            return all(self._evaluate_condition(c, context) for c in condition['match_all'])
        
        if 'match_any' in condition:
            return any(self._evaluate_condition(c, context) for c in condition['match_any'])
        
        # Direct field matching
        for key, value in condition.items():
            if key == 'query_type' and context.get('query_type') != value:
                return False
            elif key == 'has_context' and bool(context.get('context')) != value:
                return False
            elif key == 'context_size' and 'context' in context:
                size = len(context['context'])
                if 'min' in value and size < value['min']:
                    return False
                if 'max' in value and size > value['max']:
                    return False
            elif key == 'input_length' and 'query' in context:
                length = len(context['query'])
                if 'min' in value and length < value['min']:
                    return False
                if 'max' in value and length > value['max']:
                    return False
            elif key == 'custom':
                for custom_key, custom_value in value.items():
                    if context.get(custom_key) != custom_value:
                        return False
        
        return True