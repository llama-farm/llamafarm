"""Prompt template data models."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator


class TemplateType(str, Enum):
    """Types of prompt templates."""
    BASIC = "basic"
    CHAT = "chat"
    FEW_SHOT = "few_shot"
    ADVANCED = "advanced"
    DOMAIN_SPECIFIC = "domain_specific"
    AGENTIC = "agentic"
    LANGGRAPH_WORKFLOW = "langgraph_workflow"


class TemplateComplexity(str, Enum):
    """Template complexity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXPERT = "expert"


class TemplateMetadata(BaseModel):
    """Metadata for prompt templates."""
    use_case: str = Field(..., description="Primary use case for this template")
    complexity: TemplateComplexity = Field(default=TemplateComplexity.LOW)
    domain: str = Field(default="general", description="Domain or subject area")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    author: Optional[str] = Field(None, description="Template author")
    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(default_factory=datetime.now)
    version: str = Field(default="1.0.0", description="Template version")
    description: Optional[str] = Field(None, description="Template description")
    examples: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Usage examples"
    )
    performance_notes: Optional[str] = Field(
        None, 
        description="Performance characteristics and optimization notes"
    )


class PromptTemplate(BaseModel):
    """Core prompt template model."""
    
    template_id: str = Field(..., description="Unique template identifier")
    name: str = Field(..., description="Human-readable template name")
    type: TemplateType = Field(..., description="Template type/category")
    template: str = Field(..., description="The actual prompt template string")
    input_variables: List[str] = Field(
        default_factory=list, 
        description="Required input variables"
    )
    optional_variables: List[str] = Field(
        default_factory=list,
        description="Optional input variables with defaults"
    )
    metadata: TemplateMetadata = Field(
        default_factory=TemplateMetadata,
        description="Template metadata"
    )
    
    # Advanced features
    global_prompts: List[str] = Field(
        default_factory=list,
        description="Global prompt IDs to apply to this template"
    )
    preprocessing_steps: List[str] = Field(
        default_factory=list,
        description="Preprocessing functions to apply"
    )
    postprocessing_steps: List[str] = Field(
        default_factory=list, 
        description="Postprocessing functions to apply"
    )
    validation_rules: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input validation rules"
    )
    
    # LangGraph integration
    langgraph_config: Optional[Dict[str, Any]] = Field(
        None,
        description="LangGraph workflow configuration"
    )
    
    # Context and conditions
    conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Conditions for template selection"
    )
    fallback_templates: List[str] = Field(
        default_factory=list,
        description="Fallback template IDs if this template fails"
    )
    
    @validator("template_id")
    def validate_template_id(cls, v):
        """Validate template ID format."""
        if not v or not isinstance(v, str):
            raise ValueError("template_id must be a non-empty string")
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("template_id must contain only letters, numbers, underscores, and hyphens")
        return v
    
    @validator("input_variables", "optional_variables")
    def validate_variables(cls, v):
        """Validate variable names."""
        for var in v:
            if not isinstance(var, str) or not var.replace("_", "").isalnum():
                raise ValueError(f"Variable name '{var}' is invalid")
        return v
    
    def render(self, variables: Dict[str, Any], **kwargs) -> str:
        """Render the template with provided variables."""
        from jinja2 import Template
        
        # Merge variables with kwargs
        all_vars = {**variables, **kwargs}
        
        # Validate required variables
        missing_vars = set(self.input_variables) - set(all_vars.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        # Create Jinja2 template
        jinja_template = Template(self.template)
        
        # Render with all variables
        return jinja_template.render(**all_vars)
    
    def validate_inputs(self, variables: Dict[str, Any]) -> bool:
        """Validate input variables against validation rules."""
        for var_name, rules in self.validation_rules.items():
            if var_name in variables:
                value = variables[var_name]
                
                # Type validation
                if "type" in rules:
                    expected_type_str = rules["type"]
                    # Map string type names to actual Python types
                    type_mapping = {
                        "str": str,
                        "int": int,
                        "float": float,
                        "bool": bool,
                        "list": list,
                        "dict": dict,
                        "tuple": tuple,
                        "set": set
                    }
                    
                    if expected_type_str in type_mapping:
                        expected_type = type_mapping[expected_type_str]
                        if not isinstance(value, expected_type):
                            raise ValueError(
                                f"Variable '{var_name}' must be of type {expected_type_str}"
                            )
                    else:
                        # For unknown types, just skip validation
                        pass
                
                # Length validation for strings
                if isinstance(value, str):
                    if "min_length" in rules and len(value) < rules["min_length"]:
                        raise ValueError(
                            f"Variable '{var_name}' must be at least {rules['min_length']} characters"
                        )
                    if "max_length" in rules and len(value) > rules["max_length"]:
                        raise ValueError(
                            f"Variable '{var_name}' must be at most {rules['max_length']} characters"
                        )
                
                # Range validation for numbers
                if isinstance(value, (int, float)):
                    if "min_value" in rules and value < rules["min_value"]:
                        raise ValueError(
                            f"Variable '{var_name}' must be at least {rules['min_value']}"
                        )
                    if "max_value" in rules and value > rules["max_value"]:
                        raise ValueError(
                            f"Variable '{var_name}' must be at most {rules['max_value']}"
                        )
        
        return True
    
    def get_required_variables(self) -> List[str]:
        """Get list of all required variables."""
        return self.input_variables.copy()
    
    def get_all_variables(self) -> List[str]:
        """Get list of all variables (required + optional)."""
        return self.input_variables + self.optional_variables
    
    def is_compatible_with(self, context: Dict[str, Any]) -> bool:
        """Check if template is compatible with given context."""
        # Check conditions
        for condition_key, condition_value in self.conditions.items():
            context_value = context.get(condition_key)
            
            if isinstance(condition_value, list):
                if context_value not in condition_value:
                    return False
            elif context_value != condition_value:
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Create template from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_file(cls, file_path: str) -> "PromptTemplate":
        """Load template from JSON/YAML file."""
        import json
        import yaml
        from pathlib import Path
        
        path = Path(file_path)
        
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls.from_dict(data)
    
    def save_to_file(self, file_path: str) -> None:
        """Save template to JSON/YAML file."""
        import json
        import yaml
        from pathlib import Path
        
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        with open(path, 'w', encoding='utf-8') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            else:
                json.dump(data, f, indent=2, default=str)