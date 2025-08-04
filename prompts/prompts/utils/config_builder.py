"""Configuration builder utility for creating prompt system configurations."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ..models.config import PromptConfig, GlobalPromptConfig
from ..models.strategy import PromptStrategy, StrategyType, StrategyRule, RuleOperator
from .template_loader import TemplateLoader


class ConfigBuilder:
    """
    Utility class for building prompt system configurations from
    templates and other components.
    """
    
    def __init__(self, templates_dir: str = "templates"):
        """Initialize the config builder."""
        self.template_loader = TemplateLoader(templates_dir)
        self.config_data = {
            "name": "Generated Prompts Configuration",
            "version": "1.0.0",
            "enabled": True,
            "default_strategy": "context_aware_strategy",
            "global_prompts": [],
            "templates": {},
            "strategies": {},
            "fallback_behavior": {},
            "integrations": {},
            "monitoring": {}
        }
    
    def load_templates(self) -> "ConfigBuilder":
        """Load all templates from the templates directory."""
        templates = self.template_loader.load_all_templates()
        
        # Convert to config format
        self.config_data["templates"] = {
            template_id: template.dict() 
            for template_id, template in templates.items()
        }
        
        return self
    
    def add_basic_strategies(self) -> "ConfigBuilder":
        """Add basic strategy configurations."""
        
        # Static strategy
        static_strategy = PromptStrategy(
            strategy_id="static_strategy",
            name="Static Template Selection",
            type=StrategyType.STATIC,
            description="Always uses the same template",
            config={"default_template": "qa_basic"},
            enabled=True
        )
        
        # Rule-based strategy
        rules = [
            StrategyRule(
                rule_id="medical_domain_rule",
                name="Medical Domain Rule",
                field="domain",
                operator=RuleOperator.EQUALS,
                value="medical",
                template_id="medical_qa",
                priority=10
            ),
            StrategyRule(
                rule_id="summary_rule",
                name="Summary Request Rule",
                field="query_type",
                operator=RuleOperator.EQUALS,
                value="summary",
                template_id="summarization",
                priority=30
            ),
            StrategyRule(
                rule_id="complex_analysis_rule",
                name="Complex Analysis Rule",
                field="complexity_level",
                operator=RuleOperator.EQUALS,
                value="high",
                template_id="chain_of_thought",
                priority=40
            )
        ]
        
        rule_based_strategy = PromptStrategy(
            strategy_id="rule_based_strategy",
            name="Rule-Based Selection",
            type=StrategyType.RULE_BASED,
            description="Select templates based on explicit rules",
            rules=rules,
            fallback_template="qa_basic",
            enabled=True
        )
        
        # Context-aware strategy
        context_aware_strategy = PromptStrategy(
            strategy_id="context_aware_strategy",
            name="Context-Aware Selection",
            type=StrategyType.CONTEXT_AWARE,
            description="Intelligent selection based on multiple context factors",
            config={
                "domain_templates": {
                    "medical": "medical_qa",
                    "software": "code_analysis"
                },
                "complexity_templates": {
                    "high": "chain_of_thought",
                    "medium": "qa_detailed",
                    "low": "qa_basic"
                },
                "intent_templates": {
                    "summarize": "summarization",
                    "chat": "chat_assistant"
                }
            },
            fallback_template="qa_basic",
            enabled=True
        )
        
        # Add strategies to config
        strategies = {
            "static_strategy": static_strategy.dict(),
            "rule_based_strategy": rule_based_strategy.dict(),
            "context_aware_strategy": context_aware_strategy.dict()
        }
        
        self.config_data["strategies"] = strategies
        return self
    
    def add_global_prompts(self) -> "ConfigBuilder":
        """Add basic global prompt configurations."""
        
        global_prompts = [
            GlobalPromptConfig(
                global_id="system_context",
                name="System Context",
                description="Provides general system context for all prompts",
                system_prompt="You are a helpful AI assistant integrated with LlamaFarm, a comprehensive document processing and RAG system. You have access to retrieved documents and should provide accurate, helpful responses based on the available information.",
                applies_to=["*"],
                priority=10,
                enabled=True
            ),
            GlobalPromptConfig(
                global_id="quality_guidelines",
                name="Quality Guidelines",
                description="Ensures high-quality responses",
                prefix_prompt="Please provide a clear, accurate, and helpful response. If you're uncertain about something, acknowledge the uncertainty.",
                applies_to=["*"],
                excludes=["debug_*", "test_*"],
                priority=20,
                enabled=True
            ),
            GlobalPromptConfig(
                global_id="domain_medical",
                name="Medical Domain Context",
                description="Medical domain expertise context",
                system_prompt="You are analyzing medical documents and should be precise, cite sources when available, and note any limitations in the provided information. Always recommend consulting healthcare professionals for medical decisions.",
                applies_to=["medical_*"],
                conditions={"domain": "medical"},
                priority=30,
                enabled=True
            )
        ]
        
        self.config_data["global_prompts"] = [gp.dict() for gp in global_prompts]
        return self
    
    def add_fallback_behavior(self) -> "ConfigBuilder":
        """Add fallback behavior configuration."""
        self.config_data["fallback_behavior"] = {
            "strategy_fallback_chain": ["context_aware_strategy", "rule_based_strategy", "static_strategy"],
            "template_fallback_chain": ["qa_basic", "chat_assistant"],
            "error_handling": {
                "log_errors": True,
                "return_error_details": False
            }
        }
        return self
    
    def add_integrations(self) -> "ConfigBuilder":
        """Add integration configurations."""
        self.config_data["integrations"] = {
            "rag_system": {
                "enabled": True,
                "context_mapping": {
                    "documents": "context",
                    "query": "query",
                    "domain": "domain"
                }
            },
            "langgraph": {
                "enabled": False,
                "workflow_endpoint": "http://localhost:8000/workflows"
            }
        }
        return self
    
    def add_monitoring(self) -> "ConfigBuilder":
        """Add monitoring configuration."""
        self.config_data["monitoring"] = {
            "enabled": True,
            "metrics": ["execution_time", "template_usage", "error_rate", "fallback_rate"],
            "logging_level": "INFO"
        }
        return self
    
    def add_environments(self) -> "ConfigBuilder":
        """Add environment-specific configurations."""
        self.config_data["environments"] = {
            "development": {
                "monitoring": {
                    "logging_level": "DEBUG"
                },
                "fallback_behavior": {
                    "return_error_details": True
                }
            },
            "production": {
                "monitoring": {
                    "logging_level": "WARNING"
                },
                "global_prompts": [
                    {
                        "global_id": "production_disclaimer",
                        "name": "Production Disclaimer",
                        "suffix_prompt": "\n\n---\nNote: This response was generated by an AI system. Please verify important information independently.",
                        "applies_to": ["*"],
                        "priority": 200,
                        "enabled": True
                    }
                ]
            }
        }
        return self
    
    def set_metadata(self, name: str = None, version: str = None, description: str = None) -> "ConfigBuilder":
        """Set configuration metadata."""
        if name:
            self.config_data["name"] = name
        if version:
            self.config_data["version"] = version
        if description:
            self.config_data["description"] = description
        return self
    
    def build(self) -> PromptConfig:
        """Build the final PromptConfig object."""
        return PromptConfig(**self.config_data)
    
    def save(self, output_file: str, format: str = "json") -> str:
        """Save the configuration to a file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if format.lower() in ['yaml', 'yml']:
                yaml.dump(self.config_data, f, default_flow_style=False, sort_keys=False)
            else:
                json.dump(self.config_data, f, indent=2, default=str)
        
        return str(output_path)
    
    def get_summary(self) -> Dict[str, any]:
        """Get a summary of the configuration."""
        return {
            "name": self.config_data.get("name", "Unnamed Configuration"),
            "total_templates": len(self.config_data.get("templates", {})),
            "total_strategies": len(self.config_data.get("strategies", {})),
            "total_global_prompts": len(self.config_data.get("global_prompts", [])),
            "template_categories": list(set(
                t.get("type", "unknown") 
                for t in self.config_data.get("templates", {}).values()
            )),
            "template_domains": list(set(
                t.get("metadata", {}).get("domain", "unknown")
                for t in self.config_data.get("templates", {}).values()
            )),
            "loader_summary": self.template_loader.get_load_summary()
        }


def build_default_config(templates_dir: str = "templates", output_file: str = None) -> PromptConfig:
    """
    Build a default configuration with all available templates and basic strategies.
    
    Args:
        templates_dir: Directory containing template files
        output_file: Optional output file to save configuration
        
    Returns:
        PromptConfig object
    """
    builder = ConfigBuilder(templates_dir)
    
    config = (builder
        .set_metadata(
            name="LlamaFarm Default Prompts Configuration",
            version="1.0.0",
            description="Default configuration with comprehensive prompt templates and strategies"
        )
        .load_templates()
        .add_basic_strategies()
        .add_global_prompts()
        .add_fallback_behavior()
        .add_integrations()
        .add_monitoring()
        .add_environments()
        .build()
    )
    
    if output_file:
        config.save_to_file(output_file)
    
    return config