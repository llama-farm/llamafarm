"""
Configuration loader for analysis strategies.
"""

from pathlib import Path
from typing import Any

import yaml

from .strategies import AnalysisRule, ResponseValidationConfig


class ConfigLoader:
    """Loads configuration for analysis strategies"""
    
    @staticmethod
    def load_config(config_path: str | None = None) -> dict[str, Any]:
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "analysis_config.yaml"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            return ConfigLoader._get_default_config()
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception:
            return ConfigLoader._get_default_config()
    
    @staticmethod
    def _get_default_config() -> dict[str, Any]:
        """Get default configuration if file loading fails"""
        return {
            "analysis": {
                "default_namespace": "test",
                "confidence_threshold": 0.7,
                "enable_fuzzy_matching": True
            },
            "response_validation": {
                "min_response_length": 50,
                "enable_hallucination_detection": True,
                "enable_count_query_validation": True,
                "template_indicators": [
                    "[number of projects]", "[project list]", "[namespace]"
                ],
                "inability_phrases": [
                    "i don't have access", "cannot directly"
                ],
                "hallucination_indicators": [
                    "project 1", "project 2", "project 3"
                ]
            },
            "rules": {
                "excluded_namespaces": ["the", "a", "an", "my", "projects", "project"]
            }
        }
    
    @staticmethod
    def create_validation_config(
        config_dict: dict[str, Any]) -> ResponseValidationConfig:
        """Create ResponseValidationConfig from dictionary"""
        validation_config = config_dict.get("response_validation", {})
        
        return ResponseValidationConfig(
            template_indicators=validation_config.get("template_indicators", []),
            inability_phrases=validation_config.get("inability_phrases", []),
            hallucination_indicators=validation_config.get(
                "hallucination_indicators", []),
            min_response_length=validation_config.get("min_response_length", 50),
            enable_hallucination_detection=validation_config.get(
                "enable_hallucination_detection", True),
            enable_count_query_validation=validation_config.get(
                "enable_count_query_validation", True)
        )
    
    @staticmethod
    def create_analysis_rules(config_dict: dict[str, Any]) -> dict[str, Any]:
        """Create analysis rules from configuration"""
        rules = config_dict.get("rules", {})
        analysis = config_dict.get("analysis", {})
        
        return {
            "namespace_rules": [
                AnalysisRule(
                    name=rule["name"],
                    patterns=rule["patterns"],
                    keywords=rule["keywords"],
                    weight=rule.get("weight", 1.0),
                    enabled=rule.get("enabled", True)
                )
                for rule in rules.get("namespace_patterns", [])
            ],
            "action_rules": [
                AnalysisRule(
                    name=rule["name"],
                    patterns=rule["patterns"],
                    keywords=rule["keywords"],
                    weight=rule.get("weight", 1.0),
                    enabled=rule.get("enabled", True)
                )
                for rule in rules.get("action_patterns", [])
            ],
            "excluded_namespaces": set(rules.get("excluded_namespaces", [])),
            "default_namespace": analysis.get("default_namespace", "test"),
            "confidence_threshold": analysis.get("confidence_threshold", 0.7)
        }