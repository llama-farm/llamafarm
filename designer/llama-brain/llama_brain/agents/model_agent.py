"""Model configuration agent."""

from typing import Dict, Any, List
from .base_agent import BaseAgent


class ModelAgent(BaseAgent):
    """Agent specialized in model configurations."""
    
    def __init__(self):
        super().__init__("models")
    
    async def create_config(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Create a model configuration using the LlamaFarm models system."""
        output_path = self.settings.generated_configs_dir / self.component_name / f"generated_{requirements.get('use_case', 'custom')}.yaml"
        
        # Use the LlamaFarm client to create the config
        result = await self.client.create_model_config(requirements, str(output_path))
        
        if result["success"]:
            return {
                "config": result["config"],
                "file_path": result["config_path"],
                "validation_output": result.get("validation_output", ""),
                "success": True
            }
        else:
            raise ValueError(f"Failed to create model config: {result.get('error')}")
    
    async def _create_development_config(self, model: str) -> Dict[str, Any]:
        """Create a development configuration."""
        return {
            "providers": {
                "primary": {
                    "provider": "ollama",
                    "model": model,
                    "base_url": "http://localhost:11434",
                    "timeout": 30,
                    "max_retries": 3,
                    "temperature": 0.7
                }
            },
            "routing": {
                "default_provider": "primary",
                "fallback_enabled": False
            },
            "monitoring": {
                "enabled": True,
                "log_level": "INFO",
                "track_usage": True,
                "track_performance": True
            },
            "caching": {
                "enabled": True,
                "ttl_seconds": 3600,
                "max_size": 1000
            }
        }
    
    async def _create_production_config(self, providers: List[str], primary_model: str) -> Dict[str, Any]:
        """Create a production configuration."""
        config_providers = {}
        
        # Configure each provider
        for i, provider in enumerate(providers):
            if provider == "openai":
                config_providers[f"openai_{i}"] = {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "${OPENAI_API_KEY}",
                    "timeout": 30,
                    "max_retries": 3,
                    "rate_limit": {
                        "requests_per_minute": 1000,
                        "tokens_per_minute": 150000
                    }
                }
            elif provider == "anthropic":
                config_providers[f"anthropic_{i}"] = {
                    "provider": "anthropic",
                    "model": "claude-3-haiku-20240307",
                    "api_key": "${ANTHROPIC_API_KEY}",
                    "timeout": 30,
                    "max_retries": 3
                }
            elif provider == "ollama":
                config_providers[f"ollama_{i}"] = {
                    "provider": "ollama",
                    "model": primary_model,
                    "base_url": "http://localhost:11434",
                    "timeout": 60
                }
        
        return {
            "providers": config_providers,
            "routing": {
                "default_provider": list(config_providers.keys())[0],
                "fallback_enabled": True,
                "fallback_order": list(config_providers.keys()),
                "health_check_interval": 60
            },
            "monitoring": {
                "enabled": True,
                "log_level": "WARNING",
                "track_usage": True,
                "track_performance": True,
                "metrics_export": {
                    "enabled": True,
                    "format": "prometheus",
                    "endpoint": "/metrics"
                }
            },
            "caching": {
                "enabled": True,
                "ttl_seconds": 7200,
                "max_size": 10000,
                "redis_url": "${REDIS_URL}"
            }
        }
    
    async def _create_multi_provider_config(self, providers: List[str]) -> Dict[str, Any]:
        """Create a multi-provider configuration with intelligent routing."""
        config_providers = {}
        
        # Configure providers with different use cases
        for provider in providers:
            if provider == "openai":
                config_providers["openai_fast"] = {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "${OPENAI_API_KEY}",
                    "use_cases": ["quick_responses", "simple_tasks"]
                }
                config_providers["openai_powerful"] = {
                    "provider": "openai", 
                    "model": "gpt-4o",
                    "api_key": "${OPENAI_API_KEY}",
                    "use_cases": ["complex_reasoning", "high_quality"]
                }
            elif provider == "anthropic":
                config_providers["anthropic_balanced"] = {
                    "provider": "anthropic",
                    "model": "claude-3-sonnet-20240229",
                    "api_key": "${ANTHROPIC_API_KEY}",
                    "use_cases": ["balanced", "analysis"]
                }
            elif provider == "ollama":
                config_providers["ollama_local"] = {
                    "provider": "ollama",
                    "model": "llama3.2:3b",
                    "base_url": "http://localhost:11434",
                    "use_cases": ["private", "offline", "cost_effective"]
                }
        
        return {
            "providers": config_providers,
            "routing": {
                "strategy": "intelligent",
                "rules": [
                    {
                        "condition": "length < 100",
                        "provider": "openai_fast"
                    },
                    {
                        "condition": "complexity == 'high'",
                        "provider": "openai_powerful"
                    },
                    {
                        "condition": "privacy_required == true",
                        "provider": "ollama_local"
                    }
                ],
                "default_provider": list(config_providers.keys())[0],
                "fallback_enabled": True
            },
            "cost_optimization": {
                "enabled": True,
                "budget_limit_monthly": 1000,
                "prefer_cheaper_models": True,
                "cost_tracking": True
            }
        }
    
    async def edit_config(self, config: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
        """Edit an existing model configuration."""
        result = self._merge_configs(config, changes)
        
        # Validate the merged configuration
        validation = await self.validate_config(result)
        if not validation["valid"]:
            raise ValueError(f"Invalid configuration after edit: {validation['errors']}")
        
        return result
    
    async def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a model configuration."""
        errors = []
        warnings = []
        
        # Check required fields
        if "providers" not in config:
            errors.append("Missing 'providers' section")
        else:
            providers = config["providers"]
            if not providers:
                errors.append("No providers configured")
            
            # Validate each provider
            for name, provider_config in providers.items():
                if "provider" not in provider_config:
                    errors.append(f"Provider '{name}' missing 'provider' field")
                if "model" not in provider_config:
                    errors.append(f"Provider '{name}' missing 'model' field")
                
                # Check for API keys in non-local providers
                provider_type = provider_config.get("provider")
                if provider_type in ["openai", "anthropic", "cohere"] and "api_key" not in provider_config:
                    warnings.append(f"Provider '{name}' missing API key")
        
        # Check routing configuration
        if "routing" in config:
            routing = config["routing"]
            default_provider = routing.get("default_provider")
            if default_provider and default_provider not in config.get("providers", {}):
                errors.append(f"Default provider '{default_provider}' not found in providers")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": self._generate_suggestions(config)
        }
    
    def _generate_suggestions(self, config: Dict[str, Any]) -> List[str]:
        """Generate suggestions for improving the configuration."""
        suggestions = []
        
        providers = config.get("providers", {})
        
        # Suggest fallback configuration
        if len(providers) > 1 and not config.get("routing", {}).get("fallback_enabled"):
            suggestions.append("Consider enabling fallback routing for better reliability")
        
        # Suggest monitoring
        if not config.get("monitoring", {}).get("enabled"):
            suggestions.append("Enable monitoring to track model usage and performance")
        
        # Suggest caching
        if not config.get("caching", {}).get("enabled"):
            suggestions.append("Enable caching to reduce costs and improve response times")
        
        # Check for local models only
        if all(p.get("provider") == "ollama" for p in providers.values()):
            suggestions.append("Consider adding cloud providers for better availability")
        
        return suggestions