"""
Strategy management for fine-tuning.

This module provides strategy management functionality similar to the RAG system,
allowing users to choose from pre-configured fine-tuning strategies.
"""

import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from .base import BaseStrategy, FineTuningConfig

logger = logging.getLogger(__name__)


class StrategyManager:
    """Manager for fine-tuning strategies."""
    
    def __init__(self, strategies_file: Optional[Path] = None):
        """Initialize strategy manager."""
        if strategies_file is None:
            # Default to strategies.yaml in the same directory as this module
            strategies_file = Path(__file__).parent.parent / "strategies.yaml"
        
        self.strategies_file = strategies_file
        self._strategies: Dict[str, Dict[str, Any]] = {}
        self._load_strategies()
    
    def _load_strategies(self) -> None:
        """Load strategies from YAML file."""
        try:
            if self.strategies_file.exists():
                with open(self.strategies_file, 'r') as f:
                    data = yaml.safe_load(f)
                    self._strategies = data.get("strategies", {})
                logger.debug(f"Loaded {len(self._strategies)} strategies")
            else:
                logger.warning(f"Strategies file not found: {self.strategies_file}")
        except Exception as e:
            logger.error(f"Failed to load strategies: {e}")
    
    def list_strategies(self) -> List[str]:
        """Get list of available strategy names."""
        return list(self._strategies.keys())
    
    def get_strategy_info(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific strategy."""
        return self._strategies.get(strategy_name)
    
    def get_strategy_config(self, strategy_name: str, overrides: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Convert strategy to configuration dictionary.
        
        Args:
            strategy_name: Name of strategy to convert
            overrides: Optional configuration overrides
            
        Returns:
            Configuration dictionary or None if strategy not found
        """
        strategy = self._strategies.get(strategy_name)
        if not strategy:
            return None
        
        # Start with strategy components
        config = strategy.get("components", {}).copy()
        
        # Apply overrides if provided
        if overrides:
            config = self._merge_configs(config, overrides)
        
        return config
    
    def recommend_strategies(self, **criteria) -> List[Dict[str, Any]]:
        """
        Recommend strategies based on criteria.
        
        Args:
            **criteria: Filtering criteria (hardware, model_size, use_case, etc.)
            
        Returns:
            List of recommended strategies with scores
        """
        recommendations = []
        
        for name, strategy in self._strategies.items():
            score = self._calculate_strategy_score(strategy, criteria)
            if score > 0:
                rec = strategy.copy()
                rec["strategy_id"] = name
                rec["score"] = score
                recommendations.append(rec)
        
        # Sort by score (highest first)
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        
        return recommendations
    
    def _calculate_strategy_score(self, strategy: Dict[str, Any], criteria: Dict[str, Any]) -> float:
        """Calculate how well a strategy matches the criteria."""
        score = 0.0
        max_score = 0.0
        
        # Hardware matching
        if "hardware" in criteria:
            max_score += 3.0
            hardware_req = strategy.get("hardware_requirements", {})
            strategy_hardware = (hardware_req.get("type") or "").lower()
            criteria_hardware = (criteria["hardware"] or "").lower()
            
            if strategy_hardware == criteria_hardware:
                score += 3.0
            elif "gpu" in strategy_hardware and "gpu" in criteria_hardware:
                score += 2.0
            elif strategy_hardware in criteria_hardware or criteria_hardware in strategy_hardware:
                score += 1.0
        
        # Model size matching
        if "model_size" in criteria:
            max_score += 2.0
            recommended_models = strategy.get("recommended_models", [])
            criteria_size = (criteria["model_size"] or "").lower()
            
            for model in recommended_models:
                if criteria_size in model.lower():
                    score += 2.0
                    break
            else:
                # Partial match
                if any(size in model.lower() for model in recommended_models for size in ["3b", "8b", "70b"]):
                    score += 1.0
        
        # Use case matching
        if "use_case" in criteria:
            max_score += 2.0
            strategy_use_cases = strategy.get("use_cases", [])
            criteria_use_case = (criteria["use_case"] or "").lower()
            
            for use_case in strategy_use_cases:
                if criteria_use_case in use_case.lower() or use_case.lower() in criteria_use_case:
                    score += 2.0
                    break
            else:
                # Partial match on keywords
                keywords = ["coding", "creative", "domain", "production", "testing"]
                for keyword in keywords:
                    if (keyword in criteria_use_case and 
                        any(keyword in uc.lower() for uc in strategy_use_cases)):
                        score += 1.0
                        break
        
        # Normalize score
        if max_score > 0:
            score = score / max_score
        
        return score
    
    def _merge_configs(self, base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration overrides with base configuration."""
        import copy
        
        result = copy.deepcopy(base_config)
        
        def deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> None:
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    deep_merge(target[key], value)
                else:
                    target[key] = value
        
        deep_merge(result, overrides)
        return result


class Strategy(BaseStrategy):
    """Concrete strategy implementation."""
    
    def to_config(self, overrides: Optional[Dict[str, Any]] = None) -> FineTuningConfig:
        """Convert strategy to fine-tuning configuration."""
        config_dict = self.strategy_config.get("components", {}).copy()
        
        if overrides:
            config_dict = self._merge_configs(config_dict, overrides)
        
        # Ensure required fields
        if "base_model" not in config_dict:
            config_dict["base_model"] = {"name": "llama3.2-3b"}
        if "method" not in config_dict:
            config_dict["method"] = {"type": "lora"}
        if "framework" not in config_dict:
            config_dict["framework"] = {"type": "pytorch"}
        if "training_args" not in config_dict:
            config_dict["training_args"] = {"output_dir": "./fine_tuned_models"}
        if "dataset" not in config_dict:
            config_dict["dataset"] = {"path": ""}
        
        return FineTuningConfig(**config_dict)
    
    def validate_hardware(self) -> List[str]:
        """Validate hardware requirements."""
        errors = []
        
        hardware_req = self.strategy_config.get("hardware_requirements", {})
        required_memory = hardware_req.get("memory_gb", 0)
        
        # Basic validation (could be extended with actual hardware detection)
        if required_memory > 64:
            errors.append(f"Strategy requires {required_memory}GB memory, which is very high")
        
        hardware_type = hardware_req.get("type", "").lower()
        if hardware_type == "cloud" and "gpu_count" in hardware_req:
            gpu_count = hardware_req["gpu_count"]
            if gpu_count > 8:
                errors.append(f"Strategy requires {gpu_count} GPUs, which may be expensive")
        
        return errors
    
    def _merge_configs(self, base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration overrides with base configuration."""
        import copy
        
        result = copy.deepcopy(base_config)
        
        def deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> None:
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    deep_merge(target[key], value)
                else:
                    target[key] = value
        
        deep_merge(result, overrides)
        return result