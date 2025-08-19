"""
Image Strategy Manager for vision and image recognition configurations.

This module manages image-specific strategies while integrating with the main StrategyManager.
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


class ImageStrategyManager:
    """Manages image recognition strategies using the strategy-based approach."""
    
    def __init__(self, strategies_file: Optional[Path] = None):
        """Initialize image strategy manager.
        
        Args:
            strategies_file: Path to image strategies YAML file
        """
        if strategies_file:
            self.strategies_file = Path(strategies_file)
        else:
            # Default to models/image_strategies.yaml
            self.strategies_file = Path(__file__).parent.parent / "image_strategies.yaml"
        
        # Load strategies
        self.strategies = self._load_strategies()
        self.use_case_mapping = self._load_use_case_mapping()
    
    def _load_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load image strategies from YAML file."""
        if not self.strategies_file.exists():
            logger.warning(f"Image strategies file not found: {self.strategies_file}")
            return self._get_default_strategies()
        
        try:
            with open(self.strategies_file) as f:
                data = yaml.safe_load(f) or {}
                strategies_list = data.get("strategies", [])
                
                # Convert array of strategies to dict keyed by name, filtering for image strategies
                strategies_dict = {}
                for strategy in strategies_list:
                    if isinstance(strategy, dict) and "name" in strategy:
                        # Only include strategies that have image-related components or start with "image_"
                        strategy_name = strategy["name"]
                        components = strategy.get("components", {})
                        has_image_component = any(comp_type in components for comp_type in ["image_recognizer", "image_trainer"])
                        is_image_strategy = strategy_name.startswith("image_")
                        
                        if has_image_component or is_image_strategy:
                            strategies_dict[strategy_name] = strategy
                    else:
                        logger.warning(f"Invalid image strategy format: {strategy}")
                
                return strategies_dict
        except Exception as e:
            logger.error(f"Failed to load image strategies: {e}")
            return self._get_default_strategies()
    
    def _get_default_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Get default image strategies when no file is found."""
        return {
            "yolo_default": {
                "name": "yolo_default",
                "description": "Default YOLO object detection strategy",
                "components": {
                    "image_recognizer": {
                        "type": "yolo",
                        "config": {
                            "model": "yolov8n.pt",
                            "device": "auto",
                            "confidence": 0.5,
                            "nms_threshold": 0.45
                        }
                    }
                },
                "use_cases": ["object_detection", "general_vision"]
            }
        }
    
    def _load_use_case_mapping(self) -> Dict[str, List[str]]:
        """Load image use case to strategy mappings."""
        if not self.strategies_file.exists():
            return {
                "object_detection": ["yolo_default", "yolo_performance"],
                "segmentation": ["yolo_segmentation"],
                "classification": ["yolo_classification"],
                "tracking": ["yolo_tracking"]
            }
        
        try:
            with open(self.strategies_file) as f:
                data = yaml.safe_load(f) or {}
                return data.get("use_case_mapping", {})
        except Exception as e:
            logger.error(f"Failed to load image use case mappings: {e}")
            return {}
    
    def get_strategy(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific image strategy by name.
        
        Args:
            name: Strategy name
            
        Returns:
            Strategy configuration or None if not found
        """
        return deepcopy(self.strategies.get(name))
    
    def list_strategies(self) -> List[str]:
        """List all available image strategy names.
        
        Returns:
            List of strategy names
        """
        return list(self.strategies.keys())
    
    def get_strategies_for_use_case(self, use_case: str) -> List[str]:
        """Get recommended image strategies for a use case.
        
        Args:
            use_case: Use case name (e.g., 'object_detection', 'segmentation')
            
        Returns:
            List of recommended strategy names
        """
        return self.use_case_mapping.get(use_case, [])
    
    def build_image_recognizer_config(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Build image recognizer configuration from strategy.
        
        Args:
            strategy_name: Strategy name
            
        Returns:
            Image recognizer configuration or None
        """
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            return None
        
        components = strategy.get("components", {})
        recognizer_config = components.get("image_recognizer")
        
        if recognizer_config:
            # Expand environment variables in config
            return self._expand_env_vars(recognizer_config)
        
        return None
    
    def build_image_trainer_config(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Build image trainer configuration from strategy.
        
        Args:
            strategy_name: Strategy name
            
        Returns:
            Image trainer configuration or None
        """
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            return None
        
        components = strategy.get("components", {})
        trainer_config = components.get("image_trainer")
        
        if trainer_config:
            # Expand environment variables in config
            return self._expand_env_vars(trainer_config)
        
        return None
    
    def get_detection_params(self, strategy_name: str) -> Dict[str, Any]:
        """Get detection parameters for a strategy.
        
        Args:
            strategy_name: Strategy name
            
        Returns:
            Detection parameters
        """
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            return {"confidence": 0.5, "nms_threshold": 0.45}
        
        return strategy.get("detection_params", {
            "confidence": 0.5,
            "nms_threshold": 0.45
        })
    
    def get_visualization_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get visualization configuration for a strategy.
        
        Args:
            strategy_name: Strategy name
            
        Returns:
            Visualization configuration
        """
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            return {}
        
        return strategy.get("visualization", {})
    
    def get_optimization_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get optimization configuration for a strategy.
        
        Args:
            strategy_name: Strategy name
            
        Returns:
            Optimization configuration
        """
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            return {}
        
        return strategy.get("optimization", {})
    
    def validate_strategy(self, strategy_name: str) -> List[str]:
        """Validate an image strategy configuration.
        
        Args:
            strategy_name: Strategy name
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            return [f"Image strategy '{strategy_name}' not found"]
        
        # Check required fields
        if "name" not in strategy:
            errors.append("Strategy missing 'name' field")
        if "description" not in strategy:
            errors.append("Strategy missing 'description' field")
        if "components" not in strategy:
            errors.append("Strategy missing 'components' field")
        
        # Validate component types
        components = strategy.get("components", {})
        valid_component_types = ["image_recognizer", "image_trainer"]
        
        for comp_type, comp_config in components.items():
            if comp_type not in valid_component_types:
                errors.append(f"Invalid image component type: {comp_type}")
            
            if not isinstance(comp_config, dict):
                errors.append(f"Component {comp_type} configuration must be a dictionary")
            elif "type" not in comp_config:
                errors.append(f"Component {comp_type} missing 'type' field")
        
        return errors
    
    def _expand_env_vars(self, config: Union[Dict, List, str, Any]) -> Union[Dict, List, str, Any]:
        """Recursively expand environment variables in configuration.
        
        Args:
            config: Configuration to expand
            
        Returns:
            Configuration with expanded environment variables
        """
        if isinstance(config, dict):
            return {k: self._expand_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._expand_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Expand ${VAR_NAME} format
            if config.startswith("${") and config.endswith("}"):
                var_name = config[2:-1]
                return os.getenv(var_name, config)
            return config
        else:
            return config
    
    def get_supported_models(self, strategy_name: str) -> List[str]:
        """Get list of supported models for a strategy.
        
        Args:
            strategy_name: Strategy name
            
        Returns:
            List of supported model names
        """
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            return []
        
        return strategy.get("supported_models", [])
    
    def get_default_model(self, strategy_name: str) -> Optional[str]:
        """Get default model for a strategy.
        
        Args:
            strategy_name: Strategy name
            
        Returns:
            Default model name or None
        """
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            return None
        
        components = strategy.get("components", {})
        recognizer_config = components.get("image_recognizer", {})
        config = recognizer_config.get("config", {})
        
        return config.get("model")
    
    def export_strategy(self, strategy_name: str, output_path: Path, format: str = "yaml") -> bool:
        """Export an image strategy to a file.
        
        Args:
            strategy_name: Strategy name
            output_path: Output file path
            format: Output format (yaml or json)
            
        Returns:
            True if successful
        """
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            logger.error(f"Image strategy '{strategy_name}' not found")
            return False
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == "json":
                with open(output_path, "w") as f:
                    json.dump(strategy, f, indent=2)
            else:  # yaml
                with open(output_path, "w") as f:
                    yaml.dump(strategy, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Exported image strategy '{strategy_name}' to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export image strategy: {e}")
            return False