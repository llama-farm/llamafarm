"""
Strategy Loader

Loads strategy configurations from YAML files and validates them.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
import logging

from .config import StrategyConfig

logger = logging.getLogger(__name__)


class StrategyLoader:
    """Loads and manages strategy configurations."""
    
    def __init__(self, strategies_file: Optional[str] = None, strategies_dir: Optional[str] = None):
        """
        Initialize strategy loader.
        
        Args:
            strategies_file: Path to default strategies YAML file
            strategies_dir: Directory containing strategy YAML files
        """
        if strategies_file is None:
            # Default to default_strategies.yaml in the parent directory
            self.strategies_file = Path(__file__).parent.parent / "default_strategies.yaml"
        else:
            self.strategies_file = Path(strategies_file)
        
        self.strategies_dir = Path(strategies_dir) if strategies_dir else None
        
        self._strategies: Dict[str, StrategyConfig] = {}
        self._loaded = False
    
    def load_strategies(self) -> Dict[str, StrategyConfig]:
        """
        Load all strategies from configured sources.
        
        Returns:
            Dictionary mapping strategy names to StrategyConfig objects
        """
        if self._loaded:
            return self._strategies
        
        # Load default strategies
        loaded_any = False
        
        if self.strategies_file and self.strategies_file.exists():
            self._load_strategies_from_file(self.strategies_file)
            loaded_any = True
        
        # Load from directory if specified
        if self.strategies_dir and self.strategies_dir.exists():
            self._load_strategies_from_directory(self.strategies_dir)
            loaded_any = True
        
        if not loaded_any:
            logger.warning(
                "No strategies loaded: neither strategies_file (%s) nor strategies_dir (%s) exists.",
                self.strategies_file, self.strategies_dir
            )
        
        self._loaded = True
        logger.info(f"Loaded {len(self._strategies)} strategies")
        
        return self._strategies
    
    def _load_strategies_from_file(self, file_path: Path) -> None:
        """Load strategies from a single YAML file."""
        try:
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Strategies file not found: {file_path}")
            return
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in strategies file {file_path}: {e}")
            return
        except IOError as e:
            logger.error(f"Failed to read strategies file {file_path}: {e}")
            return
        
        if not data:
            logger.warning(f"Empty strategies file: {file_path}")
            return
            
        # Skip metadata fields
        strategy_names = [
            key for key in data.keys() 
            if not key.startswith('usage_') and not key in ['description', 'schema', 'validation']
        ]
        
        for strategy_name in strategy_names:
            strategy_data = data[strategy_name]
            
            # Skip if not a complete strategy definition
            if not isinstance(strategy_data, dict) or "templates" not in strategy_data:
                continue
            
            try:
                # Convert templates section to expected format
                self._normalize_strategy_data(strategy_data)
                
                strategy_config = StrategyConfig.from_dict(strategy_data)
                self._strategies[strategy_name] = strategy_config
                logger.debug(f"Loaded strategy: {strategy_name} from {file_path}")
            except ValueError as e:
                logger.error(f"Invalid strategy configuration for {strategy_name} in {file_path}: {e}")
                continue
            except KeyError as e:
                logger.error(f"Missing required field in strategy {strategy_name} from {file_path}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error loading strategy {strategy_name} from {file_path}: {e}")
                continue
    
    def _load_strategies_from_directory(self, directory: Path) -> None:
        """Load strategies from all YAML files in a directory."""
        # Support both .yaml and .yml extensions
        yaml_files = list(directory.glob("*.yaml")) + list(directory.glob("*.yml"))
        
        for yaml_file in yaml_files:
            if yaml_file.name == "default_strategies.yaml":
                continue  # Skip if already loaded
            
            try:
                with open(yaml_file, 'r') as file:
                    data = yaml.safe_load(file)
                
                # Handle single strategy per file
                if 'name' in data and 'templates' in data:
                    strategy_name = yaml_file.stem  # Use filename as strategy ID
                    self._normalize_strategy_data(data)
                    strategy_config = StrategyConfig.from_dict(data)
                    self._strategies[strategy_name] = strategy_config
                    logger.debug(f"Loaded strategy: {strategy_name} from {yaml_file}")
                else:
                    # Handle multiple strategies in one file
                    self._load_strategies_from_file(yaml_file)
                    
            except Exception as e:
                logger.error(f"Failed to load strategy from {yaml_file}: {e}")
    
    def _normalize_strategy_data(self, data: Dict[str, Any]) -> None:
        """Normalize strategy data to expected format."""
        if 'templates' in data:
            templates = data['templates']
            
            # Ensure default template exists
            if 'default' not in templates:
                raise ValueError("Strategy must have a default template")
            
            # Normalize default template
            if isinstance(templates['default'], str):
                templates['default'] = {'template': templates['default']}
            elif isinstance(templates['default'], dict) and 'template' not in templates['default']:
                if 'type' in templates['default']:
                    templates['default']['template'] = templates['default'].pop('type')
            
            # Normalize fallback template if exists
            if 'fallback' in templates:
                if isinstance(templates['fallback'], str):
                    templates['fallback'] = {'template': templates['fallback']}
                elif isinstance(templates['fallback'], dict) and 'template' not in templates['fallback']:
                    if 'type' in templates['fallback']:
                        templates['fallback']['template'] = templates['fallback'].pop('type')
            
            # Normalize specialized templates
            if 'specialized' in templates:
                normalized_specialized = []
                for spec in templates['specialized']:
                    if 'template' not in spec and 'type' in spec:
                        spec['template'] = spec.pop('type')
                    normalized_specialized.append(spec)
                templates['specialized'] = normalized_specialized
    
    def get_strategy(self, name: str) -> Optional[StrategyConfig]:
        """
        Get a specific strategy by name.
        
        Args:
            name: Strategy name
            
        Returns:
            StrategyConfig object or None if not found
        """
        strategies = self.load_strategies()
        return strategies.get(name)
    
    def list_strategies(self) -> List[str]:
        """
        List all available strategy names.
        
        Returns:
            List of strategy names
        """
        strategies = self.load_strategies()
        return list(strategies.keys())
    
    def get_strategies_by_use_case(self, use_case: str) -> List[StrategyConfig]:
        """
        Get strategies that support a specific use case.
        
        Args:
            use_case: Use case to search for
            
        Returns:
            List of matching StrategyConfig objects
        """
        strategies = self.load_strategies()
        matching = []
        
        for strategy in strategies.values():
            if use_case in strategy.use_cases:
                matching.append(strategy)
        
        return matching
    
    def get_strategies_by_performance(self, performance_profile: str) -> List[StrategyConfig]:
        """
        Get strategies with a specific performance profile.
        
        Args:
            performance_profile: Performance profile to match
            
        Returns:
            List of matching StrategyConfig objects
        """
        strategies = self.load_strategies()
        matching = []
        
        for strategy in strategies.values():
            if strategy.performance_profile.value == performance_profile:
                matching.append(strategy)
        
        return matching
    
    def get_strategies_by_complexity(self, complexity: str) -> List[StrategyConfig]:
        """
        Get strategies with a specific complexity level.
        
        Args:
            complexity: Complexity level to match
            
        Returns:
            List of matching StrategyConfig objects
        """
        strategies = self.load_strategies()
        matching = []
        
        for strategy in strategies.values():
            if strategy.complexity.value == complexity:
                matching.append(strategy)
        
        return matching
    
    def get_required_templates(self) -> Set[str]:
        """
        Get all template IDs referenced by loaded strategies.
        
        Returns:
            Set of template IDs
        """
        strategies = self.load_strategies()
        template_ids = set()
        
        for strategy in strategies.values():
            # Add default template
            template_ids.add(strategy.templates.default.template)
            
            # Add fallback if exists
            if strategy.templates.fallback:
                template_ids.add(strategy.templates.fallback.template)
            
            # Add specialized templates
            for spec in strategy.templates.specialized:
                template_ids.add(spec.template)
            
            # Add templates from selection rules
            for rule in strategy.selection_rules:
                template_ids.add(rule.template)
        
        return template_ids
    
    def validate_strategy(self, strategy_config: StrategyConfig) -> List[str]:
        """
        Validate a strategy configuration.
        
        Args:
            strategy_config: Strategy to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Basic validation
        if not strategy_config.name:
            errors.append("Strategy name is required")
        
        if not strategy_config.description:
            errors.append("Strategy description is required")
        
        if not strategy_config.templates.default:
            errors.append("Default template is required")
        
        # Check for duplicate rule names
        rule_names = [rule.name for rule in strategy_config.selection_rules]
        if len(rule_names) != len(set(rule_names)):
            errors.append("Duplicate selection rule names found")
        
        # Validate template references
        template_ids = {strategy_config.templates.default.template}
        if strategy_config.templates.fallback:
            template_ids.add(strategy_config.templates.fallback.template)
        for spec in strategy_config.templates.specialized:
            template_ids.add(spec.template)
        
        for rule in strategy_config.selection_rules:
            if rule.template not in template_ids:
                # This might be an external template reference
                logger.warning(f"Rule '{rule.name}' references external template: {rule.template}")
        
        return errors
    
    def save_strategy(self, strategy_name: str, strategy_config: StrategyConfig, 
                     file_path: Optional[Path] = None) -> None:
        """
        Save a strategy to a YAML file.
        
        Args:
            strategy_name: Name/ID for the strategy
            strategy_config: Strategy configuration to save
            file_path: Optional path to save to (defaults to strategies_dir/name.yaml)
        """
        if file_path is None:
            if self.strategies_dir:
                file_path = self.strategies_dir / f"{strategy_name}.yaml"
            else:
                raise ValueError("No file path specified and no strategies_dir configured")
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and save
        data = strategy_config.to_dict()
        
        with open(file_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved strategy '{strategy_name}' to {file_path}")
        
        # Add to loaded strategies
        self._strategies[strategy_name] = strategy_config