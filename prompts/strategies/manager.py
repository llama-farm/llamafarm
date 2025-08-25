"""
Strategy Manager

High-level interface for working with prompt strategies.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from .config import StrategyConfig, TemplateConfig
from .loader import StrategyLoader
from prompts.core.engines.template_registry import TemplateRegistry
from prompts.core.engines.template_engine import TemplateEngine
from prompts.core.models.context import PromptContext

logger = logging.getLogger(__name__)


class StrategyManager:
    """Manages prompt strategies and template execution."""
    
    def __init__(self, 
                 strategies_file: Optional[str] = None,
                 strategies_dir: Optional[str] = None,
                 template_registry: Optional[TemplateRegistry] = None,
                 template_engine: Optional[TemplateEngine] = None):
        """
        Initialize strategy manager.
        
        Args:
            strategies_file: Path to default strategies YAML file
            strategies_dir: Directory containing strategy YAML files
            template_registry: Template registry instance
            template_engine: Template engine instance
        """
        self.loader = StrategyLoader(strategies_file, strategies_dir)
        self.template_registry = template_registry or TemplateRegistry()
        self.template_engine = template_engine or TemplateEngine()
        
        self._strategies_cache: Dict[str, StrategyConfig] = {}
        self._execution_stats: Dict[str, Dict[str, int]] = {}
    
    def load_strategies(self) -> Dict[str, StrategyConfig]:
        """Load all available strategies."""
        strategies = self.loader.load_strategies()
        self._strategies_cache.update(strategies)
        return strategies
    
    def get_strategy(self, name: str) -> Optional[StrategyConfig]:
        """Get a specific strategy by name."""
        if name in self._strategies_cache:
            return self._strategies_cache[name]
        
        strategy = self.loader.get_strategy(name)
        if strategy:
            self._strategies_cache[name] = strategy
        
        return strategy
    
    def list_strategies(self) -> List[str]:
        """List all available strategy names."""
        return self.loader.list_strategies()
    
    def execute_strategy(self, 
                        strategy_name: str,
                        inputs: Dict[str, Any],
                        context: Optional[Dict[str, Any]] = None,
                        override_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute a strategy with given inputs.
        
        Args:
            strategy_name: Name of the strategy to execute
            inputs: Input variables for the template
            context: Additional context for template selection
            override_config: Configuration overrides
            
        Returns:
            Rendered prompt string
            
        Raises:
            ValueError: If strategy not found or execution fails
        """
        # Get strategy
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        # Prepare context
        full_context = {**inputs}
        if context:
            full_context.update(context)
        
        # Get appropriate template
        template_config = strategy.get_template_for_context(full_context)
        if not template_config:
            raise ValueError(f"No suitable template found for context in strategy '{strategy_name}'")
        
        # Apply transforms to inputs
        transformed_inputs = self._apply_input_transforms(inputs, template_config.input_transforms)
        
        # Merge configurations
        final_config = {}
        
        # Start with template defaults
        template = self.template_registry.get_template(template_config.template)
        if template and hasattr(template, 'config'):
            final_config.update(template.config)
        
        # Apply strategy template config
        final_config.update(template_config.config)
        
        # Apply global config
        if strategy.global_config.temperature is not None:
            final_config['temperature'] = strategy.global_config.temperature
        if strategy.global_config.max_tokens is not None:
            final_config['max_tokens'] = strategy.global_config.max_tokens
        
        # Apply overrides
        if override_config:
            final_config.update(override_config)
        
        # Create prompt context
        # Extract query from inputs for PromptContext
        query = transformed_inputs.get('query', transformed_inputs.get('message', ''))
        
        # Convert context strings to document format if needed
        raw_context = transformed_inputs.get('context', [])
        documents = []
        for i, doc in enumerate(raw_context):
            if isinstance(doc, str):
                documents.append({
                    'content': doc,
                    'title': f'Context {i+1}',
                    'relevance_score': 1.0
                })
            elif isinstance(doc, dict):
                documents.append(doc)
        
        prompt_context = PromptContext(
            query=query,
            query_type=context.get('query_type') if context else None,
            documents=documents,
            template_metadata={
                'template_id': template_config.template,
                'inputs': transformed_inputs,
                'config': final_config,
                'strategy': strategy_name,
                'performance_profile': strategy.performance_profile.value,
                'complexity': strategy.complexity.value
            }
        )
        
        # Get the template from registry
        template = self.template_registry.get_template(template_config.template)
        if not template:
            raise ValueError(f"Template '{template_config.template}' not found")
        
        # Load template defaults safely
        render_vars = transformed_inputs.copy()
        
        try:
            templates_dir = Path(__file__).parent.parent / "templates"
            template_category = str(template.type.value) if hasattr(template.type, 'value') else str(template.type)
            template_dir = templates_dir / template_category / template.template_id
            defaults_file = template_dir / "defaults.json"
            
            if defaults_file.exists():
                with open(defaults_file) as f:
                    defaults = json.load(f)
                    # Add config from defaults
                    config = defaults.get('config', {})
                    
                    # Merge template config with strategy config
                    if hasattr(template_config, 'config') and template_config.config:
                        config.update(template_config.config)
                    
                    render_vars['config'] = config
                    
                    # Merge input defaults
                    for key, default_value in defaults.get('input_defaults', {}).items():
                        if key not in render_vars:
                            render_vars[key] = default_value
            else:
                # Fallback to basic config if no defaults file
                config = template_config.config if hasattr(template_config, 'config') and template_config.config else {}
                render_vars['config'] = config
                
        except Exception as e:
            logger.warning(f"Could not load template defaults for {template.template_id}: {e}")
            # Fallback to basic config
            config = template_config.config if hasattr(template_config, 'config') and template_config.config else {}
            render_vars['config'] = config
        
        # Apply system prompts
        rendered_prompt = self._apply_system_prompts(
            self.template_engine.render_template(template, render_vars),
            strategy.global_config.system_prompts
        )
        
        # Apply output transforms
        final_prompt = self._apply_output_transforms(
            rendered_prompt,
            template_config.output_transforms
        )
        
        # Update stats
        self._update_execution_stats(strategy_name, template_config.template)
        
        return final_prompt
    
    def _apply_input_transforms(self, inputs: Dict[str, Any], transforms: List[Any]) -> Dict[str, Any]:
        """Apply input transformations."""
        transformed = inputs.copy()
        
        for transform in transforms:
            if hasattr(transform, 'input') and transform.input in transformed:
                value = transformed[transform.input]
                
                if transform.transform == "lowercase":
                    transformed[transform.input] = str(value).lower()
                elif transform.transform == "uppercase":
                    transformed[transform.input] = str(value).upper()
                elif transform.transform == "trim":
                    transformed[transform.input] = str(value).strip()
                elif transform.transform == "truncate":
                    max_length = transform.params.get('max_length', 1000)
                    transformed[transform.input] = str(value)[:max_length]
                elif transform.transform == "clean_whitespace":
                    transformed[transform.input] = ' '.join(str(value).split())
                elif transform.transform == "escape_html":
                    import html
                    transformed[transform.input] = html.escape(str(value))
        
        return transformed
    
    def _apply_output_transforms(self, output: str, transforms: List[Any]) -> str:
        """Apply output transformations."""
        result = output
        
        for transform in transforms:
            if transform.transform == "uppercase":
                result = result.upper()
            elif transform.transform == "lowercase":
                result = result.lower()
            elif transform.transform == "trim":
                result = result.strip()
            elif transform.transform == "parse_json":
                # Extract JSON from output
                import json
                import re
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                        result = json.dumps(parsed, indent=2)
                    except:
                        pass
            elif transform.transform == "extract_code":
                # Extract code blocks
                import re
                code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', result, re.DOTALL)
                if code_blocks:
                    result = '\n\n'.join(code_blocks)
            elif transform.transform == "clean_markdown":
                # Remove markdown formatting
                import re
                result = re.sub(r'\*\*(.*?)\*\*', r'\1', result)  # Bold
                result = re.sub(r'\*(.*?)\*', r'\1', result)      # Italic
                result = re.sub(r'`(.*?)`', r'\1', result)        # Code
                result = re.sub(r'^#+\s+', '', result, flags=re.MULTILINE)  # Headers
            elif transform.transform == "validate_format":
                # Validate output format (implementation depends on requirements)
                pass
        
        return result
    
    def _apply_system_prompts(self, prompt: str, system_prompts: List[Any]) -> str:
        """Apply system prompts to the rendered prompt."""
        if not system_prompts:
            return prompt
        
        # Sort by priority
        sorted_prompts = sorted(system_prompts, key=lambda p: p.priority, reverse=True)
        
        prefixes = []
        suffixes = []
        system_messages = []
        
        for sp in sorted_prompts:
            if sp.position == "prefix":
                prefixes.append(sp.content)
            elif sp.position == "suffix":
                suffixes.append(sp.content)
            elif sp.position == "system":
                system_messages.append(sp.content)
        
        # Combine components
        result_parts = []
        
        if system_messages:
            result_parts.append("System: " + "\n".join(system_messages))
        
        if prefixes:
            result_parts.extend(prefixes)
        
        result_parts.append(prompt)
        
        if suffixes:
            result_parts.extend(suffixes)
        
        return "\n\n".join(result_parts)
    
    def _update_execution_stats(self, strategy_name: str, template_id: str) -> None:
        """Update execution statistics."""
        if strategy_name not in self._execution_stats:
            self._execution_stats[strategy_name] = {}
        
        if template_id not in self._execution_stats[strategy_name]:
            self._execution_stats[strategy_name][template_id] = 0
        
        self._execution_stats[strategy_name][template_id] += 1
    
    def get_execution_stats(self) -> Dict[str, Dict[str, int]]:
        """Get execution statistics."""
        return self._execution_stats.copy()
    
    def recommend_strategies(self,
                           use_case: Optional[str] = None,
                           performance: Optional[str] = None,
                           complexity: Optional[str] = None) -> List[StrategyConfig]:
        """
        Recommend strategies based on criteria.
        
        Args:
            use_case: Desired use case
            performance: Performance profile preference
            complexity: Complexity level preference
            
        Returns:
            List of recommended strategies
        """
        all_strategies = list(self.load_strategies().values())
        recommendations = []
        
        for strategy in all_strategies:
            score = 0
            
            if use_case and use_case in strategy.use_cases:
                score += 10
            
            if performance and strategy.performance_profile.value == performance:
                score += 5
            
            if complexity and strategy.complexity.value == complexity:
                score += 3
            
            if score > 0:
                recommendations.append((strategy, score))
        
        # Sort by score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return [strategy for strategy, _ in recommendations]
    
    def create_strategy(self,
                       name: str,
                       description: str,
                       default_template: str,
                       **kwargs) -> StrategyConfig:
        """
        Create a new strategy.
        
        Args:
            name: Strategy name
            description: Strategy description
            default_template: Default template ID
            **kwargs: Additional strategy configuration
            
        Returns:
            Created StrategyConfig
        """
        from .config import TemplatesConfig, TemplateConfig
        
        # Create templates config
        templates = TemplatesConfig(
            default=TemplateConfig(template=default_template)
        )
        
        # Create strategy
        strategy = StrategyConfig(
            name=name,
            description=description,
            templates=templates,
            **kwargs
        )
        
        # Validate
        errors = self.loader.validate_strategy(strategy)
        if errors:
            raise ValueError(f"Invalid strategy configuration: {', '.join(errors)}")
        
        return strategy
    
    def save_strategy(self, strategy_name: str, strategy: StrategyConfig, 
                     file_path: Optional[str] = None) -> None:
        """Save a strategy to file."""
        self.loader.save_strategy(strategy_name, strategy, 
                                 Path(file_path) if file_path else None)
        
        # Update cache
        self._strategies_cache[strategy_name] = strategy
    
    def get_template_usage(self) -> Dict[str, List[str]]:
        """Get mapping of templates to strategies that use them."""
        template_usage: Dict[str, List[str]] = {}
        
        for strategy_name, strategy in self.load_strategies().items():
            # Add default template
            template_id = strategy.templates.default.template
            if template_id not in template_usage:
                template_usage[template_id] = []
            template_usage[template_id].append(strategy_name)
            
            # Add fallback
            if strategy.templates.fallback:
                template_id = strategy.templates.fallback.template
                if template_id not in template_usage:
                    template_usage[template_id] = []
                template_usage[template_id].append(strategy_name)
            
            # Add specialized
            for spec in strategy.templates.specialized:
                template_id = spec.template
                if template_id not in template_usage:
                    template_usage[template_id] = []
                template_usage[template_id].append(strategy_name)
        
        return template_usage