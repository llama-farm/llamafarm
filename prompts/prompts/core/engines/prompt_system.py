"""Main prompt system orchestrator."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..models.config import PromptConfig
from ..models.context import ExecutionContext, PromptContext
from ..models.template import PromptTemplate
from .global_prompt_manager import GlobalPromptManager
from .strategy_engine import StrategyEngine
from .template_engine import TemplateEngine
from .template_registry import TemplateRegistry


class PromptSystem:
    """
    Main prompt system that orchestrates template selection, global prompt application,
    and template rendering. This is the primary interface for the prompts system.
    """
    
    def __init__(self, config: PromptConfig):
        """Initialize the prompt system with configuration."""
        self.config = config
        
        # Initialize core components
        self.template_registry = TemplateRegistry()
        self.strategy_engine = StrategyEngine()
        self.template_engine = TemplateEngine()
        self.global_prompt_manager = GlobalPromptManager()
        
        # Load configuration
        self._load_config(config)
        
        # System metrics
        self.execution_count = 0
        self.error_count = 0
        self.fallback_count = 0
    
    def _load_config(self, config: PromptConfig) -> None:
        """Load configuration into system components."""
        # Load templates
        for template_id, template in config.templates.items():
            self.template_registry.register_template(template)
        
        # Load strategies
        for strategy_id, strategy in config.strategies.items():
            self.strategy_engine.register_strategy(strategy)
        
        # Load global prompts
        for global_prompt in config.global_prompts:
            self.global_prompt_manager.add_global_prompt(global_prompt)
        
        # Set default strategy
        self.default_strategy_id = config.default_strategy
    
    def execute_prompt(
        self,
        query: str,
        context: Optional[PromptContext] = None,
        variables: Optional[Dict[str, Any]] = None,
        strategy_override: Optional[str] = None,
        template_override: Optional[str] = None
    ) -> ExecutionContext:
        """
        Execute a prompt request with full orchestration.
        
        Args:
            query: The user's query/request
            context: Optional prompt context (created automatically if not provided)
            variables: Additional variables for template rendering
            strategy_override: Override strategy selection
            template_override: Override template selection
            
        Returns:
            ExecutionContext with execution results and metadata
        """
        # Create execution context
        execution_id = str(uuid.uuid4())
        
        if context is None:
            context = PromptContext(query=query)
        else:
            context.query = query
        
        exec_context = ExecutionContext(
            execution_id=execution_id,
            prompt_context=context
        )
        
        try:
            self.execution_count += 1
            
            # Step 1: Select template
            template_id, strategy_id = self._select_template(
                context, 
                strategy_override, 
                template_override
            )
            
            if not template_id:
                exec_context.add_error("No suitable template found")
                self.error_count += 1
                return exec_context
            
            exec_context.mark_template_selected(
                template_id, 
                strategy_id, 
                f"Selected via {strategy_id}"
            )
            
            # Step 2: Get template
            template = self.template_registry.get_template(template_id)
            if not template:
                exec_context.add_error(f"Template '{template_id}' not found in registry")
                self.error_count += 1
                return exec_context
            
            # Step 3: Prepare variables
            render_variables = self._prepare_variables(
                template, 
                context, 
                variables or {}
            )
            exec_context.input_variables = render_variables
            
            # Step 4: Apply global prompts and render
            rendered_prompt = self._render_with_global_prompts(
                template,
                render_variables,
                context,
                exec_context
            )
            
            exec_context.mark_rendering_complete(rendered_prompt)
            exec_context.mark_execution_complete()
            
            return exec_context
            
        except Exception as e:
            exec_context.add_error(f"Execution failed: {str(e)}")
            exec_context.mark_execution_complete()
            self.error_count += 1
            return exec_context
    
    def _select_template(
        self,
        context: PromptContext,
        strategy_override: Optional[str] = None,
        template_override: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """Select the best template for the given context."""
        
        # Direct template override
        if template_override:
            if self.template_registry.has_template(template_override):
                return template_override, "override"
            else:
                # Template override not found, continue with strategy
                pass
        
        # Use strategy to select template
        strategy_id = strategy_override or self.default_strategy_id
        strategy = self.strategy_engine.get_strategy(strategy_id)
        
        if not strategy:
            # Fallback to first available strategy
            available_strategies = self.strategy_engine.list_strategies()
            if available_strategies:
                strategy = available_strategies[0]
                strategy_id = strategy.strategy_id
            else:
                # No strategies available, try to find any template
                templates = self.template_registry.list_templates()
                if templates:
                    return templates[0].template_id, "fallback"
                return None, None
        
        # Get selection context
        selection_context = context.to_selection_context()
        
        # Select template using strategy
        selected_template_id = strategy.select_template(selection_context)
        
        if selected_template_id and self.template_registry.has_template(selected_template_id):
            return selected_template_id, strategy_id
        
        # Strategy didn't return a valid template, try fallback
        if hasattr(strategy, 'fallback_template') and strategy.fallback_template:
            if self.template_registry.has_template(strategy.fallback_template):
                self.fallback_count += 1
                return strategy.fallback_template, f"{strategy_id}_fallback"
        
        # Try fallback chain
        if hasattr(strategy, 'fallback_chain') and strategy.fallback_chain:
            for fallback_template_id in strategy.fallback_chain:
                if self.template_registry.has_template(fallback_template_id):
                    self.fallback_count += 1
                    return fallback_template_id, f"{strategy_id}_fallback_chain"
        
        # Last resort: any available template
        templates = self.template_registry.list_templates()
        if templates:
            self.fallback_count += 1
            return templates[0].template_id, "last_resort"
        
        return None, None
    
    def _prepare_variables(
        self,
        template: PromptTemplate,
        context: PromptContext,
        additional_variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare variables for template rendering."""
        variables = {}
        
        # Add context variables
        variables.update({
            "query": context.query,
            "user_id": context.user_id,
            "session_id": context.session_id,
            "domain": context.domain,
            "timestamp": context.timestamp.isoformat(),
        })
        
        # Add document context if available
        if context.documents:
            # Format documents for template
            formatted_docs = []
            for doc in context.documents:
                if isinstance(doc, dict):
                    content = doc.get("content", str(doc))
                    formatted_docs.append(content)
                else:
                    formatted_docs.append(str(doc))
            
            variables["context"] = "\\n\\n".join(formatted_docs)
            variables["documents"] = context.documents
            variables["document_count"] = context.document_count
        
        # Add custom attributes
        variables.update(context.custom_attributes)
        
        # Add additional variables (these override context variables)
        variables.update(additional_variables)
        
        return variables
    
    def _render_with_global_prompts(
        self,
        template: PromptTemplate,
        variables: Dict[str, Any],
        context: PromptContext,
        exec_context: ExecutionContext
    ) -> str:
        """Render template with global prompts applied."""
        
        # Get applicable global prompts
        global_prompts = self.global_prompt_manager.get_applicable_prompts(
            template.template_id,
            context.to_selection_context()
        )
        
        # Track applied global prompts
        exec_context.applied_global_prompts = [gp.global_id for gp in global_prompts]
        
        # Apply global prompts to template
        enhanced_template = self.global_prompt_manager.apply_global_prompts(
            template,
            global_prompts
        )
        
        # Render the enhanced template
        rendered_prompt = self.template_engine.render_template(
            enhanced_template,
            variables
        )
        
        return rendered_prompt
    
    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get a template by ID."""
        return self.template_registry.get_template(template_id)
    
    def list_templates(self, filter_by: Optional[Dict[str, Any]] = None) -> List[PromptTemplate]:
        """List available templates with optional filtering."""
        return self.template_registry.list_templates(filter_by)
    
    def search_templates(self, query: str) -> List[PromptTemplate]:
        """Search templates by query."""
        return self.template_registry.search_templates(query)
    
    def add_template(self, template: PromptTemplate) -> bool:
        """Add a new template to the system."""
        try:
            self.template_registry.register_template(template)
            return True
        except Exception:
            return False
    
    def remove_template(self, template_id: str) -> bool:
        """Remove a template from the system."""
        return self.template_registry.unregister_template(template_id)
    
    def validate_template(self, template: PromptTemplate) -> List[str]:
        """Validate a template and return any errors."""
        errors = []
        
        try:
            # Basic validation
            if not template.template_id:
                errors.append("Template ID is required")
            
            if not template.template:
                errors.append("Template content is required")
            
            # Test rendering with dummy variables
            dummy_vars = {}
            for var in template.input_variables:
                # Use validation rules to create appropriate dummy values
                if var in template.validation_rules:
                    rules = template.validation_rules[var]
                    var_type = rules.get("type", "str")
                    if var_type == "list":
                        dummy_vars[var] = [{"title": f"Test Document", "content": f"Test content for {var}"}]
                    elif var_type == "dict":
                        dummy_vars[var] = {"test_key": f"test_value_for_{var}"}
                    elif var_type == "int":
                        dummy_vars[var] = 42
                    elif var_type == "float":
                        dummy_vars[var] = 3.14
                    elif var_type == "bool":
                        dummy_vars[var] = True
                    else:
                        dummy_vars[var] = f"test_{var}"
                else:
                    dummy_vars[var] = f"test_{var}"
            
            self.template_engine.render_template(template, dummy_vars)
            
        except Exception as e:
            errors.append(f"Template validation failed: {str(e)}")
        
        return errors
    
    def test_template(
        self,
        template_id: str,
        test_variables: Dict[str, Any],
        test_context: Optional[PromptContext] = None
    ) -> ExecutionContext:
        """Test a template with provided variables and context."""
        if test_context is None:
            test_context = PromptContext(query="test query")
        
        return self.execute_prompt(
            query=test_context.query,
            context=test_context,
            variables=test_variables,
            template_override=template_id
        )
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "total_executions": self.execution_count,
            "total_errors": self.error_count,
            "total_fallbacks": self.fallback_count,
            "error_rate": self.error_count / max(1, self.execution_count),
            "fallback_rate": self.fallback_count / max(1, self.execution_count),
            "templates_count": len(self.template_registry.list_templates()),
            "strategies_count": len(self.strategy_engine.list_strategies()),
            "global_prompts_count": len(self.global_prompt_manager.list_global_prompts()),
        }
    
    def reset_stats(self) -> None:
        """Reset system statistics."""
        self.execution_count = 0
        self.error_count = 0
        self.fallback_count = 0
    
    def reload_config(self, config: PromptConfig) -> None:
        """Reload system configuration."""
        # Clear existing configuration
        self.template_registry.clear()
        self.strategy_engine.clear()
        self.global_prompt_manager.clear()
        
        # Reset stats
        self.reset_stats()
        
        # Load new configuration
        self.config = config
        self._load_config(config)