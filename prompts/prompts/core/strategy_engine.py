"""Strategy engine for prompt selection logic."""

from typing import Any, Dict, List, Optional
import logging

from ..models.strategy import PromptStrategy, StrategyType


logger = logging.getLogger(__name__)


class StrategyEngine:
    """
    Engine for managing and executing prompt selection strategies.
    """
    
    def __init__(self):
        """Initialize the strategy engine."""
        self.strategies: Dict[str, PromptStrategy] = {}
        self.default_strategy_id: Optional[str] = None
        
        # Strategy execution stats
        self.execution_stats: Dict[str, Dict[str, int]] = {}
    
    def register_strategy(self, strategy: PromptStrategy) -> None:
        """Register a new strategy."""
        self.strategies[strategy.strategy_id] = strategy
        
        # Initialize stats
        self.execution_stats[strategy.strategy_id] = {
            "total_executions": 0,
            "successful_selections": 0,
            "fallback_uses": 0,
            "errors": 0
        }
        
        logger.info(f"Registered strategy: {strategy.strategy_id}")
    
    def unregister_strategy(self, strategy_id: str) -> bool:
        """Unregister a strategy."""
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
            if strategy_id in self.execution_stats:
                del self.execution_stats[strategy_id]
            logger.info(f"Unregistered strategy: {strategy_id}")
            return True
        return False
    
    def get_strategy(self, strategy_id: str) -> Optional[PromptStrategy]:
        """Get a strategy by ID."""
        return self.strategies.get(strategy_id)
    
    def list_strategies(self) -> List[PromptStrategy]:
        """List all registered strategies."""
        return list(self.strategies.values())
    
    def select_template(
        self,
        strategy_id: str,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Select a template using the specified strategy.
        
        Args:
            strategy_id: ID of strategy to use
            context: Context for template selection
            
        Returns:
            Selected template ID or None
        """
        strategy = self.strategies.get(strategy_id)
        if not strategy:
            logger.error(f"Strategy not found: {strategy_id}")
            return None
        
        # Update execution stats
        stats = self.execution_stats.get(strategy_id, {})
        stats["total_executions"] = stats.get("total_executions", 0) + 1
        
        try:
            # Execute strategy selection
            selected_template = self._execute_strategy(strategy, context)
            
            if selected_template:
                stats["successful_selections"] = stats.get("successful_selections", 0) + 1
                logger.debug(f"Strategy {strategy_id} selected template: {selected_template}")
            else:
                logger.warning(f"Strategy {strategy_id} returned no template")
            
            return selected_template
            
        except Exception as e:
            stats["errors"] = stats.get("errors", 0) + 1
            logger.error(f"Error in strategy {strategy_id}: {str(e)}")
            return None
    
    def _execute_strategy(
        self,
        strategy: PromptStrategy,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Execute a specific strategy."""
        if not strategy.enabled:
            return None
        
        if strategy.type == StrategyType.STATIC:
            return self._execute_static_strategy(strategy, context)
        elif strategy.type == StrategyType.RULE_BASED:
            return self._execute_rule_based_strategy(strategy, context)
        elif strategy.type == StrategyType.CONTEXT_AWARE:
            return self._execute_context_aware_strategy(strategy, context)
        elif strategy.type == StrategyType.A_B_TEST:
            return self._execute_ab_test_strategy(strategy, context)
        elif strategy.type == StrategyType.PERFORMANCE_BASED:
            return self._execute_performance_based_strategy(strategy, context)
        elif strategy.type == StrategyType.ML_DRIVEN:
            return self._execute_ml_driven_strategy(strategy, context)
        elif strategy.type == StrategyType.LANGGRAPH_ORCHESTRATED:
            return self._execute_langgraph_strategy(strategy, context)
        else:
            logger.warning(f"Unknown strategy type: {strategy.type}")
            return None
    
    def _execute_static_strategy(
        self,
        strategy: PromptStrategy,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Execute static strategy (always returns the same template)."""
        return strategy.config.get("default_template")
    
    def _execute_rule_based_strategy(
        self,
        strategy: PromptStrategy,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Execute rule-based strategy."""
        # Sort rules by priority (lower number = higher priority) 
        sorted_rules = sorted(strategy.rules, key=lambda r: r.priority)
        
        for rule in sorted_rules:
            if rule.evaluate(context):
                logger.debug(f"Rule {rule.rule_id} matched, selecting template {rule.template_id}")
                return rule.template_id
        
        # No rules matched, use fallback
        return strategy.fallback_template
    
    def _execute_context_aware_strategy(
        self,
        strategy: PromptStrategy,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Execute context-aware strategy."""
        # Enhanced context-aware selection logic
        
        # Check for domain-specific templates
        domain = context.get("domain", "general")
        if domain != "general":
            domain_template = f"{domain}_specialized"
            # In a real implementation, you'd check if this template exists
            
        # Check complexity level
        complexity = context.get("query_complexity", "low")
        complexity_template = strategy.config.get(f"{complexity}_complexity_template")
        if complexity_template:
            return complexity_template
        
        # Check document types
        doc_types = context.get("document_types", [])
        if doc_types:
            primary_type = doc_types[0]
            type_template = strategy.config.get(f"{primary_type}_template")
            if type_template:
                return type_template
        
        # Check user role
        user_role = context.get("user_role")
        if user_role:
            role_template = strategy.config.get(f"{user_role}_template")
            if role_template:
                return role_template
        
        # Fallback to rule-based selection
        return self._execute_rule_based_strategy(strategy, context)
    
    def _execute_ab_test_strategy(
        self,
        strategy: PromptStrategy,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Execute A/B testing strategy."""
        import hashlib
        
        # Get user identifier for consistent assignment
        user_id = context.get("user_id", context.get("session_id", "anonymous"))
        
        # Create hash for consistent assignment
        hash_obj = hashlib.md5(f"{strategy.strategy_id}_{user_id}".encode())
        hash_value = int(hash_obj.hexdigest(), 16) % 100
        
        # Determine variant based on traffic split
        cumulative = 0
        for variant in strategy.variants:
            template_id = variant["template_id"]
            split_percent = strategy.traffic_split.get(template_id, 0) * 100
            
            if hash_value < cumulative + split_percent:
                logger.debug(f"A/B test assigned user {user_id} to variant {template_id}")
                return template_id
            
            cumulative += split_percent
        
        # If no variant assigned, use fallback
        return strategy.fallback_template
    
    def _execute_performance_based_strategy(
        self,
        strategy: PromptStrategy,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Execute performance-based strategy."""
        # This would integrate with monitoring/metrics system
        # For now, fallback to rule-based selection
        logger.debug("Performance-based strategy not fully implemented, using rule-based fallback")
        return self._execute_rule_based_strategy(strategy, context)
    
    def _execute_ml_driven_strategy(
        self,
        strategy: PromptStrategy,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Execute ML-driven strategy."""
        # This would use trained models for template selection
        # For now, fallback to context-aware selection
        logger.debug("ML-driven strategy not fully implemented, using context-aware fallback")
        return self._execute_context_aware_strategy(strategy, context)
    
    def _execute_langgraph_strategy(
        self,
        strategy: PromptStrategy,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Execute LangGraph-orchestrated strategy."""
        # This would integrate with LangGraph workflows
        # For now, fallback to rule-based selection
        logger.debug("LangGraph strategy not fully implemented, using rule-based fallback")
        return self._execute_rule_based_strategy(strategy, context)
    
    def validate_strategy(self, strategy: PromptStrategy) -> List[str]:
        """Validate a strategy configuration."""
        errors = []
        
        # Basic validation
        if not strategy.strategy_id:
            errors.append("Strategy ID is required")
        
        if not strategy.name:
            errors.append("Strategy name is required")
        
        # Type-specific validation
        if strategy.type == StrategyType.STATIC:
            if not strategy.config.get("default_template"):
                errors.append("Static strategy requires default_template in config")
        
        elif strategy.type == StrategyType.RULE_BASED:
            if not strategy.rules:
                errors.append("Rule-based strategy requires at least one rule")
            
            # Validate rules
            rule_ids = set()
            for rule in strategy.rules:
                if rule.rule_id in rule_ids:
                    errors.append(f"Duplicate rule ID: {rule.rule_id}")
                rule_ids.add(rule.rule_id)
        
        elif strategy.type == StrategyType.A_B_TEST:
            if not strategy.variants:
                errors.append("A/B test strategy requires variants")
            
            if not strategy.traffic_split:
                errors.append("A/B test strategy requires traffic_split")
            
            # Validate traffic split sums to 1.0
            total_split = sum(strategy.traffic_split.values())
            if abs(total_split - 1.0) > 0.01:
                errors.append(f"Traffic split must sum to 1.0, got {total_split}")
        
        return errors
    
    def test_strategy(
        self,
        strategy_id: str,
        test_contexts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Test a strategy with multiple contexts."""
        strategy = self.strategies.get(strategy_id)
        if not strategy:
            return {"error": f"Strategy {strategy_id} not found"}
        
        results = {
            "strategy_id": strategy_id,
            "test_cases": [],
            "summary": {
                "total_tests": len(test_contexts),
                "successful_selections": 0,
                "fallback_uses": 0,
                "errors": 0
            }
        }
        
        for i, context in enumerate(test_contexts):
            try:
                selected_template = self._execute_strategy(strategy, context)
                
                test_result = {
                    "test_case": i + 1,
                    "context": context,
                    "selected_template": selected_template,
                    "success": selected_template is not None,
                    "used_fallback": selected_template == strategy.fallback_template
                }
                
                results["test_cases"].append(test_result)
                
                if selected_template:
                    results["summary"]["successful_selections"] += 1
                    if selected_template == strategy.fallback_template:
                        results["summary"]["fallback_uses"] += 1
                        
            except Exception as e:
                test_result = {
                    "test_case": i + 1,
                    "context": context,
                    "error": str(e),
                    "success": False
                }
                results["test_cases"].append(test_result)
                results["summary"]["errors"] += 1
        
        return results
    
    def get_strategy_stats(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get execution statistics for a strategy."""
        if strategy_id not in self.execution_stats:
            return None
        
        stats = self.execution_stats[strategy_id].copy()
        
        # Calculate derived metrics
        total_executions = stats.get("total_executions", 0)
        if total_executions > 0:
            stats["success_rate"] = stats.get("successful_selections", 0) / total_executions
            stats["fallback_rate"] = stats.get("fallback_uses", 0) / total_executions
            stats["error_rate"] = stats.get("errors", 0) / total_executions
        else:
            stats["success_rate"] = 0.0
            stats["fallback_rate"] = 0.0
            stats["error_rate"] = 0.0
        
        return stats
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all strategies."""
        return {
            strategy_id: self.get_strategy_stats(strategy_id)
            for strategy_id in self.strategies.keys()
        }
    
    def reset_stats(self, strategy_id: Optional[str] = None) -> None:
        """Reset statistics for a strategy or all strategies."""
        if strategy_id:
            if strategy_id in self.execution_stats:
                self.execution_stats[strategy_id] = {
                    "total_executions": 0,
                    "successful_selections": 0,
                    "fallback_uses": 0,
                    "errors": 0
                }
        else:
            for sid in self.execution_stats:
                self.execution_stats[sid] = {
                    "total_executions": 0,
                    "successful_selections": 0,
                    "fallback_uses": 0,
                    "errors": 0
                }
    
    def clear(self) -> None:
        """Clear all strategies and stats."""
        self.strategies.clear()
        self.execution_stats.clear()
        self.default_strategy_id = None