"""Prompt strategy data models."""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class StrategyType(str, Enum):
    """Types of prompt selection strategies."""
    STATIC = "static"
    RULE_BASED = "rule_based"
    CONTEXT_AWARE = "context_aware"
    ML_DRIVEN = "ml_driven"
    A_B_TEST = "a_b_test"
    PERFORMANCE_BASED = "performance_based"
    LANGGRAPH_ORCHESTRATED = "langgraph_orchestrated"


class RuleOperator(str, Enum):
    """Operators for rule conditions."""
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX_MATCH = "regex_match"


class StrategyRule(BaseModel):
    """Individual rule for rule-based strategy."""
    
    rule_id: str = Field(..., description="Unique rule identifier")
    name: Optional[str] = Field(None, description="Human-readable rule name")
    description: Optional[str] = Field(None, description="Rule description")
    
    # Condition specification
    field: str = Field(..., description="Context field to evaluate")
    operator: RuleOperator = Field(..., description="Comparison operator")
    value: Any = Field(..., description="Value to compare against")
    
    # Rule behavior
    template_id: str = Field(..., description="Template to select if rule matches")
    priority: int = Field(default=100, description="Rule priority (lower = higher priority)")
    enabled: bool = Field(default=True, description="Whether rule is active")
    
    # Advanced conditions
    additional_conditions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Additional AND conditions"
    )
    or_conditions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Alternative OR conditions"
    )
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Rule tags")
    created_by: Optional[str] = Field(None, description="Rule author")
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate if this rule matches the given context."""
        if not self.enabled:
            return False
        
        # Evaluate main condition
        if not self._evaluate_condition(self.field, self.operator, self.value, context):
            return False
        
        # Evaluate additional AND conditions
        for condition in self.additional_conditions:
            if not self._evaluate_condition(
                condition["field"],
                RuleOperator(condition["operator"]),
                condition["value"],
                context
            ):
                return False
        
        # Evaluate OR conditions (at least one must match if specified)
        if self.or_conditions:
            or_matched = False
            for condition in self.or_conditions:
                if self._evaluate_condition(
                    condition["field"],
                    RuleOperator(condition["operator"]),
                    condition["value"],
                    context
                ):
                    or_matched = True
                    break
            if not or_matched:
                return False
        
        return True
    
    def _evaluate_condition(
        self, 
        field: str, 
        operator: RuleOperator, 
        value: Any, 
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate a single condition."""
        # Get field value from context (support nested fields)
        field_value = self._get_nested_value(context, field)
        
        if operator == RuleOperator.EQUALS:
            return field_value == value
        elif operator == RuleOperator.NOT_EQUALS:
            return field_value != value
        elif operator == RuleOperator.GREATER_THAN:
            return field_value > value
        elif operator == RuleOperator.LESS_THAN:
            return field_value < value
        elif operator == RuleOperator.GREATER_EQUAL:
            return field_value >= value
        elif operator == RuleOperator.LESS_EQUAL:
            return field_value <= value
        elif operator == RuleOperator.IN:
            return field_value in value
        elif operator == RuleOperator.NOT_IN:
            return field_value not in value
        elif operator == RuleOperator.CONTAINS:
            return str(value) in str(field_value)
        elif operator == RuleOperator.NOT_CONTAINS:
            return str(value) not in str(field_value)
        elif operator == RuleOperator.STARTS_WITH:
            return str(field_value).startswith(str(value))
        elif operator == RuleOperator.ENDS_WITH:
            return str(field_value).endswith(str(value))
        elif operator == RuleOperator.REGEX_MATCH:
            import re
            return bool(re.match(str(value), str(field_value)))
        
        return False
    
    def _get_nested_value(self, context: Dict[str, Any], field: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = field.split('.')
        value = context
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value


class PromptStrategy(BaseModel):
    """Prompt selection strategy configuration."""
    
    strategy_id: str = Field(..., description="Unique strategy identifier")
    name: str = Field(..., description="Human-readable strategy name")
    type: StrategyType = Field(..., description="Strategy type")
    description: Optional[str] = Field(None, description="Strategy description")
    
    # Strategy configuration
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy-specific configuration"
    )
    
    # Rule-based strategy
    rules: List[StrategyRule] = Field(
        default_factory=list,
        description="Rules for rule-based strategies"
    )
    
    # Fallback behavior
    fallback_template: Optional[str] = Field(
        None,
        description="Template to use if no rules match"
    )
    fallback_chain: List[str] = Field(
        default_factory=list,
        description="Chain of fallback templates"
    )
    
    # A/B testing
    variants: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Template variants for A/B testing"
    )
    traffic_split: Dict[str, float] = Field(
        default_factory=dict,
        description="Traffic split percentages for variants"
    )
    
    # Performance tracking
    performance_metrics: List[str] = Field(
        default_factory=list,
        description="Metrics to track for this strategy"
    )
    
    # LangGraph integration
    langgraph_workflow: Optional[str] = Field(
        None,
        description="LangGraph workflow ID for orchestrated strategies"
    )
    
    # Metadata
    enabled: bool = Field(default=True, description="Whether strategy is active")
    priority: int = Field(default=100, description="Strategy priority")
    tags: List[str] = Field(default_factory=list, description="Strategy tags")
    created_by: Optional[str] = Field(None, description="Strategy author")
    
    def select_template(self, context: Dict[str, Any]) -> Optional[str]:
        """Select template ID based on strategy and context."""
        if not self.enabled:
            return None
        
        if self.type == StrategyType.STATIC:
            return self.config.get("default_template")
        
        elif self.type == StrategyType.RULE_BASED:
            return self._select_by_rules(context)
        
        elif self.type == StrategyType.CONTEXT_AWARE:
            return self._select_context_aware(context)
        
        elif self.type == StrategyType.A_B_TEST:
            return self._select_a_b_test(context)
        
        elif self.type == StrategyType.PERFORMANCE_BASED:
            return self._select_performance_based(context)
        
        # Fallback to default
        return self.fallback_template
    
    def _select_by_rules(self, context: Dict[str, Any]) -> Optional[str]:
        """Select template using rule-based logic."""
        # Sort rules by priority (lower number = higher priority)
        sorted_rules = sorted(self.rules, key=lambda r: r.priority)
        
        for rule in sorted_rules:
            if rule.evaluate(context):
                return rule.template_id
        
        return self.fallback_template
    
    def _select_context_aware(self, context: Dict[str, Any]) -> Optional[str]:
        """Select template using context-aware logic."""
        # Implementation for context-aware selection
        # This would typically involve ML models or heuristics
        
        # For now, fall back to rule-based selection
        return self._select_by_rules(context)
    
    def _select_a_b_test(self, context: Dict[str, Any]) -> Optional[str]:
        """Select template for A/B testing."""
        import random
        
        if not self.variants:
            return self.fallback_template
        
        # Get user ID or session ID for consistent assignment
        user_id = context.get("user_id", context.get("session_id", "default"))
        
        # Use hash of user ID for consistent assignment
        import hashlib
        hash_obj = hashlib.md5(str(user_id).encode())
        hash_value = int(hash_obj.hexdigest(), 16) % 100
        
        # Determine which variant based on traffic split
        cumulative = 0
        for variant in self.variants:
            template_id = variant["template_id"]
            split_percent = self.traffic_split.get(template_id, 0) * 100
            
            if hash_value < cumulative + split_percent:
                return template_id
            
            cumulative += split_percent
        
        return self.fallback_template
    
    def _select_performance_based(self, context: Dict[str, Any]) -> Optional[str]:
        """Select template based on performance metrics."""
        # This would integrate with monitoring system
        # For now, fall back to rule-based selection
        return self._select_by_rules(context)
    
    def add_rule(self, rule: StrategyRule) -> None:
        """Add a new rule to the strategy."""
        self.rules.append(rule)
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule from the strategy."""
        original_count = len(self.rules)
        self.rules = [r for r in self.rules if r.rule_id != rule_id]
        return len(self.rules) < original_count
    
    def get_rule(self, rule_id: str) -> Optional[StrategyRule]:
        """Get a specific rule by ID."""
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        return None
    
    def validate_config(self) -> List[str]:
        """Validate strategy configuration and return any errors."""
        errors = []
        
        if self.type == StrategyType.RULE_BASED and not self.rules:
            errors.append("Rule-based strategy requires at least one rule")
        
        if self.type == StrategyType.A_B_TEST:
            if not self.variants:
                errors.append("A/B test strategy requires variants")
            
            total_split = sum(self.traffic_split.values())
            if abs(total_split - 1.0) > 0.01:
                errors.append(f"Traffic split must sum to 1.0, got {total_split}")
        
        # Validate rule IDs are unique
        rule_ids = [r.rule_id for r in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            errors.append("Rule IDs must be unique")
        
        return errors