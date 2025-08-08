"""
Strategy classes for handling analysis and response validation logic.
Extracted from analyzers.py for better maintainability and testability.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.logging import FastAPIStructLogger

# Initialize logger
logger = FastAPIStructLogger()

# Lazy import to avoid circular dependencies
def get_config_loader():
    from .config_loader import ConfigLoader
    return ConfigLoader


@dataclass
class AnalysisRule:
    """Configuration for a single analysis rule"""
    name: str
    patterns: list[str]
    keywords: list[str]
    weight: float = 1.0
    enabled: bool = True


@dataclass
class ResponseValidationConfig:
    """Configuration for response validation strategies"""
    template_indicators: list[str]
    inability_phrases: list[str]
    hallucination_indicators: list[str]
    min_response_length: int = 50
    enable_hallucination_detection: bool = True
    enable_count_query_validation: bool = True


class BaseAnalysisStrategy(ABC):
    """Base class for analysis strategies"""
    
    @abstractmethod
    def analyze(self, message: str, context: dict | None = None) -> dict:
        """Analyze a message and return analysis results"""
        pass


class RuleBasedAnalysisStrategy(BaseAnalysisStrategy):
    """Rule-based analysis strategy using configurable patterns"""
    
    def __init__(self, config: dict | None = None):
        if config is None:
            # Load configuration from file
            config_loader = get_config_loader()
            full_config = config_loader.load_config()
            rules_config = config_loader.create_analysis_rules(full_config)
            
            self.namespace_rules = rules_config["namespace_rules"]
            self.action_rules = rules_config["action_rules"]
            self.excluded_namespaces = rules_config["excluded_namespaces"]
            self.config = {
                "default_namespace": rules_config["default_namespace"],
                "confidence_threshold": rules_config["confidence_threshold"]
            }
        else:
            # Use provided config (for testing or custom setups),
            # but merge with sane defaults
            config_loader = get_config_loader()
            defaults_full = config_loader.load_config()
            defaults_rules = config_loader.create_analysis_rules(defaults_full)

            self.namespace_rules = config.get(
                "namespace_rules", defaults_rules["namespace_rules"]
            )
            self.action_rules = config.get(
                "action_rules", defaults_rules["action_rules"]
            )
            self.excluded_namespaces = set(
                config.get(
                    "excluded_namespaces", defaults_rules["excluded_namespaces"]
                )
            )
            self.config = {
                "default_namespace": config.get(
                    "default_namespace", defaults_rules["default_namespace"]
                ),
                "confidence_threshold": config.get(
                    "confidence_threshold", defaults_rules["confidence_threshold"]
                ),
            }
    

    
    def analyze(self, message: str, context: dict | None = None) -> dict:
        """Analyze message using rule-based approach"""
        context = context or {}
        message_lower = message.lower()
        
        # Determine action
        action_score = {"create": 0, "list": 0}
        for rule in self.action_rules:
            if not rule.enabled:
                continue
                
            keyword_matches = sum(
                1 for keyword in rule.keywords if keyword in message_lower)
            if keyword_matches > 0:
                # Extract action type from rule name
                if "create" in rule.name.lower():
                    action_score["create"] += keyword_matches * rule.weight
                elif "list" in rule.name.lower():
                    action_score["list"] += keyword_matches * rule.weight
        
        action = "create" if action_score["create"] > action_score["list"] else "list"
        
        # Extract namespace
        namespace = self._extract_namespace(message_lower)
        
        # Extract project ID for create actions
        project_id = (
            self._extract_project_id(message_lower) 
            if action == "create" else None
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            action_score, namespace, project_id, action
        )
        
        return {
            "action": action,
            "namespace": namespace,
            "project_id": project_id,
            "confidence": confidence,
            "reasoning": (
                f"Rule-based analysis: action={action}, namespace={namespace}"
            )
        }
    
    def _extract_namespace(self, message_lower: str) -> str:
        """Extract namespace using configurable rules"""
        import re
        
        for rule in self.namespace_rules:
            if not rule.enabled:
                continue
                
            for pattern in rule.patterns:
                if match := re.search(pattern, message_lower):
                    namespace = match.group(1)
                    if namespace not in self.excluded_namespaces:
                        return namespace
        
        return self.config["default_namespace"]
    
    def _extract_project_id(self, message_lower: str) -> str | None:
        """Extract project ID using configurable rules"""
        import re
        
        # Use patterns from action rules that contain project ID patterns
        for rule in self.action_rules:
            if not rule.enabled or "create" not in rule.name.lower():
                continue
                
            for pattern in rule.patterns:
                if match := re.search(pattern, message_lower):
                    return match.group(1)
        
        # Fallback regex coverage to catch common phrasing when rules miss
        fallback_patterns = [
            r"create\s+(?:project\s+)?(?:called\s+)?['\"]?([A-Za-z0-9._-]+)['\"]?",
            r"new\s+project\s+['\"]?([A-Za-z0-9._-]+)['\"]?",
            r"project\s+['\"]?([A-Za-z0-9._-]+)['\"]?",
        ]
        for pattern in fallback_patterns:
            if match := re.search(pattern, message_lower):
                return match.group(1)

        return None
    
    def _calculate_confidence(
        self, action_score: dict, namespace: str, project_id: str | None, action: str
        ) -> float:
        """Calculate confidence score based on analysis results"""
        base_confidence = 0.7  # Base confidence for rule-based
        
        # Boost confidence if we found specific indicators
        if max(action_score.values()) > 1:
            base_confidence += 0.1
        
        # Boost if namespace was explicitly mentioned
        if namespace != self.config["default_namespace"]:
            base_confidence += 0.1
        
        # Boost if project_id found for create actions
        if action == "create" and project_id:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)


class ResponseValidationStrategy:
    """
    Strategy for validating responses and determining if manual execution is needed
    """
    
    def __init__(self, config: ResponseValidationConfig | None = None):
        # Load defaults from file
        config_loader = get_config_loader()
        full_config = config_loader.load_config()
        defaults = config_loader.create_validation_config(full_config)

        if config is None:
            self.config = defaults
        else:
            # Merge provided config with defaults to ensure all fields are populated
            self.config = ResponseValidationConfig(
                template_indicators=(
                    config.template_indicators or defaults.template_indicators
                ),
                inability_phrases=(
                    config.inability_phrases or defaults.inability_phrases
                ),
                hallucination_indicators=(
                    config.hallucination_indicators
                    or defaults.hallucination_indicators
                ),
                min_response_length=(
                    config.min_response_length or defaults.min_response_length
                ),
                enable_hallucination_detection=(
                    config.enable_hallucination_detection
                    if config.enable_hallucination_detection is not None
                    else defaults.enable_hallucination_detection
                ),
                enable_count_query_validation=(
                    config.enable_count_query_validation
                    if config.enable_count_query_validation is not None
                    else defaults.enable_count_query_validation
                ),
            )
    

    
    def needs_manual_execution(self, response: str, original_message: str) -> bool:
        """Determine if manual tool execution is needed"""
        from .analyzers import MessageAnalyzer  # Import here to avoid circular import
        
        if not MessageAnalyzer.is_project_related(original_message):
            return False
        
        # Check template response
        if self._is_template_response(response):
            logger.info("Template response detected, triggering manual execution")
            return True
        
        # Check inability statements
        if self._contains_inability_phrases(response):
            logger.info("Inability phrases detected, triggering manual execution")
            return True
        
        # Check response length
        if len(response.strip()) < self.config.min_response_length:
            logger.info("Response too short, triggering manual execution")
            return True
        
        # Check for hallucinated data
        if (
            self.config.enable_hallucination_detection 
            and self._is_hallucinated_response(response)
        ):
            logger.info(
                "Hallucinated response detected, triggering manual execution"
            )
            return True
        
        # Check count queries
        if (
            self.config.enable_count_query_validation 
            and self._is_suspicious_count_response(response, original_message)
        ):
            logger.info(
                "Suspicious count response detected, triggering manual execution"
            )
            return True
        
        return False
    
    def _is_template_response(self, response: str) -> bool:
        """Check if response contains template placeholders"""
        response_lower = response.lower()
        return any(
            indicator.lower() in response_lower 
            for indicator in self.config.template_indicators
        )
    
    def _contains_inability_phrases(self, response: str) -> bool:
        """Check if response contains inability statements"""
        response_lower = response.lower()
        return any(
            phrase in response_lower
            for phrase in self.config.inability_phrases
        )
    
    def _is_hallucinated_response(self, response: str) -> bool:
        """Check for signs of hallucinated project data"""
        response_lower = response.lower()
        return any(
            indicator in response_lower
            for indicator in self.config.hallucination_indicators
        )
    
    def _is_suspicious_count_response(
        self, response: str, original_message: str
    ) -> bool:
        """Check for suspicious numeric responses to count queries"""
        original_lower = original_message.lower()
        response_lower = response.lower()
        
        # Check if original message is asking for counts
        count_keywords = ["how many", "count", "number of", "total"]
        is_count_query = any(keyword in original_lower for keyword in count_keywords)
        
        if not is_count_query:
            return False
        
        # Check if response contains numbers but not "found"
        has_numbers = any(char.isdigit() for char in response)
        has_found = "found" in response_lower
        
        return has_numbers and not has_found


class AnalysisStrategyFactory:
    """Factory for creating analysis strategies"""
    
    @staticmethod
    def create_strategy(
        strategy_type: str, config: dict | None = None) -> BaseAnalysisStrategy:
        """Create an analysis strategy of the specified type"""
        if strategy_type == "rule_based":
            return RuleBasedAnalysisStrategy(config)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    @staticmethod
    def create_validation_strategy(
        config: ResponseValidationConfig | None = None
        ) -> ResponseValidationStrategy:
        """Create a response validation strategy"""
        return ResponseValidationStrategy(config)