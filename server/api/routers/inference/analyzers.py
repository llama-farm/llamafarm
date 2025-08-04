import re
from typing import Optional
from .models import ProjectAction

# Constants
TEMPLATE_INDICATORS = [
    "[number of projects]", "[project list]", "[namespace]", "[projects]",
    "{{", "}}", "${", "[total]", "[count]", "**[list of projects",
    "**[projects", "[list of projects", "I will use the project tool",
    "To view the list, I will", "**[namespace", "currently **[",
]

PROJECT_KEYWORDS = ["project", "list", "create", "show", "namespace"]

NAMESPACE_PATTERNS = [
    r"in\s+(\w+)\s+namespace",
    r"namespace\s+(\w+)",
    r"in\s+(\w+)(?:\s|$)",
    r"from\s+(\w+)(?:\s|$)",
    r"(\w+)\s+namespace"
]

PROJECT_ID_PATTERNS = [
    r"create\s+(?:project\s+)?(?:called\s+)?['\"]?(\w+)['\"]?",
    r"new\s+project\s+['\"]?(\w+)['\"]?",
    r"project\s+['\"]?(\w+)['\"]?"
]

CREATE_KEYWORDS = ["create", "new", "add", "make"]
EXCLUDED_NAMESPACES = ["the", "a", "an", "my", "projects", "project"]

class MessageAnalyzer:
    """Handles message analysis and parameter extraction"""
    
    @staticmethod
    def extract_namespace(message: str) -> str:
        """Extract namespace from user message or return default"""
        message_lower = message.lower()
        
        # Look for explicit namespace mentions
        for pattern in NAMESPACE_PATTERNS:
            match = re.search(pattern, message_lower)
            if match:
                namespace = match.group(1)
                if namespace not in EXCLUDED_NAMESPACES:
                    return namespace
        
        return "test"  # default

    @staticmethod
    def extract_project_id(message: str) -> Optional[str]:
        """Extract project ID from create project messages"""
        message_lower = message.lower()
        
        for pattern in PROJECT_ID_PATTERNS:
            match = re.search(pattern, message_lower)
            if match:
                return match.group(1)
        
        return None

    @staticmethod
    def determine_action(message: str) -> ProjectAction:
        """Determine if user wants to create or list projects"""
        message_lower = message.lower()
        return ProjectAction.CREATE if any(word in message_lower for word in CREATE_KEYWORDS) else ProjectAction.LIST

    @staticmethod
    def is_project_related(message: str) -> bool:
        """Check if message is project-related"""
        return any(word in message.lower() for word in PROJECT_KEYWORDS)

class ResponseAnalyzer:
    """Handles response analysis and validation"""
    
    @staticmethod
    def is_template_response(response: str) -> bool:
        """Detect if response contains template placeholders"""
        response_lower = response.lower()
        return any(indicator.lower() in response_lower for indicator in TEMPLATE_INDICATORS)

    @staticmethod
    def needs_manual_execution(response: str, message: str) -> bool:
        """Determine if manual tool execution is needed"""
        if not MessageAnalyzer.is_project_related(message):
            return False
            
        return (
            ResponseAnalyzer.is_template_response(response) or
            len(response) < 50 or
            "I don't have access" in response or
            "cannot directly" in response or
            "[list of projects" in response or
            "**[" in response or
            "I will use the project tool" in response
        ) 