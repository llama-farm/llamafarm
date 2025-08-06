import re

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

from core.settings import settings

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

# Structured output models for LLM-based analysis
class ProjectAnalysis(BaseModel):
    """Structured output for project-related message analysis"""
    action: str = Field(description="The action to take: 'create' or 'list'")
    namespace: str | None = Field(
        description="The namespace mentioned, or None if not specified")
    project_id: str | None = Field(
        description="The project ID/name for create actions, or None if not specified")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Brief explanation of the analysis")

class LLMAnalyzer:
    """LLM-based message analyzer for more flexible project action detection"""
    
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the instructor client for structured outputs"""
        try:
            ollama_client = OpenAI(
                base_url=settings.ollama_host,
                api_key=settings.ollama_api_key,
            )
            self.client = instructor.from_openai(
                ollama_client, 
                mode=instructor.Mode.JSON,
                )
        except Exception as e:
            print(f"Warning: Failed to initialize LLM analyzer client: {e}")
            self.client = None
    
    def analyze_project_intent(self, message: str) -> ProjectAnalysis:
        """
        Use LLM to analyze user intent for project-related actions.
        Falls back to rule-based analysis if LLM is unavailable.
        """
        if not self.client:
            return self._fallback_analysis(message)
        
        try:
            system_prompt = """
You are an expert at analyzing user messages to determine project management actions.

Analyze the user's message and determine:
1. What action they want to take (create or list)
2. If they specified a namespace
3. If they specified a project ID/name (for create actions)
4. Your confidence in this analysis
5. Brief reasoning for your decision

Rules:
- "create", "new", "add", "make" usually indicate CREATE action
- "list", "show", "display", "view", "get" usually indicate LIST action
- Look for namespace patterns like "in X namespace", "namespace X", "in X"
- For create actions, look for project names/IDs
- Default namespace is "test" if not specified
- Be flexible with natural language variations

Examples:
- "create project myapp" → create, namespace: test, project_id: myapp
- "list projects in production" → list, namespace: production, project_id: null
- "show me my projects" → list, namespace: test, project_id: null
- "make a new project called demo in dev namespace" 
→ create, namespace: dev, project_id: demo
"""

            response = self.client.chat.completions.create(
                model=settings.ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this message: {message}"}
                ],
                response_model=ProjectAnalysis,
                temperature=0.1,
                max_retries=2
            )
            
            return response
            
        except Exception as e:
            print(f"Warning: LLM analysis failed, falling back to rule-based: {e}")
            return self._fallback_analysis(message)
    
    def _fallback_analysis(self, message: str) -> ProjectAnalysis:
        """Fallback to rule-based analysis when LLM is unavailable"""
        action = MessageAnalyzer.determine_action_legacy(message)
        namespace = MessageAnalyzer.extract_namespace(message)
        project_id = MessageAnalyzer.extract_project_id(message) 
        if action == ProjectAction.CREATE:
            project_id = MessageAnalyzer.extract_project_id(message)
        else:
            project_id = None
        
        return ProjectAnalysis(
            action=action.value,
            namespace=namespace,
            project_id=project_id,
            confidence=0.7,  # Lower confidence for rule-based
            reasoning="Rule-based fallback analysis (LLM unavailable)"
        )

class MessageAnalyzer:
    """Handles message analysis and parameter extraction"""
    
    # Class-level LLM analyzer instance
    _llm_analyzer = None
    
    @classmethod
    def get_llm_analyzer(cls) -> LLMAnalyzer:
        """Get or create LLM analyzer instance"""
        if cls._llm_analyzer is None:
            cls._llm_analyzer = LLMAnalyzer()
        return cls._llm_analyzer
    
    @staticmethod
    def analyze_with_llm(
        message: str,
        request_namespace: str | None = None,
        request_project_id: str | None = None,
    ) -> ProjectAnalysis:
        """
        Enhanced analysis using LLM with request field override support.
        This is the new primary method for message analysis.
        """
        # Get LLM analysis
        analyzer = MessageAnalyzer.get_llm_analyzer()
        analysis = analyzer.analyze_project_intent(message)
        
        # Override with request fields if provided (suggestion 2)
        if request_namespace is not None:
            analysis.namespace = request_namespace
            analysis.reasoning += " (namespace overridden from request field)"
        
        if request_project_id is not None:
            analysis.project_id = request_project_id
            analysis.reasoning += " (project_id overridden from request field)"
        
        # Use default namespace if still None
        if analysis.namespace is None:
            analysis.namespace = "test"
        
        return analysis
    
    @staticmethod
    def extract_namespace(message: str) -> str:
        """Extract namespace from user message or return default (legacy method)"""
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
    def extract_project_id(message: str) -> str | None:
        """Extract project ID from create project messages (legacy method)"""
        message_lower = message.lower()
        
        for pattern in PROJECT_ID_PATTERNS:
            match = re.search(pattern, message_lower)
            if match:
                return match.group(1)
        
        return None

    @staticmethod
    def determine_action_legacy(message: str) -> ProjectAction:
        """Determine if user wants to create or list projects (legacy method)"""
        message_lower = message.lower()
        return ProjectAction.CREATE if any(
            word in message_lower for word in CREATE_KEYWORDS) else ProjectAction.LIST
    
    @staticmethod
    def determine_action(message: str) -> ProjectAction:
        """Determine if user wants to create or list projects (enhanced method)"""
        analysis = MessageAnalyzer.analyze_with_llm(message)
        return (
            ProjectAction.CREATE
            if analysis.action.lower() == "create"
            else ProjectAction.LIST
        )

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
        return any(
            indicator.lower() in response_lower for indicator in TEMPLATE_INDICATORS)

    @staticmethod
    def needs_manual_execution(response: str, message: str) -> bool:
        """Determine if manual tool execution is needed"""
        if not MessageAnalyzer.is_project_related(message):
            return False
        
        # Check for obvious template/placeholder responses
        if ResponseAnalyzer.is_template_response(response):
            return True
            
        # Check for explicit inability statements
        if any(phrase in response.lower() for phrase in [
            "i don't have access", "cannot directly", "i will use the project tool",
            "let me check", "i'll need to", "i need to check"
        ]):
            return True
        
        # Check for very short responses (likely incomplete)
        if len(response.strip()) < 50:
            return True
        
        # NEW: Check for signs of hallucinated project data
        # If the response contains specific project information but seems generic/fake
        response_lower = response.lower()
        hallucination_indicators = [
            "project 1", "project 2", "project 3",  # Generic project names
            "example project", "sample project", "test project",
            "your projects:", "following projects:", "* project",
            "you have the following", "here are your projects"
        ]
        
        # If it looks like hallucinated project data, force tool execution
        if any(indicator in response_lower for indicator in hallucination_indicators):
            print(
                "🔧 [ResponseAnalyzer] Detected potential hallucinated project data, "
                "forcing tool execution"
                )
            return True
        
        # For project queries asking for specific counts/numbers, be more aggressive
        if (
            any(word in message.lower() for word in 
            ["how many", "count", "number of", "total"]) 
            and
            any(char.isdigit() for char in response) and "found" not in response_lower):
            print(
                "🔧 [ResponseAnalyzer] Count query with suspicious numeric response, "
                "forcing tool execution"
                )
            return True
        
        return False 