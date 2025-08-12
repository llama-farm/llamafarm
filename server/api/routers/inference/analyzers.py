import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

from core.logging import FastAPIStructLogger
from core.settings import settings

from .models import ProjectAction
from .strategies import (
    AnalysisStrategyFactory,
    ResponseValidationStrategy,
)

# Initialize logger
logger = FastAPIStructLogger()

# Constants
PROJECT_KEYWORDS = ["project", "list", "create", "show", "namespace"]

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
            logger.warning("Failed to initialize LLM analyzer client", error=str(e))
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

            return self.client.chat.completions.create(
                model=settings.ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this message: {message}"}
                ],
                response_model=ProjectAnalysis,
                temperature=0.1,
                max_retries=2
            )
            
            
        except Exception as e:
            logger.warning(
                "LLM analysis failed, falling back to rule-based", 
                error=str(e)
            )
            return self._fallback_analysis(message)
    
    def _fallback_analysis(self, message: str) -> ProjectAnalysis:
        """Fallback to rule-based analysis when LLM is unavailable"""
        # Use the new strategy-based approach
        strategy = AnalysisStrategyFactory.create_strategy("rule_based")
        result = strategy.analyze(message)
        
        return ProjectAnalysis(
            action=result["action"],
            namespace=result["namespace"],
            project_id=result["project_id"],
            confidence=result["confidence"],
            reasoning=result["reasoning"] + " (LLM unavailable)"
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
    
    # Class-level validation strategy instance
    _validation_strategy = None
    
    @classmethod
    def get_validation_strategy(cls) -> ResponseValidationStrategy:
        """Get or create validation strategy instance"""
        if cls._validation_strategy is None:
            cls._validation_strategy = (
                AnalysisStrategyFactory.create_validation_strategy()
            )
        return cls._validation_strategy
    
    @staticmethod
    def is_template_response(response: str) -> bool:
        """Detect if response contains template placeholders"""
        strategy = ResponseAnalyzer.get_validation_strategy()
        return strategy._is_template_response(response)

    @staticmethod
    def needs_manual_execution(response: str, message: str) -> bool:
        """Determine if manual tool execution is needed"""
        strategy = ResponseAnalyzer.get_validation_strategy()
        return strategy.needs_manual_execution(response, message) 