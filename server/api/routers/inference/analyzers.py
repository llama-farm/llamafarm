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
        description="The namespace mentioned, or None if not specified"
    )
    project_id: str | None = Field(
        description="The project ID/name for create actions, or None if not specified"
    )
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
                base_url=f"{settings.ollama_host}/v1",
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
                    {"role": "user", "content": f"Analyze this message: {message}"},
                ],
                response_model=ProjectAnalysis,
                temperature=0.1,
                max_retries=2,
            )

        except Exception as e:
            logger.warning(
                "LLM analysis failed, falling back to rule-based", error=str(e)
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
            reasoning=result["reasoning"] + " (LLM unavailable)",
        )


# Generic, LLM-first intent classification for routing tools/actions
class IntentAnalysis(BaseModel):
    """Structured intent for selecting a tool and action with parameters"""

    tool_name: str = Field(description="One of the available tools by name")
    action: str = Field(description="Action to perform for the selected tool")
    parameters: dict | None = Field(
        default_factory=dict,
        description="Key-value parameters for the tool input schema",
    )
    namespace: str | None = Field(
        default=None, description="Project namespace if relevant"
    )
    project_id: str | None = Field(default=None, description="Project id if relevant")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Why this tool and action were chosen")


class IntentClassifier:
    """LLM-first classifier to select tool/action and parameters from a user message"""

    def __init__(self):
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        try:
            ollama_client = OpenAI(
                base_url=f"{settings.ollama_host}/v1",
                api_key=settings.ollama_api_key,
            )
            self.client = instructor.from_openai(
                ollama_client, mode=instructor.Mode.JSON
            )
        except Exception as e:
            logger.warning("Failed to initialize IntentClassifier client", error=str(e))
            self.client = None

    def classify(
        self,
        message: str,
        available_tools: list[str],
        request_namespace: str | None = None,
        request_project_id: str | None = None,
    ) -> IntentAnalysis:
        """Classify a user message into tool/action selection.

        Falls back to heuristics if LLM is unavailable.
        """
        if not self.client:
            return self._fallback_classify(
                message, available_tools, request_namespace, request_project_id
            )

        try:
            tools_as_bullets = "\n".join(f"- {t}" for t in available_tools)
            system_prompt = f"""
You are a routing assistant. Given a user message and a list of available tools, select the best tool,
the action to perform, and any parameters required by that tool. Use only these tools:
{tools_as_bullets}

Guidelines:
- If the user is asking about projects (list/create), choose 'projects' with action 'list' or 'create'.
- If the user wants to view or change project configuration, choose 'project_config' with actions like
  'get_schema', 'analyze_config', 'suggest_changes', or 'modify_config'.
- If the user wants help crafting prompts via follow-up Q&A, choose 'prompt_engineering' with actions like
  'start_engineering' (use user_input), 'continue_conversation' (use user_response),
  'generate_prompts', or 'save_prompts'.
- Prefer using the provided namespace/project_id if given. If missing, leave them null.
- Provide concise reasoning and a confidence between 0 and 1.
"""

            analysis: IntentAnalysis = self.client.chat.completions.create(
                model=settings.ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"message: {message}\n\n"
                            f"namespace: {request_namespace or 'null'}\n"
                            f"project_id: {request_project_id or 'null'}\n"
                        ),
                    },
                ],
                response_model=IntentAnalysis,
                temperature=0.1,
                max_retries=2,
            )

            # Ensure request overrides are respected when provided
            if request_namespace is not None:
                analysis.namespace = request_namespace
            if request_project_id is not None:
                analysis.project_id = request_project_id

            return analysis
        except Exception as e:
            logger.warning(
                "LLM intent classification failed; using heuristics", error=str(e)
            )
            return self._fallback_classify(
                message, available_tools, request_namespace, request_project_id
            )

    def _fallback_classify(
        self,
        message: str,
        available_tools: list[str],
        request_namespace: str | None,
        request_project_id: str | None,
    ) -> IntentAnalysis:
        """Rule-based fallback classification"""
        msg = message.lower()
        tool = "project_config" if "config" in msg or "schema" in msg else None
        if any(k in msg for k in ["project", "namespace", "list", "create"]):
            tool = "projects"
        if any(k in msg for k in ["prompt", "generate", "few shot", "examples"]):
            tool = "prompt_engineering"

        if tool is None or tool not in available_tools:
            # Default to prompt_engineering for conversational assistance
            tool = (
                "prompt_engineering"
                if "prompt_engineering" in available_tools
                else available_tools[0]
            )

        # Basic action heuristics
        action = "start_engineering"
        params: dict = {}
        if tool == "projects":
            action = (
                "create" if "create" in msg or "new" in msg or "make" in msg else "list"
            )
        elif tool == "project_config":
            if any(k in msg for k in ["analyze", "status", "current"]):
                action = "analyze_config"
            elif any(k in msg for k in ["suggest", "recommend"]):
                action = "suggest_changes"
            elif any(k in msg for k in ["modify", "change", "set", "update"]):
                action = "modify_config"
            else:
                action = "get_schema"
            if action in ("suggest_changes", "modify_config"):
                params["user_intent"] = message
        elif tool == "prompt_engineering":
            action = "start_engineering"
            params["user_input"] = message

        return IntentAnalysis(
            tool_name=tool,
            action=action,
            parameters=params,
            namespace=request_namespace,
            project_id=request_project_id,
            confidence=0.55,
            reasoning="Heuristic fallback classification",
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
