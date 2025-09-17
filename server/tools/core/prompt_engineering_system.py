"""
Prompt Engineering System

A comprehensive system for analyzing project context, conducting intelligent
conversations with users, and generating optimized prompts for AI applications.

The system integrates with the Project Schema System to understand project
context and safely update prompt configurations.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

# Import the project schema system
from tools.core.project_schema_system import (
    ProjectConfigManipulator,
    LLMConfigurationAssistant
)

logger = logging.getLogger(__name__)


class PromptType(Enum):
    """Types of prompts that can be generated"""
    SYSTEM = "system"  # System-level behavior setting
    TASK_SPECIFIC = "task_specific"  # Specific task execution
    FEW_SHOT = "few_shot"  # Examples-based learning
    CHAIN_OF_THOUGHT = "chain_of_thought"  # Step-by-step reasoning
    ROLE_BASED = "role_based"  # Specific role/persona
    CONVERSATIONAL = "conversational"  # Chat/dialogue
    INSTRUCTIONAL = "instructional"  # How-to/tutorial style


class PromptPattern(Enum):
    """Common prompt patterns and structures"""
    IDENTITY_PURPOSE = "identity_purpose"  # Who you are + what you do
    CONTEXT_TASK_FORMAT = "context_task_format"  # Context + Task + Output format
    EXAMPLES_THEN_TASK = "examples_then_task"  # Show examples, then ask
    STEP_BY_STEP = "step_by_step"  # Break down reasoning
    PERSONA_SCENARIO = "persona_scenario"  # Act as X in scenario Y
    CONSTRAINT_OPTIMIZATION = "constraint_optimization"  # Do X while following Y rules


class AIProvider(Enum):
    """AI providers with different prompt optimization needs"""
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    GENERIC = "generic"


@dataclass
class ProjectContext:
    """Rich context about a project for prompt optimization"""
    namespace: str
    project_id: str
    domain: Optional[str] = None  # e.g., "customer_support", "technical_writing"
    industry: Optional[str] = None  # e.g., "healthcare", "finance"
    use_cases: List[str] = field(default_factory=list)
    target_audience: Optional[str] = None
    ai_provider: AIProvider = AIProvider.GENERIC
    model_name: Optional[str] = None
    existing_prompts: List[Dict[str, Any]] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)
    rag_enabled: bool = False
    tone: Optional[str] = None  # "professional", "casual", "technical"
    constraints: List[str] = field(default_factory=list)


@dataclass
class PromptRequirement:
    """A single prompt requirement gathered from conversation"""
    name: str
    purpose: str
    prompt_type: PromptType
    pattern: PromptPattern
    context_needed: List[str] = field(default_factory=list)
    examples_needed: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    priority: int = 1  # 1=high, 2=medium, 3=low


@dataclass
class ConversationState:
    """Tracks the current state of the prompt engineering conversation"""
    project_context: ProjectContext
    requirements: List[PromptRequirement] = field(default_factory=list)
    current_requirement: Optional[PromptRequirement] = None
    questions_asked: List[str] = field(default_factory=list)
    user_responses: Dict[str, str] = field(default_factory=dict)
    conversation_phase: str = "discovery"  # discovery, refinement, generation, review
    next_questions: List[str] = field(default_factory=list)
    completion_percentage: float = 0.0


@dataclass
class GeneratedPrompt:
    """A generated prompt with metadata"""
    name: str
    content: str
    prompt_type: PromptType
    pattern: PromptPattern
    provider_optimized: AIProvider
    estimated_tokens: int
    quality_score: float
    sections: Dict[str, str] = field(default_factory=dict)
    raw_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProjectContextAnalyzer:
    """Analyzes project configuration to understand context for prompt engineering"""

    def __init__(self, namespace: str, project_id: str):
        self.namespace = namespace
        self.project_id = project_id
        self.manipulator = ProjectConfigManipulator(namespace, project_id)

    def analyze_project_context(self) -> ProjectContext:
        """Analyze project configuration to build rich context"""
        try:
            config = self.manipulator.load_config()

            # Determine AI provider
            provider = AIProvider.GENERIC
            if config.runtime:
                if config.runtime.provider.value == "openai":
                    provider = AIProvider.OPENAI
                elif config.runtime.provider.value == "ollama":
                    provider = AIProvider.OLLAMA

            # Extract existing prompts
            existing_prompts = []
            for prompt in config.prompts:
                existing_prompts.append({
                    "name": prompt.name,
                    "sections": [{"title": s.title, "content": s.content} for s in prompt.sections] if prompt.sections else [],
                    "raw_text": prompt.raw_text
                })

            # Check if RAG is configured
            rag_enabled = len(config.rag.strategies) > 0

            # Extract dataset information
            datasets = [dataset.name for dataset in config.datasets]

            context = ProjectContext(
                namespace=self.namespace,
                project_id=self.project_id,
                ai_provider=provider,
                model_name=config.runtime.model if config.runtime else None,
                existing_prompts=existing_prompts,
                datasets=datasets,
                rag_enabled=rag_enabled
            )

            # Try to infer domain from project structure
            context.domain = self._infer_domain(config)

            return context

        except Exception as e:
            logger.error(f"Failed to analyze project context: {e}")
            # Return minimal context
            return ProjectContext(
                namespace=self.namespace,
                project_id=self.project_id
            )

    def _infer_domain(self, config) -> Optional[str]:
        """Infer application domain from project configuration"""
        project_name_lower = config.name.lower()

        domain_keywords = {
            "customer_support": ["support", "help", "customer", "service"],
            "technical_writing": ["docs", "documentation", "technical", "guide"],
            "content_creation": ["content", "blog", "writing", "marketing"],
            "data_analysis": ["data", "analysis", "analytics", "insights"],
            "education": ["education", "learning", "tutorial", "teach"],
            "healthcare": ["health", "medical", "patient", "clinical"],
            "finance": ["finance", "financial", "banking", "investment"],
            "legal": ["legal", "law", "contract", "compliance"],
            "research": ["research", "academic", "study", "analysis"]
        }

        for domain, keywords in domain_keywords.items():
            if any(keyword in project_name_lower for keyword in keywords):
                return domain

        # Check datasets for domain clues
        for dataset in config.datasets:
            dataset_name_lower = dataset.name.lower()
            for domain, keywords in domain_keywords.items():
                if any(keyword in dataset_name_lower for keyword in keywords):
                    return domain

        return None


class PromptConversationManager:
    """Manages the conversational flow for gathering prompt requirements"""

    def __init__(self, project_context: ProjectContext):
        self.project_context = project_context
        self.conversation_state = ConversationState(project_context=project_context)

    def start_conversation(self, user_initial_input: str) -> Tuple[str, List[str]]:
        """Start the prompt engineering conversation"""
        # Analyze initial input to understand what they want
        initial_analysis = self._analyze_initial_input(user_initial_input)

        # Generate opening questions based on context
        opening_questions = self._generate_opening_questions(initial_analysis)

        self.conversation_state.conversation_phase = "discovery"
        self.conversation_state.next_questions = opening_questions

        # Create personalized opening message
        opening_message = self._create_opening_message(initial_analysis)

        return opening_message, opening_questions

    def continue_conversation(self, user_response: str, question_answered: str) -> Tuple[str, List[str], bool]:
        """Continue the conversation with follow-up questions"""
        # Record the response
        self.conversation_state.user_responses[question_answered] = user_response

        # Analyze the response and update requirements
        self._process_user_response(user_response, question_answered)

        # Generate next questions
        next_questions = self._generate_next_questions()

        # Check if we have enough information
        is_complete = self._assess_completion()

        # Generate response message
        response_message = self._generate_response_message(is_complete)

        self.conversation_state.next_questions = next_questions

        return response_message, next_questions, is_complete

    def _analyze_initial_input(self, user_input: str) -> Dict[str, Any]:
        """Analyze the user's initial input to understand their needs"""
        analysis = {
            "intent": "general",
            "prompt_types": [],
            "domain_hints": [],
            "use_case_hints": [],
            "tone_hints": []
        }

        input_lower = user_input.lower()

        # Detect intent
        if any(word in input_lower for word in ["customer", "support", "help desk"]):
            analysis["intent"] = "customer_support"
        elif any(word in input_lower for word in ["technical", "documentation", "explain"]):
            analysis["intent"] = "technical_writing"
        elif any(word in input_lower for word in ["creative", "content", "marketing"]):
            analysis["intent"] = "content_creation"
        elif any(word in input_lower for word in ["analyze", "data", "insights"]):
            analysis["intent"] = "data_analysis"

        # Detect prompt types mentioned
        if any(word in input_lower for word in ["system", "behavior", "personality"]):
            analysis["prompt_types"].append(PromptType.SYSTEM)
        if any(word in input_lower for word in ["example", "few shot", "demonstrate"]):
            analysis["prompt_types"].append(PromptType.FEW_SHOT)
        if any(word in input_lower for word in ["step by step", "reasoning", "think"]):
            analysis["prompt_types"].append(PromptType.CHAIN_OF_THOUGHT)

        # Detect tone preferences
        if any(word in input_lower for word in ["professional", "formal"]):
            analysis["tone_hints"].append("professional")
        elif any(word in input_lower for word in ["casual", "friendly", "conversational"]):
            analysis["tone_hints"].append("casual")
        elif any(word in input_lower for word in ["technical", "precise", "detailed"]):
            analysis["tone_hints"].append("technical")

        return analysis

    def _generate_opening_questions(self, initial_analysis: Dict[str, Any]) -> List[str]:
        """Generate opening questions based on initial analysis"""
        questions = []

        # Always ask about primary use case if not clear
        if initial_analysis["intent"] == "general":
            questions.append(
                "What is the primary use case for your AI application? "
                "(e.g., customer support, content creation, data analysis, technical documentation)"
            )

        # Ask about target audience
        questions.append(
            "Who is the target audience for your AI responses? "
            "(e.g., customers, employees, technical experts, general public)"
        )

        # Ask about tone/style preferences
        if not initial_analysis["tone_hints"]:
            questions.append(
                "What tone and style should your AI use? "
                "(e.g., professional and formal, friendly and conversational, technical and precise)"
            )

        # Ask about specific tasks if project has RAG
        if self.project_context.rag_enabled:
            questions.append(
                "I see your project has RAG (document search) enabled. How should the AI use "
                "information from your documents? Should it cite sources, summarize findings, "
                "or provide detailed explanations?"
            )

        # Ask about constraints
        questions.append(
            "Are there any important constraints or guidelines the AI should always follow? "
            "(e.g., never give medical advice, always ask for clarification, stay within company policy)"
        )

        return questions[:3]  # Limit to 3 initial questions

    def _create_opening_message(self, initial_analysis: Dict[str, Any]) -> str:
        """Create a personalized opening message"""
        parts = [
            "I'll help you create optimized prompts for your project! 🚀"
        ]

        # Add context-specific opening
        if self.project_context.domain:
            parts.append(f"I can see this is a {self.project_context.domain} project.")

        if self.project_context.ai_provider != AIProvider.GENERIC:
            parts.append(f"I'll optimize the prompts for {self.project_context.ai_provider.value}.")

        if self.project_context.existing_prompts:
            parts.append(f"I noticed you have {len(self.project_context.existing_prompts)} existing prompts - I can help improve or expand on those.")

        parts.append("Let me ask a few questions to understand your needs better:")

        return " ".join(parts)

    def _process_user_response(self, user_response: str, question_answered: str):
        """Process user response and update conversation state"""
        # Update project context based on responses
        if "use case" in question_answered.lower():
            self.project_context.use_cases.append(user_response)
            # Try to infer domain from use case
            if not self.project_context.domain:
                use_case_lower = user_response.lower()
                if "support" in use_case_lower or "customer" in use_case_lower:
                    self.project_context.domain = "customer_support"
                elif "content" in use_case_lower or "writing" in use_case_lower:
                    self.project_context.domain = "content_creation"

        elif "audience" in question_answered.lower():
            self.project_context.target_audience = user_response

        elif "tone" in question_answered.lower():
            self.project_context.tone = user_response

        elif "constraints" in question_answered.lower() or "guidelines" in question_answered.lower():
            self.project_context.constraints.append(user_response)

        # Update completion percentage
        self._update_completion_percentage()

    def _generate_next_questions(self) -> List[str]:
        """Generate next questions based on current state"""
        questions = []
        responses = self.conversation_state.user_responses

        # Progressive question logic
        if len(responses) == 1:
            # Second round - get more specific
            if self.project_context.domain == "customer_support":
                questions.append(
                    "What types of customer inquiries should the AI handle? "
                    "(e.g., billing questions, technical issues, product information)"
                )
            elif self.project_context.domain == "content_creation":
                questions.append(
                    "What type of content should the AI create? "
                    "(e.g., blog posts, social media, email campaigns, product descriptions)"
                )
            else:
                questions.append(
                    "What specific tasks should the AI excel at in your use case?"
                )

        elif len(responses) == 2:
            # Third round - ask about examples and edge cases
            questions.append(
                "Can you provide an example of an ideal interaction or output you'd want from the AI?"
            )
            questions.append(
                "What are some challenging scenarios or edge cases the AI should handle well?"
            )

        elif len(responses) >= 3:
            # Final questions - refinement
            if not any("example" in q.lower() for q in self.conversation_state.questions_asked):
                questions.append(
                    "Should I include specific examples in the prompts to guide the AI's behavior?"
                )

        return questions[:2]  # Limit to 2 questions at a time

    def _assess_completion(self) -> bool:
        """Assess if we have enough information to generate good prompts"""
        responses = self.conversation_state.user_responses

        # Need at least basic information
        has_use_case = any("use case" in q.lower() for q in responses.keys()) or self.project_context.domain
        has_audience = self.project_context.target_audience is not None
        has_tone = self.project_context.tone is not None

        # Basic completion criteria
        basic_complete = len(responses) >= 3 and has_use_case and (has_audience or has_tone)

        # Advanced completion criteria
        advanced_complete = len(responses) >= 4 and has_use_case and has_audience and has_tone

        return advanced_complete or (basic_complete and len(responses) >= 5)

    def _update_completion_percentage(self):
        """Update the completion percentage"""
        total_questions = 5  # Estimated total questions needed
        answered = len(self.conversation_state.user_responses)
        self.conversation_state.completion_percentage = min(answered / total_questions * 100, 100)

    def _generate_response_message(self, is_complete: bool) -> str:
        """Generate response message based on conversation state"""
        if is_complete:
            return (
                "Perfect! I have enough information to generate optimized prompts for you. "
                f"Based on our conversation, I'll create prompts for {self.project_context.domain or 'your use case'} "
                f"with a {self.project_context.tone or 'appropriate'} tone. "
                "Let me generate some options for you!"
            )
        else:
            completion = self.conversation_state.completion_percentage
            return (
                f"Great responses! I'm about {completion:.0f}% of the way to having "
                "everything I need to create excellent prompts for you. "
                "Let me ask a couple more questions:"
            )


class PromptGenerator:
    """Generates optimized prompts based on requirements and context"""

    def __init__(self):
        self.prompt_templates = self._load_prompt_templates()

    def generate_prompts(self, conversation_state: ConversationState) -> List[GeneratedPrompt]:
        """Generate prompts based on conversation state"""
        context = conversation_state.project_context
        prompts = []

        # Generate primary system prompt
        system_prompt = self._generate_system_prompt(context, conversation_state)
        prompts.append(system_prompt)

        # Generate task-specific prompts if needed
        if context.use_cases:
            for use_case in context.use_cases:
                task_prompt = self._generate_task_prompt(context, use_case)
                prompts.append(task_prompt)

        # Generate few-shot prompt if examples were mentioned
        if self._should_generate_few_shot(conversation_state):
            few_shot_prompt = self._generate_few_shot_prompt(context, conversation_state)
            prompts.append(few_shot_prompt)

        # Generate constraint prompt if many constraints
        if len(context.constraints) > 2:
            constraint_prompt = self._generate_constraint_prompt(context)
            prompts.append(constraint_prompt)

        return prompts

    def _generate_system_prompt(self, context: ProjectContext, conv_state: ConversationState) -> GeneratedPrompt:
        """Generate a comprehensive system prompt"""
        sections = {}

        # Identity and Purpose
        identity_parts = []
        if context.domain:
            domain_identity = {
                "customer_support": "You are a helpful customer support agent",
                "technical_writing": "You are a technical writing assistant",
                "content_creation": "You are a creative content creator",
                "data_analysis": "You are a data analysis expert",
                "education": "You are an educational assistant",
                "healthcare": "You are a healthcare information assistant",
                "finance": "You are a financial analysis assistant",
                "legal": "You are a legal research assistant",
                "research": "You are a research assistant"
            }
            identity_parts.append(domain_identity.get(context.domain, "You are an AI assistant"))
        else:
            identity_parts.append("You are an AI assistant")

        # Add specific purpose based on use cases
        if context.use_cases:
            purpose = f"specialized in {', '.join(context.use_cases)}"
            identity_parts.append(purpose)

        sections["IDENTITY AND PURPOSE"] = ". ".join(identity_parts) + "."

        # Communication Style
        style_parts = []
        if context.tone:
            style_parts.append(f"Maintain a {context.tone} tone in all interactions")

        if context.target_audience:
            style_parts.append(f"Tailor your responses for {context.target_audience}")

        if context.ai_provider == AIProvider.OPENAI:
            style_parts.append("Be concise but comprehensive in your responses")
        elif context.ai_provider == AIProvider.OLLAMA:
            style_parts.append("Provide clear, well-structured responses")

        if style_parts:
            sections["COMMUNICATION STYLE"] = ". ".join(style_parts) + "."

        # Capabilities and Context
        capability_parts = []
        if context.rag_enabled:
            capability_parts.append(
                "You have access to relevant documents and information through the knowledge base. "
                "Use this information to provide accurate, source-based responses."
            )

        if context.datasets:
            capability_parts.append(f"You can reference information from these datasets: {', '.join(context.datasets)}")

        if capability_parts:
            sections["CAPABILITIES"] = " ".join(capability_parts)

        # Guidelines and Constraints
        if context.constraints:
            guidelines = []
            for constraint in context.constraints:
                if not constraint.endswith('.'):
                    constraint += '.'
                guidelines.append(constraint)
            sections["GUIDELINES"] = " ".join(guidelines)

        # Convert sections to content format
        content_sections = []
        for title, content in sections.items():
            content_sections.append({
                "title": title,
                "content": [content]
            })

        # Create raw text version
        raw_text_parts = []
        for title, content in sections.items():
            raw_text_parts.append(f"## {title}\n{content}")
        raw_text = "\n\n".join(raw_text_parts)

        return GeneratedPrompt(
            name="system_prompt",
            content=json.dumps(content_sections),
            prompt_type=PromptType.SYSTEM,
            pattern=PromptPattern.IDENTITY_PURPOSE,
            provider_optimized=context.ai_provider,
            estimated_tokens=int(len(raw_text.split()) * 1.3),  # Rough estimate
            quality_score=0.85,  # Base quality score
            sections={title: content for title, content in sections.items()},
            raw_text=raw_text,
            metadata={"generated_from": "conversation", "context": context.domain}
        )

    def _generate_task_prompt(self, context: ProjectContext, use_case: str) -> GeneratedPrompt:
        """Generate a task-specific prompt"""
        task_name = f"{use_case.lower().replace(' ', '_')}_assistant"

        # Task-specific templates
        task_templates = {
            "customer_support": {
                "identity": "You are a customer support specialist",
                "tasks": [
                    "Listen carefully to customer concerns",
                    "Provide clear, actionable solutions",
                    "Escalate complex issues when appropriate",
                    "Maintain empathy and professionalism"
                ],
                "format": "Always acknowledge the customer's concern, provide a clear solution or next steps, and offer additional help"
            },
            "technical_writing": {
                "identity": "You are a technical documentation expert",
                "tasks": [
                    "Explain complex concepts clearly",
                    "Provide step-by-step instructions",
                    "Use appropriate technical terminology",
                    "Include relevant examples and code snippets"
                ],
                "format": "Structure responses with clear headings, numbered steps where appropriate, and practical examples"
            }
        }

        # Default template for unknown use cases
        default_template = {
            "identity": f"You are an expert in {use_case}",
            "tasks": [
                "Understand user requirements clearly",
                "Provide accurate and helpful responses",
                "Ask clarifying questions when needed"
            ],
            "format": "Provide clear, structured responses that directly address the user's needs"
        }

        template = task_templates.get(use_case.lower(), default_template)

        # Build sections
        sections: dict[str, str] = {}
        sections["ROLE"] = template["identity"]
        sections["KEY RESPONSIBILITIES"] = ". ".join(template["tasks"])
        sections["RESPONSE FORMAT"] = template["format"]

        # Convert to content format
        content_sections = []
        for title, content in sections.items():
            content_sections.append({
                "title": title,
                "content": [content]
            })

        raw_text = "\n\n".join([f"## {title}\n{content}" for title, content in sections.items()])

        return GeneratedPrompt(
            name=task_name,
            content=json.dumps(content_sections),
            prompt_type=PromptType.TASK_SPECIFIC,
            pattern=PromptPattern.CONTEXT_TASK_FORMAT,
            provider_optimized=context.ai_provider,
            estimated_tokens=int(len(raw_text.split()) * 1.3),
            quality_score=0.80,
            sections=sections,
            raw_text=raw_text,
            metadata={"use_case": use_case, "task_specific": True}
        )

    def _generate_few_shot_prompt(self, context: ProjectContext, conv_state: ConversationState) -> GeneratedPrompt:
        """Generate a few-shot prompt with examples"""
        # Look for examples in user responses
        examples = []
        for question, response in conv_state.user_responses.items():
            if "example" in question.lower():
                examples.append(response)

        sections = {}
        sections["EXAMPLES"] = (
            "Here are examples of ideal interactions:\n\n" +
            "\n\n".join([f"Example {i+1}: {ex}" for i, ex in enumerate(examples)])
        )
        sections["INSTRUCTION"] = "Follow these examples as a guide for your responses. Match the style, depth, and approach shown above."

        content_sections = []
        for title, content in sections.items():
            content_sections.append({
                "title": title,
                "content": [content]
            })

        raw_text = "\n\n".join([f"## {title}\n{content}" for title, content in sections.items()])

        return GeneratedPrompt(
            name="few_shot_examples",
            content=json.dumps(content_sections),
            prompt_type=PromptType.FEW_SHOT,
            pattern=PromptPattern.EXAMPLES_THEN_TASK,
            provider_optimized=context.ai_provider,
            estimated_tokens=int(len(raw_text.split()) * 1.3),
            quality_score=0.75,
            sections=sections,
            raw_text=raw_text,
            metadata={"has_examples": len(examples)}
        )

    def _generate_constraint_prompt(self, context: ProjectContext) -> GeneratedPrompt:
        """Generate a constraints-focused prompt"""
        sections = {}

        constraints_list = []
        for i, constraint in enumerate(context.constraints, 1):
            constraints_list.append(f"{i}. {constraint}")

        sections["IMPORTANT CONSTRAINTS"] = "You must always follow these guidelines:\n\n" + "\n".join(constraints_list)
        sections["COMPLIANCE"] = "Before responding, mentally check that your response complies with all constraints above."

        content_sections = []
        for title, content in sections.items():
            content_sections.append({
                "title": title,
                "content": [content]
            })

        raw_text = "\n\n".join([f"## {title}\n{content}" for title, content in sections.items()])

        return GeneratedPrompt(
            name="constraints_guide",
            content=json.dumps(content_sections),
            prompt_type=PromptType.INSTRUCTIONAL,
            pattern=PromptPattern.CONSTRAINT_OPTIMIZATION,
            provider_optimized=context.ai_provider,
            estimated_tokens=int(len(raw_text.split()) * 1.3),
            quality_score=0.70,
            sections=sections,
            raw_text=raw_text,
            metadata={"constraint_count": len(context.constraints)}
        )

    def _should_generate_few_shot(self, conv_state: ConversationState) -> bool:
        """Determine if a few-shot prompt should be generated"""
        # Check if user provided examples
        for question in conv_state.user_responses.keys():
            if "example" in question.lower():
                return True
        return False

    def _load_prompt_templates(self) -> Dict[str, Any]:
        """Load prompt templates for different patterns"""
        # This would typically load from files or database
        # For now, return empty dict - templates are built dynamically
        return {}


class PromptOptimizer:
    """Optimizes prompts for specific AI providers and models"""

    def optimize_for_provider(self, prompt: GeneratedPrompt, provider: AIProvider, model: Optional[str] = None) -> GeneratedPrompt:
        """Optimize a prompt for a specific provider and model"""
        optimized_prompt = GeneratedPrompt(
            name=prompt.name,
            content=prompt.content,
            prompt_type=prompt.prompt_type,
            pattern=prompt.pattern,
            provider_optimized=provider,
            estimated_tokens=prompt.estimated_tokens,
            quality_score=prompt.quality_score,
            sections=prompt.sections.copy(),
            raw_text=prompt.raw_text,
            metadata=prompt.metadata.copy()
        )

        if provider == AIProvider.OPENAI:
            optimized_prompt = self._optimize_for_openai(optimized_prompt, model)
        elif provider == AIProvider.OLLAMA:
            optimized_prompt = self._optimize_for_ollama(optimized_prompt, model)
        elif provider == AIProvider.ANTHROPIC:
            optimized_prompt = self._optimize_for_anthropic(optimized_prompt, model)

        return optimized_prompt

    def _optimize_for_openai(self, prompt: GeneratedPrompt, model: Optional[str] = None) -> GeneratedPrompt:
        """Optimize prompt for OpenAI models"""
        # OpenAI models work well with structured prompts
        if prompt.prompt_type == PromptType.SYSTEM:
            # Add system role clarity
            if "IDENTITY AND PURPOSE" in prompt.sections:
                identity = prompt.sections["IDENTITY AND PURPOSE"]
                if not identity.startswith("You are"):
                    prompt.sections["IDENTITY AND PURPOSE"] = f"You are {identity}"

        # OpenAI models benefit from explicit output format instructions
        if "RESPONSE FORMAT" not in prompt.sections and prompt.prompt_type != PromptType.FEW_SHOT:
            prompt.sections["RESPONSE FORMAT"] = (
                "Provide clear, well-structured responses. Use markdown formatting where appropriate. "
                "Be concise but comprehensive."
            )

        # Rebuild raw text
        prompt.raw_text = "\n\n".join([f"## {title}\n{content}" for title, content in prompt.sections.items()])
        prompt.estimated_tokens = int(len(prompt.raw_text.split()) * 1.3)
        prompt.quality_score += 0.05  # Slight boost for optimization

        return prompt

    def _optimize_for_ollama(self, prompt: GeneratedPrompt, model: Optional[str] = None) -> GeneratedPrompt:
        """Optimize prompt for Ollama/local models"""
        # Ollama models often benefit from clearer, more explicit instructions

        # Add more explicit role definition
        if prompt.prompt_type == PromptType.SYSTEM and "IDENTITY AND PURPOSE" in prompt.sections:
            identity = prompt.sections["IDENTITY AND PURPOSE"]
            # Make role more explicit for local models
            if not "assistant" in identity.lower():
                identity += " Always act as a helpful assistant."
            prompt.sections["IDENTITY AND PURPOSE"] = identity

        # Local models benefit from explicit instruction to think step by step
        if prompt.prompt_type == PromptType.TASK_SPECIFIC:
            prompt.sections["APPROACH"] = (
                "Take a moment to understand the request fully. "
                "Think through your response step by step before answering."
            )

        # Rebuild raw text
        prompt.raw_text = "\n\n".join([f"## {title}\n{content}" for title, content in prompt.sections.items()])
        prompt.estimated_tokens = int(len(prompt.raw_text.split()) * 1.3)
        prompt.quality_score += 0.03  # Slight boost for optimization

        return prompt

    def _optimize_for_anthropic(self, prompt: GeneratedPrompt, model: Optional[str] = None) -> GeneratedPrompt:
        """Optimize prompt for Anthropic models"""
        # Anthropic models work well with conversational, human-like prompts

        # Make language more conversational
        if prompt.prompt_type == PromptType.SYSTEM:
            for section_title, content in prompt.sections.items():
                if section_title == "IDENTITY AND PURPOSE":
                    # Make it more conversational
                    content = content.replace("You are an AI assistant", "I'm here to help as your AI assistant")
                    prompt.sections[section_title] = content

        # Claude models appreciate politeness and consideration
        if "COMMUNICATION STYLE" in prompt.sections:
            style = prompt.sections["COMMUNICATION STYLE"]
            if "polite" not in style.lower():
                style += " Always be polite and considerate in your responses."
            prompt.sections["COMMUNICATION STYLE"] = style

        # Rebuild raw text
        prompt.raw_text = "\n\n".join([f"## {title}\n{content}" for title, content in prompt.sections.items()])
        prompt.estimated_tokens = int(len(prompt.raw_text.split()) * 1.3)
        prompt.quality_score += 0.04  # Slight boost for optimization

        return prompt


class PromptEngineeringCore:
    """Main orchestrator for the prompt engineering system"""

    def __init__(self, namespace: str, project_id: str):
        self.namespace = namespace
        self.project_id = project_id
        self.context_analyzer = ProjectContextAnalyzer(namespace, project_id)
        self.conversation_manager: Optional[PromptConversationManager] = None
        self.prompt_generator = PromptGenerator()
        self.prompt_optimizer = PromptOptimizer()
        self.config_manipulator = ProjectConfigManipulator(namespace, project_id)

    def start_prompt_engineering(self, user_input: str) -> Dict[str, Any]:
        """Start the prompt engineering process"""
        # Analyze project context
        context = self.context_analyzer.analyze_project_context()

        # Initialize conversation manager
        self.conversation_manager = PromptConversationManager(context)

        # Start conversation
        opening_message, questions = self.conversation_manager.start_conversation(user_input)

        return {
            "success": True,
            "phase": "discovery",
            "message": opening_message,
            "questions": questions,
            "context_summary": {
                "domain": context.domain,
                "ai_provider": context.ai_provider.value,
                "has_rag": context.rag_enabled,
                "existing_prompts": len(context.existing_prompts)
            }
        }

    def continue_conversation(self, user_response: str, question_answered: str) -> Dict[str, Any]:
        """Continue the prompt engineering conversation"""
        if not self.conversation_manager:
            return {"success": False, "error": "Conversation not started"}

        response_message, next_questions, is_complete = self.conversation_manager.continue_conversation(
            user_response, question_answered
        )

        result = {
            "success": True,
            "message": response_message,
            "questions": next_questions,
            "is_complete": is_complete,
            "completion_percentage": self.conversation_manager.conversation_state.completion_percentage
        }

        if is_complete:
            result["phase"] = "ready_for_generation"
        else:
            result["phase"] = "discovery"

        return result

    def generate_prompts(self) -> Dict[str, Any]:
        """Generate optimized prompts based on conversation"""
        if not self.conversation_manager:
            return {"success": False, "error": "Conversation not started"}

        # Generate prompts
        generated_prompts = self.prompt_generator.generate_prompts(
            self.conversation_manager.conversation_state
        )

        # Optimize for the project's AI provider
        context = self.conversation_manager.conversation_state.project_context
        optimized_prompts = []

        for prompt in generated_prompts:
            optimized = self.prompt_optimizer.optimize_for_provider(
                prompt, context.ai_provider, context.model_name
            )
            optimized_prompts.append(optimized)

        # Prepare for saving
        prompt_configs = []
        for prompt in optimized_prompts:
            config = {
                "name": prompt.name,
                "sections": [
                    {"title": title, "content": [content]}
                    for title, content in prompt.sections.items()
                ],
                "raw_text": prompt.raw_text
            }
            prompt_configs.append(config)

        return {
            "success": True,
            "prompts_generated": len(optimized_prompts),
            "prompts": [
                {
                    "name": p.name,
                    "type": p.prompt_type.value,
                    "pattern": p.pattern.value,
                    "estimated_tokens": p.estimated_tokens,
                    "quality_score": p.quality_score,
                    "sections": p.sections,
                    "raw_text": (p.raw_text[:200] + "..." if p.raw_text and len(p.raw_text) > 200 else p.raw_text) if p.raw_text else ""  # Preview
                }
                for p in optimized_prompts
            ],
            "ready_to_save": True,
            "_full_configs": prompt_configs  # For internal use
        }

    def save_prompts_to_project(self, generated_prompts_result: Dict[str, Any]) -> Dict[str, Any]:
        """Save generated prompts to the project configuration"""
        if not generated_prompts_result.get("success") or not generated_prompts_result.get("_full_configs"):
            return {"success": False, "error": "No valid prompts to save"}

        try:
            # Load current configuration
            self.config_manipulator.load_config()

            # Apply the prompt changes
            prompt_configs = generated_prompts_result["_full_configs"]
            change = self.config_manipulator.apply_change(
                "prompts",
                prompt_configs,
                f"Added {len(prompt_configs)} AI-generated prompts"
            )

            # Save the configuration
            self.config_manipulator.save_config()

            return {
                "success": True,
                "message": f"Successfully saved {len(prompt_configs)} prompts to your project configuration",
                "prompts_saved": len(prompt_configs),
                "change_details": {
                    "field_path": change.field_path,
                    "change_type": change.change_type,
                    "description": change.description
                }
            }

        except Exception as e:
            logger.error(f"Failed to save prompts: {e}")
            return {
                "success": False,
                "error": f"Failed to save prompts: {str(e)}"
            }
