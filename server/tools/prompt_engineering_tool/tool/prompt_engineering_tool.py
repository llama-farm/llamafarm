"""
Prompt Engineering Tool

Provides LLMs with comprehensive prompt engineering capabilities including:
- Project context analysis for prompt optimization
- Intelligent conversational requirements gathering
- Multi-pattern prompt generation
- Provider-specific optimization
- Safe integration with project configuration
"""

import json
import logging
from typing import Any, Dict, List, Literal, Optional

from atomic_agents import BaseIOSchema, BaseTool

# Import the prompt engineering system
from tools.core.prompt_engineering_system import (
    PromptEngineeringCore
)

logger = logging.getLogger(__name__)


class PromptEngineeringToolInput(BaseIOSchema):
    """Input schema for prompt engineering tool."""
    action: Literal[
        "start_engineering", "continue_conversation", "generate_prompts",
        "save_prompts", "analyze_context"
    ]
    namespace: str
    project_id: str
    user_input: Optional[str] = None  # Required for start_engineering
    user_response: Optional[str] = None  # Required for continue_conversation
    question_answered: Optional[str] = None  # Required for continue_conversation
    generated_prompts_data: Optional[Dict[str, Any]] = None  # Required for save_prompts


class PromptEngineeringToolOutput(BaseIOSchema):
    """Output schema for prompt engineering tool."""
    success: bool
    message: str
    action: str
    data: Optional[Dict[str, Any]] = None
    phase: Optional[str] = None  # discovery, generation, complete
    next_questions: Optional[List[str]] = None
    completion_percentage: Optional[float] = None
    errors: Optional[List[str]] = None


class PromptEngineeringTool(
    BaseTool[PromptEngineeringToolInput, PromptEngineeringToolOutput]
):
    """Tool for comprehensive prompt engineering with conversational interface."""

    def __init__(self):
        super().__init__()
        # Store active engineering sessions
        self._sessions: Dict[str, PromptEngineeringCore] = {}

    def run(
        self, input_data: PromptEngineeringToolInput
    ) -> PromptEngineeringToolOutput:
        """Execute the prompt engineering tool based on the specified action."""
        try:
            logger.info(
                f"Executing prompt engineering tool - action: {input_data.action}, "
                f"namespace: {input_data.namespace}, "
                f"project_id: {input_data.project_id}"
            )

            session_key = f"{input_data.namespace}/{input_data.project_id}"

            if input_data.action == "analyze_context":
                return self._analyze_context(
                    input_data.namespace, input_data.project_id
                )

            elif input_data.action == "start_engineering":
                if not input_data.user_input:
                    return PromptEngineeringToolOutput(
                        success=False,
                        message="user_input is required for start_engineering action",
                        action=input_data.action,
                        errors=["Missing required parameter: user_input"]
                    )
                return self._start_engineering(
                    input_data.namespace, input_data.project_id, input_data.user_input
                )

            elif input_data.action == "continue_conversation":
                if not input_data.user_response or not input_data.question_answered:
                    return PromptEngineeringToolOutput(
                        success=False,
                        message=(
                            "user_response and question_answered are required for "
                            "continue_conversation action"
                        ),
                        action=input_data.action,
                        errors=[
                            "Missing required parameters: user_response, "
                            "question_answered"
                        ]
                    )
                return self._continue_conversation(
                    session_key, input_data.user_response, input_data.question_answered
                )

            elif input_data.action == "generate_prompts":
                return self._generate_prompts(session_key)

            elif input_data.action == "save_prompts":
                if not input_data.generated_prompts_data:
                    return PromptEngineeringToolOutput(
                        success=False,
                        message=(
                            "generated_prompts_data is required for save_prompts action"
                        ),
                        action=input_data.action,
                        errors=["Missing required parameter: generated_prompts_data"]
                    )
                return self._save_prompts(
                    session_key, input_data.generated_prompts_data
                )

            else:
                return PromptEngineeringToolOutput(
                    success=False,
                    message=f"Unknown action: {input_data.action}",
                    action=input_data.action,
                    errors=[f"Unsupported action: {input_data.action}"]
                )

        except Exception as e:
            logger.exception(f"Unexpected error in prompt engineering tool: {e}")
            return PromptEngineeringToolOutput(
                success=False,
                message=f"Tool execution failed: {str(e)}",
                action=input_data.action,
                errors=[str(e)]
            )

    def _analyze_context(
        self, namespace: str, project_id: str
    ) -> PromptEngineeringToolOutput:
        """Analyze project context to provide insights for prompt engineering"""
        try:
            # Create temporary engineering core to analyze context
            core = PromptEngineeringCore(namespace, project_id)
            context = core.context_analyzer.analyze_project_context()

            # Create summary of context
            context_summary = {
                "project_info": {
                    "namespace": context.namespace,
                    "project_id": context.project_id,
                    "domain": context.domain,
                    "ai_provider": context.ai_provider.value
                },
                "current_setup": {
                    "model_name": context.model_name,
                    "has_rag": context.rag_enabled,
                    "datasets_count": len(context.datasets),
                    "existing_prompts_count": len(context.existing_prompts)
                },
                "optimization_opportunities": [],
                "recommendations": []
            }

            # Add optimization opportunities
            if not context.existing_prompts:
                context_summary["optimization_opportunities"].append(
                    "No prompts configured - perfect opportunity to create "
                    "optimized prompts"
                )
            elif len(context.existing_prompts) < 2:
                context_summary["optimization_opportunities"].append(
                    "Consider adding specialized prompts for different use cases"
                )

            if context.rag_enabled and not context.domain:
                context_summary["optimization_opportunities"].append(
                    "RAG is enabled but domain is unclear - prompts could be "
                    "optimized for document search"
                )

            # Add recommendations
            if context.ai_provider != context.ai_provider.GENERIC:
                context_summary["recommendations"].append(
                    f"I can optimize prompts specifically for "
                    f"{context.ai_provider.value}"
                )

            if context.domain:
                context_summary["recommendations"].append(
                    f"Detected {context.domain} domain - I can create specialized "
                    f"prompts for this use case"
                )

            # Create user-friendly message
            message_parts = [
                f"Analyzed project context for {namespace}/{project_id}:",
                f"• AI Provider: {context.ai_provider.value}",
                f"• Domain: {context.domain or 'Not specified'}",
                f"• RAG Enabled: {'Yes' if context.rag_enabled else 'No'}",
                f"• Existing Prompts: {len(context.existing_prompts)}",
            ]

            if context_summary["optimization_opportunities"]:
                message_parts.append("• Optimization opportunities identified")

            message = "\n".join(message_parts)

            return PromptEngineeringToolOutput(
                success=True,
                message=message,
                action="analyze_context",
                data=context_summary,
                phase="analysis"
            )

        except Exception as e:
            logger.error(f"Failed to analyze context for {namespace}/{project_id}: {e}")
            return PromptEngineeringToolOutput(
                success=False,
                message=f"Failed to analyze project context: {str(e)}",
                action="analyze_context",
                errors=[str(e)]
            )

    def _start_engineering(
        self, namespace: str, project_id: str, user_input: str
    ) -> PromptEngineeringToolOutput:
        """Start the prompt engineering process"""
        try:
            session_key = f"{namespace}/{project_id}"

            # Create new engineering core
            core = PromptEngineeringCore(namespace, project_id)
            self._sessions[session_key] = core

            # Start the engineering process
            result = core.start_prompt_engineering(user_input)

            if result["success"]:
                return PromptEngineeringToolOutput(
                    success=True,
                    message=result["message"],
                    action="start_engineering",
                    data=result.get("context_summary"),
                    phase=result["phase"],
                    next_questions=result.get("questions", [])
                )
            else:
                return PromptEngineeringToolOutput(
                    success=False,
                    message="Failed to start prompt engineering",
                    action="start_engineering",
                    errors=["Engineering initialization failed"]
                )

        except Exception as e:
            logger.error(
                f"Failed to start engineering for {namespace}/{project_id}: {e}"
            )
            return PromptEngineeringToolOutput(
                success=False,
                message=f"Failed to start prompt engineering: {str(e)}",
                action="start_engineering",
                errors=[str(e)]
            )

    def _continue_conversation(
        self, session_key: str, user_response: str, question_answered: str
    ) -> PromptEngineeringToolOutput:
        """Continue the prompt engineering conversation"""
        try:
            if session_key not in self._sessions:
                return PromptEngineeringToolOutput(
                    success=False,
                    message=(
                        "No active prompt engineering session found. Please start "
                        "with 'start_engineering' action."
                    ),
                    action="continue_conversation",
                    errors=["Session not found"]
                )

            core = self._sessions[session_key]
            result = core.continue_conversation(user_response, question_answered)

            if result["success"]:
                return PromptEngineeringToolOutput(
                    success=True,
                    message=result["message"],
                    action="continue_conversation",
                    phase=result.get("phase"),
                    next_questions=result.get("questions", []),
                    completion_percentage=result.get("completion_percentage"),
                    data={
                        "is_complete": result.get("is_complete", False),
                        "ready_for_generation": (
                            result.get("phase") == "ready_for_generation"
                        )
                    }
                )
            else:
                return PromptEngineeringToolOutput(
                    success=False,
                    message="Failed to process conversation",
                    action="continue_conversation",
                    errors=["Conversation processing failed"]
                )

        except Exception as e:
            logger.error(f"Failed to continue conversation for {session_key}: {e}")
            return PromptEngineeringToolOutput(
                success=False,
                message=f"Failed to continue conversation: {str(e)}",
                action="continue_conversation",
                errors=[str(e)]
            )

    def _generate_prompts(self, session_key: str) -> PromptEngineeringToolOutput:
        """Generate optimized prompts based on conversation"""
        try:
            if session_key not in self._sessions:
                return PromptEngineeringToolOutput(
                    success=False,
                    message="No active prompt engineering session found.",
                    action="generate_prompts",
                    errors=["Session not found"]
                )

            core = self._sessions[session_key]
            result = core.generate_prompts()

            if result["success"]:
                # Create user-friendly summary
                summary_parts = [
                    f"Successfully generated {result['prompts_generated']} "
                    f"optimized prompts:",
                ]

                for prompt in result["prompts"]:
                    summary_parts.append(
                        f"• {prompt['name']} ({prompt['type']}) - "
                        f"Quality: {prompt['quality_score']:.1%}, "
                        f"~{prompt['estimated_tokens']:.0f} tokens"
                    )

                summary_parts.append("Ready to save to your project configuration!")

                return PromptEngineeringToolOutput(
                    success=True,
                    message="\n".join(summary_parts),
                    action="generate_prompts",
                    phase="generation_complete",
                    data=result  # Include full result for saving
                )
            else:
                return PromptEngineeringToolOutput(
                    success=False,
                    message=result.get("error", "Failed to generate prompts"),
                    action="generate_prompts",
                    errors=[result.get("error", "Generation failed")]
                )

        except Exception as e:
            logger.error(f"Failed to generate prompts for {session_key}: {e}")
            return PromptEngineeringToolOutput(
                success=False,
                message=f"Failed to generate prompts: {str(e)}",
                action="generate_prompts",
                errors=[str(e)]
            )

    def _save_prompts(
        self, session_key: str, generated_prompts_data: Dict[str, Any]
    ) -> PromptEngineeringToolOutput:
        """Save generated prompts to project configuration"""
        try:
            if session_key not in self._sessions:
                return PromptEngineeringToolOutput(
                    success=False,
                    message="No active prompt engineering session found.",
                    action="save_prompts",
                    errors=["Session not found"]
                )

            core = self._sessions[session_key]
            result = core.save_prompts_to_project(generated_prompts_data)

            if result["success"]:
                # Clean up session after successful save
                del self._sessions[session_key]

                return PromptEngineeringToolOutput(
                    success=True,
                    message=result["message"],
                    action="save_prompts",
                    phase="complete",
                    data={
                        "prompts_saved": result["prompts_saved"],
                        "change_details": result.get("change_details")
                    }
                )
            else:
                return PromptEngineeringToolOutput(
                    success=False,
                    message=result.get("error", "Failed to save prompts"),
                    action="save_prompts",
                    errors=[result.get("error", "Save failed")]
                )

        except Exception as e:
            logger.error(f"Failed to save prompts for {session_key}: {e}")
            return PromptEngineeringToolOutput(
                success=False,
                message=f"Failed to save prompts: {str(e)}",
                action="save_prompts",
                errors=[str(e)]
            )


# Example usage
if __name__ == "__main__":
    # Example: Analyze project context
    tool = PromptEngineeringTool()

    result = tool.run(PromptEngineeringToolInput(
        action="analyze_context",
        namespace="test",
        project_id="sample"
    ))

    print(f"Context analysis: {result.message}")

    # Example: Start engineering process
    result = tool.run(PromptEngineeringToolInput(
        action="start_engineering",
        namespace="test",
        project_id="sample",
        user_input=(
            "I want to create prompts for a customer support chatbot that "
            "helps with billing questions"
        )
    ))

    print(f"Started engineering: {result.message}")
    if result.next_questions:
        print(f"Questions: {result.next_questions}")
