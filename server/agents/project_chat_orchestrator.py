import json
import os
import sys
from pathlib import Path

import instructor
from atomic_agents import BaseIOSchema  # type: ignore
from atomic_agents.agents.atomic_agent import (  # type: ignore
    ChatHistory,
    SystemPromptGenerator,
)
from openai import AsyncOpenAI

from agents.agent import LFAgent, LFAgentConfig
from core.settings import settings
from context_providers.docs_context_provider import DocsContextProvider

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
from config.datamodel import (  # noqa: E402
    PromptFormat,
    LlamaFarmConfig,
    Prompt,
    Provider,
)

from core.logging import FastAPIStructLogger  # noqa: E402

logger = FastAPIStructLogger(__name__)


class ProjectChatOrchestratorAgentInputSchema(BaseIOSchema):
    """
    Input schema for the project chat orchestrator agent.
    """

    chat_message: str


class ProjectChatOrchestratorAgentOutputSchema(BaseIOSchema):
    """
    Output schema for the project chat orchestrator agent.
    This schema is intentionally simple to ensure compatibility with Ollama's JSON parsing.
    """

    chat_message: str


class ProjectChatOrchestratorAgent(LFAgent):
    def __init__(
        self,
        project_config: LlamaFarmConfig,
        project_dir: str,
    ):
        # Build base history from config
        history = _get_history(project_config)
        client = _get_client(project_config)

        lf_config = LFAgentConfig(
            client=client,
            model=project_config.runtime.model,
            history=history,
            system_prompt_generator=LFSystemPromptGenerator(
                project_config=project_config
            ),
            model_api_parameters=project_config.runtime.model_api_parameters,
        )

        super().__init__(config=lf_config)

        # Session-scoped persistence context
        self._namespace = project_config.namespace
        self._project_id = project_config.name
        self._project_dir = project_dir
        self._persist_enabled = False
        self._is_new_session = True  # Track if this is a new session for greeting logic

        # Register docs context provider
        self.docs_context_provider = DocsContextProvider(title="Relevant Documentation")
        self.register_context_provider("docs", self.docs_context_provider)

    def enable_persistence(
        self,
        *,
        session_id: str,
    ) -> None:
        """Enable disk persistence for this agent and restore history.

        Use this when the agent was constructed without context (e.g., via a factory
        mocked in tests) but we still want to persist session history.
        """
        try:
            self._persist_enabled = True
            self._session_id = session_id

            # Check if history exists before restoration
            history_path = self._history_file_path()
            has_existing_history = history_path and history_path.exists() and history_path.stat().st_size > 2

            self._restore_persisted_history()

            # Determine if this is a new session based on whether we restored any history
            self._is_new_session = not has_existing_history

            # Inject greeting if enabled
            self._inject_greeting_if_needed()

        except Exception:
            logger.warning("Failed to enable persistence", exc_info=True)

    def _inject_greeting_if_needed(self) -> None:
        """Inject appropriate greeting based on session status."""
        # Check if greetings are disabled via settings
        if not settings.lf_dev_mode_greeting_enabled:
            return

        # Only inject greeting for project_seed (dev mode)
        if self._project_id != "project_seed":
            return

        try:
            if self._is_new_session:
                # New user greeting
                greeting_content = (
                    "Welcome to LlamaFarm dev mode! 🦙\n\n"
                    "I can help you with:\n"
                    "- Getting started: `lf init`, `lf start`\n"
                    "- Dataset management: `lf datasets create/upload/process`\n"
                    "- RAG queries: `lf rag query`\n"
                    "- Configuration: editing `llamafarm.yaml`\n\n"
                    "Ask me anything about LlamaFarm, or type `/help` for chat commands!"
                )
            else:
                # Returning user greeting
                # Count previous turns (user messages only)
                turn_count = sum(1 for msg in self.history.get_history()
                               if getattr(msg, "role", None) == "user" or
                               (isinstance(msg, dict) and msg.get("role") == "user"))

                greeting_content = (
                    f"Welcome back! 👋\n\n"
                    f"Last session: {turn_count} message{'s' if turn_count != 1 else ''}.\n"
                    f"How can I help you with LlamaFarm today?"
                )

            # Check if we've already injected this type of greeting
            # For new sessions, check for "Welcome to LlamaFarm dev mode"
            # For returning sessions, check for "Welcome back"
            expected_greeting_text = "Welcome to LlamaFarm dev mode" if self._is_new_session else "Welcome back"

            recent_messages = list(self.history.get_history())[-3:]  # Check last 3 messages
            has_greeting = False
            for msg in recent_messages:
                role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
                if role == "assistant":
                    content_obj = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
                    # Extract plain text from content
                    content = None
                    if isinstance(content_obj, dict):
                        content = content_obj.get("chat_message")
                    elif hasattr(content_obj, "chat_message"):
                        content = getattr(content_obj, "chat_message", None)
                    elif isinstance(content_obj, str):
                        # Handle JSON-serialized schemas
                        try:
                            parsed = json.loads(content_obj)
                            if isinstance(parsed, dict) and "chat_message" in parsed:
                                content = parsed["chat_message"]
                            else:
                                content = content_obj
                        except (json.JSONDecodeError, TypeError):
                            content = content_obj
                    if content and expected_greeting_text in content:
                        has_greeting = True
                        break

            if not has_greeting:
                # Add greeting as schema object - ChatHistory.add_message expects schema, not plain strings
                greeting_schema = ProjectChatOrchestratorAgentOutputSchema(chat_message=greeting_content)
                self.history.add_message("assistant", greeting_schema)
                logger.info("Injected greeting", is_new_session=self._is_new_session)

        except Exception:
            logger.warning("Failed to inject greeting", exc_info=True)

    def reset_history(self):
        super().reset_history()
        # Clear persisted history by removing the file
        path = self._history_file_path()
        if path:
            path.unlink(missing_ok=True)

    # -------------------- Persistence helpers --------------------
    def _history_file_path(self) -> Path | None:
        if not self._persist_enabled or not self._session_id:
            return None
        base_dir = Path(self._project_dir)
        sessions_dir = base_dir / "sessions" / self._session_id
        try:
            sessions_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Best-effort; if mkdir fails, disable persistence
            logger.warning(
                "Failed to create sessions directory",
                path=str(sessions_dir),
                exc_info=True,
            )
            return None
        return sessions_dir / "history.json"

    def _restore_persisted_history(self) -> None:
        path = self._history_file_path()
        if not path:
            return
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning(
                "Failed to read/parse history file", path=str(path), exc_info=True
            )
            return

        # Add messages into history in order
        items = data if isinstance(data, list) else []
        for item in items:
            try:
                role = item.get("role")
                content = item.get("content", "")
                if not role or not isinstance(content, str):
                    continue
                if role == "user":
                    schema = ProjectChatOrchestratorAgentInputSchema(
                        chat_message=content
                    )
                elif role == "assistant":
                    schema = ProjectChatOrchestratorAgentOutputSchema(
                        chat_message=content
                    )
                else:
                    # Skip system or unknown roles; system prompts are handled separately
                    continue
                self.history.add_message(role, schema)
            except Exception:
                # Skip malformed entries defensively
                continue

    def _persist_history(self) -> None:
        path = self._history_file_path()
        if not path:
            return
        try:
            # Serialize only user/assistant messages to a simple JSON array
            serialized: list[dict[str, str]] = []
            # ChatHistory stores internal message structures; access via .history when available
            for msg in self.history.get_history():
                try:
                    role = getattr(msg, "role", None) or (
                        msg.get("role") if isinstance(msg, dict) else None
                    )
                    content_obj = getattr(msg, "content", None) or (
                        msg.get("content") if isinstance(msg, dict) else None
                    )
                    # For our IO schemas, content is an object with chat_message
                    content = None
                    if isinstance(content_obj, dict):
                        content = content_obj.get("chat_message")
                    elif hasattr(content_obj, "chat_message"):
                        content = getattr(content_obj, "chat_message", None)
                    elif isinstance(content_obj, str):
                        # ChatHistory serializes schemas to JSON strings internally
                        # Try to parse and extract chat_message
                        try:
                            parsed = json.loads(content_obj)
                            if isinstance(parsed, dict) and "chat_message" in parsed:
                                content = parsed["chat_message"]
                            else:
                                content = content_obj
                        except (json.JSONDecodeError, TypeError):
                            content = content_obj
                    if role in ("user", "assistant") and isinstance(content, str):
                        serialized.append({"role": role, "content": content})
                except Exception:
                    continue

            tmp_path = Path(str(path) + ".tmp")
            tmp_path.write_text(
                json.dumps(serialized, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp_path, path)
        except Exception:
            logger.warning(
                "Failed to persist history",
                path=str(path) if path else None,
                exc_info=True,
            )

    # -------------------- Execution overrides --------------------
    async def run_async(self, user_input):
        response = await super().run_async(user_input)
        try:
            self._persist_history()
        except Exception:
            logger.warning("History persistence failed after run_async", exc_info=True)
        return response

    async def run_async_stream(self, user_input):
        async for chunk in super().run_async_stream(user_input):
            yield chunk
        try:
            self._persist_history()
        except Exception:
            logger.warning(
                "History persistence failed after run_async_stream", exc_info=True
            )


class LFSystemPromptGenerator(SystemPromptGenerator):
    def __init__(self, project_config: LlamaFarmConfig):
        logger.info(f"Project config: {project_config}")
        self.system_prompts = [
            prompt
            for prompt in (project_config.prompts or [])
            if prompt.role == "system"
        ]
        super().__init__()

    def generate_prompt(self) -> str:
        # return "\nYou are a helpful assistant that can answer questions and help with tasks."
        prompt_parts = []
        for prompt in self.system_prompts:
            prompt_parts.append(prompt.content)
            prompt_parts.append("")

        if self.context_providers:
            prompt_parts.append("# EXTRA INFORMATION AND CONTEXT")
            for provider in self.context_providers.values():
                info = provider.get_info()
                if info:
                    prompt_parts.append(f"## {provider.title}")
                    prompt_parts.append(info)
                    prompt_parts.append("")

        return "\n".join(prompt_parts)


def _prompt_to_content_schema(prompt: Prompt) -> BaseIOSchema:
    if prompt.role == "assistant":
        return ProjectChatOrchestratorAgentOutputSchema(
            chat_message=prompt.content,
        )
    elif prompt.role == "user":
        return ProjectChatOrchestratorAgentInputSchema(
            chat_message=prompt.content,
        )
    else:
        raise ValueError(f"Unsupported role: {prompt.role}")


def _populate_history_with_non_system_prompts(
    history: ChatHistory, project_config: LlamaFarmConfig
):
    for prompt in project_config.prompts or []:
        # Only add non-system prompts to the history
        if prompt.role != "system":
            history.add_message(
                role=prompt.role,
                content=_prompt_to_content_schema(prompt),
            )


def _get_history(project_config: LlamaFarmConfig) -> ChatHistory:
    history = ChatHistory()
    _populate_history_with_non_system_prompts(history, project_config)
    return history


def _get_client(
    project_config: LlamaFarmConfig,
) -> instructor.client.AsyncInstructor | AsyncOpenAI:
    mode = _determine_instructor_mode(project_config)

    if project_config.runtime.provider == Provider.openai:
        openaiClient = AsyncOpenAI(
            api_key=project_config.runtime.api_key,
            base_url=project_config.runtime.base_url,
        )
        if project_config.runtime.prompt_format == PromptFormat.structured:
            return instructor.from_openai(openaiClient, mode=mode)
        else:
            return openaiClient

    if project_config.runtime.provider == Provider.ollama:
        openaiClient = AsyncOpenAI(
            api_key=project_config.runtime.api_key or settings.ollama_api_key,
            base_url=project_config.runtime.base_url or f"{settings.ollama_host}/v1",
        )
        if project_config.runtime.prompt_format == PromptFormat.structured:
            return instructor.from_openai(openaiClient, mode=mode)
        else:
            return openaiClient

    else:
        raise ValueError(f"Unsupported provider: {project_config.runtime.provider}")


def _determine_instructor_mode(project_config: LlamaFarmConfig) -> instructor.Mode:
    # Use the configured instructor mode or default based on provider
    if project_config.runtime.instructor_mode is not None:
        # It's a string value
        mode_str = project_config.runtime.instructor_mode

        # Map the configured mode string to instructor.Mode
        try:
            mode = instructor.mode.Mode[mode_str.upper()]
            logger.debug(f"Using configured instructor mode: {mode}")
        except KeyError as e:
            # Invalid mode specified
            raise ValueError(
                f"Invalid instructor_mode '{mode_str}'. "
                f"Common modes include: tools, json, md_json, anthropic_tools, gemini_json. "
                f"See instructor documentation for full list of supported modes."
            ) from e
    elif project_config.runtime.provider == Provider.ollama:
        # Default to MD_JSON for Ollama as it's most compatible
        mode = instructor.Mode.MD_JSON
        logger.debug("Using MD_JSON mode for Ollama provider (default)")
    else:
        mode = instructor.Mode.TOOLS
        logger.debug("Using TOOLS mode (default for non-Ollama)")

    logger.debug(f"Instructor mode: {mode}")
    return mode


class ProjectChatOrchestratorAgentFactory:
    @staticmethod
    def create_agent(
        project_config: LlamaFarmConfig, project_dir: str
    ) -> ProjectChatOrchestratorAgent:
        runtime = project_config.runtime
        pf = runtime.prompt_format or PromptFormat.unstructured
        logger.info("Creating chat agent", prompt_format=pf.value, model=runtime.model)
        return ProjectChatOrchestratorAgent(project_config, project_dir=project_dir)
