import json
import os
from pathlib import Path

from services.runtime_service.runtime_service import RuntimeService

from config.datamodel import LlamaFarmConfig, Provider

from agents.llamagent.agent import LFAgent, LFAgentConfig
from agents.llamagent.clients.client import LFAgentClient
from agents.llamagent.clients.openai import LFAgentClientOpenAI
from agents.llamagent.history import LFAgentChatMessage, LFAgentHistory
from agents.llamagent.system_prompt_generator import LFAgentSystemPromptGenerator
from core.logging import FastAPIStructLogger
from services.model_service import ModelService

logger = FastAPIStructLogger(__name__)

CLIENT_CLASSES = {
    Provider.openai: LFAgentClientOpenAI,
    Provider.ollama: LFAgentClientOpenAI,
    Provider.lemonade: LFAgentClientOpenAI,
}


class ChatOrchestratorAgent(LFAgent):
    _persist_enabled: bool
    _project_dir: str
    _project_config: LlamaFarmConfig
    model_name: str
    _session_id: str | None = None

    def __init__(
        self,
        *,
        project_config: LlamaFarmConfig,
        project_dir: str,
        model_name: str | None = None,
    ):
        self._project_config = project_config
        self._project_dir = project_dir
        self._session_id = None
        self._persist_enabled = False
        self.model_name = (
            model_name or ModelService.get_model(project_config, model_name).name
        )

        history = self._get_history(project_config)

        model_config = ModelService.get_model(project_config, self.model_name)
        provider = RuntimeService.get_provider(model_config)
        client = provider.get_client()

        system_prompt_generator = LFAgentSystemPromptGenerator(
            prompts=self._get_prompts_for_model(self.model_name)
        )
        config = LFAgentConfig(
            history=history,
            system_prompt_generator=system_prompt_generator,
            client=client,
        )

        super().__init__(config=config)

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
            self._restore_persisted_history()

        except Exception:
            logger.warning("Failed to enable persistence", exc_info=True)

    def reset_history(self):
        super().reset_history()
        # Clear persisted history by removing the file
        path = self._history_file_path
        if path:
            path.unlink(missing_ok=True)

    def _populate_history_with_non_system_prompts(
        self, history: LFAgentHistory, project_config: LlamaFarmConfig
    ):
        prompts = self._get_prompts_for_model(self.model_name)
        for prompt in prompts:
            # Only add non-system prompts to the history
            if prompt.role != "system":
                history.add_message(prompt)

    def _get_history(self, project_config: LlamaFarmConfig) -> LFAgentHistory:
        history = LFAgentHistory()
        self._populate_history_with_non_system_prompts(history, project_config)
        return history

    def _get_prompts_for_model(self, model_name: str) -> list[LFAgentChatMessage]:
        model_config = ModelService.get_model(self._project_config, model_name)
        provider = RuntimeService.get_provider(model_config)
        Client = provider.get_client().__class__

        return [
            Client.prompt_to_message(prompt)
            for prompt in self._project_config.prompts or []
        ]

    @property
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
        path = self._history_file_path
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
                self.history.add_message(
                    message=LFAgentChatMessage(
                        role=role,
                        content=content,
                    )
                )
            except Exception:
                # Skip malformed entries defensively
                continue

    def _persist_history(self) -> None:
        path = self._history_file_path
        if not path:
            return
        try:
            history = self.history.get_history()
            tmp_path = Path(str(path) + ".tmp")
            tmp_path.write_text(
                json.dumps(history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp_path, path)
        except Exception:
            logger.warning(
                "Failed to persist history",
                path=str(path) if path else None,
                exc_info=True,
            )


class ChatOrchestratorAgentFactory:
    @staticmethod
    def create_agent(
        *,
        project_config: LlamaFarmConfig,
        project_dir: str,
        model_name: str | None = None,
        session_id: str | None = None,
    ) -> LFAgent:
        agent = ChatOrchestratorAgent(
            project_config=project_config,
            project_dir=project_dir,
            model_name=model_name,
        )
        if session_id:
            agent.enable_persistence(session_id=session_id)

        # TODO: If project config contains an MCP server, enable tools.
        return agent
