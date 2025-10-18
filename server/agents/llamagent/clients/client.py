from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from config.datamodel import Model, Prompt

from agents.llamagent.history import LFAgentChatMessage


class LFAgentClient(ABC):
    _model_name: str
    _model_config: Model

    def __init__(self, *, model_config: Model):
        self._model_name = model_config.name
        self._model_config = model_config

    @property
    def model_name(self) -> str:
        return self._model_name

    @staticmethod
    @abstractmethod
    def prompt_to_message(prompt: Prompt) -> LFAgentChatMessage:
        """
        Converts a llamafarm Prompt to a LFAgentChatMessage.
        """
        pass

    @abstractmethod
    async def chat(self, *, messages: list[LFAgentChatMessage]) -> str:
        pass

    @abstractmethod
    async def stream_chat(
        self, *, messages: list[LFAgentChatMessage]
    ) -> AsyncGenerator[str, None]:
        ...
        # Async generator - implementations should use async def with yield
        # Type checkers: return type is AsyncGenerator when async def uses yield
        yield  # type: ignore
