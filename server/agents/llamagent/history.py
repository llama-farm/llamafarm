from typing import Literal

from pydantic import BaseModel, Field


class LlamAgentChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer", "function"] = (
        Field(..., description="The role of the message")
    )
    content: str = Field(..., description="The content of the message")


class LlamAgentHistory:
    history: list[LlamAgentChatMessage]

    def __init__(self):
        self.history = []

    def add_message(self, message: LlamAgentChatMessage):
        self.history.append(message)

    def get_history(self) -> list[LlamAgentChatMessage]:
        return self.history
