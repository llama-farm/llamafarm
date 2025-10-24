from typing import Literal

from pydantic import BaseModel, Field


class LFAgentChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer", "function"] = (
        Field(..., description="The role of the message")
    )
    content: str = Field(..., description="The content of the message")


class LFAgentHistory:
    history: list[LFAgentChatMessage]

    def __init__(self):
        self.history = []

    def add_message(self, message: LFAgentChatMessage):
        self.history.append(message)

    def get_history(self) -> list[dict]:
        return [{"role": msg.role, "content": msg.content} for msg in self.history]
