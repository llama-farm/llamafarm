from typing import TypeAlias

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

LFChatCompletionDeveloperMessageParam: TypeAlias = ChatCompletionDeveloperMessageParam
LFChatCompletionSystemMessageParam: TypeAlias = ChatCompletionSystemMessageParam
LFChatCompletionUserMessageParam: TypeAlias = ChatCompletionUserMessageParam
LFChatCompletionAssistantMessageParam: TypeAlias = ChatCompletionAssistantMessageParam
LFChatCompletionToolMessageParam: TypeAlias = ChatCompletionToolMessageParam
LFChatCompletionFunctionMessageParam: TypeAlias = ChatCompletionFunctionMessageParam
LFChatCompletionMessageParam: TypeAlias = (
    LFChatCompletionDeveloperMessageParam
    | LFChatCompletionSystemMessageParam
    | LFChatCompletionUserMessageParam
    | LFChatCompletionAssistantMessageParam
    | LFChatCompletionToolMessageParam
    | LFChatCompletionFunctionMessageParam
)


class LFAgentHistory:
    history: list[LFChatCompletionMessageParam]

    def __init__(self):
        self.history = []

    def add_message(self, message: LFChatCompletionMessageParam):
        self.history.append(message)

    def get_history(self) -> list[dict]:
        return [dict(msg) for msg in self.history]

    @staticmethod
    def message_from_dict(data: dict) -> LFChatCompletionMessageParam:
        """Convert a dictionary (e.g., loaded from disk) to a message param.

        Returns the dict cast as the appropriate message type based on role.
        """
        role = data.get("role")
        if role == "system":
            return LFChatCompletionSystemMessageParam(
                role="system", content=data.get("content", "")
            )
        elif role == "user":
            return LFChatCompletionUserMessageParam(
                role="user", content=data.get("content", "")
            )
        elif role == "assistant":
            # Only include optional fields if they have valid values
            # OpenAI rejects empty strings for 'name' and empty arrays for 'tool_calls'
            params: dict = {
                "role": "assistant",
                "content": data.get("content"),
            }
            if data.get("audio"):
                params["audio"] = data["audio"]
            if data.get("name"):  # Only include if non-empty
                params["name"] = data["name"]
            if data.get("refusal"):
                params["refusal"] = data["refusal"]
            if data.get("tool_calls"):  # Only include if non-empty
                params["tool_calls"] = data["tool_calls"]
            return LFChatCompletionAssistantMessageParam(**params)
        elif role == "tool":
            return LFChatCompletionToolMessageParam(
                role="tool",
                content=data.get("content", ""),
                tool_call_id=data.get("tool_call_id", ""),
            )
        elif role == "developer":
            return LFChatCompletionDeveloperMessageParam(
                role="developer", content=data.get("content", "")
            )
        elif role == "function":
            return LFChatCompletionFunctionMessageParam(
                role="function",
                content=data.get("content", ""),
                name=data.get("name", ""),
            )
        else:
            # Default to user message if role is unknown
            return LFChatCompletionUserMessageParam(
                role="user", content=data.get("content", "")
            )
