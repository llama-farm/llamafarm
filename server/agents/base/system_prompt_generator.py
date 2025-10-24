from pydantic import BaseModel, Field

from agents.base.context_provider import LFAgentContextProvider
from agents.base.history import LFAgentChatMessage


class LFAgentPrompt(BaseModel):
    role: str = Field(..., description="The role of the prompt")
    content: str = Field(..., description="The content of the prompt")


class LFAgentSystemPromptGenerator:
    system_prompts: list[LFAgentPrompt]
    context_providers: dict[str, LFAgentContextProvider]

    def __init__(
        self,
        prompts: list[LFAgentChatMessage],
        context_providers: dict[str, LFAgentContextProvider] | None = None,
    ):
        self.system_prompts = [
            LFAgentPrompt(role="system", content=prompt.content)
            for prompt in (prompts or [])
            if prompt.role == "system"
        ]

        self.context_providers = context_providers if context_providers else {}

    def generate_prompt(self) -> str:
        # return "\nYou are a helpful assistant that can answer "
        # "questions and help with tasks."
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
