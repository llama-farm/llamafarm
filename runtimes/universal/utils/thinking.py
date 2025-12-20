"""
Thinking/reasoning model utilities.

Provides support for models like Qwen3 that use <think>...</think> tags
for chain-of-thought reasoning.
"""

import re
from dataclasses import dataclass


@dataclass
class ParsedThinkingResponse:
    """Parsed response from a thinking model."""

    thinking: str | None  # Content inside <think>...</think> tags
    content: (
        str  # Final answer (content after </think> or full response if no thinking)
    )
    thinking_complete: bool  # Whether thinking was properly closed with </think>


def parse_thinking_response(response: str) -> ParsedThinkingResponse:
    """Parse a response that may contain <think>...</think> tags.

    Extracts thinking content and final answer from model responses.
    Handles cases where thinking is incomplete (no closing tag).

    Args:
        response: Raw model response text

    Returns:
        ParsedThinkingResponse with thinking and content separated

    Examples:
        >>> parse_thinking_response("<think>Let me think...</think>The answer is 42.")
        ParsedThinkingResponse(thinking="Let me think...", content="The answer is 42.", thinking_complete=True)

        >>> parse_thinking_response("<think>Still thinking...")
        ParsedThinkingResponse(thinking="Still thinking...", content="", thinking_complete=False)

        >>> parse_thinking_response("No thinking here, just answer.")
        ParsedThinkingResponse(thinking=None, content="No thinking here, just answer.", thinking_complete=True)
    """
    # Pattern to match <think>...</think> with content after
    think_pattern = re.compile(
        r"<think>\s*(.*?)\s*</think>\s*(.*)",
        re.DOTALL | re.IGNORECASE,
    )

    match = think_pattern.match(response)
    if match:
        thinking = match.group(1).strip()
        content = match.group(2).strip()
        # Recursively clean any remaining </think> tags from content
        # (model sometimes outputs multiple closing tags in /no_think mode)
        content = re.sub(r"^\s*</think>\s*", "", content, flags=re.IGNORECASE)
        return ParsedThinkingResponse(
            thinking=thinking if thinking else None,
            content=content.strip(),
            thinking_complete=True,
        )

    # Check for stray </think> anywhere (happens with /no_think mode)
    # The model outputs empty think block or just closing tag(s)
    # Remove ALL </think> tags from the response
    cleaned = re.sub(r"</think>\s*", "", response, flags=re.IGNORECASE)
    if cleaned != response:
        # We removed some </think> tags
        return ParsedThinkingResponse(
            thinking=None,
            content=cleaned.strip(),
            thinking_complete=True,
        )

    # Check for incomplete thinking (opening tag but no closing)
    incomplete_pattern = re.compile(r"<think>\s*(.*)", re.DOTALL | re.IGNORECASE)
    incomplete_match = incomplete_pattern.match(response)
    if incomplete_match:
        thinking = incomplete_match.group(1).strip()
        return ParsedThinkingResponse(
            thinking=thinking if thinking else None,
            content="",
            thinking_complete=False,
        )

    # No thinking tags at all
    return ParsedThinkingResponse(
        thinking=None,
        content=response.strip(),
        thinking_complete=True,
    )


def inject_thinking_control(
    messages: list[dict],
    enable_thinking: bool,
) -> list[dict]:
    """Inject thinking control into messages using Qwen's soft switch.

    Qwen3 models support /think and /no_think soft switches in prompts
    to control whether the model uses thinking mode.

    Args:
        messages: List of chat messages
        enable_thinking: True to force thinking, False to disable

    Returns:
        Modified messages list with thinking control injected
    """

    # Make a copy to avoid modifying the original
    messages = [dict(m) for m in messages]

    # Find the last user message and append the control
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            content = messages[i].get("content", "")
            if enable_thinking:
                # Only add /think if not already present
                if "/think" not in content and "/no_think" not in content:
                    messages[i]["content"] = f"{content} /think"
            else:
                # Only add /no_think if not already present
                if "/think" not in content and "/no_think" not in content:
                    messages[i]["content"] = f"{content} /no_think"
            break

    return messages


class ThinkingBudgetProcessor:
    """Logits processor that enforces a thinking token budget.

    When the thinking budget is reached, this processor forces the model
    to generate </think> and proceed to the answer.

    This is used with llama-cpp-python's logits_processor parameter.
    Uses numpy arrays (required by llama-cpp-python since PR #499).
    """

    def __init__(
        self,
        llama,
        max_thinking_tokens: int,
        think_end_tokens: list[int] | None = None,
    ):
        """Initialize the thinking budget processor.

        Args:
            llama: The Llama instance (for tokenization)
            max_thinking_tokens: Maximum tokens to allow for thinking
            think_end_tokens: Token IDs for </think> (auto-detected if None)
        """
        self.llama = llama
        self.max_thinking_tokens = max_thinking_tokens
        self.thinking_tokens = 0  # Only counts tokens INSIDE <think>
        self.in_thinking = False
        self.thinking_ended = False
        self.forcing_end = False  # True while forcing </think> sequence

        # Try to get the token IDs for </think>
        if think_end_tokens is None:
            try:
                # Tokenize </think> to get its token IDs
                self.think_end_tokens = llama.tokenize(
                    b"</think>", add_bos=False, special=True
                )
            except Exception:
                # Fallback - will use soft switch instead
                self.think_end_tokens = None
        else:
            self.think_end_tokens = think_end_tokens

        self._force_token_idx = 0

    def __call__(self, input_ids, scores):
        """Process logits to enforce thinking budget.

        Args:
            input_ids: numpy array of token IDs generated so far
            scores: numpy array of logits for next token (modified in-place)

        Returns:
            Modified scores array (numpy)
        """
        import numpy as np

        # Convert to numpy if needed (for compatibility)
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)

        # Check current state by looking at generated text
        if not self.thinking_ended and not self.forcing_end:
            try:
                # Convert input_ids to list if numpy array
                ids = (
                    input_ids.tolist()
                    if hasattr(input_ids, "tolist")
                    else list(input_ids)
                )
                text = self.llama.detokenize(ids).decode("utf-8", errors="ignore")

                if "<think>" in text.lower() and not self.in_thinking:
                    self.in_thinking = True
                if "</think>" in text.lower():
                    self.thinking_ended = True
                    self.in_thinking = False
            except Exception:
                pass

        # Count tokens only while in thinking mode
        if self.in_thinking and not self.thinking_ended:
            self.thinking_tokens += 1

        # If in thinking, over budget, and have end tokens - start forcing </think>
        if (
            self.in_thinking
            and not self.thinking_ended
            and self.thinking_tokens >= self.max_thinking_tokens
            and self.think_end_tokens
            and not self.forcing_end
        ):
            self.forcing_end = True

        # If we are actively forcing the end token sequence
        if self.forcing_end and self._force_token_idx < len(self.think_end_tokens):
            target_token = self.think_end_tokens[self._force_token_idx]
            self._force_token_idx += 1

            # Set all logits to -inf except the target token
            scores[:] = -np.inf
            if target_token < len(scores):
                scores[target_token] = 0.0
        elif self.forcing_end and self._force_token_idx >= len(self.think_end_tokens):
            # Finalize state when forcing completes
            self.thinking_ended = True
            self.in_thinking = False

        return scores
