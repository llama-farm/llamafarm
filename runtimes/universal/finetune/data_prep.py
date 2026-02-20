"""Dataset preparation and validation for fine-tuning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of dataset validation."""
    
    valid: bool
    format: str | None = None
    errors: list[str] | None = None
    warnings: list[str] | None = None
    num_examples: int = 0


def auto_detect_format(data: list[dict]) -> str:
    """Auto-detect dataset format from structure.
    
    Returns: "sharegpt", "alpaca", "chat", "raw", or "unknown"
    """
    if not data:
        return "unknown"
    
    sample = data[0]
    
    # ShareGPT format
    if "conversations" in sample:
        return "sharegpt"
    
    # Chat format (OpenAI-style)
    if "messages" in sample:
        return "chat"
    
    # Alpaca format
    if "instruction" in sample and "output" in sample:
        return "alpaca"
    
    # Raw text format
    if "text" in sample:
        return "raw"
    
    return "unknown"


def validate_sft_dataset(data: list[dict], expected_format: str | None = None) -> ValidationResult:
    """Validate dataset for supervised fine-tuning.
    
    Args:
        data: Dataset to validate
        expected_format: Optional format hint ("auto", "sharegpt", "alpaca", "chat", "raw")
    
    Returns:
        ValidationResult with validation details
    """
    errors = []
    warnings = []
    
    if not data:
        return ValidationResult(valid=False, errors=["Dataset is empty"])
    
    # Auto-detect format if not specified
    detected_format = auto_detect_format(data)
    if expected_format and expected_format != "auto" and expected_format != detected_format:
        warnings.append(f"Expected {expected_format}, detected {detected_format}")
    
    format_to_use = expected_format if expected_format and expected_format != "auto" else detected_format
    
    # Validate based on format
    if format_to_use == "sharegpt":
        for i, item in enumerate(data):
            if "conversations" not in item:
                errors.append(f"Example {i}: missing 'conversations' field")
                continue
            
            convs = item["conversations"]
            if not isinstance(convs, list):
                errors.append(f"Example {i}: 'conversations' must be a list")
                continue
            
            for j, turn in enumerate(convs):
                if not isinstance(turn, dict):
                    errors.append(f"Example {i}, turn {j}: must be a dict")
                    continue
                if "from" not in turn or "value" not in turn:
                    errors.append(f"Example {i}, turn {j}: missing 'from' or 'value'")
                if turn.get("from") not in ["human", "gpt", "system"]:
                    warnings.append(f"Example {i}, turn {j}: unusual 'from' value: {turn.get('from')}")
    
    elif format_to_use == "chat":
        for i, item in enumerate(data):
            if "messages" not in item:
                errors.append(f"Example {i}: missing 'messages' field")
                continue
            
            msgs = item["messages"]
            if not isinstance(msgs, list):
                errors.append(f"Example {i}: 'messages' must be a list")
                continue
            
            for j, msg in enumerate(msgs):
                if not isinstance(msg, dict):
                    errors.append(f"Example {i}, message {j}: must be a dict")
                    continue
                if "role" not in msg or "content" not in msg:
                    errors.append(f"Example {i}, message {j}: missing 'role' or 'content'")
                if msg.get("role") not in ["user", "assistant", "system"]:
                    warnings.append(f"Example {i}, message {j}: unusual role: {msg.get('role')}")
    
    elif format_to_use == "alpaca":
        for i, item in enumerate(data):
            if "instruction" not in item:
                errors.append(f"Example {i}: missing 'instruction' field")
            if "output" not in item:
                errors.append(f"Example {i}: missing 'output' field")
            # "input" is optional
    
    elif format_to_use == "raw":
        for i, item in enumerate(data):
            if "text" not in item:
                errors.append(f"Example {i}: missing 'text' field")
    
    else:
        errors.append(f"Unknown or unsupported format: {format_to_use}")
    
    return ValidationResult(
        valid=len(errors) == 0,
        format=format_to_use,
        errors=errors if errors else None,
        warnings=warnings if warnings else None,
        num_examples=len(data)
    )


def validate_cpt_dataset(data: list[dict]) -> ValidationResult:
    """Validate dataset for continued pre-training.
    
    Args:
        data: Dataset to validate (must have 'text' field in each example)
    
    Returns:
        ValidationResult with validation details
    """
    errors = []
    
    if not data:
        return ValidationResult(valid=False, format="raw", errors=["Dataset is empty"])
    
    for i, item in enumerate(data):
        if "text" not in item:
            errors.append(f"Example {i}: missing 'text' field")
        elif not isinstance(item["text"], str):
            errors.append(f"Example {i}: 'text' must be a string")
        elif len(item["text"].strip()) == 0:
            errors.append(f"Example {i}: 'text' is empty")
    
    return ValidationResult(
        valid=len(errors) == 0,
        format="raw",
        errors=errors if errors else None,
        num_examples=len(data)
    )


def format_sharegpt(data: list[dict], template: str | None = None) -> list[dict]:
    """Convert or validate ShareGPT format.
    
    Args:
        data: Dataset (already in ShareGPT or convertible)
        template: Chat template hint (unused for ShareGPT, kept for API consistency)
    
    Returns:
        Normalized ShareGPT format
    """
    result = []
    for item in data:
        if "conversations" in item:
            # Already ShareGPT
            result.append(item)
        elif "messages" in item:
            # Convert from chat format
            conversations = []
            for msg in item["messages"]:
                role_map = {"user": "human", "assistant": "gpt", "system": "system"}
                conversations.append({
                    "from": role_map.get(msg["role"], msg["role"]),
                    "value": msg["content"]
                })
            result.append({"conversations": conversations})
        else:
            # Pass through (will fail validation later)
            result.append(item)
    
    return result


def format_alpaca(data: list[dict]) -> list[dict]:
    """Convert to or validate Alpaca instruction format.
    
    Args:
        data: Dataset (already Alpaca or convertible)
    
    Returns:
        Normalized Alpaca format
    """
    result = []
    for item in data:
        if "instruction" in item and "output" in item:
            # Already Alpaca
            result.append(item)
        elif "messages" in item:
            # Convert from chat format (simple heuristic)
            msgs = item["messages"]
            instruction = ""
            output = ""
            input_text = ""
            
            for msg in msgs:
                if msg["role"] == "user":
                    if not instruction:
                        instruction = msg["content"]
                    else:
                        input_text += msg["content"] + "\n"
                elif msg["role"] == "assistant":
                    output = msg["content"]
            
            result.append({
                "instruction": instruction,
                "input": input_text.strip(),
                "output": output
            })
        else:
            # Pass through
            result.append(item)
    
    return result


def format_raw_text(data: list[dict]) -> list[dict]:
    """Convert to raw text format for CPT.
    
    Args:
        data: Dataset with 'text' field or convertible
    
    Returns:
        Normalized raw text format [{"text": "..."}, ...]
    """
    result = []
    for item in data:
        if "text" in item:
            # Already raw
            result.append({"text": item["text"]})
        elif "conversations" in item or "messages" in item:
            # Flatten conversation to text
            text_parts = []
            
            if "conversations" in item:
                for turn in item["conversations"]:
                    text_parts.append(f"{turn['from']}: {turn['value']}")
            elif "messages" in item:
                for msg in item["messages"]:
                    text_parts.append(f"{msg['role']}: {msg['content']}")
            
            result.append({"text": "\n".join(text_parts)})
        elif "instruction" in item:
            # Flatten Alpaca to text
            parts = [f"Instruction: {item['instruction']}"]
            if item.get("input"):
                parts.append(f"Input: {item['input']}")
            parts.append(f"Output: {item['output']}")
            result.append({"text": "\n".join(parts)})
        else:
            # Try to extract any text
            text = str(item.get("text", ""))
            result.append({"text": text})
    
    return result
