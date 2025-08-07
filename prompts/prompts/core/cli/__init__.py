"""CLI interfaces for the prompts system."""

# Export the main CLI entry points
from .legacy_cli import cli as legacy_cli
from .strategy_cli import cli as strategy_cli

__all__ = ['legacy_cli', 'strategy_cli']