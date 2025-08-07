"""Main entry point for the prompts package when run with -m."""

from .core.cli.strategy_cli import cli

if __name__ == '__main__':
    cli()