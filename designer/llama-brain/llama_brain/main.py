"""Main entry point for Llama Brain."""

import asyncio
from llama_brain.server.main import main as server_main


def main():
    """Main entry point."""
    server_main()


if __name__ == "__main__":
    main()