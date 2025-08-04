"""
Pytest configuration and fixtures for the server tests.
"""

import sys
from pathlib import Path

# Add the parent directory to the Python path so we can import modules
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

# Add server directory to path for service imports
server_root = Path(__file__).parent.parent
sys.path.insert(0, str(server_root))