"""Auto-skip integration tests unless EDGE_URL is explicitly set.

CI collects tests under this directory alongside the unit tests, but the
integration suite needs a reachable edge runtime — so opt-in on the
EDGE_URL environment variable. Developers who want to run it locally
set EDGE_URL=http://<host>:11540 and pytest picks it up normally.
"""

from __future__ import annotations

import os

import pytest

collect_ignore_glob = []

if not os.environ.get("EDGE_URL"):
    collect_ignore_glob = ["test_*.py"]


def pytest_collection_modifyitems(config, items):
    if os.environ.get("EDGE_URL"):
        return
    skip = pytest.mark.skip(
        reason="Integration tests require EDGE_URL pointing to a live edge runtime",
    )
    for item in items:
        if "tests/integration/" in str(item.path).replace("\\", "/"):
            item.add_marker(skip)
