"""
Unit tests for ProjectService.

This module includes a test to verify that creating a project in a reserved
namespace raises ReservedNamespaceError.
"""

import pytest

from api.errors import ReservedNamespaceError
from services.project_service import ProjectService


def test_create_project_reserved_namespace_raises_error():
    """Ensure creating a project in a reserved namespace is rejected."""
    with pytest.raises(ReservedNamespaceError, match="Namespace llamafarm is reserved"):
        ProjectService.create_project("llamafarm", "any_project")
