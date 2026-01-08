"""Tests for RouterDatasetService (Phase E6)."""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from services.router_dataset_service import RouterDatasetService


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_router_dir(temp_project_dir):
    """Mock the router directory to use temp dir."""
    def get_router_dir(namespace, project_id, router_name):
        return Path(temp_project_dir) / namespace / project_id / router_name

    def patched_get_datasets_dir(namespace, project_id, router_name):
        return get_router_dir(namespace, project_id, router_name) / "datasets"

    def patched_get_route_dataset_path(namespace, project_id, router_name, route_name):
        datasets_dir = get_router_dir(namespace, project_id, router_name) / "datasets"
        safe_route_name = route_name.replace("/", "_").replace("\\", "_")
        return datasets_dir / f"{safe_route_name}.json"

    with patch.object(
        RouterDatasetService, "get_datasets_dir",
        classmethod(lambda cls, ns, pid, rn: patched_get_datasets_dir(ns, pid, rn))
    ):
        with patch.object(
            RouterDatasetService, "get_route_dataset_path",
            classmethod(lambda cls, ns, pid, rn, route: patched_get_route_dataset_path(ns, pid, rn, route))
        ):
            yield temp_project_dir


class TestRouterDatasetService:
    """Tests for RouterDatasetService."""

    def test_save_utterances_creates_dataset(self, mock_router_dir):
        """Test that save_utterances creates a dataset file."""
        result = RouterDatasetService.save_utterances(
            namespace="default",
            project_id="test_project",
            router_name="test_router",
            route_name="billing",
            utterances=["what is my bill", "payment question"],
        )

        assert result["count"] == 2
        assert result["route_name"] == "billing"
        assert Path(result["path"]).exists()

        # Verify file contents
        with open(result["path"]) as f:
            data = json.load(f)
        assert data["utterances"] == ["what is my bill", "payment question"]

    def test_save_utterances_append_mode(self, mock_router_dir):
        """Test that append mode adds to existing utterances."""
        # First save
        RouterDatasetService.save_utterances(
            namespace="default",
            project_id="test_project",
            router_name="test_router",
            route_name="billing",
            utterances=["first utterance", "second utterance"],
        )

        # Second save with append
        result = RouterDatasetService.save_utterances(
            namespace="default",
            project_id="test_project",
            router_name="test_router",
            route_name="billing",
            utterances=["third utterance"],
            mode="append",
        )

        assert result["count"] == 3
        utterances = RouterDatasetService.load_utterances(
            "default", "test_project", "test_router", "billing"
        )
        assert len(utterances) == 3

    def test_save_utterances_overwrite_mode(self, mock_router_dir):
        """Test that overwrite mode replaces existing utterances."""
        # First save
        RouterDatasetService.save_utterances(
            namespace="default",
            project_id="test_project",
            router_name="test_router",
            route_name="billing",
            utterances=["first utterance", "second utterance"],
        )

        # Second save with overwrite
        result = RouterDatasetService.save_utterances(
            namespace="default",
            project_id="test_project",
            router_name="test_router",
            route_name="billing",
            utterances=["only this one"],
            mode="overwrite",
        )

        assert result["count"] == 1
        utterances = RouterDatasetService.load_utterances(
            "default", "test_project", "test_router", "billing"
        )
        assert utterances == ["only this one"]

    def test_save_utterances_deduplicates(self, mock_router_dir):
        """Test that duplicate utterances are removed."""
        result = RouterDatasetService.save_utterances(
            namespace="default",
            project_id="test_project",
            router_name="test_router",
            route_name="billing",
            utterances=["hello", "HELLO", "Hello", "world"],
        )

        # Should only have 2 unique (case-insensitive)
        assert result["count"] == 2

    def test_load_utterances_returns_list(self, mock_router_dir):
        """Test that load_utterances returns the saved list."""
        utterances = ["test 1", "test 2", "test 3"]
        RouterDatasetService.save_utterances(
            namespace="default",
            project_id="test_project",
            router_name="test_router",
            route_name="support",
            utterances=utterances,
        )

        loaded = RouterDatasetService.load_utterances(
            "default", "test_project", "test_router", "support"
        )
        assert loaded == utterances

    def test_load_utterances_missing_returns_empty(self, mock_router_dir):
        """Test that loading missing dataset returns empty list."""
        loaded = RouterDatasetService.load_utterances(
            "default", "test_project", "test_router", "nonexistent"
        )
        assert loaded == []

    def test_list_datasets(self, mock_router_dir):
        """Test listing all datasets for a router."""
        # Create multiple datasets
        RouterDatasetService.save_utterances(
            namespace="default",
            project_id="test_project",
            router_name="test_router",
            route_name="billing",
            utterances=["bill 1", "bill 2"],
        )
        RouterDatasetService.save_utterances(
            namespace="default",
            project_id="test_project",
            router_name="test_router",
            route_name="support",
            utterances=["support 1"],
        )

        datasets = RouterDatasetService.list_datasets(
            "default", "test_project", "test_router"
        )

        assert len(datasets) == 2
        route_names = {d["route_name"] for d in datasets}
        assert route_names == {"billing", "support"}

    def test_delete_dataset(self, mock_router_dir):
        """Test deleting a dataset."""
        RouterDatasetService.save_utterances(
            namespace="default",
            project_id="test_project",
            router_name="test_router",
            route_name="billing",
            utterances=["to be deleted"],
        )

        # Verify exists
        assert RouterDatasetService.load_utterances(
            "default", "test_project", "test_router", "billing"
        ) == ["to be deleted"]

        # Delete
        deleted = RouterDatasetService.delete_dataset(
            "default", "test_project", "test_router", "billing"
        )
        assert deleted is True

        # Verify gone
        assert RouterDatasetService.load_utterances(
            "default", "test_project", "test_router", "billing"
        ) == []

    def test_delete_nonexistent_returns_false(self, mock_router_dir):
        """Test that deleting non-existent dataset returns False."""
        deleted = RouterDatasetService.delete_dataset(
            "default", "test_project", "test_router", "nonexistent"
        )
        assert deleted is False
