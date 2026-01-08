"""Tests for RouterStorageService - project-specific router storage."""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from services.router_storage_service import RouterStorageService


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_project_service(temp_project_dir):
    """Mock ProjectService to return our temp directory."""
    with patch("services.router_storage_service.ProjectService") as mock:
        mock.get_project_dir.return_value = temp_project_dir
        yield mock


class TestRouterStorageService:
    """Tests for RouterStorageService."""

    def test_get_routers_dir(self, mock_project_service, temp_project_dir):
        """Test getting the routers directory path."""
        routers_dir = RouterStorageService.get_routers_dir("default", "test_project")

        assert routers_dir == Path(temp_project_dir) / "lf_data" / "routers"
        mock_project_service.get_project_dir.assert_called_with("default", "test_project")

    def test_get_router_dir(self, mock_project_service, temp_project_dir):
        """Test getting a specific router directory path."""
        router_dir = RouterStorageService.get_router_dir(
            "default", "test_project", "my_router"
        )

        assert router_dir == Path(temp_project_dir) / "lf_data" / "routers" / "my_router"

    def test_get_router_dir_prevents_path_traversal(self, mock_project_service):
        """Test that path traversal attempts are rejected."""
        with pytest.raises(ValueError, match="Invalid router name"):
            RouterStorageService.get_router_dir("default", "test", "../etc/passwd")

        with pytest.raises(ValueError, match="Invalid router name"):
            RouterStorageService.get_router_dir("default", "test", "foo/bar")

        with pytest.raises(ValueError, match="Invalid router name"):
            RouterStorageService.get_router_dir("default", "test", "foo\\bar")

    def test_save_router_config(self, mock_project_service, temp_project_dir):
        """Test saving router configuration."""
        config = {
            "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "default_model": "general_llm",
            "similarity_threshold": 0.7,
            "routes": [
                {
                    "name": "billing",
                    "target_model": "billing_model",
                    "utterances": ["what is my bill"],
                }
            ],
        }

        config_path = RouterStorageService.save_router_config(
            "default", "test_project", "my_router", config
        )

        # Verify file was created
        assert config_path.exists()
        assert config_path.name == "config.json"

        # Verify content
        with open(config_path) as f:
            saved_config = json.load(f)

        assert saved_config["embedder_model"] == config["embedder_model"]
        assert saved_config["default_model"] == config["default_model"]
        assert saved_config["similarity_threshold"] == config["similarity_threshold"]
        assert saved_config["routes"] == config["routes"]

        # Verify metadata was added
        assert "_metadata" in saved_config
        assert saved_config["_metadata"]["namespace"] == "default"
        assert saved_config["_metadata"]["project_id"] == "test_project"
        assert saved_config["_metadata"]["router_name"] == "my_router"
        assert "saved_at" in saved_config["_metadata"]

    def test_load_router_config(self, mock_project_service, temp_project_dir):
        """Test loading router configuration."""
        # First save a config
        config = {
            "embedder_model": "BAAI/bge-small-en-v1.5",
            "default_model": "default_llm",
            "similarity_threshold": 0.8,
            "routes": [],
        }
        RouterStorageService.save_router_config(
            "default", "test_project", "my_router", config
        )

        # Now load it
        loaded_config = RouterStorageService.load_router_config(
            "default", "test_project", "my_router"
        )

        assert loaded_config is not None
        assert loaded_config["embedder_model"] == config["embedder_model"]
        assert loaded_config["default_model"] == config["default_model"]
        assert loaded_config["similarity_threshold"] == config["similarity_threshold"]

    def test_load_router_config_not_found(self, mock_project_service):
        """Test loading non-existent router returns None."""
        result = RouterStorageService.load_router_config(
            "default", "test_project", "nonexistent"
        )
        assert result is None

    def test_save_embeddings(self, mock_project_service, temp_project_dir):
        """Test saving router embeddings."""
        # Create some fake embeddings data
        embeddings_data = b"fake_embeddings_binary_data_here"

        embeddings_path = RouterStorageService.save_embeddings(
            "default", "test_project", "my_router", embeddings_data
        )

        # Verify file was created
        assert embeddings_path.exists()
        assert embeddings_path.name == "embeddings.pt"

        # Verify content
        with open(embeddings_path, "rb") as f:
            saved_data = f.read()
        assert saved_data == embeddings_data

    def test_load_embeddings(self, mock_project_service, temp_project_dir):
        """Test loading router embeddings."""
        # First save embeddings
        embeddings_data = b"test_embeddings_data_123"
        RouterStorageService.save_embeddings(
            "default", "test_project", "my_router", embeddings_data
        )

        # Now load them
        loaded_data = RouterStorageService.load_embeddings(
            "default", "test_project", "my_router"
        )

        assert loaded_data == embeddings_data

    def test_load_embeddings_not_found(self, mock_project_service):
        """Test loading non-existent embeddings returns None."""
        result = RouterStorageService.load_embeddings(
            "default", "test_project", "nonexistent"
        )
        assert result is None

    def test_list_routers(self, mock_project_service, temp_project_dir):
        """Test listing all routers in a project."""
        # Create multiple routers
        for i, name in enumerate(["router_a", "router_b", "router_c"]):
            config = {
                "embedder_model": "test-model",
                "default_model": f"default_{i}",
                "similarity_threshold": 0.7,
                "routes": [],
            }
            RouterStorageService.save_router_config(
                "default", "test_project", name, config
            )
            # Add embeddings to some
            if i % 2 == 0:
                RouterStorageService.save_embeddings(
                    "default", "test_project", name, b"embeddings"
                )

        # List routers
        routers = RouterStorageService.list_routers("default", "test_project")

        assert len(routers) == 3
        names = {r["name"] for r in routers}
        assert names == {"router_a", "router_b", "router_c"}

        # Check has_embeddings flag
        for router in routers:
            if router["name"] in ["router_a", "router_c"]:
                assert router["has_embeddings"] is True
            else:
                assert router["has_embeddings"] is False

    def test_list_routers_empty_project(self, mock_project_service):
        """Test listing routers in empty project returns empty list."""
        routers = RouterStorageService.list_routers("default", "empty_project")
        assert routers == []

    def test_delete_router(self, mock_project_service, temp_project_dir):
        """Test deleting a router."""
        # Create a router
        config = {"embedder_model": "test", "default_model": "default", "routes": []}
        RouterStorageService.save_router_config(
            "default", "test_project", "to_delete", config
        )
        RouterStorageService.save_embeddings(
            "default", "test_project", "to_delete", b"embeddings"
        )

        # Verify it exists
        assert RouterStorageService.router_exists("default", "test_project", "to_delete")

        # Delete it
        result = RouterStorageService.delete_router(
            "default", "test_project", "to_delete"
        )

        assert result is True
        assert not RouterStorageService.router_exists(
            "default", "test_project", "to_delete"
        )

    def test_delete_router_not_found(self, mock_project_service):
        """Test deleting non-existent router returns False."""
        result = RouterStorageService.delete_router(
            "default", "test_project", "nonexistent"
        )
        assert result is False

    def test_router_exists(self, mock_project_service, temp_project_dir):
        """Test checking if router exists."""
        # Non-existent router
        assert not RouterStorageService.router_exists(
            "default", "test_project", "my_router"
        )

        # Create router
        config = {"embedder_model": "test", "default_model": "default", "routes": []}
        RouterStorageService.save_router_config(
            "default", "test_project", "my_router", config
        )

        # Now it should exist
        assert RouterStorageService.router_exists(
            "default", "test_project", "my_router"
        )

    def test_router_exists_requires_config(self, mock_project_service, temp_project_dir):
        """Test that router_exists requires config.json, not just directory."""
        # Create directory without config
        router_dir = RouterStorageService.ensure_router_dir(
            "default", "test_project", "empty_router"
        )
        assert router_dir.exists()

        # Should not be considered existing without config
        assert not RouterStorageService.router_exists(
            "default", "test_project", "empty_router"
        )
