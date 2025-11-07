"""
Tests for the projects API endpoints.

This module contains tests for project CRUD operations via the API.
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from config.datamodel import (
    Dataset,
    LlamaFarmConfig,
    Model,
    PromptMessage,
    PromptSet,
    Provider,
    Runtime,
    Version,
)
from fastapi.testclient import TestClient

from api.main import llama_farm_api
from services.project_service import ProjectService


@pytest.fixture
def client():
    """Create a test client for the API."""
    app = llama_farm_api()
    return TestClient(app)


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config():
    """Create a mock LlamaFarm configuration for testing."""
    return LlamaFarmConfig(
        version=Version.v1,
        name="test_project",
        namespace="test_namespace",
        prompts=[
            PromptSet(
                name="default",
                messages=[
                    PromptMessage(role="system", content="You are a helpful assistant.")
                ],
            )
        ],
        rag={
            "databases": [
                {
                    "name": "test_db",
                    "type": "ChromaStore",
                    "config": {},
                    "embedding_strategies": [
                        {
                            "name": "test_embedding",
                            "type": "OllamaEmbedder",
                            "config": {"model": "nomic-embed-text"},
                        }
                    ],
                    "retrieval_strategies": [
                        {
                            "name": "test_retrieval",
                            "type": "BasicSimilarityStrategy",
                            "config": {},
                            "default": True,
                        }
                    ],
                }
            ],
            "data_processing_strategies": [
                {
                    "name": "test_strategy",
                    "parsers": [
                        {
                            "type": "CSVParser_LlamaIndex",
                            "config": {},
                            "file_extensions": [".csv"],
                        }
                    ],
                }
            ],
        },
        datasets=[
            Dataset(
                name="test_dataset",
                data_processing_strategy="test_strategy",
                database="test_db",
                files=["test_file.csv"],
            )
        ],
        runtime=Runtime(
            models=[
                Model(
                    name="default",
                    provider=Provider.openai,
                    model="llama3.1:8b",
                    api_key="test_key",
                    base_url="http://localhost:11434/v1",
                )
            ]
        ),
    )


class TestDeleteProjectAPI:
    """Test cases for the DELETE /v1/projects/{namespace}/{project_id} endpoint."""

    def test_delete_project_not_found(self, client, temp_data_dir):
        """Test that deleting a non-existent project returns 404."""
        with patch("core.settings.settings.lf_data_dir", temp_data_dir):
            response = client.delete("/v1/projects/test_ns/nonexistent")
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    def test_delete_project_success(self, client, temp_data_dir, mock_config):
        """Test successful project deletion via API."""
        with patch("core.settings.settings.lf_data_dir", temp_data_dir):
            # Create a project
            namespace = "test_ns"
            project_id = "test_proj"
            project_dir = ProjectService.get_project_dir(namespace, project_id)
            os.makedirs(project_dir, exist_ok=True)
            ProjectService.save_config(namespace, project_id, mock_config)

            # Verify project exists
            assert os.path.exists(project_dir)

            # Delete the project via API
            response = client.delete(f"/v1/projects/{namespace}/{project_id}")

            # Verify response
            assert response.status_code == 200
            response_data = response.json()
            assert "project" in response_data
            assert response_data["project"]["namespace"] == namespace
            # Note: project.name gets normalized to project_id when saved
            assert response_data["project"]["name"] == project_id

            # Verify project directory was deleted
            assert not os.path.exists(project_dir)

    def test_delete_project_with_sessions_via_api(
        self, client, temp_data_dir, mock_config
    ):
        """Test that deleting a project via API removes both disk and in-memory sessions."""
        from unittest.mock import MagicMock

        from api.routers.projects import projects
        
        with patch("core.settings.settings.lf_data_dir", temp_data_dir):
            # Create project with session
            namespace = "test_ns"
            project_id = "test_proj_sessions"
            project_dir = ProjectService.get_project_dir(namespace, project_id)
            os.makedirs(project_dir, exist_ok=True)
            ProjectService.save_config(namespace, project_id, mock_config)

            # Create session directory on disk
            sessions_dir = Path(project_dir) / "sessions" / "session_123"
            sessions_dir.mkdir(parents=True, exist_ok=True)
            (sessions_dir / "history.json").write_text('{"messages": []}')

            # Create in-memory session record
            session_key = "test_ns:test_proj_sessions:session_123"
            mock_agent = MagicMock()
            projects.agent_sessions[session_key] = projects.SessionRecord(
                namespace=namespace,
                project_id=project_id,
                agent=mock_agent,
                created_at=0.0,
                last_used=0.0,
                request_count=1,
            )

            # Verify session exists on disk and in memory
            assert sessions_dir.exists()
            assert session_key in projects.agent_sessions

            # Delete project via API
            response = client.delete(f"/v1/projects/{namespace}/{project_id}")

            # Verify success
            assert response.status_code == 200

            # Verify project and sessions are gone (both disk and memory)
            assert not os.path.exists(project_dir)
            assert not sessions_dir.exists()
            assert session_key not in projects.agent_sessions

    def test_delete_project_with_data_via_api(
        self, client, temp_data_dir, mock_config
    ):
        """Test that deleting a project via API removes entire directory with data."""
        with patch("core.settings.settings.lf_data_dir", temp_data_dir):
            # Create project with data files
            namespace = "test_ns"
            project_id = "test_proj_data"
            project_dir = ProjectService.get_project_dir(namespace, project_id)
            os.makedirs(project_dir, exist_ok=True)
            ProjectService.save_config(namespace, project_id, mock_config)

            # Create some data files to simulate datasets
            data_dir = Path(project_dir) / "data"
            data_dir.mkdir(parents=True)
            (data_dir / "test_file.pdf").write_bytes(b"test content")

            # Verify project and data exist
            assert os.path.exists(project_dir)
            assert (data_dir / "test_file.pdf").exists()

            # Delete project via API
            response = client.delete(f"/v1/projects/{namespace}/{project_id}")

            # Verify success
            assert response.status_code == 200

            # Verify entire project directory is gone
            assert not os.path.exists(project_dir)
            assert not data_dir.exists()

    def test_delete_project_permission_error_returns_403(
        self, client, temp_data_dir, mock_config
    ):
        """Test that permission errors return 403."""
        with patch("core.settings.settings.lf_data_dir", temp_data_dir):
            # Create project
            namespace = "test_ns"
            project_id = "test_proj_perms"
            project_dir = ProjectService.get_project_dir(namespace, project_id)
            os.makedirs(project_dir, exist_ok=True)
            ProjectService.save_config(namespace, project_id, mock_config)

            # Mock ProjectService.delete_project to raise PermissionError
            def mock_delete_permission(*args, **kwargs):
                raise PermissionError("Permission denied: test error")

            with patch.object(
                ProjectService, "delete_project", side_effect=mock_delete_permission
            ):
                response = client.delete(f"/v1/projects/{namespace}/{project_id}")

                # Verify 403 response
                assert response.status_code == 403
                assert "permission denied" in response.json()["detail"].lower()

    def test_delete_project_unexpected_error_returns_500(
        self, client, temp_data_dir, mock_config
    ):
        """Test that unexpected errors return 500."""
        with patch("core.settings.settings.lf_data_dir", temp_data_dir):
            # Create project
            namespace = "test_ns"
            project_id = "test_proj_error"
            project_dir = ProjectService.get_project_dir(namespace, project_id)
            os.makedirs(project_dir, exist_ok=True)
            ProjectService.save_config(namespace, project_id, mock_config)

            # Mock ProjectService.delete_project to raise unexpected error
            def mock_delete_error(*args, **kwargs):
                raise RuntimeError("Unexpected error during deletion")

            with patch.object(
                ProjectService, "delete_project", side_effect=mock_delete_error
            ):
                response = client.delete(f"/v1/projects/{namespace}/{project_id}")

                # Verify 500 response
                assert response.status_code == 500
                assert "failed to delete project" in response.json()["detail"].lower()

