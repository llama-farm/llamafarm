"""Tests for PromptsService."""

from unittest.mock import patch

import pytest
from config.datamodel import LlamaFarmConfig, Model, PromptMessage, PromptSet, Runtime

from services.prompts_service import PromptNotFoundError, PromptsService


@pytest.fixture
def mock_config():
    """Create a mock config with some prompts."""
    return LlamaFarmConfig(
        version="v1",
        name="test-project",
        namespace="default",
        runtime=Runtime(
            models=[
                Model(
                    name="default",
                    provider="ollama",
                    model="llama3",
                )
            ]
        ),
        prompts=[
            PromptSet(
                name="system_default",
                messages=[
                    PromptMessage(role="system", content="You are a helpful assistant.")
                ],
            ),
            PromptSet(
                name="code_assistant",
                messages=[
                    PromptMessage(
                        role="system", content="You are an expert programmer."
                    ),
                    PromptMessage(
                        role="user", content="Help me write clean, maintainable code."
                    ),
                ],
            ),
        ],
    )


class TestPromptsService:
    """Tests for PromptsService."""

    def test_list_prompts(self, mock_config):
        """Test listing all prompts."""
        with patch.object(PromptsService, "list_prompts") as mock_list:
            mock_list.return_value = mock_config.prompts
            prompts = PromptsService.list_prompts("default", "test-project")

            assert len(prompts) == 2
            assert prompts[0].name == "system_default"
            assert prompts[1].name == "code_assistant"

    def test_get_prompt_found(self, mock_config):
        """Test getting a prompt that exists."""
        with patch("services.prompts_service.ProjectService.load_config") as mock_load:
            mock_load.return_value = mock_config

            prompt = PromptsService.get_prompt(
                "default", "test-project", "system_default"
            )

            assert prompt.name == "system_default"
            assert len(prompt.messages) == 1
            assert prompt.messages[0].role == "system"

    def test_get_prompt_not_found(self, mock_config):
        """Test getting a prompt that doesn't exist."""
        with patch("services.prompts_service.ProjectService.load_config") as mock_load:
            mock_load.return_value = mock_config

            with pytest.raises(PromptNotFoundError) as exc_info:
                PromptsService.get_prompt("default", "test-project", "nonexistent")

            assert "nonexistent" in str(exc_info.value)

    def test_create_prompt_success(self, mock_config):
        """Test creating a new prompt."""
        with (
            patch("services.prompts_service.ProjectService.load_config") as mock_load,
            patch("services.prompts_service.ProjectService.save_config") as mock_save,
        ):
            mock_load.return_value = mock_config

            new_messages = [
                PromptMessage(role="system", content="You are a data analyst.")
            ]

            prompt = PromptsService.create_prompt(
                "default", "test-project", "data_analyst", new_messages
            )

            assert prompt.name == "data_analyst"
            assert len(prompt.messages) == 1
            mock_save.assert_called_once()

    def test_create_prompt_duplicate_name(self, mock_config):
        """Test creating a prompt with existing name fails."""
        with patch("services.prompts_service.ProjectService.load_config") as mock_load:
            mock_load.return_value = mock_config

            new_messages = [PromptMessage(role="system", content="Duplicate.")]

            with pytest.raises(ValueError) as exc_info:
                PromptsService.create_prompt(
                    "default", "test-project", "system_default", new_messages
                )

            assert "already exists" in str(exc_info.value)

    def test_create_prompt_invalid_name(self, mock_config):
        """Test creating a prompt with invalid name fails."""
        with patch("services.prompts_service.ProjectService.load_config") as mock_load:
            mock_load.return_value = mock_config

            new_messages = [PromptMessage(role="system", content="Test.")]

            with pytest.raises(ValueError) as exc_info:
                PromptsService.create_prompt(
                    "default", "test-project", "Invalid-Name!", new_messages
                )

            assert "Invalid prompt name" in str(exc_info.value)

    def test_update_prompt_success(self, mock_config):
        """Test updating an existing prompt."""
        with (
            patch("services.prompts_service.ProjectService.load_config") as mock_load,
            patch("services.prompts_service.ProjectService.save_config") as mock_save,
        ):
            mock_load.return_value = mock_config

            new_messages = [PromptMessage(role="system", content="Updated content.")]

            prompt = PromptsService.update_prompt(
                "default", "test-project", "system_default", new_messages
            )

            assert prompt.name == "system_default"
            assert prompt.messages[0].content == "Updated content."
            mock_save.assert_called_once()

    def test_update_prompt_not_found(self, mock_config):
        """Test updating a nonexistent prompt fails."""
        with patch("services.prompts_service.ProjectService.load_config") as mock_load:
            mock_load.return_value = mock_config

            new_messages = [PromptMessage(role="system", content="Updated.")]

            with pytest.raises(PromptNotFoundError):
                PromptsService.update_prompt(
                    "default", "test-project", "nonexistent", new_messages
                )

    def test_delete_prompt_success(self, mock_config):
        """Test deleting a prompt without dependencies."""
        with (
            patch("services.prompts_service.ProjectService.load_config") as mock_load,
            patch("services.prompts_service.ProjectService.save_config") as mock_save,
        ):
            mock_load.return_value = mock_config

            deleted = PromptsService.delete_prompt(
                "default", "test-project", "code_assistant"
            )

            assert deleted.name == "code_assistant"
            mock_save.assert_called_once()

    def test_delete_prompt_not_found(self, mock_config):
        """Test deleting a nonexistent prompt fails."""
        with patch("services.prompts_service.ProjectService.load_config") as mock_load:
            mock_load.return_value = mock_config

            with pytest.raises(PromptNotFoundError):
                PromptsService.delete_prompt("default", "test-project", "nonexistent")

    def test_validate_name_valid(self):
        """Test valid prompt names pass validation."""
        # These should not raise
        PromptsService._validate_name("test")
        PromptsService._validate_name("my_prompt")
        PromptsService._validate_name("prompt123")
        PromptsService._validate_name("a")

    def test_validate_name_invalid(self):
        """Test invalid prompt names fail validation."""
        invalid_names = [
            "Test",  # uppercase
            "123test",  # starts with number
            "_test",  # starts with underscore
            "test-name",  # contains hyphen
            "test.name",  # contains dot
            "test name",  # contains space
        ]

        for name in invalid_names:
            with pytest.raises(ValueError):
                PromptsService._validate_name(name)
