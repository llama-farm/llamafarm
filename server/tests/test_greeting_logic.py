"""
Tests for greeting logic in ProjectChatOrchestratorAgent.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from agents.project_chat_orchestrator import ProjectChatOrchestratorAgent
from config.datamodel import LlamaFarmConfig, Runtime, Provider
from openai import AsyncOpenAI


@pytest.fixture
def mock_project_config():
    """Create a mock project_seed config."""
    return LlamaFarmConfig(
        version="v1",
        name="project_seed",
        namespace="llamafarm",
        runtime=Runtime(
            provider=Provider.ollama,
            model="gemma3:1b",
            base_url="http://localhost:11434/v1",
        ),
        prompts=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            }
        ],
    )


@pytest.fixture
def temp_project_dir(tmp_path):
    """Create a temporary project directory."""
    project_dir = tmp_path / "projects" / "llamafarm" / "project_seed"
    project_dir.mkdir(parents=True)
    return str(project_dir)


class TestGreetingLogic:
    """Test suite for greeting injection logic."""

    @patch("agents.project_chat_orchestrator._get_client")
    def test_new_session_greeting(self, mock_get_client, mock_project_config, temp_project_dir):
        """Test that new sessions get a welcome greeting."""
        # Create a proper AsyncOpenAI mock
        mock_client = MagicMock(spec=AsyncOpenAI)
        mock_get_client.return_value = mock_client

        agent = ProjectChatOrchestratorAgent(
            project_config=mock_project_config,
            project_dir=temp_project_dir,
        )

        # Enable persistence with a new session ID
        session_id = "test-session-123"
        agent.enable_persistence(session_id=session_id)

        # Check that greeting was injected
        history = agent.history.get_history()
        assistant_messages = [
            msg for msg in history
            if getattr(msg, "role", None) == "assistant"
            or (isinstance(msg, dict) and msg.get("role") == "assistant")
        ]

        assert len(assistant_messages) >= 1, "Should have at least one assistant message (greeting)"

        # Extract content from first assistant message
        first_msg = assistant_messages[0]
        if hasattr(first_msg, "content"):
            content_obj = first_msg.content
        elif isinstance(first_msg, dict):
            content_obj = first_msg.get("content")
        else:
            content_obj = None

        # Extract chat_message from content
        if hasattr(content_obj, "chat_message"):
            content = content_obj.chat_message
        elif isinstance(content_obj, dict):
            content = content_obj.get("chat_message", "")
        else:
            content = str(content_obj) if content_obj else ""

        assert "Welcome to LlamaFarm dev mode" in content
        assert "lf init" in content or "Getting started" in content

    @patch("agents.project_chat_orchestrator._get_client")
    def test_returning_session_greeting(
        self, mock_get_client, mock_project_config, temp_project_dir
    ):
        """Test that returning sessions get a welcome back greeting."""
        # Create a proper AsyncOpenAI mock
        mock_client = MagicMock(spec=AsyncOpenAI)
        mock_get_client.return_value = mock_client

        # Create an agent and save some history
        agent = ProjectChatOrchestratorAgent(
            project_config=mock_project_config,
            project_dir=temp_project_dir,
        )

        session_id = "test-session-456"
        agent.enable_persistence(session_id=session_id)

        # Add some fake history to simulate previous session
        from agents.project_chat_orchestrator import (
            ProjectChatOrchestratorAgentInputSchema,
            ProjectChatOrchestratorAgentOutputSchema,
        )

        agent.history.add_message(
            "user",
            ProjectChatOrchestratorAgentInputSchema(chat_message="Hello"),
        )
        agent.history.add_message(
            "assistant",
            ProjectChatOrchestratorAgentOutputSchema(chat_message="Hi there!"),
        )

        # Persist this history
        agent._persist_history()

        # Create a new agent with the same session ID
        agent2 = ProjectChatOrchestratorAgent(
            project_config=mock_project_config,
            project_dir=temp_project_dir,
        )
        agent2.enable_persistence(session_id=session_id)

        # Check that "welcome back" greeting was injected
        history = agent2.history.get_history()
        assistant_messages = [
            msg for msg in history
            if getattr(msg, "role", None) == "assistant"
            or (isinstance(msg, dict) and msg.get("role") == "assistant")
        ]

        # Should have: original "Hi there!" + welcome back greeting
        assert len(assistant_messages) >= 2

        # Check the last assistant message for welcome back
        last_msg = assistant_messages[-1]
        if hasattr(last_msg, "content"):
            content_obj = last_msg.content
        elif isinstance(last_msg, dict):
            content_obj = last_msg.get("content")
        else:
            content_obj = None

        if hasattr(content_obj, "chat_message"):
            content = content_obj.chat_message
        elif isinstance(content_obj, dict):
            content = content_obj.get("chat_message", "")
        else:
            content = str(content_obj) if content_obj else ""

        assert "Welcome back" in content
        assert "message" in content.lower()  # Should mention message count

    @patch("agents.project_chat_orchestrator.settings")
    @patch("agents.project_chat_orchestrator._get_client")
    def test_greeting_disabled_via_env(
        self, mock_get_client, mock_settings, mock_project_config, temp_project_dir
    ):
        """Test that greetings can be disabled via settings."""
        # Create a proper AsyncOpenAI mock
        mock_client = MagicMock(spec=AsyncOpenAI)
        mock_get_client.return_value = mock_client

        # Mock settings to disable greetings
        mock_settings.lf_dev_mode_greeting_enabled = False

        agent = ProjectChatOrchestratorAgent(
            project_config=mock_project_config,
            project_dir=temp_project_dir,
        )

        session_id = "test-session-789"
        agent.enable_persistence(session_id=session_id)

        # Check that no greeting was injected
        history = agent.history.get_history()
        assistant_messages = [
            msg for msg in history
            if getattr(msg, "role", None) == "assistant"
            or (isinstance(msg, dict) and msg.get("role") == "assistant")
        ]

        assert len(assistant_messages) == 0, "Should have no assistant messages when greetings disabled"

    @patch("agents.project_chat_orchestrator._get_client")
    def test_greeting_only_for_project_seed(
        self, mock_get_client, mock_project_config, temp_project_dir
    ):
        """Test that greetings are only injected for project_seed."""
        # Create a proper AsyncOpenAI mock
        mock_client = MagicMock(spec=AsyncOpenAI)
        mock_get_client.return_value = mock_client

        # Change config to a different project name
        config = mock_project_config.model_copy()
        config.name = "some-other-project"

        agent = ProjectChatOrchestratorAgent(
            project_config=config,
            project_dir=temp_project_dir,
        )

        session_id = "test-session-other"
        agent.enable_persistence(session_id=session_id)

        # Check that no greeting was injected for non-project_seed
        history = agent.history.get_history()
        assistant_messages = [
            msg for msg in history
            if getattr(msg, "role", None) == "assistant"
            or (isinstance(msg, dict) and msg.get("role") == "assistant")
        ]

        assert len(assistant_messages) == 0, "Should have no greetings for non-project_seed projects"

    @patch("agents.project_chat_orchestrator._get_client")
    def test_greeting_not_for_similar_project_names(
        self, mock_get_client, mock_project_config, temp_project_dir
    ):
        """Test that greetings are NOT injected for project names similar to project_seed."""
        # Create a proper AsyncOpenAI mock
        mock_client = MagicMock(spec=AsyncOpenAI)
        mock_get_client.return_value = mock_client

        # Test with project_seed_2 (should NOT get greeting)
        config = mock_project_config.model_copy()
        config.name = "project_seed_2"

        agent = ProjectChatOrchestratorAgent(
            project_config=config,
            project_dir=temp_project_dir,
        )

        session_id = "test-session-similar"
        agent.enable_persistence(session_id=session_id)

        # Check that no greeting was injected for similar project name
        history = agent.history.get_history()
        assistant_messages = [
            msg for msg in history
            if getattr(msg, "role", None) == "assistant"
            or (isinstance(msg, dict) and msg.get("role") == "assistant")
        ]

        assert len(assistant_messages) == 0, "Should have no greetings for project_seed_2 (only exact match 'project_seed' gets greeting)"

    @patch("agents.project_chat_orchestrator._get_client")
    def test_greeting_not_duplicated(
        self, mock_get_client, mock_project_config, temp_project_dir
    ):
        """Test that greeting is not duplicated if already present."""
        # Create a proper AsyncOpenAI mock
        mock_client = MagicMock(spec=AsyncOpenAI)
        mock_get_client.return_value = mock_client

        agent = ProjectChatOrchestratorAgent(
            project_config=mock_project_config,
            project_dir=temp_project_dir,
        )

        session_id = "test-session-dup"
        agent.enable_persistence(session_id=session_id)

        # Get initial assistant message count
        history = agent.history.get_history()
        initial_assistant_count = sum(
            1
            for msg in history
            if getattr(msg, "role", None) == "assistant"
            or (isinstance(msg, dict) and msg.get("role") == "assistant")
        )

        # Call enable_persistence again (simulating reconnection)
        agent.enable_persistence(session_id=session_id)

        # Check that greeting count didn't increase
        history = agent.history.get_history()
        final_assistant_count = sum(
            1
            for msg in history
            if getattr(msg, "role", None) == "assistant"
            or (isinstance(msg, dict) and msg.get("role") == "assistant")
        )

        assert (
            final_assistant_count == initial_assistant_count
        ), "Greeting should not be duplicated"
