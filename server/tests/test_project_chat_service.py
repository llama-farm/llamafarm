"""
Unit tests for ProjectChatService.

Tests the project chat service including:
- RAG parameter resolution
- RAG search integration
- Chat orchestration
- Streaming chat
"""

import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from config.datamodel import (
    Database,
    DatabaseRetrievalStrategy,
    DatabaseRetrievalType,
    DatabaseType,
    LlamaFarmConfig,
    Model,
    PromptMessage,
    PromptSet,
    Provider,
    RAGStrategyConfigurationSchema,
    Runtime,
    Version,
)

from agents.base.history import LFChatCompletionUserMessageParam
from agents.chat_orchestrator import ChatOrchestratorAgent
from services.project_chat_service import ProjectChatService, RAGParameters
from services.project_service import ProjectService


@pytest.fixture
def base_config():
    """Create base config without RAG."""
    return LlamaFarmConfig(
        version=Version.v1,
        name="test-project",
        namespace="test",
        runtime=Runtime(
            models=[
                Model(
                    name="default",
                    provider=Provider.ollama,
                    model="llama3.2:latest",
                    base_url="http://localhost:11434/v1",
                    api_key="ollama",
                )
            ]
        ),
        prompts=[
            PromptSet(
                name="default",
                messages=[PromptMessage(role="system", content="You are helpful")],
            )
        ],
    )


@pytest.fixture
def config_with_rag():
    """Create config with RAG configuration."""
    return LlamaFarmConfig(
        version=Version.v1,
        name="test-project",
        namespace="test",
        runtime=Runtime(
            models=[
                Model(
                    name="default",
                    provider=Provider.openai,
                    model="gpt-4",
                    base_url="https://api.openai.com/v1",
                    api_key="test-key",
                )
            ]
        ),
        prompts=[
            PromptSet(
                name="default",
                messages=[PromptMessage(role="system", content="You are helpful")],
            )
        ],
        rag=RAGStrategyConfigurationSchema(
            default_database="main_db",
            databases=[
                Database(
                    name="main_db",
                    type=DatabaseType.ChromaStore,
                    retrieval_strategies=[
                        DatabaseRetrievalStrategy(
                            name="default_strategy",
                            type=DatabaseRetrievalType.VectorRetriever,
                            default=True,
                            config={"top_k": 5, "score_threshold": 0.7},
                        )
                    ],
                )
            ],
        ),
    )


class TestRAGParameters:
    """Test suite for RAGParameters."""

    def test_init_disabled(self):
        """Test RAGParameters with RAG disabled."""
        params = RAGParameters(rag_enabled=False)
        assert not params.rag_enabled
        assert params.database is None
        assert params.retrieval_strategy is None
        assert params.rag_top_k is None
        assert params.rag_score_threshold is None

    def test_init_enabled(self):
        """Test RAGParameters with RAG enabled."""
        params = RAGParameters(
            rag_enabled=True,
            database="main_db",
            retrieval_strategy="default",
            rag_top_k=10,
            rag_score_threshold=0.8,
        )
        assert params.rag_enabled
        assert params.database == "main_db"
        assert params.retrieval_strategy == "default"
        assert params.rag_top_k == 10
        assert params.rag_score_threshold == 0.8


class TestProjectChatService:
    """Test suite for ProjectChatService."""

    def test_resolve_rag_parameters_disabled_by_flag(self, config_with_rag):
        """Test RAG disabled explicitly via flag."""
        service = ProjectChatService()
        params = service._resolve_rag_parameters(
            project_config=config_with_rag,
            rag_enabled=False,
        )
        assert not params.rag_enabled

    def test_resolve_rag_parameters_no_rag_config(self, base_config):
        """Test RAG resolution when config has no RAG."""
        service = ProjectChatService()
        params = service._resolve_rag_parameters(
            project_config=base_config,
            rag_enabled=None,
        )
        assert not params.rag_enabled

    def test_resolve_rag_parameters_enabled_with_defaults(self, config_with_rag):
        """Test RAG resolution using config defaults."""
        service = ProjectChatService()
        params = service._resolve_rag_parameters(
            project_config=config_with_rag,
            rag_enabled=True,
        )
        assert params.rag_enabled
        assert params.database == "main_db"
        assert params.retrieval_strategy == "default_strategy"
        assert params.rag_top_k == 5
        assert params.rag_score_threshold == 0.7

    def test_resolve_rag_parameters_with_overrides(self, config_with_rag):
        """Test RAG resolution with explicit overrides."""
        service = ProjectChatService()
        params = service._resolve_rag_parameters(
            project_config=config_with_rag,
            rag_enabled=True,
            database="main_db",
            retrieval_strategy="default_strategy",
            rag_top_k=10,
            rag_score_threshold=0.9,
        )
        assert params.rag_enabled
        assert params.database == "main_db"
        assert params.retrieval_strategy == "default_strategy"
        assert params.rag_top_k == 10
        assert params.rag_score_threshold == 0.9

    def test_resolve_rag_parameters_no_default_database(self):
        """Test RAG when no default database is marked."""
        config = LlamaFarmConfig(
            version=Version.v1,
            name="test-project",
            namespace="test",
            runtime=Runtime(
                models=[
                    Model(
                        name="default",
                        provider=Provider.openai,
                        model="gpt-4",
                        base_url="https://api.openai.com/v1",
                        api_key="test-key",
                    )
                ]
            ),
            prompts=[],
            rag=RAGStrategyConfigurationSchema(
                databases=[
                    Database(
                        name="db1",
                        type=DatabaseType.ChromaStore,
                        retrieval_strategies=[
                            DatabaseRetrievalStrategy(
                                name="strat1",
                                type=DatabaseRetrievalType.VectorRetriever,
                                config={},
                            )
                        ],
                    ),
                    Database(
                        name="db2",
                        type=DatabaseType.ChromaStore,
                        retrieval_strategies=[
                            DatabaseRetrievalStrategy(
                                name="strat2",
                                type=DatabaseRetrievalType.VectorRetriever,
                                config={},
                            )
                        ],
                    ),
                ]
            ),
        )

        service = ProjectChatService()
        params = service._resolve_rag_parameters(
            project_config=config,
            rag_enabled=True,
        )
        # Should use first database
        assert params.database == "db1"

    def test_resolve_rag_parameters_no_strategies(self):
        """Test RAG when database has no retrieval strategies."""
        config = LlamaFarmConfig(
            version=Version.v1,
            name="test-project",
            namespace="test",
            runtime=Runtime(
                models=[
                    Model(
                        name="default",
                        provider=Provider.openai,
                        model="gpt-4",
                        base_url="https://api.openai.com/v1",
                        api_key="test-key",
                    )
                ]
            ),
            prompts=[],
            rag=RAGStrategyConfigurationSchema(
                databases=[
                    Database(
                        name="db1", type=DatabaseType.ChromaStore, retrieval_strategies=[]
                    )
                ]
            ),
        )

        service = ProjectChatService()
        params = service._resolve_rag_parameters(
            project_config=config,
            rag_enabled=True,
            database="db1",
        )
        # Should disable RAG when no strategies available
        assert not params.rag_enabled

    def test_resolve_rag_model_config_enables_rag(self, config_with_rag):
        """Model rag_enabled=True enables RAG when request omits it."""
        model_cfg = Model(
            name="rag-model",
            provider=Provider.openai,
            model="gpt-4",
            rag_enabled=True,
            target_database="main_db",
        )
        service = ProjectChatService()
        params = service._resolve_rag_parameters(
            project_config=config_with_rag,
            model_config=model_cfg,
            rag_enabled=None,  # not specified by request
        )
        assert params.rag_enabled
        assert params.database == "main_db"

    def test_resolve_rag_model_config_disables_rag(self, config_with_rag):
        """Model rag_enabled=False disables RAG."""
        model_cfg = Model(
            name="no-rag-model",
            provider=Provider.openai,
            model="gpt-4",
            rag_enabled=False,
        )
        service = ProjectChatService()
        params = service._resolve_rag_parameters(
            project_config=config_with_rag,
            model_config=model_cfg,
            rag_enabled=None,
        )
        assert not params.rag_enabled

    def test_resolve_rag_request_overrides_model(self, config_with_rag):
        """Request rag_enabled overrides model-level."""
        model_cfg = Model(
            name="rag-model",
            provider=Provider.openai,
            model="gpt-4",
            rag_enabled=False,  # model says no
        )
        service = ProjectChatService()
        params = service._resolve_rag_parameters(
            project_config=config_with_rag,
            model_config=model_cfg,
            rag_enabled=True,  # request says yes
        )
        assert params.rag_enabled

    def test_resolve_rag_model_target_database(self, config_with_rag):
        """Model target_database overrides project default."""
        model_cfg = Model(
            name="rag-model",
            provider=Provider.openai,
            model="gpt-4",
            rag_enabled=True,
            target_database="main_db",
        )
        service = ProjectChatService()
        params = service._resolve_rag_parameters(
            project_config=config_with_rag,
            model_config=model_cfg,
            rag_enabled=None,
            database=None,  # not specified by request
        )
        assert params.database == "main_db"

    def test_resolve_rag_request_database_overrides_model(self, config_with_rag):
        """Request database overrides model target_database."""
        model_cfg = Model(
            name="rag-model",
            provider=Provider.openai,
            model="gpt-4",
            target_database="other_db",  # model says other_db
        )
        service = ProjectChatService()
        params = service._resolve_rag_parameters(
            project_config=config_with_rag,
            model_config=model_cfg,
            rag_enabled=True,
            database="main_db",  # request says main_db
        )
        assert params.database == "main_db"

    def test_find_model_config(self, config_with_rag):
        """Test _find_model_config looks up model by name."""
        service = ProjectChatService()
        result = service._find_model_config(config_with_rag, "default")
        assert result is not None
        assert result.name == "default"

    def test_find_model_config_not_found(self, config_with_rag):
        """Test _find_model_config returns None for unknown model."""
        service = ProjectChatService()
        result = service._find_model_config(config_with_rag, "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    @patch("services.project_chat_service.search_with_rag")
    async def test_chat_without_rag(self, mock_search, base_config):
        """Test chat without RAG."""
        mock_search.return_value = []

        with tempfile.TemporaryDirectory() as project_dir:
            # Create mock agent
            mock_agent = AsyncMock(spec=ChatOrchestratorAgent)
            mock_agent.history = MagicMock()
            mock_agent.remove_context_provider = MagicMock()
            mock_agent.model_name = "test-model"  # Required for event logging

            # Mock run_async_stream as an async generator
            async def mock_stream(*args, **kwargs):
                yield "Hello"
                yield " world"

            mock_agent.run_async_stream = mock_stream

            service = ProjectChatService()

            # Collect response
            chunks = []
            async for chunk in service.stream_chat(
                project_dir=project_dir,
                project_config=base_config,
                chat_agent=mock_agent,
                messages=[LFChatCompletionUserMessageParam(role="user", content="Hi")],
                rag_enabled=False,
            ):
                chunks.append(chunk)

            assert "".join(chunks) == "Hello world"
            # RAG search should not be called
            mock_search.assert_not_called()

    @pytest.mark.asyncio
    @patch("services.project_chat_service.ProjectChatService._perform_rag_search")
    async def test_chat_with_rag(self, mock_search, config_with_rag):
        """Test chat with RAG enabled."""
        # Mock RAG search results - return already normalized results
        mock_result1 = MagicMock()
        mock_result1.content = "Context chunk 1"
        mock_result1.metadata = {"source": "doc1.txt"}
        mock_result1.score = 0.9

        mock_result2 = MagicMock()
        mock_result2.content = "Context chunk 2"
        mock_result2.metadata = {"source": "doc2.txt"}
        mock_result2.score = 0.85

        mock_search.return_value = [mock_result1, mock_result2]

        with tempfile.TemporaryDirectory() as project_dir:
            # Create mock agent
            mock_agent = AsyncMock(spec=ChatOrchestratorAgent)
            mock_agent.history = MagicMock()
            mock_agent.register_context_provider = MagicMock()
            mock_agent.remove_context_provider = MagicMock()
            mock_agent.model_name = "test-model"  # Required for event logging

            # Mock run_async_stream as an async generator
            async def mock_stream(*args, **kwargs):
                yield "Answer"

            mock_agent.run_async_stream = mock_stream

            service = ProjectChatService()

            chunks = []
            async for chunk in service.stream_chat(
                project_dir=project_dir,
                project_config=config_with_rag,
                chat_agent=mock_agent,
                messages=[
                    LFChatCompletionUserMessageParam(role="user", content="Question")
                ],
                rag_enabled=True,
            ):
                chunks.append(chunk)

            # RAG search should be called
            mock_search.assert_called_once()
            # Context provider should be registered
            mock_agent.register_context_provider.assert_called()

    @pytest.mark.asyncio
    @patch("services.project_chat_service.search_with_rag")
    async def test_chat_method_delegates_to_stream(self, mock_search, config_with_rag):
        """Test that chat() delegates to stream_chat()."""
        mock_search.return_value = []

        with tempfile.TemporaryDirectory() as project_dir:
            mock_agent = AsyncMock(spec=ChatOrchestratorAgent)
            mock_agent.history = MagicMock()
            mock_agent.remove_context_provider = MagicMock()
            mock_agent.model_name = "gpt-4"  # Add model_name attribute

            mock_completion = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "Response text"
            mock_completion.choices = [mock_choice]

            mock_agent.run_async = AsyncMock(return_value=mock_completion)

            service = ProjectChatService()

            response = await service.chat(
                project_dir=project_dir,
                project_config=config_with_rag,
                chat_agent=mock_agent,
                messages=[
                    LFChatCompletionUserMessageParam(role="user", content="Test")
                ],
                rag_enabled=False,
            )

            mock_agent.run_async.assert_awaited_once()
            assert "Response text" in response.choices[0].message.content

    def test_clear_rag_context_provider(self, base_config):
        """Test clearing RAG context provider."""
        mock_agent = MagicMock(spec=ChatOrchestratorAgent)
        mock_agent.remove_context_provider = MagicMock()

        service = ProjectChatService()
        service._clear_rag_context_provider(mock_agent)

        mock_agent.remove_context_provider.assert_called_once_with("rag_context")

    def test_clear_rag_context_provider_handles_error(self, base_config):
        """Test that clearing context provider handles errors gracefully."""
        mock_agent = MagicMock(spec=ChatOrchestratorAgent)
        mock_agent.remove_context_provider = MagicMock(side_effect=Exception("Error"))
        mock_project_service = MagicMock(spec=ProjectService)
        mock_project_service.get_project.return_value = base_config
        with patch("services.project_service.ProjectService", mock_project_service):
            service = ProjectChatService()
            # Should not raise
            service._clear_rag_context_provider(mock_agent)

    def test_extract_config_value_from_dict(self):
        """Test extracting value from dict config."""
        service = ProjectChatService()
        config = {"top_k": 10, "score_threshold": 0.8}

        value = service._extract_config_value(config, "top_k", "test_strategy")
        assert value == 10

    def test_extract_config_value_from_object(self):
        """Test extracting value from object config."""
        service = ProjectChatService()
        config = MagicMock()
        config.top_k = 15

        value = service._extract_config_value(config, "top_k", "test_strategy")
        assert value == 15

    def test_extract_config_value_missing(self):
        """Test extracting missing value returns None."""
        service = ProjectChatService()
        config = {"other_key": "value"}

        value = service._extract_config_value(config, "top_k", "test_strategy")
        assert value is None

    @pytest.mark.asyncio
    @patch("services.project_chat_service.ProjectChatService._perform_rag_search")
    async def test_stream_chat_includes_sources_when_requested(
        self, mock_search, config_with_rag
    ):
        """Test that sources event is yielded before content when include_sources=True."""
        # Mock RAG search results
        mock_result1 = MagicMock()
        mock_result1.content = "Context chunk 1"
        mock_result1.metadata = {"source": "doc1.txt", "similarity_score": 0.9}

        mock_result2 = MagicMock()
        mock_result2.content = "Context chunk 2"
        mock_result2.metadata = {"source": "doc2.txt", "similarity_score": 0.85}

        mock_search.return_value = [mock_result1, mock_result2]

        with tempfile.TemporaryDirectory() as project_dir:
            # Create mock agent with context_providers
            mock_agent = AsyncMock(spec=ChatOrchestratorAgent)
            mock_agent.history = MagicMock()
            mock_agent.register_context_provider = MagicMock()
            mock_agent.remove_context_provider = MagicMock()
            mock_agent.model_name = "test-model"

            # Mock get_context_provider with RAG chunks
            mock_chunk1 = MagicMock()
            mock_chunk1.content = "Context chunk 1"
            mock_chunk1.metadata = {"source": "doc1.txt", "similarity_score": 0.9}

            mock_chunk2 = MagicMock()
            mock_chunk2.content = "Context chunk 2"
            mock_chunk2.metadata = {"source": "doc2.txt", "similarity_score": 0.85}

            mock_rag_context = MagicMock()
            mock_rag_context.chunks = [mock_chunk1, mock_chunk2]
            mock_agent.get_context_provider = MagicMock(return_value=mock_rag_context)

            # Mock run_async_stream as an async generator
            async def mock_stream(*args, **kwargs):
                yield MagicMock(choices=[MagicMock(delta=MagicMock(content="Answer"))])

            mock_agent.run_async_stream = mock_stream

            service = ProjectChatService()

            chunks = []
            async for chunk in service.stream_chat(
                project_dir=project_dir,
                project_config=config_with_rag,
                chat_agent=mock_agent,
                messages=[
                    LFChatCompletionUserMessageParam(role="user", content="Question")
                ],
                rag_enabled=True,
                include_sources=True,
                sources_limit=10,
            ):
                chunks.append(chunk)

            # First chunk should be sources event
            assert len(chunks) >= 1
            first_chunk = chunks[0]
            assert isinstance(first_chunk, dict)
            assert first_chunk.get("type") == "sources"
            assert isinstance(first_chunk.get("sources"), list)
            assert len(first_chunk["sources"]) == 2

            # Verify source structure
            source1 = first_chunk["sources"][0]
            assert source1["content"] == "Context chunk 1"
            assert source1["source"] == "doc1.txt"
            assert source1["score"] == 0.9

    @pytest.mark.asyncio
    @patch("services.project_chat_service.ProjectChatService._perform_rag_search")
    async def test_stream_chat_no_sources_when_disabled(
        self, mock_search, config_with_rag
    ):
        """Test that sources event is not yielded when include_sources=False."""
        mock_search.return_value = []

        with tempfile.TemporaryDirectory() as project_dir:
            mock_agent = AsyncMock(spec=ChatOrchestratorAgent)
            mock_agent.history = MagicMock()
            mock_agent.register_context_provider = MagicMock()
            mock_agent.remove_context_provider = MagicMock()
            mock_agent.model_name = "test-model"

            # Mock get_context_provider with RAG chunks
            mock_chunk = MagicMock()
            mock_chunk.content = "Context chunk"
            mock_chunk.metadata = {"source": "doc.txt", "similarity_score": 0.9}
            mock_rag_context = MagicMock()
            mock_rag_context.chunks = [mock_chunk]
            mock_agent.get_context_provider = MagicMock(return_value=mock_rag_context)

            async def mock_stream(*args, **kwargs):
                yield MagicMock(choices=[MagicMock(delta=MagicMock(content="Answer"))])

            mock_agent.run_async_stream = mock_stream

            service = ProjectChatService()

            chunks = []
            async for chunk in service.stream_chat(
                project_dir=project_dir,
                project_config=config_with_rag,
                chat_agent=mock_agent,
                messages=[
                    LFChatCompletionUserMessageParam(role="user", content="Question")
                ],
                rag_enabled=True,
                include_sources=False,  # Disabled
            ):
                chunks.append(chunk)

            # No sources event should be present
            for chunk in chunks:
                if isinstance(chunk, dict):
                    assert chunk.get("type") != "sources"

    @pytest.mark.asyncio
    @patch("services.project_chat_service.ProjectChatService._perform_rag_search")
    async def test_stream_chat_sources_respects_limit(
        self, mock_search, config_with_rag
    ):
        """Test that sources_limit parameter is respected."""
        mock_search.return_value = []

        with tempfile.TemporaryDirectory() as project_dir:
            mock_agent = AsyncMock(spec=ChatOrchestratorAgent)
            mock_agent.history = MagicMock()
            mock_agent.register_context_provider = MagicMock()
            mock_agent.remove_context_provider = MagicMock()
            mock_agent.model_name = "test-model"

            # Create 10 mock chunks but set limit to 3
            mock_chunks = []
            for i in range(10):
                mock_chunk = MagicMock()
                mock_chunk.content = f"Context chunk {i}"
                mock_chunk.metadata = {"source": f"doc{i}.txt", "similarity_score": 0.9}
                mock_chunks.append(mock_chunk)

            mock_rag_context = MagicMock()
            mock_rag_context.chunks = mock_chunks
            mock_agent.get_context_provider = MagicMock(return_value=mock_rag_context)

            async def mock_stream(*args, **kwargs):
                yield MagicMock(choices=[MagicMock(delta=MagicMock(content="Answer"))])

            mock_agent.run_async_stream = mock_stream

            service = ProjectChatService()

            chunks = []
            async for chunk in service.stream_chat(
                project_dir=project_dir,
                project_config=config_with_rag,
                chat_agent=mock_agent,
                messages=[
                    LFChatCompletionUserMessageParam(role="user", content="Question")
                ],
                rag_enabled=True,
                include_sources=True,
                sources_limit=3,  # Limit to 3
            ):
                chunks.append(chunk)

            # Find sources event
            sources_event = None
            for chunk in chunks:
                if isinstance(chunk, dict) and chunk.get("type") == "sources":
                    sources_event = chunk
                    break

            assert sources_event is not None
            assert len(sources_event["sources"]) == 3  # Should respect limit
