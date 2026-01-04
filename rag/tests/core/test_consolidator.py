"""Tests for Consolidator - Memory synthesis agent.

These tests are written FIRST following TDD methodology.
The Consolidator implementation should make these tests pass.
"""

import tempfile
from unittest.mock import MagicMock, patch


class TestConsolidatorInitialization:
    """Test Consolidator initialization."""

    def test_consolidator_initializes_with_memory_store(self):
        """Test Consolidator initializes with a MemoryStore."""
        from core.consolidator import Consolidator
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            memory = MemoryStore(config=config)

            consolidator = Consolidator(memory_store=memory)

            assert consolidator is not None
            assert consolidator.memory_store is memory
            memory.close()

    def test_consolidator_accepts_custom_config(self):
        """Test Consolidator accepts custom configuration."""
        from core.consolidator import Consolidator
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            memory = MemoryStore(config=config)

            consolidator_config = {
                "buffer_threshold": 50,
                "consolidation_interval": 300,
                "retention_days": 7,
            }
            consolidator = Consolidator(
                memory_store=memory,
                config=consolidator_config,
            )

            assert consolidator.buffer_threshold == 50
            assert consolidator.consolidation_interval == 300
            assert consolidator.retention_days == 7
            memory.close()


class TestConsolidatorWorkingMemoryReading:
    """Test reading from WorkingMemory."""

    def test_get_pending_records_reads_from_working_memory(self):
        """Test get_pending_records() reads from WorkingMemory."""
        from core.consolidator import Consolidator
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            memory = MemoryStore(config=config)

            # Add data to working memory (chat, audio, stream types go to working memory)
            memory.add(data="Test message 1", data_type="chat")
            memory.add(data="Test message 2", data_type="chat")
            memory.add(data="Audio transcription", data_type="audio")

            consolidator = Consolidator(memory_store=memory)

            # Get pending records
            records = consolidator.get_pending_records(limit=10)

            assert len(records) >= 3
            memory.close()

    def test_get_pending_records_respects_limit(self):
        """Test get_pending_records() respects the limit parameter."""
        from core.consolidator import Consolidator
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            memory = MemoryStore(config=config)

            # Add many records
            for i in range(20):
                memory.add(data=f"Message {i}", data_type="chat")

            consolidator = Consolidator(memory_store=memory)

            # Get limited records
            records = consolidator.get_pending_records(limit=5)

            assert len(records) == 5
            memory.close()


class TestConsolidatorSynthesis:
    """Test synthesis operations."""

    def test_synthesize_extracts_facts_from_records(self):
        """Test synthesize() extracts facts from records."""
        from core.consolidator import Consolidator
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            memory = MemoryStore(config=config)

            consolidator = Consolidator(memory_store=memory)

            # Mock records to synthesize
            records = [
                {"content": "User mentioned their name is John", "data_type": "chat"},
                {"content": "User is located in New York", "data_type": "chat"},
                {
                    "content": '{"heart_rate": 120, "status": "elevated"}',
                    "data_type": "telemetry",
                },
            ]

            # Synthesize without LLM (uses rule-based extraction)
            result = consolidator.synthesize(records, use_llm=False)

            assert result is not None
            assert "facts" in result
            assert "summary" in result
            memory.close()

    def test_synthesize_with_mocked_llm(self):
        """Test synthesize() with mocked LLM response."""
        from core.consolidator import Consolidator
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            memory = MemoryStore(config=config)

            # Create consolidator with a mock LLM client
            mock_llm = MagicMock()
            consolidator = Consolidator(memory_store=memory, llm_client=mock_llm)

            records = [
                {
                    "content": "Subject reported chest pain at 14:30",
                    "data_type": "chat",
                },
            ]

            # Mock LLM response
            mock_llm_response = {
                "facts": [
                    {
                        "subject": "subject",
                        "predicate": "experienced",
                        "object": "chest pain",
                    },
                    {"subject": "event", "predicate": "timestamp", "object": "14:30"},
                ],
                "summary": "Subject experienced chest pain at 14:30",
            }

            with patch.object(
                consolidator, "_call_llm", return_value=mock_llm_response
            ):
                result = consolidator.synthesize(records, use_llm=True)

            assert result is not None
            assert len(result.get("facts", [])) >= 1
            memory.close()


class TestConsolidatorFactExtraction:
    """Test fact extraction from LLM responses."""

    def test_extract_facts_parses_llm_response(self):
        """Test extract_facts() parses LLM response correctly."""
        from core.consolidator import Consolidator
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            memory = MemoryStore(config=config)

            consolidator = Consolidator(memory_store=memory)

            llm_response = {
                "facts": [
                    {"subject": "John", "predicate": "lives_in", "object": "New York"},
                    {"subject": "John", "predicate": "works_at", "object": "Acme Corp"},
                ],
                "summary": "John lives in New York and works at Acme Corp",
            }

            facts = consolidator.extract_facts(llm_response)

            assert len(facts) == 2
            assert facts[0]["subject"] == "John"
            memory.close()

    def test_extract_facts_handles_malformed_response(self):
        """Test extract_facts() handles malformed LLM responses gracefully."""
        from core.consolidator import Consolidator
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            memory = MemoryStore(config=config)

            consolidator = Consolidator(memory_store=memory)

            # Malformed response
            llm_response = {"error": "something went wrong"}

            facts = consolidator.extract_facts(llm_response)

            # Should return empty list, not raise
            assert facts == []
            memory.close()


class TestConsolidatorGraphCreation:
    """Test creating graph nodes from extracted facts."""

    def test_create_graph_nodes_from_facts(self):
        """Test create_graph_nodes() adds nodes to graph store."""
        from core.consolidator import Consolidator
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            memory = MemoryStore(config=config)

            consolidator = Consolidator(memory_store=memory)

            facts = [
                {"subject": "John", "predicate": "lives_in", "object": "New York"},
                {"subject": "John", "predicate": "works_at", "object": "Acme Corp"},
            ]

            count = consolidator.create_graph_nodes(facts)

            # Should create nodes and edges
            assert count >= 2
            memory.close()

    def test_create_graph_nodes_handles_empty_facts(self):
        """Test create_graph_nodes() handles empty facts list."""
        from core.consolidator import Consolidator
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            memory = MemoryStore(config=config)

            consolidator = Consolidator(memory_store=memory)

            count = consolidator.create_graph_nodes([])

            assert count == 0
            memory.close()


class TestConsolidatorPruning:
    """Test pruning operations."""

    def test_prune_removes_expired_records(self):
        """Test prune() removes expired records from working memory."""
        import time

        from core.consolidator import Consolidator
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "base_path": temp_dir,
                "working_memory": {
                    "path": f"{temp_dir}/working.duckdb",
                    "ttl_seconds": 1,  # Very short TTL
                },
            }
            memory = MemoryStore(config=config)

            # Add records
            memory.add(data="Test message", data_type="chat")

            consolidator = Consolidator(memory_store=memory)

            # Wait for expiration
            time.sleep(1.5)

            # Prune
            pruned = consolidator.prune()

            assert pruned >= 1
            memory.close()

    def test_prune_respects_retention_policy(self):
        """Test prune() respects retention policy settings."""
        from core.consolidator import Consolidator
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            memory = MemoryStore(config=config)

            consolidator_config = {"retention_days": 30}
            consolidator = Consolidator(
                memory_store=memory,
                config=consolidator_config,
            )

            # Prune should not remove recent records
            memory.add(data="Recent message", data_type="chat")

            pruned = consolidator.prune()

            # No records should be pruned (they're recent)
            assert pruned == 0
            memory.close()


class TestConsolidatorRunCycle:
    """Test the main consolidation cycle."""

    def test_run_cycle_processes_working_memory(self):
        """Test run_cycle() processes working memory records."""
        from core.consolidator import Consolidator
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            memory = MemoryStore(config=config)

            # Add some records
            memory.add(data="User John mentioned he lives in Boston", data_type="chat")
            memory.add(data="John works at Tech Company", data_type="chat")

            consolidator = Consolidator(memory_store=memory)

            # Run cycle (without LLM)
            result = consolidator.run_cycle(use_llm=False)

            assert result is not None
            assert "records_processed" in result
            assert "facts_extracted" in result
            memory.close()

    def test_run_cycle_skips_when_below_threshold(self):
        """Test run_cycle() skips when records below threshold."""
        from core.consolidator import Consolidator
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            memory = MemoryStore(config=config)

            # Only add 1 record
            memory.add(data="Single message", data_type="chat")

            consolidator_config = {"buffer_threshold": 10}  # Require 10 records
            consolidator = Consolidator(
                memory_store=memory,
                config=consolidator_config,
            )

            # Run cycle - should skip
            result = consolidator.run_cycle(use_llm=False)

            assert result["records_processed"] == 0
            assert result.get("skipped") is True
            memory.close()


class TestConsolidatorErrorHandling:
    """Test error handling."""

    def test_handles_llm_unavailable(self):
        """Test error handling when LLM is unavailable."""
        from core.consolidator import Consolidator
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            memory = MemoryStore(config=config)

            memory.add(data="Test message", data_type="chat")

            consolidator = Consolidator(memory_store=memory)

            # Mock LLM to raise error
            with patch.object(
                consolidator, "_call_llm", side_effect=Exception("LLM unavailable")
            ):
                # Should not raise, should fall back to rule-based
                result = consolidator.synthesize(
                    [{"content": "Test", "data_type": "chat"}],
                    use_llm=True,
                )

            assert result is not None
            memory.close()

    def test_handles_empty_working_memory(self):
        """Test handling when working memory is empty."""
        from core.consolidator import Consolidator
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            memory = MemoryStore(config=config)

            consolidator = Consolidator(memory_store=memory)

            # Run cycle on empty memory
            result = consolidator.run_cycle(use_llm=False)

            assert result["records_processed"] == 0
            memory.close()
