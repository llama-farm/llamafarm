"""Tests for Working Memory - Short-term cache with TTL.

These tests are written FIRST following TDD methodology.
The WorkingMemory implementation should make these tests pass.
"""

import tempfile
import time


class TestWorkingMemoryInitialization:
    """Test WorkingMemory initialization."""

    def test_working_memory_initializes_with_default_config(self):
        """Test WorkingMemory initializes with default configuration."""
        from components.stores.duckdb_store import WorkingMemory

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            memory = WorkingMemory(config=config)

            assert memory is not None
            assert memory.is_connected()
            memory.close()

    def test_working_memory_creates_buffer_table_with_ttl(self):
        """Test WorkingMemory creates buffer table with TTL column."""
        from components.stores.duckdb_store import WorkingMemory

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            memory = WorkingMemory(config=config)

            # Check table exists
            result = memory.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'working_memory'"
            )
            assert result[0][0] == 1

            # Check schema has required columns
            result = memory.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'working_memory'
                ORDER BY ordinal_position
                """
            )
            columns = [row[0] for row in result]
            assert "id" in columns
            assert "data_type" in columns
            assert "content" in columns
            assert "metadata" in columns
            assert "created_at" in columns
            assert "expires_at" in columns
            memory.close()

    def test_working_memory_respects_custom_ttl(self):
        """Test WorkingMemory uses custom TTL from config."""
        from components.stores.duckdb_store import WorkingMemory

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "path": f"{temp_dir}/test.duckdb",
                "ttl_seconds": 300,  # 5 minutes
            }
            memory = WorkingMemory(config=config)

            assert memory.ttl_seconds == 300
            memory.close()


class TestWorkingMemoryAddOperations:
    """Test adding records to working memory."""

    def test_add_inserts_with_automatic_timestamp(self):
        """Test add() inserts records with automatic timestamp."""
        from components.stores.duckdb_store import WorkingMemory

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb", "ttl_seconds": 3600}
            memory = WorkingMemory(config=config)

            record_id = memory.add(
                data_type="chat",
                content="Hello, how are you?",
                metadata={"sender": "user1"},
            )

            assert record_id is not None

            # Verify timestamp was set
            result = memory.execute(
                "SELECT created_at, expires_at FROM working_memory WHERE id = ?",
                [record_id],
            )
            assert result[0][0] is not None  # created_at
            assert result[0][1] is not None  # expires_at

            # Verify expiry is roughly TTL seconds in future
            created = result[0][0]
            expires = result[0][1]
            diff = (expires - created).total_seconds()
            assert abs(diff - 3600) < 5  # Within 5 seconds of expected
            memory.close()

    def test_add_multiple_data_types(self):
        """Test add() handles different data types."""
        from components.stores.duckdb_store import WorkingMemory

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            memory = WorkingMemory(config=config)

            # Add different types
            memory.add(data_type="chat", content="Hello")
            memory.add(data_type="telemetry", content='{"heart_rate": 72}')
            memory.add(data_type="audio", content="transcript of audio")

            # Verify all types stored
            result = memory.execute("SELECT DISTINCT data_type FROM working_memory")
            types = {row[0] for row in result}
            assert types == {"chat", "telemetry", "audio"}
            memory.close()

    def test_add_batch_records(self):
        """Test add_batch() inserts multiple records efficiently."""
        from components.stores.duckdb_store import WorkingMemory

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            memory = WorkingMemory(config=config)

            records = [
                {"data_type": "telemetry", "content": f"value_{i}"} for i in range(100)
            ]

            count = memory.add_batch(records)
            assert count == 100

            # Verify all records in database
            result = memory.execute("SELECT COUNT(*) FROM working_memory")
            assert result[0][0] == 100
            memory.close()


class TestWorkingMemoryQueryOperations:
    """Test querying working memory."""

    def test_get_recent_retrieves_within_ttl_window(self):
        """Test get_recent() retrieves records within TTL window."""
        from components.stores.duckdb_store import WorkingMemory

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb", "ttl_seconds": 3600}
            memory = WorkingMemory(config=config)

            # Add records
            memory.add(data_type="chat", content="Message 1")
            memory.add(data_type="chat", content="Message 2")
            memory.add(data_type="chat", content="Message 3")

            # Get recent (should return all since TTL not expired)
            records = memory.get_recent(limit=10)

            assert len(records) == 3
            memory.close()

    def test_get_recent_with_time_window(self):
        """Test get_recent() can filter by custom time window."""
        from components.stores.duckdb_store import WorkingMemory

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            memory = WorkingMemory(config=config)

            # Add records
            memory.add(data_type="chat", content="Recent message")

            # Get records from last 5 minutes
            records = memory.get_recent(minutes=5)
            assert len(records) >= 1

            # Get records from last 0 minutes (should be empty or very few)
            records = memory.get_recent(seconds=0)
            assert len(records) == 0
            memory.close()

    def test_get_by_type_filters_correctly(self):
        """Test get_by_type() filters by data type."""
        from components.stores.duckdb_store import WorkingMemory

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            memory = WorkingMemory(config=config)

            # Add different types
            memory.add(data_type="chat", content="Chat 1")
            memory.add(data_type="chat", content="Chat 2")
            memory.add(data_type="telemetry", content="Telemetry 1")
            memory.add(data_type="audio", content="Audio 1")

            # Filter by type
            chats = memory.get_by_type("chat")
            assert len(chats) == 2

            telemetry = memory.get_by_type("telemetry")
            assert len(telemetry) == 1

            audio = memory.get_by_type("audio")
            assert len(audio) == 1
            memory.close()

    def test_get_by_type_respects_limit(self):
        """Test get_by_type() respects limit parameter."""
        from components.stores.duckdb_store import WorkingMemory

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            memory = WorkingMemory(config=config)

            # Add many records
            for i in range(20):
                memory.add(data_type="chat", content=f"Message {i}")

            # Get with limit
            records = memory.get_by_type("chat", limit=5)
            assert len(records) == 5
            memory.close()


class TestWorkingMemoryPruneOperations:
    """Test pruning expired records."""

    def test_prune_removes_expired_records(self):
        """Test prune() removes expired records."""
        from components.stores.duckdb_store import WorkingMemory

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "path": f"{temp_dir}/test.duckdb",
                "ttl_seconds": 1,
            }  # 1 second TTL
            memory = WorkingMemory(config=config)

            # Add records
            memory.add(data_type="chat", content="Will expire")

            # Wait for expiry
            time.sleep(1.5)

            # Add a new record (won't be expired)
            memory.add(data_type="chat", content="Fresh")

            # Prune expired
            pruned = memory.prune()
            assert pruned >= 1  # At least 1 expired

            # Verify only fresh record remains
            records = memory.get_recent()
            assert len(records) == 1
            assert records[0]["content"] == "Fresh"
            memory.close()

    def test_prune_returns_count_of_removed(self):
        """Test prune() returns the count of removed records."""
        from components.stores.duckdb_store import WorkingMemory

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb", "ttl_seconds": 1}
            memory = WorkingMemory(config=config)

            # Add records
            for i in range(5):
                memory.add(data_type="chat", content=f"Message {i}")

            # Wait for expiry
            time.sleep(1.5)

            # Prune and check count
            pruned = memory.prune()
            assert pruned == 5
            memory.close()


class TestWorkingMemoryAutoprune:
    """Test auto-prune functionality."""

    def test_auto_prune_on_max_size_exceeded(self):
        """Test auto-prune runs when buffer exceeds max_size."""
        from components.stores.duckdb_store import WorkingMemory

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "path": f"{temp_dir}/test.duckdb",
                "max_size": 10,
                "ttl_seconds": 1,  # Short TTL so records expire quickly
            }
            memory = WorkingMemory(config=config)

            # Add records up to max size
            for i in range(10):
                memory.add(data_type="chat", content=f"Message {i}")

            # Wait a bit for some to expire
            time.sleep(1.2)

            # Add one more - should trigger auto-prune
            memory.add(data_type="chat", content="Trigger prune")

            # Total should be less than max_size + 1 if prune ran
            result = memory.execute("SELECT COUNT(*) FROM working_memory")
            count = result[0][0]

            # After prune, should have reduced count
            # (only the recent non-expired record should remain)
            assert count <= 5  # Most old ones should be pruned
            memory.close()


class TestWorkingMemoryClear:
    """Test clearing working memory."""

    def test_clear_removes_all_records(self):
        """Test clear() removes all records."""
        from components.stores.duckdb_store import WorkingMemory

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            memory = WorkingMemory(config=config)

            # Add records
            for i in range(10):
                memory.add(data_type="chat", content=f"Message {i}")

            # Verify records exist
            result = memory.execute("SELECT COUNT(*) FROM working_memory")
            assert result[0][0] == 10

            # Clear
            memory.clear()

            # Verify empty
            result = memory.execute("SELECT COUNT(*) FROM working_memory")
            assert result[0][0] == 0
            memory.close()


class TestWorkingMemoryStats:
    """Test statistics operations."""

    def test_get_stats_returns_counts(self):
        """Test get_stats() returns buffer statistics."""
        from components.stores.duckdb_store import WorkingMemory

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb", "max_size": 100}
            memory = WorkingMemory(config=config)

            # Add various types
            memory.add(data_type="chat", content="Chat 1")
            memory.add(data_type="chat", content="Chat 2")
            memory.add(data_type="telemetry", content="Telemetry 1")

            stats = memory.get_stats()

            assert stats["total_records"] == 3
            assert stats["max_size"] == 100
            assert "type_counts" in stats
            assert stats["type_counts"].get("chat", 0) == 2
            assert stats["type_counts"].get("telemetry", 0) == 1
            memory.close()


class TestWorkingMemoryErrorHandling:
    """Test error handling."""

    def test_handles_empty_buffer(self):
        """Test operations on empty buffer."""
        from components.stores.duckdb_store import WorkingMemory

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            memory = WorkingMemory(config=config)

            # Query empty buffer
            records = memory.get_recent()
            assert records == []

            # Prune empty buffer
            pruned = memory.prune()
            assert pruned == 0

            # Stats on empty buffer
            stats = memory.get_stats()
            assert stats["total_records"] == 0
            memory.close()

    def test_close_is_idempotent(self):
        """Test close() can be called multiple times."""
        from components.stores.duckdb_store import WorkingMemory

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            memory = WorkingMemory(config=config)

            memory.close()
            memory.close()  # Should not raise
            memory.close()  # Should not raise
