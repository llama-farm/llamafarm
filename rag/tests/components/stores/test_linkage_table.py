"""Tests for Linkage Table - Cross-database record linking.

These tests are written FIRST following TDD methodology.
The LinkageTable implementation should make these tests pass.
"""

import tempfile


class TestLinkageTableInitialization:
    """Test LinkageTable initialization."""

    def test_linkage_table_initializes_with_default_config(self):
        """Test LinkageTable initializes with default configuration."""
        from components.stores.duckdb_store import LinkageTable

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            table = LinkageTable(config=config)

            assert table is not None
            assert table.is_connected()
            table.close()

    def test_linkage_table_creates_mapping_table(self):
        """Test LinkageTable creates mapping table with correct schema."""
        from components.stores.duckdb_store import LinkageTable

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            table = LinkageTable(config=config)

            # Check table exists
            result = table.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'linkage'"
            )
            assert result[0][0] == 1

            # Check schema has required columns
            result = table.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'linkage'
                ORDER BY ordinal_position
                """
            )
            columns = [row[0] for row in result]
            assert "uuid" in columns
            assert "vector_id" in columns
            assert "graph_node_id" in columns
            assert "timeseries_row_id" in columns
            assert "created_at" in columns
            table.close()


class TestLinkageTableLinkOperations:
    """Test link creation operations."""

    def test_link_creates_mapping(self):
        """Test link() creates UUID -> IDs mapping."""
        from components.stores.duckdb_store import LinkageTable

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            table = LinkageTable(config=config)

            concept_uuid = table.link(
                vector_id="vec_001",
                graph_node_id="node_001",
                timeseries_row_id="ts_001",
            )

            assert concept_uuid is not None

            # Verify link exists
            result = table.execute(
                "SELECT vector_id, graph_node_id, timeseries_row_id FROM linkage WHERE uuid = ?",
                [concept_uuid],
            )
            assert len(result) == 1
            assert result[0][0] == "vec_001"
            assert result[0][1] == "node_001"
            assert result[0][2] == "ts_001"
            table.close()

    def test_link_with_custom_uuid(self):
        """Test link() accepts custom UUID."""
        from components.stores.duckdb_store import LinkageTable

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            table = LinkageTable(config=config)

            custom_uuid = "concept_rescue_event_001"
            result_uuid = table.link(
                concept_uuid=custom_uuid,
                vector_id="vec_001",
            )

            assert result_uuid == custom_uuid
            table.close()

    def test_link_with_partial_ids(self):
        """Test link() works with only some IDs provided."""
        from components.stores.duckdb_store import LinkageTable

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            table = LinkageTable(config=config)

            # Only vector ID
            uuid1 = table.link(vector_id="vec_only")
            links1 = table.get_links(uuid1)
            assert links1["vector_id"] == "vec_only"
            assert links1["graph_node_id"] is None
            assert links1["timeseries_row_id"] is None

            # Only graph ID
            uuid2 = table.link(graph_node_id="node_only")
            links2 = table.get_links(uuid2)
            assert links2["graph_node_id"] == "node_only"
            table.close()

    def test_link_updates_existing_mapping(self):
        """Test link() updates existing mapping when UUID exists."""
        from components.stores.duckdb_store import LinkageTable

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            table = LinkageTable(config=config)

            # Create initial link
            concept_uuid = table.link(vector_id="vec_001")

            # Update with additional IDs
            table.link(
                concept_uuid=concept_uuid,
                graph_node_id="node_001",
                timeseries_row_id="ts_001",
            )

            # Verify updated
            links = table.get_links(concept_uuid)
            assert links["vector_id"] == "vec_001"
            assert links["graph_node_id"] == "node_001"
            assert links["timeseries_row_id"] == "ts_001"
            table.close()


class TestLinkageTableQueryOperations:
    """Test query operations."""

    def test_get_links_retrieves_all_ids(self):
        """Test get_links() retrieves all IDs for a concept UUID."""
        from components.stores.duckdb_store import LinkageTable

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            table = LinkageTable(config=config)

            concept_uuid = table.link(
                vector_id="vec_123",
                graph_node_id="node_456",
                timeseries_row_id="ts_789",
            )

            links = table.get_links(concept_uuid)

            assert links["uuid"] == concept_uuid
            assert links["vector_id"] == "vec_123"
            assert links["graph_node_id"] == "node_456"
            assert links["timeseries_row_id"] == "ts_789"
            table.close()

    def test_get_links_returns_none_for_missing_uuid(self):
        """Test get_links() returns None for nonexistent UUID."""
        from components.stores.duckdb_store import LinkageTable

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            table = LinkageTable(config=config)

            links = table.get_links("nonexistent_uuid")
            assert links is None
            table.close()

    def test_find_by_vector_id(self):
        """Test find_by_any_id() finds UUID from vector ID."""
        from components.stores.duckdb_store import LinkageTable

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            table = LinkageTable(config=config)

            concept_uuid = table.link(vector_id="vec_special")

            found_uuid = table.find_by_any_id(vector_id="vec_special")
            assert found_uuid == concept_uuid
            table.close()

    def test_find_by_graph_node_id(self):
        """Test find_by_any_id() finds UUID from graph node ID."""
        from components.stores.duckdb_store import LinkageTable

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            table = LinkageTable(config=config)

            concept_uuid = table.link(graph_node_id="node_special")

            found_uuid = table.find_by_any_id(graph_node_id="node_special")
            assert found_uuid == concept_uuid
            table.close()

    def test_find_by_timeseries_id(self):
        """Test find_by_any_id() finds UUID from timeseries row ID."""
        from components.stores.duckdb_store import LinkageTable

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            table = LinkageTable(config=config)

            concept_uuid = table.link(timeseries_row_id="ts_special")

            found_uuid = table.find_by_any_id(timeseries_row_id="ts_special")
            assert found_uuid == concept_uuid
            table.close()

    def test_find_by_any_id_returns_none_when_not_found(self):
        """Test find_by_any_id() returns None when ID not found."""
        from components.stores.duckdb_store import LinkageTable

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            table = LinkageTable(config=config)

            found = table.find_by_any_id(vector_id="nonexistent")
            assert found is None
            table.close()

    def test_list_all_links(self):
        """Test list_all() returns all linkages."""
        from components.stores.duckdb_store import LinkageTable

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            table = LinkageTable(config=config)

            # Create several links
            table.link(vector_id="vec_1")
            table.link(vector_id="vec_2", graph_node_id="node_2")
            table.link(timeseries_row_id="ts_3")

            all_links = table.list_all()
            assert len(all_links) == 3
            table.close()


class TestLinkageTableUnlinkOperations:
    """Test unlink and deletion operations."""

    def test_unlink_removes_mapping(self):
        """Test unlink() removes mapping."""
        from components.stores.duckdb_store import LinkageTable

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            table = LinkageTable(config=config)

            concept_uuid = table.link(vector_id="vec_001")

            # Unlink
            result = table.unlink(concept_uuid)
            assert result is True

            # Verify removed
            links = table.get_links(concept_uuid)
            assert links is None
            table.close()

    def test_unlink_returns_false_for_missing_uuid(self):
        """Test unlink() returns False for nonexistent UUID."""
        from components.stores.duckdb_store import LinkageTable

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            table = LinkageTable(config=config)

            result = table.unlink("nonexistent_uuid")
            assert result is False
            table.close()

    def test_unlink_returns_component_ids(self):
        """Test unlink() returns the IDs that were linked."""
        from components.stores.duckdb_store import LinkageTable

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            table = LinkageTable(config=config)

            concept_uuid = table.link(
                vector_id="vec_001",
                graph_node_id="node_001",
                timeseries_row_id="ts_001",
            )

            # Unlink and get IDs for cascade delete
            ids = table.unlink_and_get_ids(concept_uuid)

            assert ids["vector_id"] == "vec_001"
            assert ids["graph_node_id"] == "node_001"
            assert ids["timeseries_row_id"] == "ts_001"

            # Verify mapping is removed
            links = table.get_links(concept_uuid)
            assert links is None
            table.close()


class TestLinkageTableStats:
    """Test statistics operations."""

    def test_get_stats_returns_counts(self):
        """Test get_stats() returns linkage statistics."""
        from components.stores.duckdb_store import LinkageTable

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            table = LinkageTable(config=config)

            # Create links with varying completeness
            table.link(
                vector_id="v1", graph_node_id="n1", timeseries_row_id="t1"
            )  # Full
            table.link(vector_id="v2", graph_node_id="n2")  # Partial
            table.link(vector_id="v3")  # Minimal

            stats = table.get_stats()

            assert stats["total_links"] == 3
            assert stats["links_with_vector"] == 3
            assert stats["links_with_graph"] == 2
            assert stats["links_with_timeseries"] == 1
            table.close()


class TestLinkageTableErrorHandling:
    """Test error handling."""

    def test_handles_duplicate_component_id(self):
        """Test handling when same component ID linked to different UUIDs."""
        from components.stores.duckdb_store import LinkageTable

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            table = LinkageTable(config=config)

            # Link same vector ID to two different concepts
            uuid1 = table.link(vector_id="shared_vec")
            uuid2 = table.link(vector_id="shared_vec")

            # Both should exist (this is valid - same data in multiple concepts)
            links1 = table.get_links(uuid1)
            links2 = table.get_links(uuid2)

            assert links1 is not None
            assert links2 is not None
            table.close()

    def test_close_is_idempotent(self):
        """Test close() can be called multiple times."""
        from components.stores.duckdb_store import LinkageTable

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            table = LinkageTable(config=config)

            table.close()
            table.close()  # Should not raise
            table.close()  # Should not raise
