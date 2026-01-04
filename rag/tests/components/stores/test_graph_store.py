"""Tests for Graph Store - Entity relationships and knowledge graph.

These tests are written FIRST following TDD methodology.
The GraphStore implementation should make these tests pass.

Note: DuckPGQ extension may not be available on all platforms.
Tests will skip if extension is not installed.
"""

import tempfile


class TestGraphStoreInitialization:
    """Test GraphStore initialization and schema creation."""

    def test_graph_store_initializes_with_default_config(self):
        """Test GraphStore initializes with default configuration."""
        from components.stores.duckdb_store import GraphStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = GraphStore(config=config)

            assert store is not None
            assert store.is_connected()
            store.close()

    def test_graph_store_creates_node_table(self):
        """Test GraphStore creates node table with proper schema."""
        from components.stores.duckdb_store import GraphStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = GraphStore(config=config)

            # Check nodes table exists
            result = store.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'nodes'"
            )
            assert result[0][0] == 1

            # Check schema has required columns
            result = store.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'nodes'
                ORDER BY ordinal_position
                """
            )
            columns = [row[0] for row in result]
            assert "id" in columns
            assert "name" in columns
            assert "node_type" in columns
            assert "properties" in columns
            store.close()

    def test_graph_store_creates_edge_table(self):
        """Test GraphStore creates edge table with proper schema."""
        from components.stores.duckdb_store import GraphStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = GraphStore(config=config)

            # Check edges table exists
            result = store.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'edges'"
            )
            assert result[0][0] == 1

            # Check schema has required columns
            result = store.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'edges'
                ORDER BY ordinal_position
                """
            )
            columns = [row[0] for row in result]
            assert "id" in columns
            assert "source_id" in columns
            assert "target_id" in columns
            assert "relationship" in columns
            assert "weight" in columns
            store.close()


class TestGraphStoreNodeOperations:
    """Test node CRUD operations."""

    def test_add_node_inserts_correctly(self):
        """Test add_node inserts nodes with properties."""
        from components.stores.duckdb_store import GraphStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = GraphStore(config=config)

            node_id = store.add_node(
                name="Soldier Alpha",
                node_type="person",
                properties={"rank": "Captain", "unit": "1st Battalion"},
            )

            assert node_id is not None

            # Verify node exists
            result = store.execute(
                "SELECT name, node_type FROM nodes WHERE id = ?", [node_id]
            )
            assert len(result) == 1
            assert result[0][0] == "Soldier Alpha"
            assert result[0][1] == "person"
            store.close()

    def test_add_node_with_custom_id(self):
        """Test add_node with custom ID."""
        from components.stores.duckdb_store import GraphStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = GraphStore(config=config)

            custom_id = "soldier_001"
            node_id = store.add_node(
                node_id=custom_id,
                name="Soldier One",
                node_type="person",
            )

            assert node_id == custom_id

            # Verify node exists with custom ID
            result = store.execute("SELECT id FROM nodes WHERE id = ?", [custom_id])
            assert len(result) == 1
            store.close()

    def test_get_node_retrieves_correctly(self):
        """Test get_node retrieves node by ID."""
        from components.stores.duckdb_store import GraphStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = GraphStore(config=config)

            node_id = store.add_node(
                name="Test Node",
                node_type="location",
                properties={"coordinates": "37.7749,-122.4194"},
            )

            node = store.get_node(node_id)

            assert node is not None
            assert node["name"] == "Test Node"
            assert node["node_type"] == "location"
            assert "coordinates" in node.get("properties", {})
            store.close()

    def test_delete_node_removes_correctly(self):
        """Test delete_node removes node and its edges."""
        from components.stores.duckdb_store import GraphStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = GraphStore(config=config)

            # Create nodes and edge
            node1 = store.add_node(name="Node 1", node_type="test")
            node2 = store.add_node(name="Node 2", node_type="test")
            store.add_edge(source_id=node1, target_id=node2, relationship="connects_to")

            # Delete node1
            deleted = store.delete_node(node1)
            assert deleted is True

            # Verify node is gone
            node = store.get_node(node1)
            assert node is None

            # Verify edge is also removed
            edges = store.get_edges(source_id=node1)
            assert len(edges) == 0
            store.close()

    def test_find_nodes_by_type(self):
        """Test finding nodes by type."""
        from components.stores.duckdb_store import GraphStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = GraphStore(config=config)

            # Create different node types
            store.add_node(name="Person 1", node_type="person")
            store.add_node(name="Person 2", node_type="person")
            store.add_node(name="Location 1", node_type="location")

            # Find by type
            persons = store.find_nodes(node_type="person")
            assert len(persons) == 2

            locations = store.find_nodes(node_type="location")
            assert len(locations) == 1
            store.close()


class TestGraphStoreEdgeOperations:
    """Test edge CRUD operations."""

    def test_add_edge_creates_relationship(self):
        """Test add_edge creates relationships between nodes."""
        from components.stores.duckdb_store import GraphStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = GraphStore(config=config)

            node1 = store.add_node(name="Soldier", node_type="person")
            node2 = store.add_node(name="Base Camp", node_type="location")

            edge_id = store.add_edge(
                source_id=node1,
                target_id=node2,
                relationship="located_at",
                weight=1.0,
                properties={"since": "2024-01-01"},
            )

            assert edge_id is not None

            # Verify edge exists
            result = store.execute(
                "SELECT relationship, weight FROM edges WHERE id = ?", [edge_id]
            )
            assert len(result) == 1
            assert result[0][0] == "located_at"
            assert result[0][1] == 1.0
            store.close()

    def test_get_edges_retrieves_outgoing(self):
        """Test get_edges retrieves outgoing edges from a node."""
        from components.stores.duckdb_store import GraphStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = GraphStore(config=config)

            center = store.add_node(name="Center", node_type="hub")
            target1 = store.add_node(name="Target 1", node_type="endpoint")
            target2 = store.add_node(name="Target 2", node_type="endpoint")

            store.add_edge(source_id=center, target_id=target1, relationship="connects")
            store.add_edge(source_id=center, target_id=target2, relationship="connects")

            edges = store.get_edges(source_id=center)
            assert len(edges) == 2
            store.close()

    def test_get_edges_with_relationship_filter(self):
        """Test get_edges can filter by relationship type."""
        from components.stores.duckdb_store import GraphStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = GraphStore(config=config)

            person = store.add_node(name="Person", node_type="person")
            location = store.add_node(name="Location", node_type="location")
            event = store.add_node(name="Event", node_type="event")

            store.add_edge(
                source_id=person, target_id=location, relationship="located_at"
            )
            store.add_edge(
                source_id=person, target_id=event, relationship="participated_in"
            )

            location_edges = store.get_edges(
                source_id=person, relationship="located_at"
            )
            assert len(location_edges) == 1
            assert location_edges[0]["relationship"] == "located_at"
            store.close()

    def test_delete_edge_removes_correctly(self):
        """Test delete_edge removes edge."""
        from components.stores.duckdb_store import GraphStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = GraphStore(config=config)

            node1 = store.add_node(name="Node 1", node_type="test")
            node2 = store.add_node(name="Node 2", node_type="test")
            edge_id = store.add_edge(
                source_id=node1, target_id=node2, relationship="test"
            )

            deleted = store.delete_edge(edge_id)
            assert deleted is True

            # Verify edge is gone
            edges = store.get_edges(source_id=node1)
            assert len(edges) == 0
            store.close()


class TestGraphStoreTraversal:
    """Test graph traversal operations."""

    def test_find_neighbors_retrieves_connected_nodes(self):
        """Test find_neighbors retrieves directly connected nodes."""
        from components.stores.duckdb_store import GraphStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = GraphStore(config=config)

            # Create a simple graph: center -> target1, center -> target2
            center = store.add_node(name="Center", node_type="hub")
            target1 = store.add_node(name="Target 1", node_type="endpoint")
            target2 = store.add_node(name="Target 2", node_type="endpoint")
            store.add_node(name="Unconnected", node_type="endpoint")  # Not connected

            store.add_edge(source_id=center, target_id=target1, relationship="connects")
            store.add_edge(source_id=center, target_id=target2, relationship="connects")

            neighbors = store.find_neighbors(node_id=center)
            neighbor_names = [n["name"] for n in neighbors]

            assert len(neighbors) == 2
            assert "Target 1" in neighbor_names
            assert "Target 2" in neighbor_names
            assert "Unconnected" not in neighbor_names
            store.close()

    def test_find_neighbors_includes_incoming(self):
        """Test find_neighbors can include incoming edges."""
        from components.stores.duckdb_store import GraphStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = GraphStore(config=config)

            node1 = store.add_node(name="Node 1", node_type="test")
            node2 = store.add_node(name="Node 2", node_type="test")
            node3 = store.add_node(name="Node 3", node_type="test")

            # node1 -> node2, node3 -> node2
            store.add_edge(source_id=node1, target_id=node2, relationship="points_to")
            store.add_edge(source_id=node3, target_id=node2, relationship="points_to")

            # Get all neighbors (both directions)
            neighbors = store.find_neighbors(node_id=node2, direction="both")
            assert len(neighbors) == 2  # node1 and node3
            store.close()

    def test_find_path_finds_shortest_path(self):
        """Test find_path finds paths between nodes with max depth."""
        from components.stores.duckdb_store import GraphStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = GraphStore(config=config)

            # Create chain: A -> B -> C -> D
            node_a = store.add_node(name="A", node_type="test")
            node_b = store.add_node(name="B", node_type="test")
            node_c = store.add_node(name="C", node_type="test")
            node_d = store.add_node(name="D", node_type="test")

            store.add_edge(source_id=node_a, target_id=node_b, relationship="next")
            store.add_edge(source_id=node_b, target_id=node_c, relationship="next")
            store.add_edge(source_id=node_c, target_id=node_d, relationship="next")

            # Find path from A to D
            paths = store.find_path(start_id=node_a, end_id=node_d, max_depth=5)

            assert len(paths) >= 1
            # First path should be shortest
            assert len(paths[0]) == 4  # A, B, C, D
            store.close()

    def test_find_path_respects_max_depth(self):
        """Test find_path stops at max_depth."""
        from components.stores.duckdb_store import GraphStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = GraphStore(config=config)

            # Create chain: A -> B -> C -> D -> E
            nodes = []
            for name in ["A", "B", "C", "D", "E"]:
                nodes.append(store.add_node(name=name, node_type="test"))

            for i in range(len(nodes) - 1):
                store.add_edge(
                    source_id=nodes[i], target_id=nodes[i + 1], relationship="next"
                )

            # Find path from A to E with max_depth=2 (should not find)
            paths = store.find_path(start_id=nodes[0], end_id=nodes[4], max_depth=2)
            assert len(paths) == 0  # Cannot reach in 2 hops

            # Find path with sufficient depth
            paths = store.find_path(start_id=nodes[0], end_id=nodes[4], max_depth=5)
            assert len(paths) >= 1
            store.close()

    def test_graph_handles_cycles_without_infinite_loop(self):
        """Test graph handles cycles without infinite loops."""
        from components.stores.duckdb_store import GraphStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = GraphStore(config=config)

            # Create cycle: A -> B -> C -> A
            node_a = store.add_node(name="A", node_type="test")
            node_b = store.add_node(name="B", node_type="test")
            node_c = store.add_node(name="C", node_type="test")

            store.add_edge(source_id=node_a, target_id=node_b, relationship="next")
            store.add_edge(source_id=node_b, target_id=node_c, relationship="next")
            store.add_edge(
                source_id=node_c, target_id=node_a, relationship="next"
            )  # Cycle!

            # This should complete without hanging
            neighbors = store.find_neighbors(node_id=node_a, direction="both")
            assert len(neighbors) >= 1

            # Path finding should also handle cycles
            paths = store.find_path(start_id=node_a, end_id=node_c, max_depth=5)
            assert len(paths) >= 1
            store.close()


class TestGraphStoreErrorHandling:
    """Test error handling and edge cases."""

    def test_get_nonexistent_node_returns_none(self):
        """Test getting nonexistent node returns None."""
        from components.stores.duckdb_store import GraphStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = GraphStore(config=config)

            node = store.get_node("nonexistent_id")
            assert node is None
            store.close()

    def test_add_edge_with_invalid_source_fails_gracefully(self):
        """Test add_edge with invalid source handles error."""
        from components.stores.duckdb_store import GraphStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = GraphStore(config=config)

            target = store.add_node(name="Target", node_type="test")

            # Try to add edge with nonexistent source
            # Returns None or edge_id depending on implementation
            store.add_edge(
                source_id="nonexistent",
                target_id=target,
                relationship="invalid",
            )

            # Should either fail gracefully or skip
            # Implementation can choose behavior
            store.close()

    def test_find_path_between_unconnected_nodes_returns_empty(self):
        """Test find_path between unconnected nodes returns empty."""
        from components.stores.duckdb_store import GraphStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = GraphStore(config=config)

            node1 = store.add_node(name="Isolated 1", node_type="test")
            node2 = store.add_node(name="Isolated 2", node_type="test")

            # No edges between them
            paths = store.find_path(start_id=node1, end_id=node2, max_depth=5)
            assert len(paths) == 0
            store.close()

    def test_close_is_idempotent(self):
        """Test close() can be called multiple times."""
        from components.stores.duckdb_store import GraphStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = GraphStore(config=config)

            store.close()
            store.close()  # Should not raise
            store.close()  # Should not raise


class TestGraphStoreStats:
    """Test statistics and metadata operations."""

    def test_get_stats_returns_counts(self):
        """Test get_stats returns node and edge counts."""
        from components.stores.duckdb_store import GraphStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = GraphStore(config=config)

            # Add some data
            node1 = store.add_node(name="Node 1", node_type="person")
            node2 = store.add_node(name="Node 2", node_type="location")
            store.add_edge(source_id=node1, target_id=node2, relationship="located_at")

            stats = store.get_stats()

            assert stats["node_count"] == 2
            assert stats["edge_count"] == 1
            assert (
                "person" in stats.get("node_types", {})
                or stats.get("node_types") is None
            )
            store.close()
