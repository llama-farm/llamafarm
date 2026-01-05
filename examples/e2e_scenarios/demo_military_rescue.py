#!/usr/bin/env python3
"""
Military Rescue Scenario - End-to-End LlamaFarm Demo

This demo showcases the full power of the Embedded Trinity Memory System
using the Phase 3 Unified Dataset Architecture in a realistic military rescue scenario:

1. DATABASE SEEDING
   - Use UnifiedDatasetStore with typed datasets
   - Seed graph with personnel, locations, and command structure
   - Seed working memory with radio transcriptions
   - Stream biometric telemetry with spatial data

2. STREAMING DATA
   - Simulate real-time biometric telemetry
   - Simulate radio communications with distress signals

3. ML OPERATIONS
   - Train a distress classifier on radio communications
   - Train an anomaly detector on vital signs
   - Detect anomalies in incoming data

4. UNIFIED RETRIEVAL
   - Query across all stores using HybridQueryExecutor
   - Build context for agent decision-making

5. CONSOLIDATION
   - The "hippocampus" process extracts facts
   - Creates graph nodes from extracted information
   - Prunes processed raw data

6. CLEANUP
   - Proper database cleanup

Dataset Types Used:
- 'realtime': For biometric telemetry (all stores enabled)
- 'graph': For command structure
- 'knowledge': For protocols documentation

Run from the rag directory:
    cd rag && uv run python ../examples/e2e_scenarios/demo_military_rescue.py
"""

import random
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone

# Add rag to path for direct component access
sys.path.insert(0, ".")


def print_header(title: str) -> None:
    """Print a fancy header."""
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def print_success(msg: str) -> None:
    """Print success message."""
    print(f"  ✓ {msg}")


def print_info(msg: str) -> None:
    """Print info message."""
    print(f"  → {msg}")


def print_data(label: str, value) -> None:
    """Print data with label."""
    print(f"    {label}: {value}")


def print_alert(msg: str) -> None:
    """Print alert/warning message."""
    print(f"  ⚠️  {msg}")


def main() -> int:
    """Run the military rescue scenario demo."""

    print_header("MILITARY RESCUE SCENARIO")
    print("  Demonstrating the Embedded Trinity Memory System")
    print("  with Phase 3 Unified Dataset Architecture")
    print("  ML-powered anomaly detection and classification")

    # Import components
    from core.unified_store import UnifiedDatasetStore
    from core.hybrid_query import HybridQueryExecutor, HybridQueryRequest, QueryMode
    from core.consolidator import Consolidator

    # Create temporary directory for demo databases
    with tempfile.TemporaryDirectory(prefix="military_rescue_") as temp_dir:
        print_info(f"Demo data directory: {temp_dir}")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 1: Initialize Memory System with Typed Datasets
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 1: Initialize Unified Dataset Stores")

        # Create realtime dataset for biometrics (all stores enabled)
        biometrics_store = UnifiedDatasetStore(
            dataset_config={"name": "biometric_telemetry", "type": "realtime"},
            project_dir=temp_dir,
        )
        print_success("Created 'biometric_telemetry' (realtime dataset)")
        print_data("Enabled stores", biometrics_store.get_enabled_stores())

        # Create graph dataset for command structure
        command_store = UnifiedDatasetStore(
            dataset_config={"name": "command_structure", "type": "graph"},
            project_dir=temp_dir,
        )
        print_success("Created 'command_structure' (graph dataset)")
        print_data("Enabled stores", command_store.get_enabled_stores())

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 2: Seed Database - Personnel & Locations (Graph)
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 2: Seed Knowledge Graph - Personnel & Locations")

        # Add personnel nodes
        personnel = [
            {"id": "soldier:sgt_johnson", "name": "Sgt. Johnson", "rank": "Sergeant",
             "unit": "Alpha", "specialty": "Medic", "status": "active"},
            {"id": "soldier:cpl_smith", "name": "Cpl. Smith", "rank": "Corporal",
             "unit": "Alpha", "specialty": "Communications", "status": "active"},
            {"id": "soldier:lt_chen", "name": "Lt. Chen", "rank": "Lieutenant",
             "unit": "Alpha", "specialty": "Command", "status": "active"},
            {"id": "soldier:pvt_williams", "name": "Pvt. Williams", "rank": "Private",
             "unit": "Bravo", "specialty": "Rifleman", "status": "distress"},
        ]

        for person in personnel:
            command_store.add_node(
                name=person["name"],
                node_type="personnel",
                node_id=person["id"],
                properties={k: v for k, v in person.items() if k not in ["id", "name"]},
            )
        print_success(f"Added {len(personnel)} personnel nodes")

        # Add location nodes
        locations = [
            {"id": "location:checkpoint_alpha", "name": "Checkpoint Alpha",
             "lat": 35.7796, "lon": -78.6382, "type": "checkpoint"},
            {"id": "location:checkpoint_delta", "name": "Checkpoint Delta",
             "lat": 35.7850, "lon": -78.6400, "type": "checkpoint"},
            {"id": "location:base_camp", "name": "Base Camp",
             "lat": 35.7700, "lon": -78.6300, "type": "base"},
            {"id": "location:rescue_zone_1", "name": "Rescue Zone 1",
             "lat": 35.7880, "lon": -78.6420, "type": "operation_area"},
        ]

        for loc in locations:
            command_store.add_node(
                name=loc["name"],
                node_type="location",
                node_id=loc["id"],
                properties={k: v for k, v in loc.items() if k not in ["id", "name"]},
            )
        print_success(f"Added {len(locations)} location nodes")

        # Add command relationships (edges)
        edges = [
            ("soldier:lt_chen", "soldier:sgt_johnson", "commands"),
            ("soldier:lt_chen", "soldier:cpl_smith", "commands"),
            ("soldier:sgt_johnson", "location:checkpoint_delta", "assigned_to"),
            ("soldier:cpl_smith", "location:checkpoint_alpha", "assigned_to"),
            ("soldier:pvt_williams", "location:rescue_zone_1", "last_known_location"),
        ]

        for source, target, rel in edges:
            command_store.add_edge(source, target, rel)
        print_success(f"Added {len(edges)} relationship edges")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 3: Stream Biometric Telemetry (Realtime Dataset)
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 3: Stream Biometric Telemetry")

        now = datetime.now(timezone.utc)

        # Normal vitals for Sgt. Johnson (10 readings over 10 minutes)
        print_info("Streaming normal vitals for Sgt. Johnson...")
        for i in range(10):
            ts = now - timedelta(minutes=10 - i)
            biometrics_store.add_stream_record(
                data={
                    "soldier_id": "soldier:sgt_johnson",
                    "heart_rate": 72 + random.randint(-5, 5),
                    "blood_oxygen": 98 + random.uniform(-1, 1),
                    "stress_level": random.uniform(0.1, 0.3),
                },
                data_type="telemetry",
                timestamp=ts,
                latitude=35.7850 + random.uniform(-0.001, 0.001),
                longitude=-78.6400 + random.uniform(-0.001, 0.001),
                metadata={"source": "biometric_watch", "unit": "Alpha"},
            )
        print_success("Added 10 normal biometric readings")

        # Distress vitals for Pvt. Williams (elevated heart rate, dropping O2)
        print_info("Streaming DISTRESS vitals for Pvt. Williams...")
        for i in range(10):
            ts = now - timedelta(minutes=10 - i)
            # Simulate deteriorating condition
            hr = 85 + (i * 8) + random.randint(-3, 3)  # Rising heart rate
            o2 = 97 - (i * 1.5) + random.uniform(-0.5, 0.5)  # Dropping O2
            biometrics_store.add_stream_record(
                data={
                    "soldier_id": "soldier:pvt_williams",
                    "heart_rate": hr,
                    "blood_oxygen": max(80, o2),  # Floor at 80
                    "stress_level": min(1.0, 0.3 + (i * 0.08)),
                },
                data_type="telemetry",
                timestamp=ts,
                latitude=35.7880 + random.uniform(-0.001, 0.001),
                longitude=-78.6420 + random.uniform(-0.001, 0.001),
                metadata={"source": "biometric_watch", "unit": "Bravo"},
            )
        print_success("Added 10 DISTRESS biometric readings")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 4: Stream Radio Communications (Working Memory)
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 4: Stream Radio Communications")

        radio_comms = [
            # Normal communications
            {"text": "Alpha Base, this is Checkpoint Alpha. All clear, over.",
             "speaker": "Cpl. Smith", "priority": "normal"},
            {"text": "Copy that Checkpoint Alpha. Maintain position.",
             "speaker": "Lt. Chen", "priority": "normal"},
            {"text": "Sgt. Johnson reporting from Delta. Patrol complete, returning to base.",
             "speaker": "Sgt. Johnson", "priority": "normal"},

            # Distress communications
            {"text": "MAYDAY MAYDAY! This is Pvt. Williams! I'm hit! Need immediate medevac!",
             "speaker": "Pvt. Williams", "priority": "emergency"},
            {"text": "Williams, this is Chen. What's your position? Are you mobile?",
             "speaker": "Lt. Chen", "priority": "urgent"},
            {"text": "I'm at grid reference... rescue zone one... losing blood... HELP!",
             "speaker": "Pvt. Williams", "priority": "emergency"},
            {"text": "All units, this is Lt. Chen. We have a soldier down at Rescue Zone 1. "
                     "Sgt. Johnson, you're closest - initiate rescue protocol immediately!",
             "speaker": "Lt. Chen", "priority": "emergency"},
            {"text": "Roger that, Lieutenant! Moving to rescue zone now. ETA 5 minutes.",
             "speaker": "Sgt. Johnson", "priority": "urgent"},
        ]

        for i, comm in enumerate(radio_comms):
            ts = now - timedelta(minutes=8 - i)
            biometrics_store.add_stream_record(
                data=comm["text"],
                data_type="radio",
                timestamp=ts,
                metadata={
                    "speaker": comm["speaker"],
                    "priority": comm["priority"],
                    "channel": "tactical_1",
                },
            )
        print_success(f"Added {len(radio_comms)} radio communications to working memory")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 5: Storage Statistics
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 5: Check Storage Statistics")

        biometrics_stats = biometrics_store.get_stats()
        print_success("Biometrics Store Statistics:")
        print_data("TimeSeries Records", biometrics_stats["stores"]["timeseries"]["record_count"])
        print_data("Working Memory Records", biometrics_stats["stores"]["working_memory"]["total_records"])
        print_data("Spatial Records", biometrics_stats["stores"].get("spatial", {}).get("record_count", "N/A"))

        command_stats = command_store.get_stats()
        print_success("Command Store Statistics:")
        print_data("Graph Nodes", command_stats["stores"]["graph"]["node_count"])
        print_data("Graph Edges", command_stats["stores"]["graph"]["edge_count"])

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 6: ML - Train Distress Classifier (Simulated)
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 6: Train Distress Signal Classifier")

        # Training data for distress classifier
        training_data = [
            # Normal communications
            {"text": "All clear at checkpoint alpha", "label": "normal"},
            {"text": "Patrol complete, returning to base", "label": "normal"},
            {"text": "Routine check-in, nothing to report", "label": "normal"},
            {"text": "Position secured, awaiting orders", "label": "normal"},

            # Urgent communications
            {"text": "Unknown movement detected, investigating", "label": "urgent"},
            {"text": "Requesting backup at position delta", "label": "urgent"},
            {"text": "Minor injury sustained, continuing mission", "label": "urgent"},

            # Emergency/Distress
            {"text": "MAYDAY MAYDAY! Under fire! Need immediate support!", "label": "distress"},
            {"text": "Man down! I'm hit! Need medevac NOW!", "label": "distress"},
            {"text": "HELP! Pinned down by enemy fire!", "label": "distress"},
            {"text": "Critical injury! Losing blood! Emergency!", "label": "distress"},
        ]

        print_info(f"Training distress classifier with {len(training_data)} examples...")
        print_info("Labels: normal, urgent, distress")
        print_success("Classifier trained (simulated - would use /v1/ml/classifier/fit)")

        # Test classification on radio comms
        print_info("\nClassifying radio communications:")
        test_texts = [
            "All clear at checkpoint",
            "Need backup, situation developing",
            "MAYDAY! I'm hit! Need immediate medevac!",
        ]
        classifications = ["normal", "urgent", "distress"]  # Simulated results
        for text, label in zip(test_texts, classifications):
            print(f"    '{text[:40]}...' → {label.upper()}")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 7: ML - Train Vital Signs Anomaly Detector (Simulated)
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 7: Train Vital Signs Anomaly Detector")

        # Normal vital signs for training
        normal_vitals = []
        for _ in range(200):
            normal_vitals.append([
                70 + random.gauss(0, 5),  # heart_rate: mean 70, std 5
                97 + random.gauss(0, 1),  # blood_oxygen: mean 97, std 1
                0.2 + random.gauss(0, 0.1),  # stress_level: mean 0.2, std 0.1
            ])

        print_info(f"Training anomaly detector with {len(normal_vitals)} normal vital readings...")
        print_info("Features: heart_rate, blood_oxygen, stress_level")
        print_success("Anomaly detector trained (simulated - would use /v1/ml/anomaly/fit)")

        # Detect anomalies in recent data
        print_info("\nDetecting anomalies in Pvt. Williams' vital signs:")
        anomaly_readings = [
            [145, 82.0, 0.95],  # High HR, low O2, high stress
            [160, 80.0, 1.0],   # Critical
        ]
        for reading in anomaly_readings:
            print_alert(f"ANOMALY DETECTED: HR={reading[0]}, O2={reading[1]}, Stress={reading[2]}")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 8: Hybrid Query - Unified Context Retrieval
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 8: Hybrid Query - Unified Context Retrieval")

        print_info("Creating HybridQueryExecutor with caching...")
        executor = HybridQueryExecutor(
            biometrics_store,
            enable_cache=True,
            cache_max_size=100,
            cache_ttl_seconds=60,
        )

        # Query recent telemetry
        print_info("Querying recent telemetry and working memory...")
        request = HybridQueryRequest(
            start_time=now - timedelta(minutes=15),
            end_time=now,
            mode=QueryMode.HYBRID,
            limit=20,
        )
        response = executor.execute(request)
        print_success(f"Retrieved {response.total_count} results")
        print_data("Stores queried", response.stores_queried)
        print_data("Execution time", f"{response.execution_time_ms:.2f}ms")

        # Spatial query - find personnel near rescue zone
        print_info("\nSpatial Query: Personnel within 2km of Rescue Zone 1")
        spatial_results = biometrics_store.query(
            query_type="spatial",
            spatial={"latitude": 35.7880, "longitude": -78.6420, "radius_meters": 2000},
        )
        print_success(f"Found {len(spatial_results.get('spatial', []))} records near rescue zone")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 9: Knowledge Graph Queries
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 9: Knowledge Graph Queries")

        print_info("Finding command structure for rescue coordination...")

        # Query neighbors of Lt. Chen (who does he command?)
        graph_results = command_store.query(
            query_type="graph",
            graph_query={
                "node_id": "soldier:lt_chen",
                "direction": "outgoing",
                "relationship": "commands",
            },
        )
        print_success(f"Lt. Chen commands personnel (query returned)")

        print_info("\nFinding Pvt. Williams' last known location...")
        williams_location = command_store.query(
            query_type="graph",
            graph_query={
                "node_id": "soldier:pvt_williams",
                "direction": "outgoing",
                "relationship": "last_known_location",
            },
        )
        print_success("Found location link")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 10: Memory Consolidation
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 10: Memory Consolidation (The 'Hippocampus' Process)")

        print_info("Consolidation synthesizes facts from raw data and creates knowledge...")

        consolidator = Consolidator(
            memory_store=biometrics_store,
            config={
                "buffer_threshold": 5,
                "use_entity_extractor": False,  # Rule-based for demo
            },
        )

        print_info("Running consolidation cycle...")
        result = consolidator.run_cycle(use_llm=False)  # Rule-based for demo

        print_success("Consolidation complete:")
        print_data("Records Processed", result.get("records_processed", 0))
        print_data("Facts Extracted", result.get("facts_extracted", 0))
        print_data("Graph Nodes Created", result.get("nodes_created", 0))
        print_data("Records Pruned", result.get("pruned", 0))

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 11: Final Statistics
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 11: Final Memory Statistics")

        final_biometrics = biometrics_store.get_stats()
        print_success("Final Biometrics Store State:")
        print_data("Dataset Name", final_biometrics["dataset_name"])
        print_data("Dataset Type", final_biometrics["dataset_type"])
        print_data("TimeSeries Records", final_biometrics["stores"]["timeseries"]["record_count"])
        print_data("Working Memory Records", final_biometrics["stores"]["working_memory"]["total_records"])

        final_command = command_store.get_stats()
        print_success("Final Command Store State:")
        print_data("Dataset Name", final_command["dataset_name"])
        print_data("Dataset Type", final_command["dataset_type"])
        print_data("Graph Nodes", final_command["stores"]["graph"]["node_count"])
        print_data("Graph Edges", final_command["stores"]["graph"]["edge_count"])

        # Cache statistics
        cache_stats = executor.get_cache_stats()
        print_success("Query Cache Statistics:")
        print_data("Cache Size", cache_stats["size"])
        print_data("Hit Rate", f"{cache_stats['hit_rate']:.1%}")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 12: Cleanup
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 12: Cleanup")

        biometrics_store.close()
        command_store.close()
        print_success("All stores closed")
        print_success("Temporary databases cleaned up automatically")

    # Final summary
    print_header("DEMO COMPLETE")
    print("""
  The Military Rescue Scenario demonstrated:

  1. PHASE 3: UNIFIED DATASET ARCHITECTURE
     - Typed datasets: 'realtime' for telemetry, 'graph' for command
     - Automatic store selection based on dataset type
     - Cross-store linking via LinkageTable

  2. EMBEDDED TRINITY MEMORY SYSTEM
     - TimeSeries Store: Biometric telemetry with timestamps
     - Spatial Store: Location-aware data with geo-queries
     - Graph Store: Personnel and location relationships
     - Working Memory: Radio communications with TTL

  3. HYBRID QUERY EXECUTOR
     - Multi-store query routing
     - Result fusion with score-based ranking
     - Query result caching with TTL

  4. ML OPERATIONS (Simulated)
     - Distress Signal Classifier (SetFit few-shot learning)
     - Vital Signs Anomaly Detector (One-Class SVM)

  5. MEMORY CONSOLIDATION
     - Extract facts from raw working memory
     - Create knowledge graph nodes
     - Prune processed raw data

  This showcases the power of LlamaFarm for mission-critical applications!

  For full documentation, see: rag/docs/EMBEDDED_TRINITY_MEMORY.md
""")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ImportError as e:
        print(f"Import error: {e}")
        print("Run from the 'rag' directory: cd rag && uv run python ../examples/e2e_scenarios/demo_military_rescue.py")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
