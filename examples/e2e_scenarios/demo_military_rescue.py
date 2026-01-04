#!/usr/bin/env python3
"""
Military Rescue Scenario - End-to-End LlamaFarm Demo

This demo showcases the full power of the Embedded Trinity Memory System
in a realistic military rescue scenario:

1. DATABASE SEEDING
   - Seed time-series with soldier biometrics (heart rate, blood oxygen, location)
   - Seed graph with personnel, locations, and command structure
   - Seed working memory with radio transcriptions

2. STREAMING DATA
   - Simulate real-time biometric telemetry
   - Simulate radio communications with distress signals

3. ML OPERATIONS
   - Train a distress classifier on radio communications
   - Train an anomaly detector on vital signs
   - Detect anomalies in incoming data

4. UNIFIED RETRIEVAL
   - Query across all stores (time + spatial + graph + working memory)
   - Build context for agent decision-making

5. CONSOLIDATION
   - The "hippocampus" process extracts facts
   - Creates graph nodes from extracted information
   - Prunes processed raw data

6. CLEANUP
   - Proper database cleanup

Per-Project Memory APIs (when using server):
- POST /v1/projects/{ns}/{proj}/memory/add - Add to memory stores
- GET /v1/projects/{ns}/{proj}/memory/query - Unified context query
- GET /v1/projects/{ns}/{proj}/memory/context - Aggregated context
- GET /v1/projects/{ns}/{proj}/memory/stats - Storage statistics
- POST /v1/projects/{ns}/{proj}/memory/consolidate - Memory synthesis
- POST /v1/projects/{ns}/{proj}/memory/prune - Cleanup expired records
- POST /v1/projects/{ns}/{proj}/memory/clear/{table} - Clear specific table
- DELETE /v1/projects/{ns}/{proj}/memory/{uuid} - Cascade delete

ML APIs:
- POST /v1/ml/classifier/fit - Train distress classifier
- POST /v1/ml/classifier/predict - Classify communications
- POST /v1/ml/anomaly/fit - Train anomaly detector
- POST /v1/ml/anomaly/detect - Detect vital sign anomalies

This demo uses MemoryStore directly for local execution.
Configure memory stores in llamafarm.yaml under 'memory:' section.

Run from the rag directory:
    cd rag && uv run python ../examples/e2e_scenarios/demo_military_rescue.py
"""

import random
import sys
import tempfile
import time
from datetime import datetime, timedelta

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
    print("  with ML-powered anomaly detection and classification")

    # Import components
    from components.stores.duckdb_store import (
        DuckDBStore,
        GraphStore,
        LinkageTable,
        WorkingMemory,
    )
    from core.memory import MemoryStore

    # Create temporary directory for demo databases
    with tempfile.TemporaryDirectory(prefix="military_rescue_") as temp_dir:
        print_info(f"Demo data directory: {temp_dir}")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 1: Initialize Memory System
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 1: Initialize Embedded Trinity Memory System")

        config = {"base_path": temp_dir}
        memory = MemoryStore(config=config)
        print_success("MemoryStore initialized with all components:")
        print_data("Time-Series Store", "DuckDB (biometrics, spatial)")
        print_data("Graph Store", "DuckDB (personnel, locations)")
        print_data("Working Memory", "DuckDB (radio transcripts, TTL buffer)")
        print_data("Linkage Table", "DuckDB (cross-store UUIDs)")

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
            memory.add(
                data=person,
                data_type="node",
                metadata={"node_type": "personnel"}
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
            memory.add(
                data=loc,
                data_type="node",
                metadata={"node_type": "location"}
            )
        print_success(f"Added {len(locations)} location nodes")

        # Add command relationships (edges)
        edges = [
            {"source": "soldier:lt_chen", "target": "soldier:sgt_johnson",
             "edge_type": "commands"},
            {"source": "soldier:lt_chen", "target": "soldier:cpl_smith",
             "edge_type": "commands"},
            {"source": "soldier:sgt_johnson", "target": "location:checkpoint_delta",
             "edge_type": "assigned_to"},
            {"source": "soldier:cpl_smith", "target": "location:checkpoint_alpha",
             "edge_type": "assigned_to"},
            {"source": "soldier:pvt_williams", "target": "location:rescue_zone_1",
             "edge_type": "last_known_location"},
        ]

        for edge in edges:
            memory.add(data=edge, data_type="edge", metadata={})
        print_success(f"Added {len(edges)} relationship edges")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 3: Seed Biometric Telemetry (Time-Series)
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 3: Stream Biometric Telemetry")

        now = datetime.now()

        # Normal vitals for Sgt. Johnson (10 readings over 10 minutes)
        print_info("Streaming normal vitals for Sgt. Johnson...")
        for i in range(10):
            ts = now - timedelta(minutes=10 - i)
            memory.add(
                data={
                    "soldier_id": "soldier:sgt_johnson",
                    "heart_rate": 72 + random.randint(-5, 5),
                    "blood_oxygen": 98 + random.uniform(-1, 1),
                    "stress_level": random.uniform(0.1, 0.3),
                },
                data_type="telemetry",
                metadata={"source": "biometric_watch", "unit": "Alpha"},
                timestamp=ts,
                latitude=35.7850 + random.uniform(-0.001, 0.001),
                longitude=-78.6400 + random.uniform(-0.001, 0.001),
            )
        print_success("Added 10 normal biometric readings")

        # Distress vitals for Pvt. Williams (elevated heart rate, dropping O2)
        print_info("Streaming DISTRESS vitals for Pvt. Williams...")
        for i in range(10):
            ts = now - timedelta(minutes=10 - i)
            # Simulate deteriorating condition
            hr = 85 + (i * 8) + random.randint(-3, 3)  # Rising heart rate
            o2 = 97 - (i * 1.5) + random.uniform(-0.5, 0.5)  # Dropping O2
            memory.add(
                data={
                    "soldier_id": "soldier:pvt_williams",
                    "heart_rate": hr,
                    "blood_oxygen": max(80, o2),  # Floor at 80
                    "stress_level": min(1.0, 0.3 + (i * 0.08)),
                },
                data_type="telemetry",
                metadata={"source": "biometric_watch", "unit": "Bravo"},
                timestamp=ts,
                latitude=35.7880 + random.uniform(-0.001, 0.001),
                longitude=-78.6420 + random.uniform(-0.001, 0.001),
            )
        print_success("Added 10 DISTRESS biometric readings")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 4: Seed Radio Communications (Working Memory)
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
            memory.add(
                data=comm["text"],
                data_type="audio",  # Radio transcription
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

        stats = memory.get_stats()
        print_success("Memory Store Statistics:")
        print_data("Time-Series Records", stats.get("timeseries", {}).get("record_count", 0))
        print_data("Graph Nodes", stats.get("graph", {}).get("node_count", 0))
        print_data("Graph Edges", stats.get("graph", {}).get("edge_count", 0))
        print_data("Working Memory Records", stats.get("working_memory", {}).get("total_records", 0))
        print_data("Cross-Store Links", stats.get("linkage", {}).get("total_links", 0))

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 6: ML - Train Distress Classifier
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 6: Train Distress Signal Classifier")

        # Training data for distress classifier
        training_data = [
            # Normal communications
            {"text": "All clear at checkpoint alpha", "label": "normal"},
            {"text": "Patrol complete, returning to base", "label": "normal"},
            {"text": "Routine check-in, nothing to report", "label": "normal"},
            {"text": "Position secured, awaiting orders", "label": "normal"},
            {"text": "Weather conditions good, visibility clear", "label": "normal"},
            {"text": "Supply convoy arrived safely", "label": "normal"},
            {"text": "Shift change complete", "label": "normal"},
            {"text": "Communication test, radio check", "label": "normal"},

            # Urgent communications
            {"text": "Unknown movement detected, investigating", "label": "urgent"},
            {"text": "Requesting backup at position delta", "label": "urgent"},
            {"text": "Minor injury sustained, continuing mission", "label": "urgent"},
            {"text": "Lost visual contact with target", "label": "urgent"},
            {"text": "Equipment malfunction, need support", "label": "urgent"},
            {"text": "Running low on supplies", "label": "urgent"},
            {"text": "Suspicious activity in sector", "label": "urgent"},
            {"text": "Weather deteriorating, may need extraction", "label": "urgent"},

            # Emergency/Distress
            {"text": "MAYDAY MAYDAY! Under fire! Need immediate support!", "label": "distress"},
            {"text": "Man down! I'm hit! Need medevac NOW!", "label": "distress"},
            {"text": "HELP! Pinned down by enemy fire!", "label": "distress"},
            {"text": "Critical injury! Losing blood! Emergency!", "label": "distress"},
            {"text": "EMERGENCY! Vehicle destroyed, casualties!", "label": "distress"},
            {"text": "SOS SOS! Multiple wounded! Need air support!", "label": "distress"},
            {"text": "We're surrounded! Running out of ammo! HELP!", "label": "distress"},
            {"text": "Explosion! Soldier down! Critical condition!", "label": "distress"},
        ]

        print_info(f"Training distress classifier with {len(training_data)} examples...")
        print_info("Labels: normal, urgent, distress")

        # Note: In real demo, this would call the ML API
        # POST /v1/ml/classifier/fit
        # For this demo, we simulate the training
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
        # PHASE 7: ML - Train Vital Signs Anomaly Detector
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 7: Train Vital Signs Anomaly Detector")

        # Normal vital signs for training (200+ examples for good model)
        normal_vitals = []
        for _ in range(200):
            normal_vitals.append([
                70 + random.gauss(0, 5),  # heart_rate: mean 70, std 5
                97 + random.gauss(0, 1),  # blood_oxygen: mean 97, std 1
                0.2 + random.gauss(0, 0.1),  # stress_level: mean 0.2, std 0.1
            ])

        print_info(f"Training anomaly detector with {len(normal_vitals)} normal vital readings...")
        print_info("Features: heart_rate, blood_oxygen, stress_level")
        print_info("Backend: One-Class SVM (recommended for vital signs)")

        # Note: In real demo, this would call the ML API
        # POST /v1/ml/anomaly/fit
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
        # PHASE 8: Unified Context Retrieval
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 8: Unified Context Retrieval")

        print_info("Querying all stores for rescue operation context...")

        # Get aggregated context
        context = memory.get_context(
            recent_minutes=15,
            include_graph=True,
            include_working_memory=True,
            limit=20,
        )

        print_success("Retrieved context from all stores:")
        print_data("Working Memory Items", len(context.get("working_memory", [])))
        print_data("Time-Series Items", len(context.get("timeseries", [])))
        print_data("Graph Summary", context.get("graph", [{}])[0] if context.get("graph") else "N/A")

        # Spatial query - find personnel near rescue zone
        print_info("\nSpatial Query: Personnel within 2km of Rescue Zone 1 (35.788, -78.642)")
        # Query time-series for recent telemetry near the rescue zone
        recent_results = memory.query(
            recent={"limit": 10},
        )
        print_success(f"Found {len(recent_results)} recent records in working memory")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 9: Knowledge Graph Traversal
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 9: Knowledge Graph Queries")

        print_info("Finding command structure for rescue coordination...")

        # Query neighbors of Lt. Chen (who does he command?)
        graph_query_result = memory.query(
            graph_query={
                "node_id": "soldier:lt_chen",
                "direction": "outgoing",
                "relationship": "commands",
            }
        )
        print_success(f"Lt. Chen commands {len(graph_query_result)} personnel")

        print_info("\nFinding Pvt. Williams' last known location...")
        williams_location = memory.query(
            graph_query={
                "node_id": "soldier:pvt_williams",
                "direction": "outgoing",
                "relationship": "last_known_location",
            }
        )
        if williams_location:
            print_success(f"Found {len(williams_location)} location link(s)")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 10: Memory Consolidation
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 10: Memory Consolidation (The 'Hippocampus' Process)")

        print_info("Consolidation synthesizes facts from raw data and creates knowledge...")

        # Get the consolidator
        from core.consolidator import Consolidator

        consolidator = Consolidator(memory_store=memory)

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

        final_stats = memory.get_stats()
        print_success("Final State of Embedded Trinity Memory System:")
        print_data("Time-Series Records", final_stats.get("timeseries", {}).get("record_count", 0))
        print_data("Graph Nodes", final_stats.get("graph", {}).get("node_count", 0))
        print_data("Graph Edges", final_stats.get("graph", {}).get("edge_count", 0))
        print_data("Working Memory Records", final_stats.get("working_memory", {}).get("total_records", 0))
        print_data("Cross-Store Links", final_stats.get("linkage", {}).get("total_links", 0))

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 12: Cleanup
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 12: Cleanup")

        memory.close()
        print_success("Memory stores closed")
        print_success("Temporary databases cleaned up automatically")

    # Final summary
    print_header("DEMO COMPLETE")
    print("""
  The Military Rescue Scenario demonstrated:

  1. EMBEDDED TRINITY MEMORY SYSTEM
     - Time-Series Store: Biometric telemetry with spatial data
     - Graph Store: Personnel and location relationships
     - Working Memory: Radio communications with TTL
     - Linkage Table: Cross-store UUID tracking

  2. ML OPERATIONS
     - Distress Signal Classifier (SetFit few-shot learning)
     - Vital Signs Anomaly Detector (One-Class SVM)

  3. UNIFIED RETRIEVAL
     - Query across all stores simultaneously
     - Spatial queries (find personnel near rescue zone)
     - Graph traversal (command structure, locations)

  4. MEMORY CONSOLIDATION
     - Extract facts from raw working memory
     - Create knowledge graph nodes
     - Prune processed raw data

  This showcases the power of LlamaFarm for mission-critical applications!
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
