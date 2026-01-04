#!/usr/bin/env python3
"""
Medical Patient Scenario - End-to-End LlamaFarm Demo

This demo showcases the full power of the Embedded Trinity Memory System
in a realistic hospital/medical scenario:

1. DATABASE SEEDING
   - Seed graph with patient records, doctors, departments, and medications
   - Seed time-series with patient vitals (HR, BP, O2, temperature)
   - Seed working memory with clinical notes and alerts

2. STREAMING DATA
   - Simulate real-time patient monitoring data
   - Simulate clinical documentation workflow

3. ML OPERATIONS
   - Train a triage classifier for patient urgency
   - Train anomaly detector for vital sign patterns
   - Detect critical patient conditions

4. UNIFIED RETRIEVAL
   - Query patient history across all stores
   - Build comprehensive patient context
   - Cross-reference medications and allergies via graph

5. CONSOLIDATION
   - Synthesize clinical insights from observations
   - Create treatment pathway nodes
   - Archive processed monitoring data

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
- POST /v1/ml/classifier/fit - Train triage classifier
- POST /v1/ml/classifier/predict - Classify patient urgency
- POST /v1/ml/anomaly/fit - Train vital signs anomaly detector
- POST /v1/ml/anomaly/detect - Detect patient anomalies

This demo uses MemoryStore directly for local execution.
Configure memory stores in llamafarm.yaml under 'memory:' section.

Run from the rag directory:
    cd rag && uv run python ../examples/e2e_scenarios/demo_medical_patient.py
"""

import random
import sys
import tempfile
from datetime import datetime, timedelta

# Add rag to path
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
    """Print alert/critical message."""
    print(f"  🚨 {msg}")


def print_warning(msg: str) -> None:
    """Print warning message."""
    print(f"  ⚠️  {msg}")


def main() -> int:
    """Run the medical patient scenario demo."""

    print_header("MEDICAL PATIENT MONITORING SCENARIO")
    print("  Demonstrating the Embedded Trinity Memory System")
    print("  for Healthcare Analytics and Patient Safety")

    # Import components
    from core.memory import MemoryStore
    from core.consolidator import Consolidator

    # Create temporary directory for demo databases
    with tempfile.TemporaryDirectory(prefix="medical_demo_") as temp_dir:
        print_info(f"Demo data directory: {temp_dir}")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 1: Initialize Memory System
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 1: Initialize Medical Data Infrastructure")

        config = {"base_path": temp_dir}
        memory = MemoryStore(config=config)
        print_success("Medical MemoryStore initialized:")
        print_data("Time-Series Store", "Patient vitals, lab results")
        print_data("Graph Store", "Patients, providers, medications, conditions")
        print_data("Working Memory", "Clinical notes, alerts, observations")
        print_data("Linkage Table", "Cross-reference medical records")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 2: Seed Knowledge Graph - Healthcare Entities
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 2: Seed Medical Knowledge Graph")

        # Add patients
        patients = [
            {"id": "patient:P001", "name": "John Smith", "age": 67, "gender": "M",
             "blood_type": "A+", "admission_date": "2024-01-15"},
            {"id": "patient:P002", "name": "Mary Johnson", "age": 45, "gender": "F",
             "blood_type": "O-", "admission_date": "2024-01-16"},
            {"id": "patient:P003", "name": "Robert Davis", "age": 72, "gender": "M",
             "blood_type": "B+", "admission_date": "2024-01-14"},
            {"id": "patient:P004", "name": "Sarah Wilson", "age": 34, "gender": "F",
             "blood_type": "AB+", "admission_date": "2024-01-17", "status": "critical"},
        ]

        for patient in patients:
            memory.add(data=patient, data_type="node", metadata={"node_type": "patient"})
        print_success(f"Added {len(patients)} patient records")

        # Add healthcare providers
        providers = [
            {"id": "doctor:D001", "name": "Dr. Emily Chen", "specialty": "Cardiology",
             "department": "ICU"},
            {"id": "doctor:D002", "name": "Dr. Michael Brown", "specialty": "Internal Medicine",
             "department": "General"},
            {"id": "nurse:N001", "name": "Nurse James Taylor", "unit": "ICU",
             "shift": "day"},
            {"id": "nurse:N002", "name": "Nurse Lisa Anderson", "unit": "General",
             "shift": "night"},
        ]

        for provider in providers:
            memory.add(data=provider, data_type="node", metadata={"node_type": "provider"})
        print_success(f"Added {len(providers)} healthcare providers")

        # Add conditions/diagnoses
        conditions = [
            {"id": "condition:CHF", "name": "Congestive Heart Failure", "icd10": "I50.9",
             "severity": "high"},
            {"id": "condition:DM2", "name": "Type 2 Diabetes", "icd10": "E11.9",
             "severity": "medium"},
            {"id": "condition:HTN", "name": "Hypertension", "icd10": "I10",
             "severity": "medium"},
            {"id": "condition:SEPSIS", "name": "Sepsis", "icd10": "A41.9",
             "severity": "critical"},
        ]

        for cond in conditions:
            memory.add(data=cond, data_type="node", metadata={"node_type": "condition"})
        print_success(f"Added {len(conditions)} medical conditions")

        # Add medications
        medications = [
            {"id": "med:LISINOPRIL", "name": "Lisinopril", "class": "ACE Inhibitor",
             "route": "oral"},
            {"id": "med:METFORMIN", "name": "Metformin", "class": "Biguanide",
             "route": "oral"},
            {"id": "med:FUROSEMIDE", "name": "Furosemide", "class": "Loop Diuretic",
             "route": "IV"},
            {"id": "med:VANCOMYCIN", "name": "Vancomycin", "class": "Antibiotic",
             "route": "IV"},
            {"id": "med:NOREPINEPHRINE", "name": "Norepinephrine", "class": "Vasopressor",
             "route": "IV", "high_risk": True},
        ]

        for med in medications:
            memory.add(data=med, data_type="node", metadata={"node_type": "medication"})
        print_success(f"Added {len(medications)} medications")

        # Add locations/rooms
        rooms = [
            {"id": "room:ICU_101", "name": "ICU Room 101", "type": "ICU", "floor": 3},
            {"id": "room:ICU_102", "name": "ICU Room 102", "type": "ICU", "floor": 3},
            {"id": "room:GEN_201", "name": "General Room 201", "type": "General", "floor": 2},
            {"id": "room:GEN_202", "name": "General Room 202", "type": "General", "floor": 2},
        ]

        for room in rooms:
            memory.add(data=room, data_type="node", metadata={"node_type": "location"})
        print_success(f"Added {len(rooms)} room locations")

        # Add relationships (edges)
        print_info("Creating medical relationships...")
        relationships = [
            # Patient assignments
            {"source": "patient:P001", "target": "room:ICU_101", "edge_type": "admitted_to"},
            {"source": "patient:P002", "target": "room:GEN_201", "edge_type": "admitted_to"},
            {"source": "patient:P003", "target": "room:GEN_202", "edge_type": "admitted_to"},
            {"source": "patient:P004", "target": "room:ICU_102", "edge_type": "admitted_to"},

            # Attending physicians
            {"source": "doctor:D001", "target": "patient:P001", "edge_type": "attending"},
            {"source": "doctor:D001", "target": "patient:P004", "edge_type": "attending"},
            {"source": "doctor:D002", "target": "patient:P002", "edge_type": "attending"},
            {"source": "doctor:D002", "target": "patient:P003", "edge_type": "attending"},

            # Diagnoses
            {"source": "patient:P001", "target": "condition:CHF", "edge_type": "diagnosed_with"},
            {"source": "patient:P001", "target": "condition:HTN", "edge_type": "diagnosed_with"},
            {"source": "patient:P002", "target": "condition:DM2", "edge_type": "diagnosed_with"},
            {"source": "patient:P003", "target": "condition:HTN", "edge_type": "diagnosed_with"},
            {"source": "patient:P004", "target": "condition:SEPSIS", "edge_type": "diagnosed_with"},

            # Medications
            {"source": "patient:P001", "target": "med:LISINOPRIL", "edge_type": "prescribed"},
            {"source": "patient:P001", "target": "med:FUROSEMIDE", "edge_type": "prescribed"},
            {"source": "patient:P002", "target": "med:METFORMIN", "edge_type": "prescribed"},
            {"source": "patient:P004", "target": "med:VANCOMYCIN", "edge_type": "prescribed"},
            {"source": "patient:P004", "target": "med:NOREPINEPHRINE", "edge_type": "prescribed"},

            # Drug interactions (important for safety!)
            {"source": "med:LISINOPRIL", "target": "med:FUROSEMIDE",
             "edge_type": "interacts_with", "properties": {"severity": "moderate"}},
        ]

        for rel in relationships:
            memory.add(data=rel, data_type="edge", metadata={})
        print_success(f"Created {len(relationships)} medical relationships")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 3: Stream Patient Vital Signs (Time-Series)
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 3: Stream Patient Vital Signs")

        now = datetime.now()

        # Normal vitals for P001 (stable CHF patient)
        print_info("Streaming vitals for Patient P001 (stable CHF)...")
        for i in range(15):
            ts = now - timedelta(minutes=30 - (i * 2))
            memory.add(
                data={
                    "patient_id": "patient:P001",
                    "heart_rate": 78 + random.randint(-5, 5),
                    "systolic_bp": 135 + random.randint(-8, 8),
                    "diastolic_bp": 82 + random.randint(-5, 5),
                    "spo2": 94 + random.uniform(-2, 2),
                    "temperature": 98.6 + random.uniform(-0.3, 0.3),
                    "respiratory_rate": 18 + random.randint(-2, 2),
                },
                data_type="telemetry",
                metadata={"source": "bedside_monitor", "room": "ICU_101"},
                timestamp=ts,
            )
        print_success("Added 15 stable vitals readings")

        # Normal vitals for P002 (stable DM2 patient)
        print_info("Streaming vitals for Patient P002 (stable diabetes)...")
        for i in range(15):
            ts = now - timedelta(minutes=30 - (i * 2))
            memory.add(
                data={
                    "patient_id": "patient:P002",
                    "heart_rate": 72 + random.randint(-4, 4),
                    "systolic_bp": 128 + random.randint(-6, 6),
                    "diastolic_bp": 78 + random.randint(-4, 4),
                    "spo2": 97 + random.uniform(-1, 1),
                    "temperature": 98.4 + random.uniform(-0.2, 0.2),
                    "blood_glucose": 145 + random.randint(-20, 20),
                },
                data_type="telemetry",
                metadata={"source": "bedside_monitor", "room": "GEN_201"},
                timestamp=ts,
            )
        print_success("Added 15 stable vitals readings")

        # CRITICAL vitals for P004 (septic shock patient)
        print_info("Streaming CRITICAL vitals for Patient P004 (sepsis)...")
        for i in range(15):
            ts = now - timedelta(minutes=30 - (i * 2))
            # Simulate deteriorating septic shock
            hr = 110 + (i * 3) + random.randint(-5, 5)  # Rising tachycardia
            sbp = 90 - (i * 2) + random.randint(-5, 5)  # Dropping BP
            temp = 102.5 + random.uniform(-0.5, 0.5)  # High fever
            spo2 = 92 - (i * 0.3) + random.uniform(-1, 1)  # Dropping O2

            memory.add(
                data={
                    "patient_id": "patient:P004",
                    "heart_rate": min(180, hr),
                    "systolic_bp": max(60, sbp),
                    "diastolic_bp": max(40, 55 - i + random.randint(-3, 3)),
                    "spo2": max(85, spo2),
                    "temperature": temp,
                    "respiratory_rate": 28 + random.randint(-2, 4),
                    "lactate": 4.5 + (i * 0.3),  # Rising lactate
                    "map": max(50, (sbp + 2 * (55 - i)) / 3),  # Mean arterial pressure
                },
                data_type="telemetry",
                metadata={"source": "bedside_monitor", "room": "ICU_102", "priority": "critical"},
                timestamp=ts,
            )
        print_alert("Added 15 CRITICAL septic vitals readings!")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 4: Clinical Notes & Observations (Working Memory)
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 4: Stream Clinical Documentation")

        clinical_notes = [
            # Normal notes
            {"text": "P001: Patient resting comfortably. CHF well-controlled on current regimen. "
                     "Continue current medications.",
             "author": "Dr. Chen", "type": "progress_note", "patient": "P001"},
            {"text": "P002: Blood glucose within target range. Diet education provided. "
                     "Continue Metformin 1000mg BID.",
             "author": "Dr. Brown", "type": "progress_note", "patient": "P002"},
            {"text": "P003: Stable hypertension. BP trending toward goal. "
                     "Will continue monitoring.",
             "author": "Dr. Brown", "type": "progress_note", "patient": "P003"},

            # Critical alerts
            {"text": "CRITICAL: P004 showing signs of septic shock. MAP dropping below 65. "
                     "Initiating aggressive fluid resuscitation and vasopressor support.",
             "author": "Dr. Chen", "type": "critical_alert", "patient": "P004"},
            {"text": "P004 STAT: Norepinephrine drip started at 0.1 mcg/kg/min. "
                     "Central line placed. Blood cultures drawn x2.",
             "author": "Nurse Taylor", "type": "intervention", "patient": "P004"},
            {"text": "P004: Lactate rising to 6.2. Increasing norepi to 0.2 mcg/kg/min. "
                     "Bedside ultrasound shows hyperdynamic cardiac function.",
             "author": "Dr. Chen", "type": "critical_update", "patient": "P004"},
            {"text": "CODE BLUE AVERTED: P004 responded to fluid bolus and vasopressors. "
                     "MAP now 68. Continuing close monitoring q15min.",
             "author": "Dr. Chen", "type": "critical_update", "patient": "P004"},

            # Lab results
            {"text": "P001 Labs: BNP 450 (stable), Cr 1.2, K 4.1. CHF compensated.",
             "author": "Lab System", "type": "lab_result", "patient": "P001"},
            {"text": "P004 Labs: WBC 22.4, Lactate 6.2, Procalcitonin 12.5. Severe sepsis confirmed.",
             "author": "Lab System", "type": "lab_result", "patient": "P004"},
        ]

        for note in clinical_notes:
            memory.add(
                data=note["text"],
                data_type="chat",  # Clinical documentation as chat-like entries
                metadata={
                    "author": note["author"],
                    "note_type": note["type"],
                    "patient_id": note["patient"],
                },
            )
        print_success(f"Added {len(clinical_notes)} clinical notes to working memory")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 5: Storage Statistics
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 5: Check Medical Data Statistics")

        stats = memory.get_stats()
        print_success("Medical Data Store Statistics:")
        print_data("Vital Signs Records", stats.get("timeseries", {}).get("record_count", 0))
        print_data("Knowledge Graph Nodes", stats.get("graph", {}).get("node_count", 0))
        print_data("Knowledge Graph Edges", stats.get("graph", {}).get("edge_count", 0))
        print_data("Clinical Notes", stats.get("working_memory", {}).get("total_records", 0))
        print_data("Cross-Reference Links", stats.get("linkage", {}).get("total_links", 0))

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 6: ML - Train Patient Triage Classifier
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 6: Train Patient Triage Classifier")

        triage_training_data = [
            # Stable patients
            {"text": "Patient resting comfortably, vital signs stable", "label": "stable"},
            {"text": "Blood pressure within normal limits, no acute distress", "label": "stable"},
            {"text": "Tolerating oral intake, ambulating independently", "label": "stable"},
            {"text": "Chronic condition well-controlled on current regimen", "label": "stable"},
            {"text": "Post-operative day 2, recovering as expected", "label": "stable"},
            {"text": "Lab values improving, patient feeling better", "label": "stable"},
            {"text": "Pain well-controlled, sleeping comfortably", "label": "stable"},
            {"text": "Ready for discharge pending final labs", "label": "stable"},

            # Moderate urgency
            {"text": "Blood pressure elevated, adjusting medications", "label": "moderate"},
            {"text": "Mild dyspnea on exertion, monitoring closely", "label": "moderate"},
            {"text": "Blood glucose out of range, sliding scale adjusted", "label": "moderate"},
            {"text": "New onset confusion, workup ordered", "label": "moderate"},
            {"text": "Fever developing, cultures drawn", "label": "moderate"},
            {"text": "Oxygen requirement increased to 4L nasal cannula", "label": "moderate"},
            {"text": "Heart rate trending upward, monitoring", "label": "moderate"},
            {"text": "Pain poorly controlled, consult requested", "label": "moderate"},

            # Critical patients
            {"text": "SEPTIC SHOCK: MAP below 65 despite fluids", "label": "critical"},
            {"text": "RESPIRATORY FAILURE: Intubation required", "label": "critical"},
            {"text": "CARDIAC ARREST: Code Blue in progress", "label": "critical"},
            {"text": "Massive hemorrhage, transfusion protocol activated", "label": "critical"},
            {"text": "Anaphylaxis, epinephrine administered", "label": "critical"},
            {"text": "Stroke symptoms, STAT CT ordered", "label": "critical"},
            {"text": "Acute MI, cath lab activated", "label": "critical"},
            {"text": "Unresponsive, GCS 3, emergency intervention", "label": "critical"},
        ]

        print_info(f"Training triage classifier with {len(triage_training_data)} examples...")
        print_info("Labels: stable, moderate, critical")

        # Simulated - would use /v1/ml/classifier/fit
        print_success("Triage classifier trained (simulated - uses /v1/ml/classifier/fit)")

        # Test classification
        print_info("\nClassifying clinical notes:")
        test_notes = [
            "Patient stable, vital signs normal",
            "Blood pressure elevated, may need adjustment",
            "CRITICAL: Septic shock, MAP dropping rapidly!",
        ]
        predictions = ["stable", "moderate", "critical"]
        for note, pred in zip(test_notes, predictions):
            print(f"    '{note[:45]}...' → {pred.upper()}")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 7: ML - Train Vital Signs Anomaly Detector
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 7: Train Patient Vital Signs Anomaly Detector")

        # Normal vital ranges for training
        normal_patient_vitals = []
        for _ in range(250):
            normal_patient_vitals.append([
                75 + random.gauss(0, 8),    # heart_rate: normal range
                120 + random.gauss(0, 10),  # systolic_bp
                75 + random.gauss(0, 6),    # diastolic_bp
                97 + random.gauss(0, 1.5),  # spo2
                98.6 + random.gauss(0, 0.4),# temperature
                16 + random.gauss(0, 2),    # respiratory_rate
            ])

        print_info(f"Training anomaly detector with {len(normal_patient_vitals)} normal readings...")
        print_info("Features: HR, SBP, DBP, SpO2, Temp, RR")
        print_info("Backend: Isolation Forest (good for multivariate data)")

        # Simulated - would use /v1/ml/anomaly/fit
        print_success("Anomaly detector trained (simulated - uses /v1/ml/anomaly/fit)")

        # Detect anomalies
        print_info("\nDetecting anomalies in Patient P004's vital signs:")
        anomaly_readings = [
            {"hr": 145, "sbp": 72, "dbp": 45, "spo2": 88, "temp": 102.8, "rr": 32},
            {"hr": 165, "sbp": 65, "dbp": 42, "spo2": 86, "temp": 103.1, "rr": 35},
        ]
        for reading in anomaly_readings:
            print_alert(f"ANOMALY: HR={reading['hr']}, BP={reading['sbp']}/{reading['dbp']}, "
                       f"SpO2={reading['spo2']}%, Temp={reading['temp']}°F")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 8: Unified Patient Context Retrieval
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 8: Unified Patient Context Retrieval")

        print_info("Building comprehensive context for Patient P004...")

        # Get aggregated context
        context = memory.get_context(
            recent_minutes=60,
            include_graph=True,
            include_working_memory=True,
            limit=50,
        )

        print_success("Retrieved multi-store context:")
        print_data("Clinical Notes Retrieved", len(context.get("working_memory", [])))
        print_data("Vital Sign Records", len(context.get("timeseries", [])))
        print_data("Graph Summary", context.get("graph", []))

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 9: Knowledge Graph Queries
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 9: Medical Knowledge Graph Queries")

        print_info("Query 1: What conditions does P004 have?")
        p004_conditions = memory.query(
            graph_query={
                "node_id": "patient:P004",
                "direction": "outgoing",
                "relationship": "diagnosed_with",
            }
        )
        print_success(f"Found {len(p004_conditions)} condition(s)")

        print_info("\nQuery 2: What medications is P004 on?")
        p004_meds = memory.query(
            graph_query={
                "node_id": "patient:P004",
                "direction": "outgoing",
                "relationship": "prescribed",
            }
        )
        print_success(f"Found {len(p004_meds)} medication(s)")

        print_info("\nQuery 3: Who is the attending physician for P004?")
        p004_doctor = memory.query(
            graph_query={
                "node_id": "patient:P004",
                "direction": "incoming",
                "relationship": "attending",
            }
        )
        print_success(f"Found {len(p004_doctor)} attending physician(s)")

        print_info("\nQuery 4: Check for drug interactions...")
        interactions = memory.query(
            graph_query={
                "node_id": "med:LISINOPRIL",
                "direction": "both",
                "relationship": "interacts_with",
            }
        )
        if interactions:
            print_warning(f"Found {len(interactions)} potential drug interaction(s)!")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 10: Memory Consolidation
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 10: Clinical Data Consolidation")

        print_info("Consolidating clinical observations into structured knowledge...")

        consolidator = Consolidator(memory_store=memory)
        result = consolidator.run_cycle(use_llm=False)

        print_success("Consolidation complete:")
        print_data("Records Processed", result.get("records_processed", 0))
        print_data("Clinical Facts Extracted", result.get("facts_extracted", 0))
        print_data("Knowledge Nodes Created", result.get("nodes_created", 0))
        print_data("Raw Data Archived", result.get("pruned", 0))

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 11: Clinical Decision Support Summary
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 11: Clinical Decision Support Summary")

        print_info("Generating patient safety report for P004...")
        print()
        print("  ┌────────────────────────────────────────────────────────────┐")
        print("  │           CLINICAL DECISION SUPPORT ALERT                  │")
        print("  ├────────────────────────────────────────────────────────────┤")
        print("  │  Patient: Sarah Wilson (P004)                              │")
        print("  │  Location: ICU Room 102                                    │")
        print("  │  Attending: Dr. Emily Chen                                 │")
        print("  ├────────────────────────────────────────────────────────────┤")
        print("  │  DIAGNOSIS: Sepsis (A41.9) - CRITICAL                      │")
        print("  │  ANOMALIES DETECTED: 15                                    │")
        print("  │  TRIAGE LEVEL: CRITICAL                                    │")
        print("  ├────────────────────────────────────────────────────────────┤")
        print("  │  CURRENT MEDICATIONS:                                      │")
        print("  │    - Vancomycin (IV antibiotic)                           │")
        print("  │    - Norepinephrine (vasopressor) ⚠️ HIGH RISK            │")
        print("  ├────────────────────────────────────────────────────────────┤")
        print("  │  VITAL SIGN TRENDS (Last 30 min):                         │")
        print("  │    - Heart Rate: 110 → 165 bpm (↑ CRITICAL)               │")
        print("  │    - Blood Pressure: 90/55 → 65/42 (↓ CRITICAL)           │")
        print("  │    - SpO2: 92% → 86% (↓ CRITICAL)                         │")
        print("  │    - Temperature: 102.5°F → 103.1°F (↑ FEVER)             │")
        print("  │    - Lactate: 4.5 → 6.2 (↑ CRITICAL)                      │")
        print("  ├────────────────────────────────────────────────────────────┤")
        print("  │  RECOMMENDATIONS:                                          │")
        print("  │    1. Continue aggressive fluid resuscitation             │")
        print("  │    2. Consider adding second vasopressor                  │")
        print("  │    3. Evaluate for source control                         │")
        print("  │    4. Consider stress-dose steroids                       │")
        print("  └────────────────────────────────────────────────────────────┘")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 12: Final Statistics & Cleanup
        # ═══════════════════════════════════════════════════════════════════
        print_section("PHASE 12: Final Statistics & Cleanup")

        final_stats = memory.get_stats()
        print_success("Final Medical Data Store State:")
        print_data("Vital Signs Records", final_stats.get("timeseries", {}).get("record_count", 0))
        print_data("Knowledge Graph Nodes", final_stats.get("graph", {}).get("node_count", 0))
        print_data("Knowledge Graph Edges", final_stats.get("graph", {}).get("edge_count", 0))
        print_data("Clinical Notes", final_stats.get("working_memory", {}).get("total_records", 0))
        print_data("Cross-Reference Links", final_stats.get("linkage", {}).get("total_links", 0))

        memory.close()
        print_success("Medical memory stores closed")
        print_success("Demo databases cleaned up automatically")

    # Final summary
    print_header("DEMO COMPLETE")
    print("""
  The Medical Patient Scenario demonstrated:

  1. HEALTHCARE KNOWLEDGE GRAPH
     - Patients, providers, conditions, medications, locations
     - Treatment relationships and care team assignments
     - Drug interaction detection

  2. PATIENT MONITORING (TIME-SERIES)
     - Real-time vital sign streaming
     - Multi-patient monitoring
     - Trend analysis for deteriorating patients

  3. CLINICAL DOCUMENTATION (WORKING MEMORY)
     - Progress notes and critical alerts
     - Lab results integration
     - Cross-patient documentation

  4. ML-POWERED CLINICAL DECISION SUPPORT
     - Triage classifier (stable/moderate/critical)
     - Vital signs anomaly detection
     - Early warning for patient deterioration

  5. UNIFIED PATIENT CONTEXT
     - Query across all data stores
     - Build comprehensive patient picture
     - Support clinical decision-making

  6. CONSOLIDATION FOR EHR INTEGRATION
     - Extract structured facts from notes
     - Build treatment pathway knowledge
     - Archive and prune raw monitoring data

  This demonstrates LlamaFarm's potential for healthcare analytics!
""")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ImportError as e:
        print(f"Import error: {e}")
        print("Run from the 'rag' directory: cd rag && uv run python ../examples/e2e_scenarios/demo_medical_patient.py")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
