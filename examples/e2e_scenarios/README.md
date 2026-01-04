# End-to-End LlamaFarm Scenarios

This directory contains comprehensive end-to-end demos that showcase the full power of LlamaFarm's **Embedded Trinity Memory System** combined with ML capabilities.

## Overview

The Embedded Trinity Memory System provides a federated embedded architecture:

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Vector Memory** | ChromaDB | Semantic search, embeddings |
| **Time-Series Memory** | DuckDB | Telemetry, spatial queries, analytics |
| **Graph Memory** | DuckDB | Entity relationships, knowledge graphs |
| **Working Memory** | DuckDB | Short-term buffer with TTL |
| **Linkage Table** | DuckDB | Cross-store UUID tracking |

## Scenarios

### 1. Military Rescue Scenario (`demo_military_rescue.py`)

A realistic military rescue operation demonstrating:

- **Biometric Telemetry**: Streaming soldier vital signs (heart rate, blood oxygen, location)
- **Knowledge Graph**: Personnel, locations, command structure
- **Radio Communications**: Transcriptions in working memory with priority tagging
- **Distress Detection**: ML classifier for emergency communications
- **Anomaly Detection**: Vital signs monitoring for soldier distress
- **Rescue Coordination**: Unified context retrieval across all stores

### 2. Medical Patient Scenario (`demo_medical_patient.py`)

A hospital patient monitoring system demonstrating:

- **Patient Monitoring**: Real-time vital signs streaming
- **Clinical Knowledge Graph**: Patients, providers, conditions, medications
- **Drug Interactions**: Graph-based medication safety checks
- **Clinical Documentation**: Progress notes, alerts, lab results
- **Triage Classification**: ML classifier for patient urgency
- **Anomaly Detection**: Detect deteriorating patient conditions
- **Clinical Decision Support**: Unified patient context for care teams

## Running the Demos

### Quick Start

```bash
# Run all demos
cd llamafarm
./examples/e2e_scenarios/run_all_e2e_demos.sh

# Run specific demo
./examples/e2e_scenarios/run_all_e2e_demos.sh military
./examples/e2e_scenarios/run_all_e2e_demos.sh medical
```

### Manual Execution

```bash
cd llamafarm/rag
uv run python ../examples/e2e_scenarios/demo_military_rescue.py
uv run python ../examples/e2e_scenarios/demo_medical_patient.py
```

## API Endpoints Used

These demos use (or simulate) the following LlamaFarm APIs:

### Per-Project Memory API (`/v1/projects/{namespace}/{project}/memory/*`)

Memory stores are configured per-project in `llamafarm.yaml` and accessed via:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/projects/{ns}/{proj}/memory/add` | POST | Add data to memory stores |
| `/v1/projects/{ns}/{proj}/memory/query` | GET | Unified context query |
| `/v1/projects/{ns}/{proj}/memory/context` | GET | Aggregated context |
| `/v1/projects/{ns}/{proj}/memory/stats` | GET | Storage statistics |
| `/v1/projects/{ns}/{proj}/memory/consolidate` | POST | Memory synthesis |
| `/v1/projects/{ns}/{proj}/memory/prune` | POST | Cleanup expired records |
| `/v1/projects/{ns}/{proj}/memory/clear/{table}` | POST | Clear specific table |
| `/v1/projects/{ns}/{proj}/memory/{uuid}` | DELETE | Cascade delete record |

**Note:** There's also a global Memory API at `/v1/memory/*` for non-project contexts.

### ML API (`/v1/ml/*`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/ml/classifier/fit` | POST | Train text classifier |
| `/v1/ml/classifier/predict` | POST | Classify texts |
| `/v1/ml/anomaly/fit` | POST | Train anomaly detector |
| `/v1/ml/anomaly/detect` | POST | Detect anomalies |

## Configuration

The `llamafarm.yaml` in this directory provides:

- ChromaDB configuration for semantic search
- Memory system configuration
- ML model settings
- Dataset definitions

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Unified Memory Interface                       │
│                        (MemoryStore)                              │
├──────────────┬──────────────┬──────────────┬──────────────────────┤
│   ChromaDB   │   DuckDB     │   DuckDB     │   DuckDB             │
│   (Vector)   │  (TimeSeries)│   (Graph)    │  (WorkingMem)        │
│              │              │              │                      │
│  Semantic    │  Telemetry   │  Entities    │  Short-term          │
│  Search      │  Spatial     │  Relations   │  TTL Buffer          │
└──────────────┴──────────────┴──────────────┴──────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Linkage Table    │
                    │  (Cross-Store ID) │
                    └───────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │   Consolidator    │
                    │   (Hippocampus)   │
                    └───────────────────┘
```

## Key Concepts

### The Consolidator ("Hippocampus")

Just like the hippocampus in the brain, the Consolidator:

1. **Reads** raw data from Working Memory
2. **Synthesizes** facts using rule-based or LLM extraction
3. **Creates** knowledge graph nodes from extracted facts
4. **Prunes** processed raw data to manage storage

### Unified Context Retrieval

Query across all stores simultaneously:

```python
context = memory.get_context(
    recent_minutes=10,
    include_graph=True,
    include_working_memory=True,
)
# Returns data from all stores in one call
```

### Cross-Store Linking

Every record gets a UUID that's tracked across stores:

```python
result = memory.add(data=..., data_type="telemetry")
# result["uuid"] is tracked in LinkageTable
# Enables cascade deletes and cross-store queries
```

## Cleanup

All demos use temporary directories that are automatically cleaned up. No persistent data is modified unless you configure a specific `base_path`.
