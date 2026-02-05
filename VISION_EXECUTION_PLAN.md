# Vision Capabilities - Execution Plan

> **Branch**: `feat/vision` (ONLY commit here, NEVER to main)
> **Generated**: 2025-06-05
> **Based on**: Plan.md

---

## Overview

This plan breaks down the comprehensive vision capabilities into parallel workstreams executed by sub-agents. Each agent has a focused scope and clear deliverables.

## Dependency Graph

```
                    ┌──────────────────┐
                    │  Agent 1: Types  │
                    │  & Base Classes  │
                    └────────┬─────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │ Agent 2: YOLO │ │ Agent 3: CLIP │ │ Agent 4: SAM  │
    │   Detection   │ │  Classifier   │ │ Segmentation  │
    └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
            │                 │                 │
            └────────────────┬┴─────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Agent 5: Runtime │
                    │    Routers       │
                    └────────┬─────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │ Agent 6:      │ │ Agent 7:      │ │ Agent 8:      │
    │ Server Router │ │ Streaming     │ │ Storage       │
    └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
            │                 │                 │
            └────────────────┬┴─────────────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │ Agent 9:      │ │ Agent 10:     │ │ Agent 11:     │
    │ Review Queue  │ │ Training      │ │ Image RAG     │
    └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
            │                 │                 │
            └────────────────┬┴─────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Agent 12:        │
                    │ Integration      │
                    └──────────────────┘
```

---

## Agent Assignments

### 🔷 Agent 1: Types & Base Classes (FOUNDATION)
**Priority**: P0 (Start Immediately)
**Estimated Time**: 1-2 hours
**Dependencies**: None

**Deliverables**:
```
runtimes/universal/
├── api_types/vision/
│   ├── __init__.py
│   ├── common.py           # BoundingBox, Point, Mask
│   ├── detection.py        # DetectRequest/Response
│   ├── classification.py   # ClassifyRequest/Response
│   ├── segmentation.py     # SegmentRequest/Response
│   ├── streaming.py        # StreamingConfig, FrameRequest/Response
│   ├── training.py         # TrainingConfig, TrainRequest/Response
│   └── models.py           # ModelInfo, ExportRequest/Response
├── models/vision/
│   ├── __init__.py
│   └── base.py             # VisionModel, VisionResult base classes

server/api/routers/vision/
├── __init__.py             # Router aggregation
└── types.py                # Mirror of runtime types
```

**Instructions**:
1. Create directory structures
2. Implement all Pydantic models from Plan.md Section "Component Specifications"
3. Create VisionModel abstract base class
4. Ensure types are identical between server and runtime
5. Add `__all__` exports to all `__init__.py`

---

### 🔷 Agent 2: YOLO Detection Model
**Priority**: P0 (Start after Agent 1)
**Estimated Time**: 2-3 hours
**Dependencies**: Agent 1

**Deliverables**:
```
runtimes/universal/
├── models/vision/
│   └── yolo_model.py       # YOLOModel implementation
└── tests/vision/
    └── test_yolo_model.py
```

**Instructions**:
1. Implement YOLOModel class from Plan.md
2. Support yolov8n, yolov8s, yolov8m, yolov11n
3. Implement `load()`, `infer()`, `train()`, `export()` methods
4. Handle device auto-detection (MPS, CUDA, CPU)
5. Add unit tests for detection
6. Follow existing pattern in `runtimes/universal/models/`

**Key Dependencies**:
```toml
# Add to pyproject.toml [project.optional-dependencies.vision]
ultralytics>=8.0.0
```

---

### 🔷 Agent 3: CLIP Classifier
**Priority**: P0 (Start after Agent 1)
**Estimated Time**: 2-3 hours
**Dependencies**: Agent 1

**Deliverables**:
```
runtimes/universal/
├── models/vision/
│   └── clip_classifier.py  # CLIPClassifier implementation
└── tests/vision/
    └── test_clip_classifier.py
```

**Instructions**:
1. Implement CLIPClassifier from Plan.md
2. Support zero-shot classification (text prompts)
3. Implement few-shot training with classifier head
4. Support image and text embedding generation
5. Add unit tests

**Key Dependencies**:
```toml
transformers>=4.30.0
```

---

### 🔷 Agent 4: MobileSAM Segmentation
**Priority**: P1 (Start after Agent 1)
**Estimated Time**: 2-3 hours
**Dependencies**: Agent 1

**Deliverables**:
```
runtimes/universal/
├── models/vision/
│   └── sam_model.py        # MobileSAM implementation
└── tests/vision/
    └── test_sam_model.py
```

**Instructions**:
1. Implement SAMModel class for segmentation
2. Support point prompts and box prompts
3. Handle mask generation and encoding
4. Add unit tests

**Key Dependencies**:
```toml
mobile-sam>=1.0.0  # or segment-anything
```

---

### 🔷 Agent 5: Runtime Vision Routers
**Priority**: P0 (Start after Agents 2, 3)
**Estimated Time**: 3-4 hours
**Dependencies**: Agents 2, 3, 4

**Deliverables**:
```
runtimes/universal/routers/vision/
├── __init__.py             # Update to include new routers
├── detection.py            # /v1/vision/detect endpoint
├── classification.py       # /v1/vision/classify endpoint
├── segmentation.py         # /v1/vision/segment endpoint
├── embedding.py            # /v1/vision/embed endpoint (CLIP)
└── models.py               # /v1/vision/models/* endpoints
```

**Instructions**:
1. Follow existing router pattern (see `routers/anomaly/router.py`)
2. Use dependency injection pattern (`set_*_loader` functions)
3. Implement endpoints from Plan.md API Design section
4. Wire up to model classes
5. Add to `routers/__init__.py` exports

---

### 🔷 Agent 6: Server Vision Routers & Services
**Priority**: P1 (Start after Agent 5)
**Estimated Time**: 3-4 hours
**Dependencies**: Agent 5

**Deliverables**:
```
server/
├── api/routers/vision/
│   ├── __init__.py         # Router aggregation (vision_router)
│   ├── types.py            # Already from Agent 1
│   ├── detection.py        # Proxy to runtime
│   ├── classification.py   # Proxy to runtime
│   ├── segmentation.py     # Proxy to runtime
│   └── models.py           # Model management
└── services/vision/
    ├── __init__.py
    ├── detection_service.py
    ├── classification_service.py
    ├── segmentation_service.py
    └── models_service.py
```

**Instructions**:
1. Create VisionDetectionService, etc. following Plan.md
2. Use httpx.AsyncClient for runtime communication
3. Do NOT modify `universal_runtime_service.py`
4. Add `vision_router` to server's main.py

---

### 🔷 Agent 7: Streaming Vision Detector
**Priority**: P1 (Start after Agent 5)
**Estimated Time**: 3-4 hours
**Dependencies**: Agent 5 (needs YOLO model working)

**Deliverables**:
```
runtimes/universal/
├── models/vision/
│   └── streaming_detector.py
├── routers/vision/
│   └── streaming.py        # /v1/vision/stream/* endpoints
├── services/vision/
│   └── streaming_service.py
└── tests/vision/
    └── test_streaming.py

server/
├── api/routers/vision/
│   └── streaming.py
└── services/vision/
    └── streaming_service.py
```

**Instructions**:
1. Implement StreamingVisionDetector from Plan.md
2. Session management (start, process frame, end)
3. Confidence-based routing (action/review/ok)
4. Cooldown mechanism
5. Multi-model cascade (primary → secondary escalation)

---

### 🔷 Agent 8: Storage & Metadata
**Priority**: P1 (Start after Agent 1)
**Estimated Time**: 2-3 hours
**Dependencies**: Agent 1 (types only)

**Deliverables**:
```
runtimes/universal/
├── storage/
│   ├── __init__.py
│   ├── image_store.py      # SQLite metadata store
│   ├── retention_policy.py # Cleanup logic
│   └── replay_buffer.py    # Experience replay storage
└── utils/
    └── image_utils.py      # Compression, thumbnails
```

**Instructions**:
1. Implement ImageMetadataStore from Plan.md (SQLite)
2. Create tables: images, detections, labels
3. Implement retention policy engine
4. Add replay buffer for training corrections
5. Image compression and thumbnail generation

---

### 🔷 Agent 9: Review Queue
**Priority**: P2 (Start after Agents 6, 8)
**Estimated Time**: 2-3 hours
**Dependencies**: Agents 6, 8

**Deliverables**:
```
server/
├── api/routers/vision/
│   └── review.py           # /v1/vision/review/* endpoints
└── services/vision/
    └── review_service.py
```

**Instructions**:
1. Implement review queue endpoints (list, submit decision)
2. Connect to ImageMetadataStore
3. Handle corrections (add to replay buffer)
4. Batch review functionality
5. Server-only endpoints (not mirrored in runtime)

---

### 🔷 Agent 10: Training Pipeline
**Priority**: P2 (Start after Agents 2, 8)
**Estimated Time**: 4-5 hours
**Dependencies**: Agents 2, 8

**Deliverables**:
```
runtimes/universal/
├── training/
│   ├── __init__.py
│   ├── incremental_trainer.py
│   ├── ewc.py              # Elastic Weight Consolidation
│   └── replay_sampler.py   # Experience replay sampling
├── routers/vision/
│   └── training.py         # /v1/vision/train/* endpoints
└── services/vision/
    └── training_service.py

server/
├── api/routers/vision/
│   └── training.py
└── services/vision/
    └── training_service.py
```

**Instructions**:
1. Implement incremental training from Plan.md
2. EWC regularization for catastrophic forgetting prevention
3. Experience replay with priority sampling
4. Async training job management
5. Progress tracking and metrics

---

### 🔷 Agent 11: Image RAG Integration
**Priority**: P2 (Start after Agents 3, 8)
**Estimated Time**: 3-4 hours
**Dependencies**: Agents 3 (CLIP), 8 (storage)

**Deliverables**:
```
rag/components/embedders/clip_embedder/
├── __init__.py
└── clip_embedder.py

server/
├── api/routers/vision/
│   └── image_rag.py        # /v1/vision/rag/* endpoints
└── services/vision/
    ├── image_rag_service.py
    └── image_rag_health.py
```

**Instructions**:
1. Implement CLIPEmbedder for RAG framework
2. Add to rag/core/factories.py EMBEDDER_REGISTRY
3. Implement image search (text → images, image → images)
4. Index images with auto-detection/classification
5. Health monitoring

---

### 🔷 Agent 12: Integration & Wiring
**Priority**: P0 (Final phase)
**Estimated Time**: 2-3 hours
**Dependencies**: All agents

**Deliverables**:
```
runtimes/universal/
├── server.py               # Add vision router imports
└── routers/__init__.py     # Export vision_router

server/
├── api/main.py             # Add vision_router
└── core/settings.py        # Vision config if needed

tests/integration/
└── test_vision_api.py      # End-to-end tests
```

**Instructions**:
1. Wire all routers into server.py
2. Add vision router to server's main.py
3. Verify all endpoints work end-to-end
4. Write integration tests
5. Update pyproject.toml with vision dependencies
6. Test on MPS (Mac), CUDA (Linux), CPU fallback

---

## Execution Order

### Wave 1 (Parallel - Start Immediately)
- **Agent 1**: Types & Base Classes ⭐ CRITICAL PATH
- **Agent 8**: Storage & Metadata (only needs types)

### Wave 2 (Parallel - After Agent 1 completes)
- **Agent 2**: YOLO Detection Model
- **Agent 3**: CLIP Classifier  
- **Agent 4**: MobileSAM Segmentation

### Wave 3 (After Wave 2)
- **Agent 5**: Runtime Vision Routers (needs models)

### Wave 4 (Parallel - After Agent 5)
- **Agent 6**: Server Vision Routers
- **Agent 7**: Streaming Vision Detector

### Wave 5 (Parallel - After relevant dependencies)
- **Agent 9**: Review Queue (needs Agent 6, 8)
- **Agent 10**: Training Pipeline (needs Agent 2, 8)
- **Agent 11**: Image RAG (needs Agent 3, 8)

### Wave 6 (Final)
- **Agent 12**: Integration & Wiring

---

## Git Workflow

```bash
# Every agent MUST verify branch before ANY commit
git branch --show-current  # Must show: feat/vision

# Commit pattern
git add <files>
git commit -m "feat(vision): <description>"

# NEVER do this:
git checkout main
git push origin main
```

---

## Testing Commands

```bash
# Run tests after each agent completes
cd runtimes/universal
uv run pytest tests/vision/ -v

cd server
uv run pytest tests/vision/ -v

# Full integration test
cd /Users/robthelen/clawd/projects/llamafarm-core
uv run pytest tests/integration/test_vision_api.py -v
```

---

## Success Criteria

Each agent should verify:
1. ✅ All files created in correct locations
2. ✅ Types match between server and runtime
3. ✅ Tests pass
4. ✅ No modifications to `universal_runtime_service.py`
5. ✅ Committed to `feat/vision` branch only
6. ✅ No breaking changes to existing APIs

---

## Notes

- **Restart services after changes**: LlamaFarm does NOT auto-reload
- **Kill processes**: `pkill -f "python main.py" ; pkill -f "python server.py"`
- **Check ports**: `lsof -i :14345` (server), `lsof -i :11540` (runtime)
- **Reference existing routers**: `routers/anomaly/`, `routers/classifier/` for patterns

