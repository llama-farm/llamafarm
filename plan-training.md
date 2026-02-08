# Vision Learning Loop: Automatic Cascade, Feedback, and Model Evolution

## Context

The vision system has detection, classification, segmentation, a cascade pipeline, replay buffer, auto-trainer, and review queue. But the learning loop has gaps that prevent it from being truly automatic and truly useful across deployment modes.

**Two deployment realities, one codebase:**
- **Standalone**: Laptop/desktop. One machine, multiple models of different sizes. Cascade is local (yolov8n -> yolov8m -> yolov8x). Training happens locally. Human reviews happen in Designer UI.
- **Mesh/Edge**: Phone/Jetson/drone running Atmosphere. Tiny local model, bigger models found via mesh service discovery. Training happens on a GPU node. Feedback flows back over mesh. Model updates gossip to edge devices.

The architecture must not care which mode it's in. The cascade is a **list of models to try**, and each model is either local or remote. The system doesn't distinguish -- it just calls `.detect()` on the next one in the chain.

**The 99% automatic requirement**: The only thing a human should ever need to do is occasionally glance at a review queue and tap "yes/no/fix" on bounding boxes. Everything else -- escalation, training, validation, deployment, model distribution -- happens without human intervention.

---

## The Escalation Envelope

This is the core data structure that flows through the entire system. When a model is uncertain, it doesn't just send an image -- it sends everything the next model needs to make a better decision, and everything the training pipeline needs to learn from the outcome.

### What it contains:

```python
@dataclass
class EscalationEnvelope:
    """Everything needed for the next model AND for training feedback."""

    # The image
    image_bytes: bytes          # Original full frame
    image_hash: str             # For dedup across mesh
    source_id: str              # "drone-cam-1", "laptop-webcam", etc.
    timestamp: datetime

    # What each model saw (grows as it cascades)
    opinions: list[ModelOpinion]  # Ordered: first model first

    # Bounding boxes from detection (ALWAYS present if anything was found)
    detections: list[DetectionWithMask]

    # Routing metadata
    origin_node: str            # Atmosphere node ID or "local"
    hops: int = 0               # How many models have seen this
    max_hops: int = 3           # Circuit breaker
    urgency: str = "normal"     # "normal", "important", "critical"

@dataclass
class ModelOpinion:
    """What one model thought about this image."""
    model_id: str               # "yolov8n", "yolov8x", "remote:gpu-server/yolov8x"
    node_id: str                # Where this model ran
    class_name: str             # What it thinks this is
    confidence: float           # How sure it is
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    mask_polygon: list[list[float]] | None   # Segmentation if available
    inference_time_ms: float
    timestamp: datetime

@dataclass
class DetectionWithMask:
    """A detection with its segmentation mask attached.
    This is what flows to the next model -- bbox + visual context."""
    bbox: tuple[float, float, float, float]
    crop_bytes: bytes | None    # Cropped region from the image
    mask_polygon: list[list[float]] | None
    mask_rle: str | None        # Run-length encoded for storage
    class_name: str             # Best guess so far
    confidence: float           # Best confidence so far
```

### Why this shape matters:

1. **Bounding boxes always flow forward** -- the next model sees exactly where the uncertain object is, doesn't have to re-find it
2. **Segmentation masks flow forward** -- if available, the next model gets pixel-level context of the object shape
3. **Every model's opinion is preserved** -- when model A says "airplane" at 0.4 and model B says "bird" at 0.9, the training pipeline gets BOTH opinions. This is gold for training.
4. **Same shape on laptop and mesh** -- on a laptop, `node_id` is "local" for all opinions. On mesh, it's the Atmosphere node ID. The code doesn't care.
5. **Circuit breaker** -- `max_hops` prevents infinite escalation loops

---

## How Many Hops Should It Take?

**Answer: Maximum 3, typically 1-2.**

```
Hop 0: Local fast model (yolov8n) -- always runs, ~5ms
        confidence >= 0.7  ->  DONE (action)
        confidence 0.4-0.7 ->  Hop 1
        confidence < 0.4   ->  Hop 1 (with "low_confidence" flag)

Hop 1: Local large model OR nearest mesh peer with better GPU
        (on laptop: yolov8m/yolov8x local)
        (on edge: Atmosphere routes to best available peer)
        confidence >= 0.7  ->  DONE (action) + feed back to Hop 0's training
        confidence 0.4-0.7 ->  Hop 2
        confidence < 0.4   ->  REVIEW QUEUE (auto if model available, human if not)

Hop 2: Largest available model (remote GPU, cloud, etc.)
        confidence >= 0.5  ->  DONE + feed back to Hop 0 AND Hop 1
        confidence < 0.5   ->  REVIEW QUEUE (needs human eyes)

Hop 3: NEVER automatically. Only if human explicitly requests "get another opinion"
```

**Why 3 max?** Latency. On mesh: hop 0 is 5ms, hop 1 is 50-200ms (network), hop 2 is 200-500ms. By hop 3 you're at 1s+ and the drone has moved on. On laptop: all hops are local so latency is just model inference time, but the value of a 4th model opinion is negligible.

---

## The Automatic Feedback Loop (99% Hands-Free)

```
DETECT (Hop 0)
  |
  +-- High confidence --> ACTION --> store image+detections --> done
  |                                   (also: periodic audit samples from these)
  |
  +-- Mid confidence --> ESCALATE to Hop 1
  |   |
  |   +-- Hop 1 agrees (high conf) --> ACTION
  |   |   +-- AUTO-FEEDBACK: Hop 1's answer + Hop 0's bbox
  |   |       become a training sample for Hop 0's model
  |   |       (source="escalation_resolved", priority=1.5)
  |   |
  |   +-- Hop 1 disagrees --> ESCALATE to Hop 2
  |   |   +-- Hop 2 resolves it --> AUTO-FEEDBACK to BOTH Hop 0 and Hop 1
  |   |       (source="cascade_resolved", priority=1.8)
  |   |
  |   +-- Nobody confident --> REVIEW QUEUE
  |       +-- Bounding boxes + all opinions shown to human
  |           Human taps correct answer --> TRAINING SAMPLE (priority=2.0)
  |
  +-- Low confidence --> REVIEW QUEUE (same as above)

TRAINING (automatic, periodic)
  |
  +-- Replay buffer hits threshold (50 samples default)
  +-- Auto-trainer creates dataset from buffer
  +-- Trains candidate model (few epochs, EWC to prevent forgetting)
  +-- VALIDATION GATE: run candidate against held-out verified images
  |   +-- Better? --> PROMOTE (blue/green swap)
  |   |   +-- Standalone: hot-swap model in memory, keep old as rollback
  |   |   +-- Mesh: gossip MODEL_AVAILABLE, peers pull new model
  |   +-- Worse? --> REJECT, keep current model, log why
  +-- Clear consumed samples from replay buffer

AUDIT (automatic, periodic background)
  |
  +-- Sample N recent HIGH-confidence predictions (the ones we DIDN'T escalate)
  +-- Re-run through a bigger model (local or remote)
  +-- Compare: same class? similar bbox?
  |   +-- Agreement --> mark as VERIFIED (good validation data)
  |   +-- Disagreement --> AUTO-FEEDBACK (source="audit", priority=2.0)
  |       This catches systematic errors the cascade missed
  +-- Feed disagreements into replay buffer --> triggers training
```

**Where humans come in (the 1%):**
- Reviewing the queue when all models are uncertain (bboxes + masks always visible)
- Optionally spot-checking audit results
- That's it. Everything else is automatic.

---

## Phase 1: EscalationEnvelope + Persistent Replay Buffer

**The foundation. Everything flows through these types.**

### Modify: `runtimes/universal/vision_training/replay_buffer.py`

1. Add `ModelOpinion` dataclass (model_id, node_id, class_name, confidence, bbox, mask, timestamp)
2. Add `EscalationEnvelope` as a new field on `ReplaySample` -- stores the full cascade context
3. Extend `ReplaySample`:
   - `opinions: list[ModelOpinion]` -- every model's take
   - `final_label: str` -- the resolved answer
   - `final_source: str` -- "cascade", "human", "audit"
   - `bbox: tuple[float,float,float,float] | None` -- THE bounding box
   - `mask_rle: str | None` -- segmentation mask
   - `crop_path: str | None` -- cropped bbox region on disk
4. Add `source="audit"` and `source="cascade_resolved"` to valid source types
5. **SQLite persistence** -- the `storage_dir` param already exists but is unused. Wire it:
   - `_init_db()`, `_persist_sample()`, `_remove_from_db()`, `_load_from_db()`
   - JSON columns for opinions list, separate columns for bbox/confidence/label
   - On startup: reload from SQLite into in-memory dict (fast sampling still works)

### Create: `runtimes/universal/vision_training/persistence.py`

SQLite read/write logic for replay samples. Serialize `ModelOpinion` list as JSON. Keep it simple -- one table, indexed by source and priority.

### Modify: `runtimes/universal/storage/image_store.py`

Add `detection_history` table:
```sql
detection_history (id, image_id, model_id, node_id, class_name,
                   confidence, x1, y1, x2, y2, mask_rle,
                   stage, hop_number, created_at)
```
Every model opinion gets stored. This is the audit trail AND the source for building validation sets.

---

## Phase 2: Cross-Modal Enrichment in the Cascade

**When detection is uncertain, attach segmentation before escalating.**

### Modify: `runtimes/universal/models/streaming_vision.py`

1. Add to `CascadeConfig`:
   - `segmentation_model_id: str | None` -- run seg on uncertain bboxes
   - `classification_model_id: str | None` -- run CLIP classification on uncertain crops
   - `enrich_on_escalation: bool = True`
   - `cascade_chain: list[str] = []` -- ordered list of model IDs to try (replaces single `secondary_model_id`)
2. Add `_enrich_detection()` method:
   - Crop bbox region from full image
   - If seg model configured: run segmentation, attach mask polygon
   - If classification model configured: run CLIP, attach class scores
   - Return `DetectionWithMask` with crop bytes + mask + best class guess
3. Refactor `process_frame()` to use the cascade chain:
   - Replace the binary primary/secondary logic with a loop over `cascade_chain`
   - At each hop: build `ModelOpinion`, append to envelope
   - If any hop resolves (high confidence): break, return result, feed envelope to replay buffer
   - If all hops fail: send to review queue with full envelope
4. Modify `_add_to_replay_buffer()`:
   - Accept `ModelOpinion` list, not flat YOLO strings
   - Store structured bbox + mask data alongside the YOLO-format label (backward compat)
   - Build the `CascadeHistory` from the envelope's opinions

### Modify: `runtimes/universal/api_types/vision.py`

- Add `cascade_chain`, `segmentation_model_id`, `classification_model_id`, `enrich_on_escalation` to Pydantic `CascadeConfig`
- Add `ModelOpinion` and `EscalationEnvelope` Pydantic models for API responses
- Add `hop_count` and `cascade_resolved_by` to `StreamFrameResponse`

---

## Phase 3: Remote Model Proxy (Works for Both Standalone and Mesh)

**A model that calls another LlamaFarm instance. On standalone, this is optional. On mesh, this is how Atmosphere peers route vision work.**

### Create: `runtimes/universal/models/remote_model_proxy.py`

```python
class RemoteModelProxy(DetectionModel):
    """Calls /v1/vision/detect on a remote LlamaFarm instance.

    Implements the same interface as YOLOModel. The cascade
    doesn't know or care if a model is local or remote.

    Standalone: configured manually with a peer URL
    Mesh: Atmosphere auto-discovers and registers proxies
    """
    def __init__(self, model_id, remote_url, remote_model, timeout=30.0): ...
    async def detect(self, image, confidence_threshold, classes): ...
    async def load(self): ...   # verify remote health
    async def unload(self): ... # close client
```

Key: the `detect()` method sends the `EscalationEnvelope` (serialized), not just raw image bytes. The remote node gets all prior opinions and bbox context. It returns its own `ModelOpinion` which gets appended to the envelope.

### Create: `runtimes/universal/config/federation.py`

```python
@dataclass
class PeerConfig:
    name: str
    url: str                    # Direct URL or "atmosphere://node-id"
    models: list[str]           # What models they have
    gpu_vram_gb: float = 0      # For routing decisions
    priority: int = 0           # Lower = tried first in cascade
    timeout: float = 30.0

@dataclass
class FederationConfig:
    enabled: bool = False
    peers: list[PeerConfig] = field(default_factory=list)
    auto_register_atmosphere: bool = True  # Auto-add Atmosphere peers
```

### Modify: `runtimes/universal/models/streaming_vision.py`

In `start_session()`: build the cascade chain from config. For each entry in `cascade_chain`:
- If it's a known local model ID -> use local model
- If it starts with `remote:` or `atmosphere://` -> create `RemoteModelProxy`
- The cascade loop in `process_frame()` doesn't change -- it just calls `.detect()` on whatever's next

### Create: `runtimes/universal/routers/vision/federation.py`

- `GET /v1/vision/federation/peers` -- list peers
- `POST /v1/vision/federation/peers` -- register peer
- `DELETE /v1/vision/federation/peers/{name}` -- remove
- `GET /v1/vision/federation/status` -- health/latency of all peers
- `POST /v1/vision/federation/escalate` -- **the inbound endpoint**: another node sends us an `EscalationEnvelope`, we run our model, return our `ModelOpinion`

That last endpoint is critical -- it's how a remote node acts as a cascade hop. On standalone nobody calls it. On mesh, Atmosphere routes to it.

---

## Phase 4: Audit Pipeline (Automatic Model-Checking-Model)

**A bigger model periodically reviews what the small model has been saying. 100% automatic.**

### Create: `runtimes/universal/vision_training/audit_pipeline.py`

```python
class AuditPipeline:
    """Periodically re-checks primary model predictions with a bigger model.

    Runs automatically. No human needed.

    Disagreements become training samples.
    Agreements become validation samples.
    """
    async def run_audit(self, sample_size=20) -> AuditReport:
        # 1. Pull N recent high-confidence images from image_store
        # 2. Re-run through audit_model (local large or RemoteModelProxy)
        # 3. Compare: IoU of bboxes + class name match
        # 4. Disagreements -> replay buffer (source="audit", priority=2.0)
        # 5. Agreements -> mark verified in image_store
        # 6. Return report with accuracy estimate
```

The audit model can be:
- **Standalone**: a larger local model (yolov8x)
- **Mesh**: a RemoteModelProxy pointing at a GPU node

Same code either way. The `AuditConfig.audit_model_id` can be `"yolov8x"` or `"remote:gpu-server/yolov8x"`.

### Modify: `server/services/vision/review_service.py`

1. Fix `_record_to_item()` -- populate `prediction` field with actual detections from image_store (currently returns `None`)
2. Add `reviewer_type` to `submit_review()` -- "human" or "model"
3. When `reviewer_type="model"`, auto-create training sample without human confirmation
4. **Bounding boxes in review items**: include ALL detections with bbox coords and model opinions so humans see exactly what each model thought

---

## Phase 5: Validation Gate + Blue/Green Model Swap

**Retrained model must prove it's better before going live.**

### Create: `runtimes/universal/vision_training/validation_gate.py`

```python
class ValidationGate:
    async def build_validation_set(self) -> str:
        # Auto-build from: human-verified images + audit-verified images
        # These are images we're CONFIDENT about the correct label

    async def validate_candidate(self, candidate_path, current_model_id) -> ValidationResult:
        # Run both models against validation set
        # Compare mAP, precision, recall
        # Return pass/fail with detailed metrics

    async def promote(self, candidate_path, model_id, backup=True) -> str:
        # Blue/green: load candidate alongside current
        # Run a few live frames through both
        # If candidate holds up: swap, keep old as rollback
        # On mesh: trigger MODEL_AVAILABLE gossip

    async def rollback(self, model_id) -> bool:
        # Restore previous model version
```

### Modify: `runtimes/universal/vision_training/auto_trainer.py`

1. Wire validation gate into `_wait_for_completion()`:
   - Training done -> validate candidate -> promote or reject
   - **This is the blue/green test**: candidate must beat current model
2. Fix `_create_dataset()`:
   - Build class map from actual sample labels (not hardcoded `nc: 80`)
   - Use `DetectionContext.class_name` from enhanced `ReplaySample`
3. Add `on_model_promoted` callback -- on mesh, this triggers model package creation + gossip

### Modify: `runtimes/universal/vision_training/trainer.py`

1. Return `model_path` in metrics (so validation gate can find the candidate)
2. Save as `{model_id}_v{N}.pt` for versioning
3. Wire the `# TODO: Add EWC and replay buffer integration` that's been sitting there

---

## Phase 6: Model Packaging + Mesh Distribution

**Package a trained model so it can be sent to other nodes.**

### Create: `runtimes/universal/vision_training/model_package.py`

Package format (`.tar.gz`):
```
model.pt / model.onnx      - Weights
metadata.json               - version, base_model, class_map, training_samples,
                              metrics, lineage, source_node, description
class_map.json              - {0: "person", 1: "bird", ...}
validation_metrics.json     - What it scored on the validation gate
```

`ModelPackager`:
- `create_package()` -- called automatically after validation gate promotes
- `read_package()` -- extract, validate, return model + metadata
- On mesh: package gets announced via Atmosphere `MODEL_AVAILABLE` gossip
- On standalone: packages sit in `~/.llamafarm/vision/packages/` for manual transfer

### Add to federation router:

- `POST /v1/vision/models/package` -- create package from current model
- `POST /v1/vision/models/import` -- import a package (file path, URL, or from peer)
- `GET /v1/vision/models/packages` -- list available packages
- `POST /v1/vision/models/push/{peer}` -- push package to a specific peer

---

## Phase 7: Review Queue Fix (Bounding Boxes MUST Be There)

### Modify: `server/services/vision/review_service.py`

`_record_to_item()` currently returns:
```python
"prediction": None,  # TODO: Include detection info
"confidence": 0.0,
"model": "",
```

Fix to return:
```python
"prediction": {
    "box": {"x1": det.x1, "y1": det.y1, "x2": det.x2, "y2": det.y2},
    "class_name": det.class_name,
    "confidence": det.confidence,
},
"all_opinions": [  # Every model's take, for context
    {"model": op.model_id, "class": op.class_name, "confidence": op.confidence}
    for op in detection_history
],
"confidence": det.confidence,
"model": det.model,
```

Humans see: the image, every bounding box drawn, what each model thought, and they tap the right answer. That answer goes straight into the replay buffer at priority 2.0.

### Modify: `runtimes/universal/storage/image_store.py`

Add `reviewer_type`, `reviewer_model_id`, `reviewer_confidence` columns. The review service needs to distinguish "human said this" from "audit model said this" for training priority.

### Modify: `runtimes/universal/api_types/vision.py`

Update `ReviewDecision` with `reviewer_type`, `reviewer_model_id`, `reviewer_confidence`.
Update `ReviewItem` to include `all_opinions: list[ModelOpinion]`.

---

## Phase 8: Tests

New test files (all in `runtimes/universal/tests/`):
- `test_replay_persistence.py` -- persist to SQLite, restore on restart, eviction
- `test_escalation_envelope.py` -- envelope builds correctly through cascade, opinions accumulate
- `test_cross_modal_cascade.py` -- detection+seg enrichment, crop+mask attached
- `test_remote_model_proxy.py` -- mocked HTTP, timeout handling, envelope round-trip
- `test_audit_pipeline.py` -- disagreement detection, auto-feedback to replay buffer
- `test_validation_gate.py` -- pass/reject/promote/rollback

Modify `test_vision_cascade.py` -- test 3-hop cascade chain, ModelOpinion accumulation, circuit breaker at max_hops.

---

## File Summary

### New Files (9)
| File | Purpose |
|------|---------|
| `vision_training/persistence.py` | Replay buffer SQLite persistence |
| `models/remote_model_proxy.py` | Remote LlamaFarm inference proxy |
| `config/federation.py` | Peer configuration |
| `vision_training/audit_pipeline.py` | Automatic model-checking-model |
| `vision_training/validation_gate.py` | Blue/green validation before promote |
| `vision_training/model_package.py` | Portable model packages |
| `routers/vision/federation.py` | Federation API + escalation inbound endpoint |
| `tests/test_escalation_envelope.py` | Envelope/opinion flow tests |
| `tests/test_audit_pipeline.py` | Audit disagreement tests |

### Modified Files (8)
| File | Key Changes |
|------|-------------|
| `vision_training/replay_buffer.py` | ModelOpinion, structured bbox/mask, SQLite persistence |
| `models/streaming_vision.py` | Cascade chain loop, cross-modal enrichment, envelope building |
| `storage/image_store.py` | detection_history table, reviewer columns |
| `api_types/vision.py` | ModelOpinion, EscalationEnvelope, enhanced cascade/review types |
| `vision_training/auto_trainer.py` | Validation gate, fix class map, on_model_promoted |
| `vision_training/trainer.py` | Model versioning, return model_path, wire EWC |
| `server/services/vision/review_service.py` | Fix prediction=None, bboxes in items, model-as-reviewer |
| `tests/test_vision_cascade.py` | 3-hop chain, envelope accumulation tests |

All paths relative to `runtimes/universal/` unless noted.

---

## Implementation Order

```
Phase 1: EscalationEnvelope + Persistent Replay Buffer
  |       (foundation -- everything else depends on this shape)
  v
Phase 2: Cross-Modal Enrichment in Cascade
  |       (bbox + seg + CLIP flow forward on escalation)
  v
Phase 3: Remote Model Proxy  <----------->  Phase 4: Audit Pipeline
  |       (same DetectionModel              (uses same proxy for
  |        interface, local or remote)        remote audit models)
  v                                           |
Phase 5: Validation Gate + Blue/Green         |
  |       (auto-trainer wires through here)   |
  v                                           v
Phase 6: Model Packaging  <--- triggered by Phase 5 promote
  |
  v
Phase 7: Review Queue Fix (bboxes + all opinions visible)
  |
  v
Phase 8: Tests (incremental with each phase)
```

---

## Verification

After each phase, run: `cd runtimes/universal && python -m pytest tests/ -x`

End-to-end integration test scenario:
1. Start streaming session with cascade_chain=["yolov8n", "yolov8m"]
2. Send frame with object at 0.55 confidence (mid-range)
3. Verify: seg enrichment ran, bbox+mask attached, escalated to yolov8m
4. yolov8m returns 0.85 confidence -> verify "action" returned
5. Verify: replay buffer has sample with 2 ModelOpinions, bbox, mask
6. Fill replay buffer to threshold -> auto-trainer triggers
7. Validation gate runs -> candidate promoted or rejected
8. Check review queue items have bounding boxes and all model opinions
