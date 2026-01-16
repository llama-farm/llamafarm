# ML API Alignment & Quality Assurance Plan

## Overview
This plan ensures the LlamaFarm API properly proxies all Universal Runtime ML features with consistent types, constraints, and best practices.

---

## Phase 1: Sync Types Between LlamaFarm Server and Universal Runtime

### 1.1 Add Missing Fields to Anomaly Detection Types
- [x] Add `scaler_type: Literal["standard", "robust"] = "robust"` to `AnomalyFitRequest`
- [x] Add `validation_split: float = Field(default=0.1, ge=0, le=0.5)` to `AnomalyFitRequest`
- [x] Add `patience: int = Field(default=10, ge=1)` to `AnomalyFitRequest`
- [x] Add `min_delta: float = Field(default=1e-4, ge=0)` to `AnomalyFitRequest`
- [x] Add `training_file: str | None = None` to `AnomalyFitRequest`
- [x] Add `scaler_type: Literal["standard", "robust"] = "robust"` to `AnomalyScoreRequest`

**Files modified:**
- `server/api/routers/ml/types.py`
- `server/api/routers/ml/router.py`
- `server/services/universal_runtime_service.py`

### 1.2 Add Constraints to Universal Runtime Types
- [x] Add `contamination: float = Field(default=0.1, gt=0, le=0.5)` constraint in Universal Runtime
- [x] Add `threshold: float = Field(default=0.5, ge=0, le=1.0)` constraints where applicable
- [x] Add `top_k: int = Field(default=10, ge=1, le=100)` constraints for keyword extraction

**Files to modify:**
- `runtimes/universal/server.py` (Pydantic models section)

---

## Phase 2: Expose Missing Endpoints via LlamaFarm API

### 2.1 Add Embeddings Endpoint (HIGH PRIORITY)
- [x] Create `POST /v1/ml/embeddings` endpoint
- [x] Add `EmbeddingsRequest` type
- [x] Proxy to Universal Runtime `/v1/embeddings`

### 2.2 Add Text/NLP Endpoints
- [x] Create `POST /v1/ml/nlp/language` - Language detection
- [x] Create `POST /v1/ml/nlp/language/batch` - Batch language detection
- [x] Create `POST /v1/ml/nlp/keywords` - Keyword extraction
- [x] Create `POST /v1/ml/nlp/keywords/batch` - Batch keyword extraction
- [ ] Create `POST /v1/ml/nlp/ner` - Named entity recognition (skipped - low priority)
- [x] Create `POST /v1/ml/nlp/pii/detect` - PII detection
- [x] Create `POST /v1/ml/nlp/redact` - PII redaction

### 2.3 Add Time Series Endpoints
- [x] Create `POST /v1/ml/timeseries/forecast` - Time series forecasting
- [ ] Create `POST /v1/ml/timeseries/forecast/batch` - Batch forecasting (skipped - low priority)
- [x] Create `POST /v1/ml/timeseries/changepoints` - Change point detection

### 2.4 Add Vision/Detection Endpoints
- [ ] Create `POST /v1/ml/vision/classify` - Zero-shot image classification (skipped - low priority)
- [ ] Create `POST /v1/ml/vision/classify/batch` - Batch classification (skipped - low priority)
- [x] Create `POST /v1/ml/vision/detect` - Object detection (YOLO)
- [x] Create `POST /v1/ml/vision/detect/batch` - Batch object detection
- [ ] Create `POST /v1/ml/vision/detect-open` - Open-vocabulary detection (skipped - low priority)
- [ ] Create `POST /v1/ml/vision/segment` - Image segmentation (skipped - low priority)
- [x] Create `POST /v1/ml/vision/background-remove` - Background removal

### 2.5 Add Analysis Endpoints
- [x] Create `POST /v1/ml/analysis/table-qa` - Table question answering
- [x] Create `POST /v1/ml/analysis/dataset-audit` - Dataset quality audit
- [x] Create `POST /v1/ml/analysis/drift` - Drift detection
- [x] Create `POST /v1/ml/anomaly/explain` - SHAP-based anomaly explanation

**Files modified:**
- `server/api/routers/ml/router.py` - Added 17 new endpoints
- `server/api/routers/ml/types.py` - Added 15 new request types
- `server/services/universal_runtime_service.py` - Added 15 new service methods

---

## Phase 3: Code Quality Improvements in Universal Runtime

### 3.1 Extract Shared Utilities (HIGH PRIORITY)
- [ ] Create `runtimes/universal/utils/image_utils.py`
- [ ] Extract `_load_image()` function from vision_model.py
- [ ] Update all models to use shared `load_image()`:
  - [ ] `models/vision_model.py`
  - [ ] `models/few_shot_classifier.py`
  - [ ] `models/object_detection_model.py`
  - [ ] `models/open_vocab_detection_model.py`
  - [ ] `models/background_removal_model.py`
  - [ ] `models/ocr_model.py`

### 3.2 Fix Model Cache Cleanup (MEDIUM PRIORITY)
- [ ] Add missing caches to lifespan shutdown in `server.py`:
  - [ ] `_lang_detection_models`
  - [ ] `_pii_models`
  - [ ] `_object_detection_models`
  - [ ] `_background_removal_models`
  - [ ] `_timeseries_models`
- [ ] Add missing caches to `_cleanup_idle_models()` background task

### 3.3 Security Improvements
- [ ] Add model allowlist validation for `trust_remote_code=True` in `background_removal_model.py`
- [ ] Document security implications in docstrings

### 3.4 Minor Improvements
- [ ] Add MPS fallback logging in `table_qa_model.py`
- [ ] Fix Chronos-Bolt docstring in `timeseries_model.py`
- [ ] Make text truncation configurable in `language_detection_model.py`

---

## Phase 4: Run Tests and Examples

### 4.1 Run Universal Runtime Tests
- [ ] Run `cd runtimes/universal && uv run pytest tests/ -v`
- [ ] Fix any failing tests

### 4.2 Run Example Scripts
- [ ] Run `examples/ml/test_anomaly.sh`
- [ ] Run `examples/ml/test_anomaly_explain.sh`
- [ ] Run `examples/ml/test_background_removal.sh`
- [ ] Run `examples/ml/test_classifier.sh`
- [ ] Run `examples/ml/test_clip.sh`
- [ ] Run `examples/ml/test_dataset_audit.sh`
- [ ] Run `examples/ml/test_document.sh`
- [ ] Run `examples/ml/test_drift_detection.sh`
- [ ] Run `examples/ml/test_encoder.sh`
- [ ] Run `examples/ml/test_keywords.sh`
- [ ] Run `examples/ml/test_language_detection.sh`
- [ ] Run `examples/ml/test_object_detection.sh`
- [ ] Run `examples/ml/test_ocr.sh`
- [ ] Run `examples/ml/test_pii_redaction.sh`
- [ ] Run `examples/ml/test_time_series.sh`

### 4.3 Run Demo Scripts
- [ ] Run `examples/ml/demo-full-ml-suite.sh`
- [ ] Run `examples/ml/demo-anomaly-explain.sh`
- [ ] Run `examples/ml/demo-pii-redaction.sh`
- [ ] Run `examples/ml/demo-timeseries-forecast.sh`

---

## Phase 5: Commit and Push

### 5.1 Prepare Commit
- [ ] Run linters (`ruff check`, `ruff format`)
- [ ] Ensure all tests pass
- [ ] Review changes with `git diff`

### 5.2 Create PR
- [ ] Commit with descriptive message
- [ ] Push to `feat/ml` branch
- [ ] Create PR with summary of changes

---

## Summary Table

| Phase | Priority | Items |
|-------|----------|-------|
| Phase 1: Sync Types | High | 9 items |
| Phase 2: Add Endpoints | High | 20 items |
| Phase 3: Code Quality | Medium | 15 items |
| Phase 4: Tests | High | 20 items |
| Phase 5: Commit | High | 4 items |
| **Total** | | **68 items** |

---

## Key Findings from Analysis

### API Gaps (38 endpoints missing from LlamaFarm API)
The LlamaFarm server currently only exposes ~13 out of 51 Universal Runtime endpoints (25% coverage).

**Missing high-value endpoints:**
- Embeddings (`/v1/embeddings`)
- Language detection, keywords, PII
- Time series forecasting
- Vision classification and detection
- Table QA

### Type Mismatches
- Server's `AnomalyFitRequest` missing: `scaler_type`, `validation_split`, `patience`, `min_delta`
- Server has better constraints (`contamination: gt=0, le=0.5`) that should be added to Universal Runtime

### Code Quality Issues
1. **DRY violation**: `_load_image()` duplicated in 6+ model files
2. **Incomplete cleanup**: Model caches not properly cleaned on shutdown
3. **Security**: `trust_remote_code=True` needs validation
