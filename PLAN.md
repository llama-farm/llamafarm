# Plan: Universal Runtime ML Enhancements

## Overview

This plan implements two major categories of improvements to the Universal Runtime's ML capabilities:

1. **Part 1: Core ML Infrastructure Improvements** - Fix scalability and robustness issues in the existing anomaly detection and classifier systems
2. **Part 2: New ML Tool Additions** - Add new ML capabilities for vision, NLP, time-series, and advanced anomaly detection

The work is organized into phases, with each phase independently testable and deployable.

## Agents to Use

- **llamafarm** - For ML model implementation, API endpoint design, testing
- **backend-architect** - For FastAPI endpoint patterns, async architecture
- **test-runner** - After each phase to run and verify tests
- **debugger** - If any tests fail, to fix issues
- **senior-code-reviewer** - After each phase for code quality review
- **demo-builder** - To create phase demos
- **smart-committer** - After completing each phase

## LlamaFarm API Usage

Existing endpoints to modify:
- `POST /v1/anomaly/fit` - Add async training support, RobustScaler, VAE option
- `POST /v1/anomaly/detect` - No changes to interface
- `POST /v1/classifier/fit` - Add async training support

New endpoints to add:
- `POST /v1/vision/classify-zero-shot` - Zero-shot image classification (CLIP)
- `POST /v1/vision/detect-objects` - Object detection (YOLOS)
- `POST /v1/vision/segment` - Background removal (RMBG)
- `POST /v1/text/language` - Language identification
- `POST /v1/text/keywords` - Keyword/keyphrase extraction
- `POST /v1/text/pii-redact` - PII redaction (GLiNER)
- `POST /v1/timeseries/forecast` - Time-series forecasting (Chronos-Bolt)
- `POST /v1/timeseries/changepoints` - Change point detection (Ruptures)
- `POST /v1/analysis/table-qa` - Table question answering (TAPAS)
- `POST /v1/anomaly/explain` - SHAP-based anomaly explanation
- `POST /v1/dataset/audit` - Dataset quality audit (Cleanlab)
- `GET /v1/streaming/drift` - Concept drift detection stream (River)

---

## Phase 1: Non-Blocking Training Infrastructure

### Phase 1 Tests (Define FIRST)
- [x] Test: Anomaly detector `/fit` endpoint returns immediately while training runs in background
- [x] Test: Health check `/health` responds during training
- [x] Test: Concurrent fit requests are handled gracefully (queued or rejected with proper error)
- [x] Test: Training completion updates model state correctly
- [x] Test: Error during background training is propagated to client appropriately
- [x] Test file: `runtimes/universal/tests/test_async_training.py`

### Phase 1 Demo (Define FIRST)
- [x] Demo script: `examples/ml/demo-async-training.sh`
- [x] Demo shows: Start training, immediately call health check (must respond), poll for completion
- [x] Expected output: Health checks return 200ms response times even during 30s+ training

### Phase 1 Implementation
- [x] Create `utils/training_executor.py` - Thread/process pool for CPU-bound training
- [x] Modify `AnomalyModel.fit()` to offload to executor via `loop.run_in_executor()`
- [x] Modify `ClassifierModel.fit()` to offload SetFit training to executor
- [x] Add training status tracking (in-progress, completed, failed)
- [x] Update `/v1/anomaly/fit` and `/v1/classifier/fit` endpoints to use async training
- [ ] Add optional `async_mode=true` parameter for polling-based training
- [ ] Implement training job status endpoint `/v1/training/{job_id}/status`

### Phase 1 Verification
- [x] Run tests: `cd runtimes/universal && uv run pytest tests/test_async_training.py -v`
- [x] All tests pass (11/11)
- [x] Run demo: `bash examples/ml/demo-async-training.sh`
- [x] Demo runs successfully (health checks respond during training - avg 0.2s < 0.5s threshold)

### Phase 1 Checkpoint
- [x] Tests verified passing
- [x] Demo verified working
- [x] Code review completed (senior-code-reviewer) - fixed unused variable and blind exception catch
- [x] **COMMIT**: `git commit -m "feat(universal-runtime): add non-blocking async training for ML models"`
- [x] Ready for Phase 2

---

## Phase 2: Anomaly Detection Robustness

### Phase 2 Tests (Define FIRST)
- [x] Test: RobustScaler handles outliers in training data better than StandardScaler
- [x] Test: Models trained with RobustScaler detect anomalies when training data has outliers
- [x] Test: `scaler_type` parameter correctly selects scaler
- [x] Test: Backward compatibility - existing models load correctly
- [x] Test file: `runtimes/universal/tests/test_anomaly_robustness.py`

### Phase 2 Demo (Define FIRST)
- [x] Demo script: `examples/ml/demo-robust-scaler.sh`
- [x] Demo shows: Train detector on data WITH outliers, show improved anomaly detection
- [x] Expected output: RobustScaler model detects true anomalies that StandardScaler misses

### Phase 2 Implementation
- [x] Add `RobustScaler` option to `AnomalyModel` (`scaler_type: "robust" | "standard"`)
- [x] Default new models to `RobustScaler` while maintaining backward compatibility
- [x] Store scaler type in saved model metadata
- [x] Update `/v1/anomaly/fit` to accept `scaler_type` parameter
- [x] Update model save/load to preserve scaler type

### Phase 2 Verification
- [x] Run tests: all Phase 2 tests pass (12/12)
- [x] Run demo: demo runs successfully
- [x] Verify backward compatibility with existing saved models

### Phase 2 Checkpoint
- [x] Tests verified passing
- [x] Demo verified working
- [x] Backward compatibility verified
- [x] **COMMIT**: `git commit -m "feat(anomaly): add RobustScaler option for outlier-resistant training"`
- [x] Ready for Phase 3

---

## Phase 3: Autoencoder Improvements (Early Stopping + VAE)

### Phase 3 Tests (Define FIRST)
- [x] Test: Early stopping triggers when validation loss plateaus
- [x] Test: Early stopping reduces training time on easy datasets
- [x] Test: VAE backend produces reconstruction + KL divergence loss
- [x] Test: VAE latent space is continuous (samples from prior are valid)
- [x] Test: VAE anomaly scores are statistically interpretable
- [x] Test file: `runtimes/universal/tests/test_autoencoder_improvements.py`

### Phase 3 Demo (Define FIRST)
- [x] Demo script: `examples/ml/demo-vae-anomaly.sh`
- [x] Demo shows: Train VAE on normal data, detect anomalies with probability scores
- [x] Expected output: VAE provides log-likelihood scores for anomaly detection

### Phase 3 Implementation
- [x] Add `EarlyStopping` callback to autoencoder training
  - Monitor validation loss (10% holdout)
  - Patience parameter (default: 10 epochs)
  - Restore best weights on early stop
- [x] Add `vae` backend option to `AnomalyModel`
  - Implement VAE architecture with encoder/decoder + latent sampling
  - ELBO loss = reconstruction + KL divergence
  - Anomaly score = negative log-likelihood
- [x] Add `validation_split` parameter to autoencoder fit
- [x] Update pyproject.toml if any new dependencies needed (none needed)

### Phase 3 Verification
- [x] Run tests: all Phase 3 tests pass (11/11)
- [x] Run demo: demo runs successfully
- [x] VAE anomaly detection works with at least 90% accuracy on synthetic test (100% detection rate)

### Phase 3 Checkpoint
- [x] Tests verified passing
- [x] Demo verified working
- [x] Code review completed (ruff: all checks passed)
- [x] **COMMIT**: `git commit -m "feat(anomaly): add early stopping and VAE backend for autoencoder"`
- [x] Ready for Phase 4

---

## Phase 4: Memory Management for Large Datasets

### Phase 4 Tests (Define FIRST)
- [x] Test: Streaming upload endpoint accepts CSV/Parquet files
- [x] Test: Large file (>100MB) doesn't OOM the server
- [x] Test: Streaming training uses chunked processing
- [x] Test: Memory usage stays bounded during large dataset training
- [x] Test file: `runtimes/universal/tests/test_large_dataset_training.py`

### Phase 4 Demo (Define FIRST)
- [x] Demo script: `examples/ml/demo-streaming-training.sh`
- [x] Demo shows: Upload CSV, train without OOM (100K rows trained in 1.6s)
- [x] Expected output: Memory stays bounded during training

### Phase 4 Implementation
- [x] Create `/v1/anomaly/upload-training-data` endpoint for file uploads
  - Accept CSV, Parquet, JSON Lines formats
  - Save to temp file in `~/.llamafarm/temp/streaming/`
  - Return file reference for subsequent fit call
- [x] Add `training_file` parameter to `/v1/anomaly/fit`
  - Stream data from file instead of in-memory list
  - Process in batches to limit memory
- [x] Implement streaming fit for supported backends
  - All backends (isolation_forest, autoencoder, vae, etc.)
  - Uses `fit_from_file` method in AnomalyModel
- [x] Created `utils/streaming_data.py` module
  - StreamingDataLoader for file management
  - FileReference for batch iteration
  - Support for CSV, JSON Lines, Parquet formats
- [x] Clean up temp files after training completes

### Phase 4 Verification
- [x] Run tests: all Phase 4 tests pass (11/11)
- [x] Run demo: demo runs successfully (100K rows trained)
- [x] Memory profiling shows bounded usage (<200MB for 100K rows)

### Phase 4 Checkpoint
- [x] Tests verified passing
- [x] Demo verified working
- [x] Memory efficiency verified
- [x] **COMMIT**: `git commit -m "feat(anomaly): add streaming training support for large datasets"`
- [x] Ready for Phase 5

---

## Phase 5: Zero-Shot Image Classification (CLIP)

### Phase 5 Tests (Define FIRST)
- [x] Test: CLIP model loads successfully
- [x] Test: Zero-shot classification returns probabilities for all labels
- [x] Test: Classification works with various image formats (PNG, JPEG, WebP)
- [x] Test: Labels are case-insensitive
- [x] Test: Batch processing multiple images works
- [x] Test file: `runtimes/universal/tests/test_vision_clip.py`

### Phase 5 Demo (Define FIRST)
- [x] Demo script: `examples/ml/demo-clip-classification.sh`
- [x] Demo shows: Upload image, classify as "receipt", "contract", or "ID card"
- [x] Expected output: Returns probability distribution over labels

### Phase 5 Implementation
- [x] Create `models/vision_model.py` with `CLIPVisionModel` class
  - Load `openai/clip-vit-base-patch32` from HuggingFace
  - Support image input: file path, base64, URL
  - Support custom label lists
- [x] Create `/v1/vision/classify-zero-shot` endpoint
  - Request: `{"image": "<base64 or path>", "labels": ["receipt", "contract", "id_card"]}`
  - Response: `{"label": "receipt", "score": 0.87, "all_scores": {...}}`
- [x] Add to `pyproject.toml` optional dependency: `vision = [...]`
- [x] Lazy import to avoid loading CLIP unless endpoint is called

### Phase 5 Verification
- [x] Run tests: all Phase 5 tests pass (11/11)
- [x] Run demo: demo runs successfully
- [x] Classification accuracy > 80% on test images (cat: 97.6%, horse: 100%)

### Phase 5 Learnings
- **IMPORTANT**: Large images (>500KB) can cause terminal/server issues when base64 encoded
- Demo script resizes images to max 512px before encoding to avoid memory issues
- Vision model `_load_image()` handles long base64 strings by skipping file path check for strings > 4096 chars

### Phase 5 Checkpoint
- [x] Tests verified passing
- [x] Demo verified working
- [x] Code review completed
- [x] **COMMIT**: `git commit -m "feat(vision): add zero-shot image classification with CLIP"`
- [x] Ready for Phase 6

---

## Phase 6: Language Identification

### Phase 6 Tests (Define FIRST)
- [x] Test: Language detection model loads successfully
- [x] Test: Correctly identifies English, Spanish, French, German, Chinese, Japanese
- [x] Test: Returns confidence scores
- [x] Test: Batch processing multiple texts works
- [x] Test file: `runtimes/universal/tests/test_language_detection.py`

### Phase 6 Demo (Define FIRST)
- [x] Demo script: `examples/ml/demo-language-detection.sh`
- [x] Demo shows: Detect language of multilingual texts
- [x] Expected output: Correct language with >90% confidence

### Phase 6 Implementation
- [x] Create new `LanguageDetectionModel`
  - Use `papluca/xlm-roberta-base-language-detection`
  - Pre-trained, no training needed
- [x] Create `/v1/text/language` endpoint
  - Request: `{"text": "Hello world", "top_k": 5}`
  - Response: `{"language": "en", "language_name": "English", "confidence": 0.99, "all_scores": {...}}`
- [x] Create `/v1/text/language/batch` endpoint for batch processing
- [x] Support ISO 639-1 language codes in output (20 languages supported)

### Phase 6 Verification
- [x] Run tests: all Phase 6 tests pass (17/17)
- [x] Run demo: demo runs successfully (all 7 tests pass)

### Phase 6 Checkpoint
- [x] Tests verified passing
- [x] Demo verified working
- [x] **COMMIT**: `git commit -m "feat(text): add language identification endpoint"`
- [x] Ready for Phase 7

---

## Phase 7: Keyword/Keyphrase Extraction

### Phase 7 Tests (Define FIRST)
- [x] Test: Keyword extraction returns ranked keywords
- [x] Test: N-gram generation (1-3 words) works correctly
- [x] Test: Cosine similarity ranking produces relevant keywords
- [x] Test: Works with documents of various lengths
- [x] Test file: `runtimes/universal/tests/test_keyword_extraction.py`

### Phase 7 Demo (Define FIRST)
- [x] Demo script: `examples/ml/demo-keyword-extraction.sh`
- [x] Demo shows: Extract keywords from a technical document
- [x] Expected output: Top 10 relevant keyphrases with scores

### Phase 7 Implementation
- [x] Create `utils/keyword_extractor.py`
  - Generate n-gram candidates (1-3 words)
  - Embed document using existing sentence-transformers
  - Embed candidate phrases
  - Rank by cosine similarity to document embedding
  - Return top-k keywords with scores
  - MMR (Maximal Marginal Relevance) for diversity
- [x] Create `/v1/text/keywords` endpoint
  - Request: `{"text": "...", "top_k": 10, "ngram_range": [1, 3], "diversity": 0.5}`
  - Response: `{"keywords": [{"keyword": "machine learning", "score": 0.89}, ...], "count": 10}`
- [x] Create `/v1/text/keywords/batch` endpoint for batch processing
- [x] No new dependencies (reuse existing embeddings infrastructure)

### Phase 7 Verification
- [x] Run tests: all Phase 7 tests pass (20/20)
- [x] Run demo: demo runs successfully (5 tests)

### Phase 7 Checkpoint
- [x] Tests verified passing
- [x] Demo verified working
- [x] **COMMIT**: `git commit -m "feat(text): add keyword and keyphrase extraction endpoint"`
- [x] Ready for Phase 8

---

## Phase 8: PII Redaction (GLiNER)

### Phase 8 Tests (Define FIRST)
- [x] Test: GLiNER model loads successfully
- [x] Test: Detects standard PII types (SSN, phone, email, credit card)
- [x] Test: Custom entity types work (e.g., "medical record number")
- [x] Test: Redaction replaces PII with placeholders
- [x] Test: Returns both original positions and redacted text
- [x] Test file: `runtimes/universal/tests/test_pii_detection.py`

### Phase 8 Demo (Define FIRST)
- [x] Demo script: `examples/ml/demo-pii-redaction.sh`
- [x] Demo shows: Redact PII from sample text with custom replacements
- [x] Expected output: Text with [REDACTED], [NAME], [EMAIL] placeholders

### Phase 8 Implementation
- [x] Create `models/pii_model.py` with `PIIModel` class
  - Load `urchade/gliner_small-v2.1`
  - Support dynamic entity type specification
  - Zero-shot NER without retraining
  - Regex patterns for high-precision common PII (email, phone, SSN, IP, credit card)
- [x] Create `/v1/text/pii-detect` endpoint for detection only
- [x] Create `/v1/text/pii-redact` endpoint for detection + redaction
  - Support custom replacement strings
  - Support per-entity-type replacement map
- [x] Add GLiNER dependency to pyproject.toml

### Phase 8 Verification
- [x] Run tests: all Phase 8 tests pass (22/22)
- [x] Run demo: demo runs successfully (5 tests)

### Phase 8 Checkpoint
- [x] Tests verified passing
- [x] Demo verified working
- [x] **COMMIT**: `git commit -m "feat(text): add PII redaction endpoint with GLiNER"`
- [x] Ready for Phase 9

---

## Phase 9: Object Detection (YOLOS)

### Phase 9 Tests (Define FIRST)
- [ ] Test: YOLOS model loads successfully
- [ ] Test: Detects objects with bounding boxes
- [ ] Test: Returns confidence scores for each detection
- [ ] Test: Works with various image formats
- [ ] Test file: `runtimes/universal/tests/test_object_detection.py`

### Phase 9 Demo (Define FIRST)
- [ ] Demo script: `examples/ml/demo-object-detection.sh`
- [ ] Demo shows: Detect objects in an image
- [ ] Expected output: List of objects with bounding boxes and confidence scores

### Phase 9 Implementation
- [ ] Create `/v1/vision/detect-objects` endpoint
  - Use `hustvl/yolos-tiny` (lightweight transformer-based)
  - Request: `{"image": "<base64 or path>", "confidence_threshold": 0.5}`
  - Response: `{"detections": [{"label": "car", "confidence": 0.95, "bbox": [x1, y1, x2, y2]}, ...]}`
- [ ] Add to vision model infrastructure

### Phase 9 Verification
- [ ] Run tests: all Phase 9 tests pass
- [ ] Run demo: demo runs successfully

### Phase 9 Checkpoint
- [ ] Tests verified passing
- [ ] Demo verified working
- [ ] **COMMIT**: `git commit -m "feat(vision): add object detection endpoint with YOLOS"`
- [ ] Ready for Phase 10

---

## Phase 10: Background Removal (RMBG)

### Phase 10 Tests (Define FIRST)
- [ ] Test: RMBG model loads successfully
- [ ] Test: Returns image with transparent background
- [ ] Test: Works with various image formats
- [ ] Test: Output is valid PNG with alpha channel
- [ ] Test file: `runtimes/universal/tests/test_background_removal.py`

### Phase 10 Demo (Define FIRST)
- [ ] Demo script: `examples/ml/demo-background-removal.sh`
- [ ] Demo shows: Remove background from product image
- [ ] Expected output: PNG with transparent background

### Phase 10 Implementation
- [ ] Create `/v1/vision/segment` endpoint
  - Use `briaai/RMBG-1.4`
  - Request: `{"image": "<base64 or path>", "return_mask": false}`
  - Response: `{"image": "<base64 PNG>"}` or `{"mask": "<base64 mask>"}`
- [ ] Support output formats: PNG (transparent), mask only

### Phase 10 Verification
- [ ] Run tests: all Phase 10 tests pass
- [ ] Run demo: demo runs successfully

### Phase 10 Checkpoint
- [ ] Tests verified passing
- [ ] Demo verified working
- [ ] **COMMIT**: `git commit -m "feat(vision): add background removal endpoint with RMBG"`
- [ ] Ready for Phase 11

---

## Phase 11: Time-Series Forecasting (Chronos-Bolt)

### Phase 11 Tests (Define FIRST)
- [ ] Test: Chronos model loads successfully
- [ ] Test: Forecasts future values from historical data
- [ ] Test: Returns prediction intervals (uncertainty quantification)
- [ ] Test: Handles various time-series lengths
- [ ] Test file: `runtimes/universal/tests/test_timeseries_forecast.py`

### Phase 11 Demo (Define FIRST)
- [ ] Demo script: `examples/ml/demo-timeseries-forecast.sh`
- [ ] Demo shows: Forecast next 7 days from 30 days of data
- [ ] Expected output: Point forecasts with confidence intervals

### Phase 11 Implementation
- [ ] Create `models/timeseries_model.py` with `ChronosModel` class
  - Load `amazon/chronos-bolt-small`
  - Tokenize time-series data
  - Generate forecasts with quantile predictions
- [ ] Create `/v1/timeseries/forecast` endpoint
  - Request: `{"values": [1.0, 2.0, ...], "horizon": 7, "quantiles": [0.1, 0.5, 0.9]}`
  - Response: `{"forecasts": [{"point": 2.5, "lower": 2.0, "upper": 3.0}, ...]}`
- [ ] Add to `pyproject.toml`: `trends = ["chronos-forecasting>=1.0.0"]`

### Phase 11 Verification
- [ ] Run tests: all Phase 11 tests pass
- [ ] Run demo: demo runs successfully

### Phase 11 Checkpoint
- [ ] Tests verified passing
- [ ] Demo verified working
- [ ] **COMMIT**: `git commit -m "feat(timeseries): add forecasting endpoint with Chronos-Bolt"`
- [ ] Ready for Phase 12

---

## Phase 12: Change Point Detection (Ruptures)

### Phase 12 Tests (Define FIRST)
- [x] Test: Ruptures algorithms load successfully
- [x] Test: Detects change points in synthetic signal
- [x] Test: Multiple algorithms available (Pelt, Binseg, Window, BottomUp)
- [x] Test: Returns change point indices
- [x] Test file: `runtimes/universal/tests/test_changepoint_detection.py` (37 tests)

### Phase 12 Demo (Define FIRST)
- [x] Demo script: `examples/ml/demo-changepoint-detection.sh`
- [x] Demo shows: Detect when CPU usage shifted from baseline
- [x] Expected output: Index where the change occurred

### Phase 12 Implementation
- [x] Create `utils/changepoint_detector.py`
  - Wrap Ruptures algorithms (Pelt, Binseg, Window, BottomUp)
  - Support different cost functions (l1, l2, rbf, normal, ar)
- [x] Create `/v1/timeseries/changepoints` endpoint
  - Request: `{"values": [...], "algorithm": "pelt", "penalty": 10, "min_size": 5}`
  - Response: `{"change_points": [45, 112], "segments": [{"start":0,"end":45}, ...]}`
- [x] Create `/v1/timeseries/changepoints/batch` endpoint
- [x] ruptures already in `pyproject.toml`

### Phase 12 Verification
- [x] Run tests: all 37 Phase 12 tests pass
- [x] Run demo: demo runs successfully

### Phase 12 Checkpoint
- [x] Tests verified passing
- [x] Demo verified working
- [x] **COMMIT**: `git commit -m "feat(timeseries): add change point detection endpoint with Ruptures"`
- [x] Ready for Phase 13

---

## Phase 13: Table Question Answering (TAPAS)

### Phase 13 Tests (Define FIRST)
- [x] Test: TAPAS model loads successfully
- [x] Test: Answers questions about table data
- [x] Test: Handles aggregation queries (sum, average, count)
- [x] Test: Works with JSON table format
- [x] Test file: `runtimes/universal/tests/test_table_qa.py` (23 tests)

### Phase 13 Demo (Define FIRST)
- [x] Demo script: `examples/ml/demo-table-qa.sh`
- [x] Demo shows: Ask "Which server had the highest CPU?" about metrics table
- [x] Expected output: Natural language answer with referenced cells

### Phase 13 Implementation
- [x] Create `/v1/analysis/table-qa` endpoint
  - Use `google/tapas-base-finetuned-wtq`
  - Request: `{"table": {"columns": [...], "rows": [...]}, "question": "..."}`
  - Response: `{"answer": "...", "aggregation": "AVERAGE", "cells": [...]}`
- [x] Create `/v1/analysis/table-qa/batch` endpoint for multiple questions
- [x] Create `models/table_qa_model.py` with TableQAModel class

### Phase 13 Verification
- [x] Run tests: all 23 Phase 13 tests pass
- [x] Run demo: demo runs successfully

### Phase 13 Checkpoint
- [x] Tests verified passing
- [x] Demo verified working
- [x] **COMMIT**: `git commit -m "feat(analysis): add table question answering endpoint with TAPAS"`
- [x] Ready for Phase 14

---

## Phase 14: Concept Drift Detection (River)

### Phase 14 Tests (Define FIRST)
- [x] Test: ADWIN drift detector initializes correctly
- [x] Test: Detects concept drift in simulated changing distribution
- [x] Test: Streaming updates work correctly
- [x] Test: Drift alerts are accurate
- [x] Test file: `runtimes/universal/tests/test_drift_detection.py` (31 tests)

### Phase 14 Demo (Define FIRST)
- [x] Demo script: `examples/ml/demo-drift-detection.sh`
- [x] Demo shows: Stream data, detect when distribution changes
- [x] Expected output: Drift alert with index

### Phase 14 Implementation
- [x] Create `utils/drift_detector.py` with `DriftDetector` class
  - Wrap River's ADWIN, PageHinkley, KSWIN, DDM detectors
  - Support online updates
  - Emit drift alerts
- [x] Create `/v1/streaming/drift/detect` endpoint for batch detection
- [x] Create `/v1/streaming/drift/create` endpoint for stateful detector
- [x] Create `/v1/streaming/drift/update/{id}` endpoint for streaming
- [x] Create `/v1/streaming/drift/state/{id}` endpoint for state
- [x] Add river>=0.23.0 to dependencies

### Phase 14 Verification
- [x] Run tests: all 31 Phase 14 tests pass
- [x] Run demo: demo runs successfully

### Phase 14 Checkpoint
- [x] Tests verified passing
- [x] Demo verified working
- [x] **COMMIT**: `git commit -m "feat(streaming): add concept drift detection endpoint with River"`
- [x] Ready for Phase 15

---

## Phase 15: SHAP Explanations for Anomalies

### Phase 15 Tests (Define FIRST)
- [x] Test: SHAP explainer initializes for trained anomaly model
- [x] Test: Returns feature importance for anomalous points
- [x] Test: Explanations are human-readable
- [x] Test: Works with all anomaly backends (IF, OC-SVM, LOF)
- [x] Test file: `runtimes/universal/tests/test_shap_explanations.py`

### Phase 15 Demo (Define FIRST)
- [x] Demo script: `examples/ml/demo-anomaly-explain.sh`
- [x] Demo shows: Explain WHY a data point is anomalous
- [x] Expected output: "Anomaly because 'CPU' is high AND 'Memory' is low"

### Phase 15 Implementation
- [x] Create `/v1/anomaly/explain` endpoint
  - Use SHAP KernelExplainer (model-agnostic)
  - Request: `{"model": "...", "data": [[...]], "background_samples": 100}`
  - Response: `{"explanations": [{"feature": "cpu", "importance": 0.82, "value": 95.0}, ...]}`
- [x] Add to `pyproject.toml`: `issues = [..., "shap>=0.44.0"]`
- [x] Lazy import SHAP only when endpoint is called

### Phase 15 Verification
- [x] Run tests: all Phase 15 tests pass (16 tests)
- [x] Run demo: demo runs successfully

### Phase 15 Checkpoint
- [x] Tests verified passing
- [x] Demo verified working
- [x] **COMMIT**: `git commit -m "feat(anomaly): add SHAP-based anomaly explanation endpoint"`
- [x] Ready for Phase 16

---

## Phase 16: Dataset Quality Audit (Cleanlab)

### Phase 16 Tests (Define FIRST)
- [x] Test: Cleanlab finds label errors in synthetic noisy dataset
- [x] Test: Returns confidence scores for each label
- [x] Test: Identifies near-duplicates
- [x] Test: Works with classifier training data format
- [x] Test file: `runtimes/universal/tests/test_dataset_audit.py`

### Phase 16 Demo (Define FIRST)
- [x] Demo script: `examples/ml/demo-dataset-audit.sh`
- [x] Demo shows: Find mislabeled examples in training data
- [x] Expected output: List of potentially mislabeled samples

### Phase 16 Implementation
- [x] Create `/v1/dataset/audit` endpoint
  - Use Cleanlab's `find_label_issues`
  - Request: `{"texts": [...], "labels": [...], "check_duplicates": true}`
  - Response: `{"label_issues": [{"index": 5, "given_label": "A", "suggested_label": "B", "confidence": 0.23}], "duplicates": [...]}`
- [x] Create `/v1/dataset/quality-scores` endpoint for per-sample scores
- [x] Add to `pyproject.toml`: `cleanlab>=2.9.0`

### Phase 16 Verification
- [x] Run tests: all Phase 16 tests pass (20 tests)
- [x] Run demo: demo runs successfully

### Phase 16 Checkpoint
- [x] Tests verified passing
- [x] Demo verified working
- [x] **COMMIT**: `git commit -m "feat(dataset): add quality audit endpoint with Cleanlab"`
- [x] Ready for Phase 17

---

## Phase 17: PyOD Integration (Advanced Anomaly Detection)

### Phase 17 Tests (Define FIRST)
- [ ] Test: COPOD algorithm loads and trains
- [ ] Test: HBOS algorithm loads and trains
- [ ] Test: PyOD models integrate with existing anomaly API
- [ ] Test: Model save/load works for PyOD backends
- [ ] Test file: `runtimes/universal/tests/test_pyod_integration.py`

### Phase 17 Demo (Define FIRST)
- [ ] Demo script: `examples/ml/demo-pyod-anomaly.sh`
- [ ] Demo shows: Compare COPOD vs Isolation Forest on same data
- [ ] Expected output: Both models detect anomalies, show differences

### Phase 17 Implementation
- [ ] Extend `AnomalyModel` to support PyOD backends
  - Add `copod` backend (Copula-Based, no hyperparameters!)
  - Add `hbos` backend (Histogram-Based, very fast)
  - Add `ecod` backend (Empirical Cumulative Distribution)
- [ ] Update `/v1/anomaly/fit` to accept new backends
- [ ] Add to `pyproject.toml`: `issues = [..., "pyod>=1.1.2"]`

### Phase 17 Verification
- [ ] Run tests: all Phase 17 tests pass
- [ ] Run demo: demo runs successfully

### Phase 17 Checkpoint
- [ ] Tests verified passing
- [ ] Demo verified working
- [ ] **COMMIT**: `git commit -m "feat(anomaly): add PyOD backends (COPOD, HBOS, ECOD)"`
- [ ] Ready for Phase 18

---

## Phase 18: Dependency Organization & Documentation

### Phase 18 Tests (Define FIRST)
- [ ] Test: Base install works without optional dependencies
- [ ] Test: Each optional extra installs correctly
- [ ] Test: Lazy imports work (no import errors if optional dep missing)
- [ ] Test: Helpful error messages when dependency not installed
- [ ] Test file: `runtimes/universal/tests/test_optional_dependencies.py`

### Phase 18 Demo (Define FIRST)
- [ ] Demo script: `examples/ml/demo-install-extras.sh`
- [ ] Demo shows: Install specific extras, verify functionality
- [ ] Expected output: Clean install with only needed dependencies

### Phase 18 Implementation
- [ ] Organize `pyproject.toml` optional dependencies:
  ```toml
  [project.optional-dependencies]
  # Core ML improvements (Phase 1-4)
  ml-core = ["scikit-learn>=1.3.0"]

  # Vision capabilities (Phase 5, 9, 10)
  vision = [
    "transformers>=4.35.0",
    "pillow>=10.0.0",
  ]

  # NLP tools (Phase 6, 7, 8)
  nlp = [
    "gliner>=0.1.0",
  ]

  # Time-series & trends (Phase 11, 12, 14)
  trends = [
    "ruptures>=1.1.9",
    "river>=0.21.0",
    # chronos if available
  ]

  # Advanced anomaly & data quality (Phase 15, 16, 17)
  issues = [
    "pyod>=1.1.2",
    "cleanlab>=2.5.0",
    "shap>=0.44.0",
  ]

  # All ML features
  ml-all = [
    "universal-runtime[ml-core,vision,nlp,trends,issues]",
  ]
  ```
- [ ] Add lazy import helpers to each model file
- [ ] Add clear error messages with install instructions
- [ ] Update README with feature matrix and install guide

### Phase 18 Verification
- [ ] Run tests: all Phase 18 tests pass
- [ ] Run demo: demo runs successfully
- [ ] Documentation is accurate

### Phase 18 Checkpoint
- [ ] Tests verified passing
- [ ] Demo verified working
- [ ] Documentation complete
- [ ] **COMMIT**: `git commit -m "chore(deps): organize optional ML dependencies and add lazy imports"`
- [ ] Ready for Final Integration

---

## Phase 19: Final Integration & End-to-End Testing

### Phase 19 Tests (Define FIRST)
- [ ] Test: All new endpoints respond correctly
- [ ] Test: End-to-end workflow: upload data → train → detect → explain
- [ ] Test: All demos run without errors
- [ ] Test: Performance benchmarks meet targets
- [ ] Test: Memory usage stays within bounds
- [ ] Test file: `runtimes/universal/tests/test_integration.py`

### Phase 19 Demo (Define FIRST)
- [ ] Demo script: `examples/ml/demo-full-ml-suite.sh`
- [ ] Demo shows: Complete ML pipeline showcasing multiple new features
- [ ] Expected output: All features working together

### Phase 19 Implementation
- [ ] Run all existing test suites
- [ ] Run all new test suites
- [ ] Run all demos end-to-end
- [ ] Performance benchmarking
- [ ] Memory profiling
- [ ] Security audit (security-auditor)
- [ ] Final code review (senior-code-reviewer)

### Phase 19 Verification
- [ ] All tests pass
- [ ] All demos run successfully
- [ ] Performance benchmarks met
- [ ] Security audit passed

### Phase 19 Checkpoint
- [ ] All verification complete
- [ ] **COMMIT**: `git commit -m "test(ml): add integration tests and benchmarks for all ML features"`
- [ ] Ready for merge

---

## Phase 20: Comprehensive Example Scripts

**CRITICAL**: Every new ML capability requires TWO runnable example scripts:
1. **Direct Universal Runtime** (`test_*.sh`) - Hits port 11540 directly
2. **LlamaFarm API Proxy** (`test_*_api.sh`) - Hits port 8000 via `/v1/ml/*` proxy

All scripts will be placed in `examples/ocr_and_document/` alongside existing anomaly and classifier examples.

### Phase 20 Tests (Define FIRST)
- [ ] Test: All example scripts are executable (`chmod +x`)
- [ ] Test: Each script includes health check before running
- [ ] Test: Each script has colored output and clear progress indicators
- [ ] Test: Each script cleans up resources after completion
- [ ] Test: Each script handles errors gracefully with helpful messages
- [ ] Test file: `examples/validate_ml_examples.sh`

### Phase 20 Demo (Define FIRST)
- [ ] Demo script: `examples/ocr_and_document/run_all_ml_examples.sh`
- [ ] Demo shows: Run ALL new ML example scripts sequentially
- [ ] Expected output: All examples pass with green checkmarks

### Phase 20 Implementation

#### 20.1 Vision Examples (CLIP, YOLOS, RMBG)

**Test Images & Videos:** Use files from `examples/files/`:
- Images: `cat.png`, `cat1.jpg`, `cat2.jpg`, `horse.jpg`, `ts_to_test.jpg`
- Videos: `bird.mp4`, `fish.mp4`, `polar_bear.mp4` (extract frames every 1 second)

**Video Frame Extraction:** For video testing, scripts will:
1. Use `ffmpeg` to extract frames every 1 second
2. Process each frame through the endpoint
3. Aggregate results
4. Clean up temp frames

**Zero-Shot Image Classification (CLIP):**
- [ ] `examples/ocr_and_document/test_clip.sh` - Direct Universal Runtime
  - Test zero-shot classification with sample images from `examples/files/`
  - Classify images with labels: "cat", "dog", "horse", "bird", "fish", "bear"
  - Process video files by extracting frames every 1 second
  - Show probability distribution for each image/frame
- [ ] `examples/ocr_and_document/test_clip_api.sh` - LlamaFarm API proxy
  - Same tests via `/v1/ml/vision/classify-zero-shot`

**Object Detection (YOLOS):**
- [ ] `examples/ocr_and_document/test_object_detection.sh` - Direct Universal Runtime
  - Detect objects in sample images from `examples/files/`
  - Process video files by extracting frames
  - Show bounding boxes and confidence scores
  - Handle images with no objects gracefully
- [ ] `examples/ocr_and_document/test_object_detection_api.sh` - LlamaFarm API proxy

**Background Removal (RMBG):**
- [ ] `examples/ocr_and_document/test_background_removal.sh` - Direct Universal Runtime
  - Remove background from images in `examples/files/`
  - Process video frames and save as transparent PNGs
  - Verify output is PNG with alpha channel
  - Save results to temp directory for inspection
- [ ] `examples/ocr_and_document/test_background_removal_api.sh` - LlamaFarm API proxy

#### 20.2 NLP Examples (Language Detection, Keywords, PII)

**Language Detection:**
- [ ] `examples/ocr_and_document/test_language_detection.sh` - Direct Universal Runtime
  - Detect language of multilingual text samples
  - Test: English, Spanish, French, German, Chinese, Japanese
  - Show confidence scores for each detection
- [ ] `examples/ocr_and_document/test_language_detection_api.sh` - LlamaFarm API proxy

**Keyword Extraction:**
- [ ] `examples/ocr_and_document/test_keywords.sh` - Direct Universal Runtime
  - Extract keywords from technical document
  - Test different n-gram sizes (1, 2, 3)
  - Show top 10 keywords with scores
- [ ] `examples/ocr_and_document/test_keywords_api.sh` - LlamaFarm API proxy

**PII Redaction (GLiNER):**
- [ ] `examples/ocr_and_document/test_pii_redaction.sh` - Direct Universal Runtime
  - Redact PII from sample text with SSN, phone, email
  - Show original positions and redacted text
  - Test custom entity types
- [ ] `examples/ocr_and_document/test_pii_redaction_api.sh` - LlamaFarm API proxy

#### 20.3 Time-Series Examples (Forecast, Changepoints, Drift)

**Time-Series Forecasting (Chronos):**
- [ ] `examples/ocr_and_document/test_timeseries_forecast.sh` - Direct Universal Runtime
  - Forecast from 30 days of synthetic data
  - Show point forecasts and confidence intervals
  - Visualize with ASCII chart
- [ ] `examples/ocr_and_document/test_timeseries_forecast_api.sh` - LlamaFarm API proxy

**Change Point Detection (Ruptures):**
- [ ] `examples/ocr_and_document/test_changepoints.sh` - Direct Universal Runtime
  - Detect change points in synthetic signal
  - Test multiple algorithms (Pelt, Binseg)
  - Show segment boundaries
- [ ] `examples/ocr_and_document/test_changepoints_api.sh` - LlamaFarm API proxy

**Concept Drift Detection (River):**
- [ ] `examples/ocr_and_document/test_drift_detection.sh` - Direct Universal Runtime
  - Stream data with distribution shift
  - Detect when drift occurs
  - Show drift index and confidence
- [ ] `examples/ocr_and_document/test_drift_detection_api.sh` - LlamaFarm API proxy

#### 20.4 Analysis Examples (Table QA, SHAP, Audit, PyOD)

**Table Question Answering (TAPAS):**
- [ ] `examples/ocr_and_document/test_table_qa.sh` - Direct Universal Runtime
  - Load sample CSV with server metrics
  - Ask questions: "Which server had highest load?"
  - Show answer with referenced cells
- [ ] `examples/ocr_and_document/test_table_qa_api.sh` - LlamaFarm API proxy

**Anomaly Explanation (SHAP):**
- [ ] `examples/ocr_and_document/test_anomaly_explain.sh` - Direct Universal Runtime
  - Train anomaly detector on multivariate data
  - Detect anomaly and explain WHY
  - Show feature importance breakdown
- [ ] `examples/ocr_and_document/test_anomaly_explain_api.sh` - LlamaFarm API proxy

**Dataset Quality Audit (Cleanlab):**
- [ ] `examples/ocr_and_document/test_dataset_audit.sh` - Direct Universal Runtime
  - Create synthetic dataset with label errors
  - Find mislabeled examples
  - Show confidence scores and suggestions
- [ ] `examples/ocr_and_document/test_dataset_audit_api.sh` - LlamaFarm API proxy

**Advanced Anomaly Detection (PyOD):**
- [ ] `examples/ocr_and_document/test_pyod.sh` - Direct Universal Runtime
  - Compare COPOD, HBOS, ECOD backends
  - Show score differences between algorithms
  - Demonstrate no-hyperparameter COPOD
- [ ] `examples/ocr_and_document/test_pyod_api.sh` - LlamaFarm API proxy

#### 20.5 Infrastructure Examples (Async, Robust, VAE, Streaming)

**Async Training:**
- [ ] `examples/ocr_and_document/test_async_training.sh` - Direct Universal Runtime
  - Start long training, verify health check responds
  - Poll for completion status
  - Show non-blocking behavior
- [ ] `examples/ocr_and_document/test_async_training_api.sh` - LlamaFarm API proxy

**Robust Scaler:**
- [ ] `examples/ocr_and_document/test_robust_scaler.sh` - Direct Universal Runtime
  - Train with outlier-contaminated data
  - Compare StandardScaler vs RobustScaler
  - Show improved detection with RobustScaler
- [ ] `examples/ocr_and_document/test_robust_scaler_api.sh` - LlamaFarm API proxy

**VAE Anomaly Detection:**
- [ ] `examples/ocr_and_document/test_vae_anomaly.sh` - Direct Universal Runtime
  - Train VAE on normal data
  - Detect anomalies with probability scores
  - Show early stopping behavior
- [ ] `examples/ocr_and_document/test_vae_anomaly_api.sh` - LlamaFarm API proxy

**Streaming Large Dataset:**
- [ ] `examples/ocr_and_document/test_streaming_training.sh` - Direct Universal Runtime
  - Upload large CSV file
  - Train with streaming (bounded memory)
  - Monitor memory usage
- [ ] `examples/ocr_and_document/test_streaming_training_api.sh` - LlamaFarm API proxy

### Phase 20 Example Script Template

Each script MUST follow this pattern (from existing examples):

**CRITICAL: Use environment variables from `.env` for ports:**
- `LF_RUNTIME_PORT` - Universal Runtime port (default: 11545)
- `PORT` - LlamaFarm API port (default: 8005)

**Direct Universal Runtime Script Template (`test_[feature].sh`):**
```bash
#!/bin/bash
# Test [FEATURE NAME] endpoint via Universal Runtime
#
# This script demonstrates:
# 1. [Feature capability 1]
# 2. [Feature capability 2]
# 3. [Feature capability 3]
#
# Usage: ./test_[feature].sh [PORT]
#   PORT defaults to LF_RUNTIME_PORT from .env (or 11545)

set -e

# Load port from .env if available, allow override via argument
if [ -f "$(dirname "$0")/../../.env" ]; then
    source "$(dirname "$0")/../../.env" 2>/dev/null || true
fi
RUNTIME_PORT=${1:-${LF_RUNTIME_PORT:-11545}}
BASE_URL="http://localhost:${RUNTIME_PORT}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  [Feature Name] Test${NC}"
echo -e "${BLUE}  (Direct Universal Runtime - port ${RUNTIME_PORT})${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Health check
echo -e "${YELLOW}Checking Universal Runtime health...${NC}"
if ! curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Universal Runtime not running on port ${RUNTIME_PORT}${NC}"
    echo "Start it with: nx start universal-runtime"
    exit 1
fi
echo -e "${GREEN}✓ Universal Runtime is healthy${NC}"
echo ""

# ... test implementation ...
```

**LlamaFarm API Proxy Script Template (`test_[feature]_api.sh`):**
```bash
#!/bin/bash
# Test [FEATURE NAME] endpoint via LlamaFarm API
#
# This script demonstrates:
# 1. [Feature capability 1]
# 2. [Feature capability 2]
# 3. [Feature capability 3]
#
# Usage: ./test_[feature]_api.sh [PORT]
#   PORT defaults to PORT from .env (or 8005)

set -e

# Load port from .env if available, allow override via argument
if [ -f "$(dirname "$0")/../../.env" ]; then
    source "$(dirname "$0")/../../.env" 2>/dev/null || true
fi
API_PORT=${1:-${PORT:-8005}}
BASE_URL="http://localhost:${API_PORT}/v1/ml"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  [Feature Name] Test${NC}"
echo -e "${BLUE}  (LlamaFarm API proxy - port ${API_PORT})${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Health check - LlamaFarm server
echo -e "${YELLOW}Checking LlamaFarm API health...${NC}"
if ! curl -s "http://localhost:${API_PORT}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: LlamaFarm API not running on port ${API_PORT}${NC}"
    echo "Start it with: nx start server"
    exit 1
fi
echo -e "${GREEN}✓ LlamaFarm API is healthy${NC}"

# Health check - Universal Runtime via proxy
echo -e "${YELLOW}Checking Universal Runtime via proxy...${NC}"
if ! curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Universal Runtime not available via proxy${NC}"
    echo "Start it with: nx start universal-runtime"
    exit 1
fi
echo -e "${GREEN}✓ Universal Runtime is healthy (via proxy)${NC}"
echo ""

# ... test implementation ...
```

### Phase 20 Verification
- [ ] Run: `bash examples/ocr_and_document/run_all_ml_examples.sh`
- [ ] All 38 example scripts execute successfully (19 features × 2 scripts each)
- [ ] Each script produces clear output with success/failure indicators
- [ ] No hanging processes or resource leaks

### Phase 20 Checkpoint
- [ ] All example scripts created and working
- [ ] All scripts follow consistent template
- [ ] Master runner script works
- [ ] Documentation updated with example usage
- [ ] **COMMIT**: `git commit -m "docs(examples): add comprehensive example scripts for all ML features"`
- [ ] Ready for merge

---

## Final Success Criteria

- [ ] All 20 phase checkpoints complete
- [ ] All tests pass: `cd runtimes/universal && uv run pytest -v`
- [ ] All demos run successfully
- [ ] Security audit passed
- [ ] Code review passed
- [ ] Documentation updated
- [ ] No regressions in existing functionality
- [ ] Memory usage bounded during large operations
- [ ] API response times within acceptable limits
- [ ] Backward compatibility maintained for existing saved models

---

## Dependency Summary

New optional dependencies organized by category:

```toml
[project.optional-dependencies]
# Trend Analysis & Forecasting
trends = [
    "ruptures>=1.1.9",
    "river>=0.21.0",
]

# Advanced Issue Detection
issues = [
    "pyod>=1.1.2",
    "cleanlab>=2.5.0",
    "shap>=0.44.0",
]

# Vision tools
vision = [
    # CLIP, YOLOS, RMBG use existing transformers
]

# NLP tools
pii = [
    "gliner>=0.1.0",
]
```

---

## Risk Mitigation

1. **Blocking Training**: Phase 1 addresses this first as it affects all ML operations
2. **Memory Issues**: Phase 4 adds streaming before adding memory-intensive new models
3. **Backward Compatibility**: Each phase maintains compatibility with existing saved models
4. **Dependency Bloat**: Phase 18 organizes all dependencies as lazy-loaded optionals
5. **Testing Coverage**: Each phase requires tests before implementation

---

## Execution Order Rationale

1. **Phases 1-4** (Core Infrastructure): Must be completed first as they fix existing issues and prepare the infrastructure for new models
2. **Phases 5-10** (Vision & NLP): Can be parallelized after core infrastructure is stable
3. **Phases 11-14** (Time-Series & Trends): Depend on core infrastructure
4. **Phases 15-17** (Advanced Anomaly): Build on anomaly improvements from Phase 2-4
5. **Phases 18-19** (Polish & Integration): Final organization and verification
6. **Phase 20** (Comprehensive Examples): Created AFTER each feature phase completes, but documented here for planning purposes

---

## Example Scripts Summary (Phase 20)

All example scripts will be placed in `examples/ocr_and_document/` with this naming convention:

| Feature | Direct UR (port 11540) | LlamaFarm API (port 8000) |
|---------|------------------------|---------------------------|
| **Vision** | | |
| CLIP Zero-Shot | `test_clip.sh` | `test_clip_api.sh` |
| Object Detection | `test_object_detection.sh` | `test_object_detection_api.sh` |
| Background Removal | `test_background_removal.sh` | `test_background_removal_api.sh` |
| **NLP** | | |
| Language Detection | `test_language_detection.sh` | `test_language_detection_api.sh` |
| Keyword Extraction | `test_keywords.sh` | `test_keywords_api.sh` |
| PII Redaction | `test_pii_redaction.sh` | `test_pii_redaction_api.sh` |
| **Time-Series** | | |
| Forecasting | `test_timeseries_forecast.sh` | `test_timeseries_forecast_api.sh` |
| Change Points | `test_changepoints.sh` | `test_changepoints_api.sh` |
| Drift Detection | `test_drift_detection.sh` | `test_drift_detection_api.sh` |
| **Analysis** | | |
| Table QA | `test_table_qa.sh` | `test_table_qa_api.sh` |
| SHAP Explain | `test_anomaly_explain.sh` | `test_anomaly_explain_api.sh` |
| Dataset Audit | `test_dataset_audit.sh` | `test_dataset_audit_api.sh` |
| PyOD Anomaly | `test_pyod.sh` | `test_pyod_api.sh` |
| **Infrastructure** | | |
| Async Training | `test_async_training.sh` | `test_async_training_api.sh` |
| Robust Scaler | `test_robust_scaler.sh` | `test_robust_scaler_api.sh` |
| VAE Anomaly | `test_vae_anomaly.sh` | `test_vae_anomaly_api.sh` |
| Streaming Training | `test_streaming_training.sh` | `test_streaming_training_api.sh` |

**Total: 34 new example scripts** (17 features × 2 scripts each)

Plus master runner: `run_all_ml_examples.sh`
