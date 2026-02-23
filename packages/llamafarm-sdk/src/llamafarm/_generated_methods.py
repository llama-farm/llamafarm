"""Auto-generated LlamaFarm SDK method stubs from OpenAPI spec.

Generated: 2026-02-23 17:42:06 UTC
Endpoints: 134

This file is a REFERENCE — not imported directly.
Use it to verify client.py coverage and update signatures.
"""

from __future__ import annotations

from typing import Any

from ._generated_types import *  # noqa: F403

# ======================================================================
# ADDONS (4 endpoints)
# ======================================================================

class AddonsAPI:
    def addons(self) -> list[Any]:
        """List Addons"""
        # GET /v1/addons
        ...

    def post_addons_install(self, request: AddonInstallRequest) -> AddonInstallResponse:
        """Install Addon"""
        # POST /v1/addons/install
        ...

    def addons_tasks_by_task_id(self, task_id: str) -> AddonTaskStatus:
        """Get Task Status"""
        # GET /v1/addons/tasks/{task_id}
        ...

    def post_addons_uninstall(self, request: AddonInstallRequest) -> dict[str, Any]:
        """Uninstall Addon"""
        # POST /v1/addons/uninstall
        ...


# ======================================================================
# ADTK (6 endpoints)
# ======================================================================

class AdtkAPI:
    def post_adtk_detect(self, request: ADTKDetectRequest) -> dict[str, Any]:
        """Detect Anomalies"""
        # POST /v1/adtk/detect
        ...

    def adtk_detectors(self) -> dict[str, Any]:
        """List Detectors"""
        # GET /v1/adtk/detectors
        ...

    def post_adtk_fit(self, request: ADTKFitRequest) -> dict[str, Any]:
        """Fit Detector"""
        # POST /v1/adtk/fit
        ...

    def post_adtk_load(self, request: ADTKLoadRequest) -> dict[str, Any]:
        """Load Model"""
        # POST /v1/adtk/load
        ...

    def adtk_models(self) -> dict[str, Any]:
        """List Models"""
        # GET /v1/adtk/models
        ...

    def delete_adtk_models_by_model_name(self, model_name: str) -> dict[str, Any]:
        """Delete Model"""
        # DELETE /v1/adtk/models/{model_name}
        ...


# ======================================================================
# ANOMALY (13 endpoints)
# ======================================================================

class AnomalyAPI:
    def ml_anomaly_backends(self) -> dict[str, Any]:
        """List Anomaly Backends"""
        # GET /v1/ml/anomaly/backends
        ...

    def post_ml_anomaly_detect(self, request: AnomalyScoreRequest) -> dict[str, Any]:
        """Detect Anomalies"""
        # POST /v1/ml/anomaly/detect
        ...

    def post_ml_anomaly_fit(self, request: AnomalyFitRequest) -> dict[str, Any]:
        """Fit Anomaly Detector"""
        # POST /v1/ml/anomaly/fit
        ...

    def post_ml_anomaly_load(self, request: AnomalyLoadRequest) -> dict[str, Any]:
        """Load Anomaly Model"""
        # POST /v1/ml/anomaly/load
        ...

    def ml_anomaly_models(self) -> dict[str, Any]:
        """List Anomaly Models"""
        # GET /v1/ml/anomaly/models
        ...

    def delete_ml_anomaly_models_by_filename(self, filename: str) -> dict[str, Any]:
        """Delete Anomaly Model"""
        # DELETE /v1/ml/anomaly/models/{filename}
        ...

    def post_ml_anomaly_save(self, request: AnomalySaveRequest) -> dict[str, Any]:
        """Save Anomaly Model"""
        # POST /v1/ml/anomaly/save
        ...

    def post_ml_anomaly_score(self, request: AnomalyScoreRequest) -> dict[str, Any]:
        """Score Anomalies"""
        # POST /v1/ml/anomaly/score
        ...

    def post_ml_anomaly_stream(self) -> dict[str, Any]:
        """Anomaly Stream"""
        # POST /v1/ml/anomaly/stream
        ...

    def ml_anomaly_stream_detectors(self) -> dict[str, Any]:
        """List Streaming Detectors"""
        # GET /v1/ml/anomaly/stream/detectors
        ...

    def ml_anomaly_stream_by_model_id(self, model_id: str) -> dict[str, Any]:
        """Get Streaming Detector"""
        # GET /v1/ml/anomaly/stream/{model_id}
        ...

    def delete_ml_anomaly_stream_by_model_id(self, model_id: str) -> dict[str, Any]:
        """Delete Streaming Detector"""
        # DELETE /v1/ml/anomaly/stream/{model_id}
        ...

    def post_ml_anomaly_stream_by_model_id_reset(self, model_id: str) -> dict[str, Any]:
        """Reset Streaming Detector"""
        # POST /v1/ml/anomaly/stream/{model_id}/reset
        ...


# ======================================================================
# CATBOOST (8 endpoints)
# ======================================================================

class CatboostAPI:
    def post_catboost_fit(self, request: CatBoostFitRequest) -> dict[str, Any]:
        """Fit Model"""
        # POST /v1/catboost/fit
        ...

    def catboost_info(self) -> dict[str, Any]:
        """Get Info"""
        # GET /v1/catboost/info
        ...

    def post_catboost_load(self, request: CatBoostLoadRequest) -> dict[str, Any]:
        """Load Model"""
        # POST /v1/catboost/load
        ...

    def catboost_models(self) -> dict[str, Any]:
        """List Models"""
        # GET /v1/catboost/models
        ...

    def post_catboost_predict(self, request: CatBoostPredictRequest) -> dict[str, Any]:
        """Predict"""
        # POST /v1/catboost/predict
        ...

    def post_catboost_update(self, request: CatBoostUpdateRequest) -> dict[str, Any]:
        """Update Model"""
        # POST /v1/catboost/update
        ...

    def delete_catboost_by_model_id(self, model_id: str) -> dict[str, Any]:
        """Delete Model"""
        # DELETE /v1/catboost/{model_id}
        ...

    def catboost_by_model_id_importance(self, model_id: str) -> dict[str, Any]:
        """Get Feature Importance"""
        # GET /v1/catboost/{model_id}/importance
        ...


# ======================================================================
# CLASSIFIER (6 endpoints)
# ======================================================================

class ClassifierAPI:
    def post_ml_classifier_fit(self, request: ClassifierFitRequest) -> dict[str, Any]:
        """Fit Classifier"""
        # POST /v1/ml/classifier/fit
        ...

    def post_ml_classifier_load(self, request: ClassifierLoadRequest) -> dict[str, Any]:
        """Load Classifier"""
        # POST /v1/ml/classifier/load
        ...

    def ml_classifier_models(self) -> dict[str, Any]:
        """List Classifier Models"""
        # GET /v1/ml/classifier/models
        ...

    def delete_ml_classifier_models_by_model_name(self, model_name: str) -> dict[str, Any]:
        """Delete Classifier Model"""
        # DELETE /v1/ml/classifier/models/{model_name}
        ...

    def post_ml_classifier_predict(self, request: ClassifierPredictRequest) -> dict[str, Any]:
        """Predict Classifier"""
        # POST /v1/ml/classifier/predict
        ...

    def post_ml_classifier_save(self, request: ClassifierSaveRequest) -> dict[str, Any]:
        """Save Classifier"""
        # POST /v1/ml/classifier/save
        ...


# ======================================================================
# CORE (2 endpoints)
# ======================================================================

class CoreAPI:
    def _health(self) -> dict[str, Any]:
        """Get Health"""
        # GET /health
        ...

    def _info(self) -> dict[str, Any]:
        """<Lambda>"""
        # GET /info
        ...


# ======================================================================
# DRIFT (8 endpoints)
# ======================================================================

class DriftAPI:
    def post_drift_detect(self, request: DriftDetectRequest) -> dict[str, Any]:
        """Detect Drift"""
        # POST /v1/drift/detect
        ...

    def drift_detectors(self) -> dict[str, Any]:
        """List Detectors"""
        # GET /v1/drift/detectors
        ...

    def post_drift_fit(self, request: DriftFitRequest) -> dict[str, Any]:
        """Fit Detector"""
        # POST /v1/drift/fit
        ...

    def post_drift_load(self, request: DriftLoadRequest) -> dict[str, Any]:
        """Load Model"""
        # POST /v1/drift/load
        ...

    def drift_models(self) -> dict[str, Any]:
        """List Models"""
        # GET /v1/drift/models
        ...

    def delete_drift_models_by_model_name(self, model_name: str) -> dict[str, Any]:
        """Delete Model"""
        # DELETE /v1/drift/models/{model_name}
        ...

    def post_drift_reset_by_model_name(self, model_name: str) -> dict[str, Any]:
        """Reset Detector"""
        # POST /v1/drift/reset/{model_name}
        ...

    def drift_status_by_model_name(self, model_name: str) -> dict[str, Any]:
        """Get Status"""
        # GET /v1/drift/status/{model_name}
        ...


# ======================================================================
# EXAMPLES (7 endpoints)
# ======================================================================

class ExamplesAPI:
    def examples(self) -> dict[str, Any]:
        """List Examples"""
        # GET /v1/examples
        ...

    def examples_(self) -> dict[str, Any]:
        """List Examples"""
        # GET /v1/examples/
        ...

    def examples_datasets(self) -> dict[str, Any]:
        """List All Example Datasets"""
        # GET /v1/examples/datasets
        ...

    def examples_by_example_id_datasets(self, example_id: str) -> dict[str, Any]:
        """List Example Datasets"""
        # GET /v1/examples/{example_id}/datasets
        ...

    def post_examples_by_example_id_import_data(self, example_id: str, request: ImportDataRequest) -> ImportDataResponse:
        """Import Data"""
        # POST /v1/examples/{example_id}/import-data
        ...

    def post_examples_by_example_id_import_dataset(self, example_id: str, request: ImportDatasetRequest) -> ImportDatasetResponse:
        """Import Dataset"""
        # POST /v1/examples/{example_id}/import-dataset
        ...

    def post_examples_by_example_id_import_project(self, example_id: str, request: ImportProjectRequest) -> ImportProjectResponse:
        """Import Project"""
        # POST /v1/examples/{example_id}/import-project
        ...


# ======================================================================
# EXPLAIN (3 endpoints)
# ======================================================================

class ExplainAPI:
    def explain_explainers(self) -> dict[str, Any]:
        """List Explainers"""
        # GET /v1/explain/explainers
        ...

    def post_explain_importance(self, request: FeatureImportanceRequest) -> dict[str, Any]:
        """Feature Importance"""
        # POST /v1/explain/importance
        ...

    def post_explain_shap(self, request: SHAPExplainRequest) -> dict[str, Any]:
        """Explain Shap"""
        # POST /v1/explain/shap
        ...


# ======================================================================
# HEALTH (1 endpoints)
# ======================================================================

class HealthAPI:
    def _health_liveness(self) -> dict[str, Any]:
        """Get Liveness"""
        # GET /health/liveness
        ...


# ======================================================================
# MODELS (5 endpoints)
# ======================================================================

class ModelsAPI:
    def models(self, provider: Provider | None = None) -> dict[str, Any]:
        """List Models"""
        # GET /v1/models
        ...

    def post_models_download(self, request: DownloadModelRequest) -> dict[str, Any]:
        """Download Model"""
        # POST /v1/models/download
        ...

    def post_models_validate_download(self, request: ValidateDownloadRequest) -> dict[str, Any]:
        """Validate Download"""
        # POST /v1/models/validate-download
        ...

    def models_by_model_id_quantizations(self, model_id: str) -> GGUFOptionsResponse:
        """Get Gguf Options"""
        # GET /v1/models/{model_id}/quantizations
        ...

    def delete_models_by_model_name(self, model_name: str, provider: Provider | None = None) -> dict[str, Any]:
        """Delete Model"""
        # DELETE /v1/models/{model_name}
        ...


# ======================================================================
# NLP (4 endpoints)
# ======================================================================

class NlpAPI:
    def post_nlp_classify(self, request: ClassifyRequest) -> dict[str, Any]:
        """Classify Text"""
        # POST /v1/nlp/classify
        ...

    def post_nlp_embeddings(self, request: EmbeddingRequest) -> dict[str, Any]:
        """Create Embeddings"""
        # POST /v1/nlp/embeddings
        ...

    def post_nlp_ner(self, request: NERRequest) -> dict[str, Any]:
        """Extract Entities"""
        # POST /v1/nlp/ner
        ...

    def post_nlp_rerank(self, request: RerankRequest) -> dict[str, Any]:
        """Rerank Documents"""
        # POST /v1/nlp/rerank
        ...


# ======================================================================
# POLARS (8 endpoints)
# ======================================================================

class PolarsAPI:
    def post_ml_polars_append(self, request: PolarsBufferAppendRequest) -> dict[str, Any]:
        """Append To Polars Buffer"""
        # POST /v1/ml/polars/append
        ...

    def ml_polars_buffers(self) -> PolarsBuffersListResponse:
        """List Polars Buffers"""
        # GET /v1/ml/polars/buffers
        ...

    def post_ml_polars_buffers(self, request: PolarsBufferCreateRequest) -> dict[str, Any]:
        """Create Polars Buffer"""
        # POST /v1/ml/polars/buffers
        ...

    def ml_polars_buffers_by_buffer_id(self, buffer_id: str) -> PolarsBufferStats:
        """Get Polars Buffer"""
        # GET /v1/ml/polars/buffers/{buffer_id}
        ...

    def delete_ml_polars_buffers_by_buffer_id(self, buffer_id: str) -> dict[str, Any]:
        """Delete Polars Buffer"""
        # DELETE /v1/ml/polars/buffers/{buffer_id}
        ...

    def post_ml_polars_buffers_by_buffer_id_clear(self, buffer_id: str) -> dict[str, Any]:
        """Clear Polars Buffer"""
        # POST /v1/ml/polars/buffers/{buffer_id}/clear
        ...

    def ml_polars_buffers_by_buffer_id_data(self, buffer_id: str, tail: int | None | None = None, with_features: bool | None = None) -> PolarsBufferDataResponse:
        """Get Polars Buffer Data"""
        # GET /v1/ml/polars/buffers/{buffer_id}/data
        ...

    def post_ml_polars_features(self, request: PolarsBufferFeaturesRequest) -> PolarsBufferDataResponse:
        """Compute Polars Features"""
        # POST /v1/ml/polars/features
        ...


# ======================================================================
# PROJECTS (32 endpoints)
# ======================================================================

class ProjectsAPI:
    def projects_by_namespace(self, namespace: str) -> ListProjectsResponse:
        """List projects for a namespace"""
        # GET /v1/projects/{namespace}
        ...

    def post_projects_by_namespace(self, namespace: str, request: CreateProjectRequest) -> CreateProjectResponse:
        """Create a project"""
        # POST /v1/projects/{namespace}
        ...

    def projects_by_namespace_by_project_id(self, namespace: str, project_id: str) -> GetProjectResponse:
        """Get a project"""
        # GET /v1/projects/{namespace}/{project_id}
        ...

    def put_projects_by_namespace_by_project_id(self, namespace: str, project_id: str, request: UpdateProjectRequest) -> UpdateProjectResponse:
        """Update a project"""
        # PUT /v1/projects/{namespace}/{project_id}
        ...

    def delete_projects_by_namespace_by_project_id(self, namespace: str, project_id: str) -> DeleteProjectResponse:
        """Delete a project"""
        # DELETE /v1/projects/{namespace}/{project_id}
        ...

    def post_projects_by_namespace_by_project_id_chat_completions(self, namespace: str, project_id: str, request: ChatRequest) -> ChatCompletion:
        """Chat"""
        # POST /v1/projects/{namespace}/{project_id}/chat/completions
        ...

    def delete_projects_by_namespace_by_project_id_chat_sessions(self, namespace: str, project_id: str) -> dict[str, Any]:
        """Delete All Chat Sessions"""
        # DELETE /v1/projects/{namespace}/{project_id}/chat/sessions
        ...

    def delete_projects_by_namespace_by_project_id_chat_sessions_by_session_id(self, namespace: str, project_id: str, session_id: str) -> dict[str, Any]:
        """Delete Chat Session"""
        # DELETE /v1/projects/{namespace}/{project_id}/chat/sessions/{session_id}
        ...

    def projects_by_namespace_by_project_id_chat_sessions_by_session_id_history(self, namespace: str, project_id: str, session_id: str) -> dict[str, Any]:
        """Get Chat Session History"""
        # GET /v1/projects/{namespace}/{project_id}/chat/sessions/{session_id}/history
        ...

    def projects_by_namespace_by_project_id_event_logs(self, namespace: str, project_id: str, type_: str | None | None = None, start_time: str | None | None = None, end_time: str | None | None = None, limit: int | None = None, offset: int | None = None) -> ListEventsResponse:
        """List event logs"""
        # GET /v1/projects/{namespace}/{project_id}/event_logs
        ...

    def projects_by_namespace_by_project_id_event_logs_by_event_id(self, namespace: str, project_id: str, event_id: str) -> EventDetail:
        """Get event details"""
        # GET /v1/projects/{namespace}/{project_id}/event_logs/{event_id}
        ...

    def projects_by_namespace_by_project_id_models(self, namespace: str, project_id: str) -> ListModelsResponse:
        """List all available models for this project"""
        # GET /v1/projects/{namespace}/{project_id}/models
        ...

    def projects_by_namespace_by_project_id_tasks_by_task_id(self, namespace: str, project_id: str, task_id: str) -> GetTaskResponse:
        """Get the status of an asynchronous task"""
        # GET /v1/projects/{namespace}/{project_id}/tasks/{task_id}
        ...

    def delete_projects_by_namespace_by_project_id_tasks_by_task_id(self, namespace: str, project_id: str, task_id: str) -> CancelTaskResponse:
        """Cancel a running task"""
        # DELETE /v1/projects/{namespace}/{project_id}/tasks/{task_id}
        ...

    def projects_by_namespace_by_project_datasets_(self, namespace: str, project: str, include_extra_details: bool | None = None) -> ListDatasetsResponse:
        """List Datasets"""
        # GET /v1/projects/{namespace}/{project}/datasets/
        ...

    def post_projects_by_namespace_by_project_datasets_(self, namespace: str, project: str, request: CreateDatasetRequest) -> CreateDatasetResponse:
        """Create Dataset"""
        # POST /v1/projects/{namespace}/{project}/datasets/
        ...

    def projects_by_namespace_by_project_datasets_strategies(self, namespace: str, project: str) -> AvailableStrategiesResponse:
        """List available data processing strategies and databases for the project"""
        # GET /v1/projects/{namespace}/{project}/datasets/strategies
        ...

    def delete_projects_by_namespace_by_project_datasets_by_dataset(self, namespace: str, project: str, dataset: str) -> DeleteDatasetResponse:
        """Delete Dataset"""
        # DELETE /v1/projects/{namespace}/{project}/datasets/{dataset}
        ...

    def post_projects_by_namespace_by_project_datasets_by_dataset_actions(self, namespace: str, project: str, dataset: str, request: DatasetActionRequest) -> DatasetActionResponse:
        """Execute an action on a dataset"""
        # POST /v1/projects/{namespace}/{project}/datasets/{dataset}/actions
        ...

    def post_projects_by_namespace_by_project_datasets_by_dataset_data(self, namespace: str, project: str, dataset: str, auto_process: bool | None | None = None) -> DatasetDataUploadResponse:
        """Upload a file to the dataset"""
        # POST /v1/projects/{namespace}/{project}/datasets/{dataset}/data
        ...

    def post_projects_by_namespace_by_project_datasets_by_dataset_data_bulk(self, namespace: str, project: str, dataset: str, auto_process: bool | None | None = None) -> BulkDatasetDataUploadResponse:
        """Upload multiple files to the dataset"""
        # POST /v1/projects/{namespace}/{project}/datasets/{dataset}/data/bulk
        ...

    def delete_projects_by_namespace_by_project_datasets_by_dataset_data_by_file_hash(self, namespace: str, project: str, dataset: str, file_hash: str) -> DeleteDataResponse:
        """Delete a file from the dataset"""
        # DELETE /v1/projects/{namespace}/{project}/datasets/{dataset}/data/{file_hash}
        ...

    def projects_by_namespace_by_project_rag_databases(self, namespace: str, project: str) -> DatabasesResponse:
        """Get Rag Databases"""
        # GET /v1/projects/{namespace}/{project}/rag/databases
        ...

    def post_projects_by_namespace_by_project_rag_databases(self, namespace: str, project: str, request: CreateDatabaseRequest) -> DatabaseResponse:
        """Create a new RAG database"""
        # POST /v1/projects/{namespace}/{project}/rag/databases
        ...

    def projects_by_namespace_by_project_rag_databases_by_database_name(self, namespace: str, project: str, database_name: str) -> DatabaseDetailResponse:
        """Get a single RAG database by name"""
        # GET /v1/projects/{namespace}/{project}/rag/databases/{database_name}
        ...

    def patch_projects_by_namespace_by_project_rag_databases_by_database_name(self, namespace: str, project: str, database_name: str, request: UpdateDatabaseRequest) -> DatabaseDetailResponse:
        """Update a RAG database"""
        # PATCH /v1/projects/{namespace}/{project}/rag/databases/{database_name}
        ...

    def delete_projects_by_namespace_by_project_rag_databases_by_database_name(self, namespace: str, project: str, database_name: str, delete_collection: bool | None = None) -> DeleteDatabaseResponse:
        """Delete a RAG database"""
        # DELETE /v1/projects/{namespace}/{project}/rag/databases/{database_name}
        ...

    def projects_by_namespace_by_project_rag_databases_by_database_name_documents(self, namespace: str, project: str, database_name: str, limit: int | None = None, offset: int | None = None) -> list[Any]:
        """List documents in a RAG database"""
        # GET /v1/projects/{namespace}/{project}/rag/databases/{database_name}/documents
        ...

    def post_projects_by_namespace_by_project_rag_databases_by_database_name_preview(self, namespace: str, project: str, database_name: str, request: DocumentPreviewRequest) -> DocumentPreviewResponse:
        """Preview document chunking"""
        # POST /v1/projects/{namespace}/{project}/rag/databases/{database_name}/preview
        ...

    def projects_by_namespace_by_project_rag_health(self, namespace: str, project: str, database: str | None | None = None) -> RAGHealthResponse:
        """Get Rag Health"""
        # GET /v1/projects/{namespace}/{project}/rag/health
        ...

    def post_projects_by_namespace_by_project_rag_query(self, namespace: str, project: str, request: RAGQueryRequest) -> QueryResponse:
        """Query the RAG system for semantic search"""
        # POST /v1/projects/{namespace}/{project}/rag/query
        ...

    def projects_by_namespace_by_project_rag_stats(self, namespace: str, project: str, database: str | None | None = None) -> RAGStatsResponse:
        """Get Rag Stats"""
        # GET /v1/projects/{namespace}/{project}/rag/stats
        ...


# ======================================================================
# SYSTEM (2 endpoints)
# ======================================================================

class SystemAPI:
    def system_disk(self) -> dict[str, Any]:
        """Get Disk Space"""
        # GET /v1/system/disk
        ...

    def system_version_check(self) -> dict[str, Any]:
        """Cli Upgrade"""
        # GET /v1/system/version-check
        ...


# ======================================================================
# TIMESERIES (7 endpoints)
# ======================================================================

class TimeseriesAPI:
    def timeseries_backends(self) -> TimeseriesBackendsResponse:
        """List Backends"""
        # GET /v1/timeseries/backends
        ...

    def post_timeseries_fit(self, request: TimeseriesFitRequest) -> TimeseriesFitResponse:
        """Fit Timeseries"""
        # POST /v1/timeseries/fit
        ...

    def post_timeseries_load(self, request: TimeseriesLoadRequest) -> TimeseriesFitResponse:
        """Load Timeseries"""
        # POST /v1/timeseries/load
        ...

    def timeseries_models(self) -> TimeseriesModelsResponse:
        """List Models"""
        # GET /v1/timeseries/models
        ...

    def timeseries_models_by_model_name(self, model_name: str) -> TimeseriesModelInfo:
        """Get Model"""
        # GET /v1/timeseries/models/{model_name}
        ...

    def delete_timeseries_models_by_model_name(self, model_name: str) -> TimeseriesDeleteResponse:
        """Delete Model"""
        # DELETE /v1/timeseries/models/{model_name}
        ...

    def post_timeseries_predict(self, request: TimeseriesPredictRequest) -> TimeseriesPredictResponse:
        """Predict Timeseries"""
        # POST /v1/timeseries/predict
        ...


# ======================================================================
# VISION (16 endpoints)
# ======================================================================

class VisionAPI:
    def post_vision_classify(self) -> dict[str, Any]:
        """Classify Image"""
        # POST /v1/vision/classify
        ...

    def post_vision_detect(self) -> dict[str, Any]:
        """Detect Objects"""
        # POST /v1/vision/detect
        ...

    def post_vision_detect_classify(self) -> dict[str, Any]:
        """Detect Classify"""
        # POST /v1/vision/detect_classify
        ...

    def post_vision_documents_extract(self) -> dict[str, Any]:
        """Extract From Documents"""
        # POST /v1/vision/documents/extract
        ...

    def vision_models(self) -> dict[str, Any]:
        """List Models"""
        # GET /v1/vision/models
        ...

    def post_vision_models_export(self) -> dict[str, Any]:
        """Export Model"""
        # POST /v1/vision/models/export
        ...

    def post_vision_ocr(self) -> dict[str, Any]:
        """Extract Text"""
        # POST /v1/vision/ocr
        ...

    def post_vision_review_decide(self) -> dict[str, Any]:
        """Submit Decision"""
        # POST /v1/vision/review/decide
        ...

    def vision_review_pending(self, limit: int | None = None, source: str | None | None = None) -> dict[str, Any]:
        """Get Pending"""
        # GET /v1/vision/review/pending
        ...

    def post_vision_stream_frame(self) -> dict[str, Any]:
        """Stream Frame"""
        # POST /v1/vision/stream/frame
        ...

    def vision_stream_sessions(self) -> dict[str, Any]:
        """Stream Sessions"""
        # GET /v1/vision/stream/sessions
        ...

    def post_vision_stream_start(self) -> dict[str, Any]:
        """Stream Start"""
        # POST /v1/vision/stream/start
        ...

    def post_vision_stream_stop(self) -> dict[str, Any]:
        """Stream Stop"""
        # POST /v1/vision/stream/stop
        ...

    def post_vision_train(self) -> dict[str, Any]:
        """Start Training"""
        # POST /v1/vision/train
        ...

    def vision_train_by_job_id(self, job_id: str) -> dict[str, Any]:
        """Training Status"""
        # GET /v1/vision/train/{job_id}
        ...

    def delete_vision_train_by_job_id(self, job_id: str) -> dict[str, Any]:
        """Cancel Training"""
        # DELETE /v1/vision/train/{job_id}
        ...


# ======================================================================
# {NAMESPACE} (2 endpoints)
# ======================================================================

class _Namespace_API:
    def post_by_namespace_by_project_audio_speech(self, namespace: str, project: str, request: SpeechRequest) -> dict[str, Any]:
        """Create Speech"""
        # POST /v1/{namespace}/{project}/audio/speech
        ...

    def by_namespace_by_project_audio_voices(self, namespace: str, project: str, model: str | None | None = None) -> VoiceListResponse:
        """List Voices"""
        # GET /v1/{namespace}/{project}/audio/voices
        ...

