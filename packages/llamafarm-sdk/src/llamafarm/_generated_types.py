"""Auto-generated LlamaFarm SDK types from OpenAPI spec.

Generated: 2026-02-23 17:42:06 UTC
Server: FastAPI
Schemas: 216

DO NOT EDIT — regenerate with: python scripts/generate_sdk.py
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ADTKDetectRequest(BaseModel):
    """Request to detect anomalies in time series data."""
    model: str | None = Field(None, description="Model name. If None, uses default detector settings.")
    detector: str = Field('level_shift', description="Detector type to use (ignored if model is specified)")
    data: list[dict[str, Any]] = Field(..., description="Time series data as list of {timestamp, value} dicts")
    params: dict[str, Any] | None = Field(None, description="Detector-specific parameters (e.g., c, window, side)")


class ADTKFitRequest(BaseModel):
    """Request to fit an ADTK detector."""
    model: str = Field('default', description="Model identifier")
    detector: str = Field('level_shift', description="Detector type (level_shift, seasonal, persist, volatility_shift, threshold, interquartile_range)")
    data: list[dict[str, Any]] = Field(..., description="Time series data as list of {timestamp, value} dicts")
    params: dict[str, Any] | None = Field(None, description="Detector-specific parameters (e.g., c, window, side)")
    overwrite: bool = Field(True, description="Overwrite existing model if it exists")
    description: str | None = Field(None, description="Optional model description")


class ADTKLoadRequest(BaseModel):
    """Request to load an ADTK model."""
    model: str = Field(..., description="Model identifier")


class AddonInfo(BaseModel):
    """Addon metadata."""
    name: str = Field(...)
    display_name: str = Field(...)
    description: str = Field(...)
    component: str = Field(...)
    version: str = Field(...)
    dependencies: list[str] | None = Field(None)
    packages: list[str] | None = Field(None)
    installed: bool = Field(False)
    installed_at: str | None = Field(None)


class AddonInstallRequest(BaseModel):
    """Request to install an addon."""
    name: str = Field(...)
    restart_service: bool = Field(True)


class AddonInstallResponse(BaseModel):
    """Response after initiating addon installation."""
    task_id: str = Field(...)
    status: str = Field(...)
    addon: str = Field(...)


class AddonTaskStatus(BaseModel):
    """Status of an addon installation task."""
    status: str = Field(...)
    progress: int = Field(...)
    message: str = Field(...)
    error: str | None = Field(None)


class Annotation(BaseModel):
    type_: str = Field(..., alias="type")
    url_citation: AnnotationURLCitation = Field(...)


class AnnotationURLCitation(BaseModel):
    end_index: int = Field(...)
    start_index: int = Field(...)
    title: str = Field(...)
    url: str = Field(...)


class AnomalyFitRequest(BaseModel):
    """Anomaly model fitting request. Supports two data formats: 1. Numeric arrays: data = [[1.0, 2.0], [3.0, 4.0]] 2. Dict-based with schema: data = [{\"time_ms\": 100, \"user_agent\": \"curl\"}] with schema = {\"time_ms\": \"numeric\", \"user_agent\": \"hash\"} All backends are powered by PyOD. See GET /v1/anomaly/backends for full list. Popular choices: - isolation_forest: Fast, works well out of the box (recommended legacy) - ecod: Fast and parameter-free (recommended for new projects) - hbos: Fastest algorithm, good for high dimensions Normalization methods: - standardization (default): Sigmoid 0-1 range, threshold ~0.5 - zscore: Standard deviations from mean, threshold ~2.0-3.0 - raw: Backend-native scores (higher = more anomalous)"""
    model: str = Field('default')
    backend: str = Field('isolation_forest')
    data: list[list[float]] | list[dict[str, Any]] = Field(...)
    schema_: dict[str, str] | None = Field(None, alias="schema")
    contamination: float = Field(0.1, description="Expected proportion of anomalies (0-0.5]")
    normalization: Literal['standardization', 'zscore', 'raw'] = Field('standardization')
    epochs: int = Field(100)
    batch_size: int = Field(32)
    overwrite: bool = Field(True)
    description: str | None = Field(None)


class AnomalyLoadRequest(BaseModel):
    """Request to load a pre-trained anomaly model."""
    model: str = Field(...)
    backend: str = Field('isolation_forest')


class AnomalySaveRequest(BaseModel):
    """Request to save a fitted anomaly model."""
    model: str = Field(...)
    backend: str = Field('isolation_forest')
    normalization: Literal['standardization', 'zscore', 'raw'] = Field('standardization')
    description: str | None = Field(None)


class AnomalyScoreRequest(BaseModel):
    """Anomaly scoring request. All backends are powered by PyOD. See GET /v1/anomaly/backends for full list. Normalization methods: - standardization (default): Sigmoid 0-1 range, threshold ~0.5 - zscore: Standard deviations from mean, threshold ~2.0-3.0 - raw: Backend-native scores (higher = more anomalous)"""
    model: str = Field('default')
    backend: str = Field('isolation_forest')
    data: list[list[float]] | list[dict[str, Any]] = Field(...)
    schema_: dict[str, str] | None = Field(None, alias="schema")
    normalization: Literal['standardization', 'zscore', 'raw'] = Field('standardization')
    threshold: float | None = Field(None)
    explain: bool = Field(False)


class Audio(BaseModel):
    id: str = Field(...)


class AvailableStrategiesResponse(BaseModel):
    data_processing_strategies: list[str] = Field(...)
    databases: list[str] = Field(...)


class BulkDatasetDataUploadResponse(BaseModel):
    uploaded: int = Field(..., description="Number of files uploaded")
    skipped: int = Field(0, description="Number of files skipped (duplicates)")
    failed: int = Field(0, description="Number of files that failed to upload")
    task_id: str | None = Field(None, description="Celery task ID if processing was started")
    status: str = Field('uploaded', description="Bulk upload status (processing when auto-process triggered)")


class CancelTaskResponse(BaseModel):
    """Response from cancelling a task."""
    message: str = Field(..., description="Human-readable message about the cancellation")
    task_id: str = Field(..., description="The ID of the cancelled task")
    cancelled: bool = Field(..., description="Whether the task was successfully cancelled")
    pending_tasks_cancelled: int = Field(0, description="Number of pending tasks that were cancelled")
    running_tasks_at_cancel: int = Field(0, description="Number of running tasks at the time of cancellation")
    files_reverted: int = Field(0, description="Number of files that were successfully reverted")
    files_failed_to_revert: int = Field(0, description="Number of files that failed to revert")
    errors: list[CleanupError] | None = Field(None, description="List of errors encountered during cleanup")
    already_completed: bool = Field(False, description="True if the task had already completed before cancellation was requested")
    already_cancelled: bool = Field(False, description="True if the task was already cancelled before this request")


class CatBoostFitRequest(BaseModel):
    """Request to train a CatBoost model."""
    model_id: str | None = Field(None, description="Model identifier (auto-generated if not provided)")
    model_type: Literal['classifier', 'regressor'] = Field('classifier', description="Model type: classifier or regressor")
    data: list[list[dict[str, Any]]] = Field(..., description="Training features (can include strings for categorical features)")
    labels: list[dict[str, Any]] = Field(..., description="Training labels (classes for classifier, values for regressor)")
    feature_names: list[str] | None = Field(None, description="Names for feature columns")
    cat_features: list[int] | None = Field(None, description="Indices of categorical feature columns (auto-detected if not specified)")
    iterations: int = Field(100, description="Number of boosting iterations")
    learning_rate: float = Field(0.1, description="Learning rate")
    depth: int = Field(6, description="Tree depth")
    random_state: int | None = Field(None, description="Random seed for reproducibility")
    validation_fraction: float | None = Field(None, description="Fraction of data to use for validation (0-1)")
    early_stopping_rounds: int | None = Field(None, description="Stop training if validation score doesn't improve for N rounds")


class CatBoostLoadRequest(BaseModel):
    """Request to load a model."""
    model_id: str = Field(..., description="Model identifier")


class CatBoostPredictRequest(BaseModel):
    """Request to make predictions."""
    model_id: str = Field(..., description="Model identifier")
    data: list[list[dict[str, Any]]] = Field(..., description="Features to predict on")
    return_proba: bool = Field(False, description="Return class probabilities (classifier only)")


class CatBoostUpdateRequest(BaseModel):
    """Request to incrementally update a model."""
    model_id: str = Field(..., description="Model identifier")
    data: list[list[dict[str, Any]]] = Field(..., description="New training features")
    labels: list[dict[str, Any]] = Field(..., description="New training labels")
    sample_weight: list[float] | None = Field(None, description="Optional sample weights")


class ChatCompletion(BaseModel):
    id: str = Field(...)
    choices: list[Choice] = Field(...)
    created: int = Field(...)
    model: str = Field(...)
    object: str = Field(...)
    service_tier: Literal['auto', 'default', 'flex', 'scale', 'priority'] | None = Field(None)
    system_fingerprint: str | None = Field(None)
    usage: CompletionUsage | None = Field(None)


class ChatCompletionAssistantMessageParam(BaseModel):
    role: str = Field(...)
    audio: Audio | None = Field(None)
    content: str | list[ChatCompletionContentPartTextParam | ChatCompletionContentPartRefusalParam] | None = Field(None)
    function_call: FunctionCall_Input | None = Field(None)
    name: str | None = Field(None)
    refusal: str | None = Field(None)
    tool_calls: list[ChatCompletionMessageFunctionToolCallParam | ChatCompletionMessageCustomToolCallParam] | None = Field(None)


class ChatCompletionAudio(BaseModel):
    id: str = Field(...)
    data: str = Field(...)
    expires_at: int = Field(...)
    transcript: str = Field(...)


class ChatCompletionContentPartImageParam(BaseModel):
    image_url: ImageURL = Field(...)
    type_: str = Field(..., alias="type")


class ChatCompletionContentPartInputAudioParam(BaseModel):
    input_audio: InputAudio = Field(...)
    type_: str = Field(..., alias="type")


class ChatCompletionContentPartRefusalParam(BaseModel):
    refusal: str = Field(...)
    type_: str = Field(..., alias="type")


class ChatCompletionContentPartTextParam(BaseModel):
    text: str = Field(...)
    type_: str = Field(..., alias="type")


class ChatCompletionDeveloperMessageParam(BaseModel):
    content: str | list[ChatCompletionContentPartTextParam] = Field(...)
    role: str = Field(...)
    name: str | None = Field(None)


class ChatCompletionFunctionMessageParam(BaseModel):
    content: str | None = Field(...)
    name: str = Field(...)
    role: str = Field(...)


class ChatCompletionFunctionToolParam(BaseModel):
    function: FunctionDefinition = Field(...)
    type_: str = Field(..., alias="type")


class ChatCompletionMessage(BaseModel):
    content: str | None = Field(None)
    refusal: str | None = Field(None)
    role: str = Field(...)
    annotations: list[Annotation] | None = Field(None)
    audio: ChatCompletionAudio | None = Field(None)
    function_call: FunctionCall_Output | None = Field(None)
    tool_calls: list[ChatCompletionMessageFunctionToolCall | ChatCompletionMessageCustomToolCall] | None = Field(None)


class ChatCompletionMessageCustomToolCall(BaseModel):
    id: str = Field(...)
    custom: Custom_Output = Field(...)
    type_: str = Field(..., alias="type")


class ChatCompletionMessageCustomToolCallParam(BaseModel):
    id: str = Field(...)
    custom: Custom_Input = Field(...)
    type_: str = Field(..., alias="type")


class ChatCompletionMessageFunctionToolCall(BaseModel):
    id: str = Field(...)
    function: Function_Output = Field(...)
    type_: str = Field(..., alias="type")


class ChatCompletionMessageFunctionToolCallParam(BaseModel):
    id: str = Field(...)
    function: Function_Input = Field(...)
    type_: str = Field(..., alias="type")


class ChatCompletionSystemMessageParam(BaseModel):
    content: str | list[ChatCompletionContentPartTextParam] = Field(...)
    role: str = Field(...)
    name: str | None = Field(None)


class ChatCompletionTokenLogprob(BaseModel):
    token: str = Field(...)
    bytes: list[int] | None = Field(None)
    logprob: float = Field(...)
    top_logprobs: list[TopLogprob] = Field(...)


class ChatCompletionToolMessageParam(BaseModel):
    content: str | list[ChatCompletionContentPartTextParam] = Field(...)
    role: str = Field(...)
    tool_call_id: str = Field(...)


class ChatCompletionUserMessageParam(BaseModel):
    content: str | list[ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam | ChatCompletionContentPartInputAudioParam | File] = Field(...)
    role: str = Field(...)
    name: str | None = Field(None)


class ChatRequest(BaseModel):
    messages: list[ChatCompletionDeveloperMessageParam | ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam | ChatCompletionAssistantMessageParam | ChatCompletionToolMessageParam | ChatCompletionFunctionMessageParam] = Field(...)
    model: str | None = Field(None)
    frequency_penalty: float | None = Field(None)
    logit_bias: dict[str, int] | None = Field(None)
    logprobs: bool | None = Field(None)
    max_completion_tokens: int | None = Field(None)
    max_tokens: int | None = Field(None)
    metadata: dict[str, Any] | None = Field(None)
    n: int | None = Field(None)
    parallel_tool_calls: bool | None = Field(None)
    presence_penalty: float | None = Field(None)
    response_format: dict[str, Any] | None = Field(None)
    seed: int | None = Field(None)
    stop: str | list[str] | None = Field(None)
    stream: bool = Field(False)
    stream_options: dict[str, Any] | None = Field(None)
    temperature: float | None = Field(None)
    tool_choice: str | dict[str, Any] | None = Field(None)
    tools: list[ChatCompletionFunctionToolParam] | None = Field(None)
    top_logprobs: int | None = Field(None)
    top_p: float | None = Field(None)
    user: str | None = Field(None)
    rag_enabled: bool | None = Field(None)
    database: str | None = Field(None)
    rag_retrieval_strategy: str | None = Field(None)
    rag_top_k: int | None = Field(None)
    rag_score_threshold: float | None = Field(None)
    n_ctx: int | None = Field(None)
    rag_queries: list[str] | None = Field(None, description="Custom queries for RAG retrieval. Overrides using the user message. Can be a single query or multiple queries - results are merged and deduplicated.")
    think: bool | None = Field(None)
    thinking_budget: int | None = Field(None)
    variables: dict[str, Any] | None = Field(None, description="Dynamic variables for template substitution in prompts and tools. Use {{variable_name}} syntax in config prompts/tools, then pass values here. Example: {'user_name': 'Alice', 'company': 'Acme Corp'}")
    include_sources: bool | None = Field(None, description="Include retrieved RAG chunks in streaming response as custom SSE event.")
    sources_limit: int | None = Field(10, description="Maximum number of sources to include (default 10, max 50).")


class Choice(BaseModel):
    finish_reason: Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call'] = Field(...)
    index: int = Field(...)
    logprobs: ChoiceLogprobs | None = Field(None)
    message: ChatCompletionMessage = Field(...)


class ChoiceLogprobs(BaseModel):
    content: list[ChatCompletionTokenLogprob] | None = Field(None)
    refusal: list[ChatCompletionTokenLogprob] | None = Field(None)


class ChunkPreviewInfo(BaseModel):
    """Information about a single chunk in preview."""
    chunk_index: int = Field(...)
    content: str = Field(...)
    start_position: int = Field(...)
    end_position: int = Field(...)
    char_count: int = Field(...)
    word_count: int = Field(...)
    metadata: dict[str, Any] | None = Field(None)


class ClassifierFitRequest(BaseModel):
    """Request to fit a text classifier."""
    model: str = Field(...)
    base_model: str = Field('sentence-transformers/all-MiniLM-L6-v2')
    training_data: list[dict[str, str]] = Field(...)
    num_iterations: int = Field(20)
    batch_size: int = Field(16)
    overwrite: bool = Field(True)
    description: str | None = Field(None)


class ClassifierLoadRequest(BaseModel):
    """Request to load a pre-trained classifier."""
    model: str = Field(...)


class ClassifierPredictRequest(BaseModel):
    """Request to classify texts."""
    model: str = Field(...)
    texts: list[str] = Field(...)


class ClassifierSaveRequest(BaseModel):
    """Request to save a fitted classifier."""
    model: str = Field(...)
    description: str | None = Field(None)


class ClassifyRequest(BaseModel):
    """Request for text classification."""
    input: str | list[str] = Field(...)
    model: str = Field(...)
    labels: list[str] | None = Field(None)


class CleanupError(BaseModel):
    """Error details for a file cleanup failure."""
    file_hash: str = Field(..., description="Hash of the file that failed to clean up")
    error: str = Field(..., description="Error message describing the cleanup failure")


class CompletionTokensDetails(BaseModel):
    accepted_prediction_tokens: int | None = Field(None)
    audio_tokens: int | None = Field(None)
    reasoning_tokens: int | None = Field(None)
    rejected_prediction_tokens: int | None = Field(None)


class CompletionUsage(BaseModel):
    completion_tokens: int = Field(...)
    prompt_tokens: int = Field(...)
    total_tokens: int = Field(...)
    completion_tokens_details: CompletionTokensDetails | None = Field(None)
    prompt_tokens_details: PromptTokensDetails | None = Field(None)


class ComponentHealth(BaseModel):
    """Health status of a single RAG component."""
    name: str = Field(...)
    status: str = Field(...)
    latency: float = Field(...)
    message: str | None = Field(None)


class ComponentsDefinition_Input(BaseModel):
    embedding_strategies: list[NamedEmbeddingStrategy] | None = Field(None, description="Reusable embedding strategies that can be referenced by name")
    retrieval_strategies: list[NamedRetrievalStrategy] | None = Field(None, description="Reusable retrieval strategies that can be referenced by name")
    parsers: list[NamedParserDefinition] | None = Field(None, description="Reusable parsers that can be referenced by name")
    defaults: Defaults | None = Field(None)


class ComponentsDefinition_Output(BaseModel):
    embedding_strategies: list[NamedEmbeddingStrategy] | None = Field(None, description="Reusable embedding strategies that can be referenced by name")
    retrieval_strategies: list[NamedRetrievalStrategy] | None = Field(None, description="Reusable retrieval strategies that can be referenced by name")
    parsers: list[NamedParserDefinition] | None = Field(None, description="Reusable parsers that can be referenced by name")
    defaults: Defaults | None = Field(None)


class CreateDatabaseRequest(BaseModel):
    """Request model for creating a new database."""
    name: str = Field(..., description="Unique database identifier")
    type_: str = Field(..., description="Database type (ChromaStore, QdrantStore)", alias="type")
    config: dict[str, Any] | None = Field(None, description="Database-specific configuration")
    embedding_strategies: list[dict[str, Any]] | None = Field(None, description="Embedding strategies for this database")
    embedding_strategy: str | None = Field(None, description="Reference to reusable embedding strategy defined under components.embedding_strategies")
    retrieval_strategies: list[dict[str, Any]] | None = Field(None, description="Retrieval strategies for this database")
    retrieval_strategy: str | None = Field(None, description="Reference to reusable retrieval strategy defined under components.retrieval_strategies")
    default_embedding_strategy: str | None = Field(None, description="Name of default embedding strategy")
    default_retrieval_strategy: str | None = Field(None, description="Name of default retrieval strategy")


class CreateDatasetRequest(BaseModel):
    name: str = Field(...)
    data_processing_strategy: str = Field(...)
    database: str = Field(...)


class CreateDatasetResponse(BaseModel):
    dataset: Dataset = Field(...)


class CreateProjectRequest(BaseModel):
    name: str = Field(..., description="The name of the project")
    config_template: str | None = Field(None, description="The config template to use for the project")


class CreateProjectResponse(BaseModel):
    project: Project = Field(..., description="The created project")


class Custom_Input(BaseModel):
    input: str = Field(...)
    name: str = Field(...)


class Custom_Output(BaseModel):
    input: str = Field(...)
    name: str = Field(...)


class DataProcessingStrategyDefinition(BaseModel):
    name: str = Field(..., description="Unique strategy identifier")
    description: str | None = Field(None, description="Strategy description")
    parsers: list[str | Parser] = Field(..., description="Document parsers in processing order")
    extractors: list[Extractor] | None = Field(None, description="Metadata and feature extractors")


class Database_Input(BaseModel):
    name: str = Field(..., description="Unique database identifier")
    type_: DatabaseType = Field(..., alias="type")
    config: dict[str, Any] | None = Field(None, description="Database-specific configuration")
    embedding_strategies: list[DatabaseEmbeddingStrategy] | None = Field(None, description="Multiple embedding strategies for hybrid search")
    embedding_strategy: str | None = Field(None, description="Reference to a reusable embedding strategy defined under components.embedding_strategies")
    retrieval_strategies: list[DatabaseRetrievalStrategy] | None = Field(None, description="Multiple retrieval strategies for different query patterns")
    retrieval_strategy: str | None = Field(None, description="Reference to a reusable retrieval strategy defined under components.retrieval_strategies")
    default_embedding_strategy: str | None = Field(None, description="Name of the default embedding strategy to use")
    default_retrieval_strategy: str | None = Field(None, description="Name of the default retrieval strategy to use")


class Database_Output(BaseModel):
    name: str = Field(..., description="Unique database identifier")
    type_: DatabaseType = Field(..., alias="type")
    config: dict[str, Any] | None = Field(None, description="Database-specific configuration")
    embedding_strategies: list[DatabaseEmbeddingStrategy] | None = Field(None, description="Multiple embedding strategies for hybrid search")
    embedding_strategy: str | None = Field(None, description="Reference to a reusable embedding strategy defined under components.embedding_strategies")
    retrieval_strategies: list[DatabaseRetrievalStrategy] | None = Field(None, description="Multiple retrieval strategies for different query patterns")
    retrieval_strategy: str | None = Field(None, description="Reference to a reusable retrieval strategy defined under components.retrieval_strategies")
    default_embedding_strategy: str | None = Field(None, description="Name of the default embedding strategy to use")
    default_retrieval_strategy: str | None = Field(None, description="Name of the default retrieval strategy to use")


class DatabaseDetailResponse(BaseModel):
    """Detailed response for a single database including raw config."""
    name: str = Field(...)
    type_: str = Field(..., alias="type")
    config: dict[str, Any] | None = Field(...)
    embedding_strategies: list[dict[str, Any]] | None = Field(...)
    retrieval_strategies: list[dict[str, Any]] | None = Field(...)
    default_embedding_strategy: str | None = Field(...)
    default_retrieval_strategy: str | None = Field(...)
    dependent_datasets: list[str] = Field(...)


class DatabaseEmbeddingStrategy(BaseModel):
    name: str = Field(..., description="Strategy identifier")
    type_: DatabaseEmbeddingType = Field(..., alias="type")
    config: dict[str, Any] | None = Field(None, description="Embedder configuration")
    condition: str | None = Field(None, description="Optional condition for when to use this embedding (e.g., doc.type == 'code')")
    priority: int | None = Field(0, description="Priority for automatic selection")


# Enum: DatabaseEmbeddingType
DatabaseEmbeddingType = Literal['HuggingFaceEmbedder', 'OllamaEmbedder', 'OpenAIEmbedder', 'SentenceTransformerEmbedder', 'UniversalEmbedder']

class DatabaseInfo(BaseModel):
    name: str = Field(...)
    type_: str = Field(..., alias="type")
    is_default: bool = Field(...)
    embedding_strategies: list[EmbeddingStrategyInfo] = Field(...)
    retrieval_strategies: list[RetrievalStrategyInfo] = Field(...)


class DatabaseResponse(BaseModel):
    """Response model for a single database."""
    database: DatabaseInfo = Field(...)


class DatabaseRetrievalStrategy(BaseModel):
    name: str = Field(..., description="Strategy identifier")
    type_: DatabaseRetrievalType = Field(..., alias="type")
    config: dict[str, Any] | None = Field(None, description="Retrieval configuration")
    default: bool | None = Field(False, description="Whether this is the default strategy")


# Enum: DatabaseRetrievalType
DatabaseRetrievalType = Literal['VectorRetriever', 'HybridRetriever', 'BM25Retriever', 'RerankedRetriever', 'GraphRetriever', 'ElasticRetriever', 'BasicSimilarityStrategy', 'MetadataFilteredStrategy', 'MultiQueryStrategy', 'RerankedStrategy', 'HybridUniversalStrategy', 'CrossEncoderRerankedStrategy', 'MultiTurnRAGStrategy']

# Enum: DatabaseType
DatabaseType = Literal['ChromaStore', 'QdrantStore']

class DatabasesResponse(BaseModel):
    databases: list[DatabaseInfo] = Field(...)
    default_database: str | None = Field(...)


class Dataset(BaseModel):
    name: str = Field(..., description="Dataset name")
    auto_process: bool | None = Field(True, description="Whether to automatically process uploads into the vector store")
    data_processing_strategy: str | None = Field('universal_rag', description="RAG data processing strategy to use for the dataset. Defaults to universal_rag if not specified.")
    database: str = Field(..., description="RAG database to use for the dataset")


class DatasetActionRequest(BaseModel):
    action_type: DatasetActionType = Field(..., description="The type of action to execute")
    file_hash: str | None = Field(None, description="File hash for delete_file_chunks action")
    parser_overrides: dict[str, dict[str, Any]] | None = Field(None, description="Optional parser config overrides for PROCESS actions")


class DatasetActionResponse(BaseModel):
    message: str = Field(..., description="The status message")
    task_uri: str = Field(..., description="The URI for tracking the task")
    task_id: str = Field(..., description="The Celery task ID")


# Enum: DatasetActionType
DatasetActionType = Literal['process', 'delete_file_chunks', 'delete_dataset_chunks']

class DatasetDataUploadResponse(BaseModel):
    filename: str = Field(..., description="The name of the uploaded file")
    hash: str = Field(..., description="The hash of the uploaded file")
    processed: bool = Field(..., description="Whether the file has been processed")
    skipped: bool = Field(False, description="Whether the file was skipped (duplicate)")
    task_id: str | None = Field(None, description="Celery task ID if processing was started")
    status: str | None = Field(None, description="Upload status (processing, uploaded, skipped, or error)")


class DatasetDetails(BaseModel):
    files_metadata: list[MetadataFileContent] = Field(...)


class DatasetWithFileDetails(BaseModel):
    name: str = Field(..., description="Dataset name")
    auto_process: bool | None = Field(True, description="Whether to automatically process uploads into the vector store")
    data_processing_strategy: str | None = Field('universal_rag', description="RAG data processing strategy to use for the dataset. Defaults to universal_rag if not specified.")
    database: str = Field(..., description="RAG database to use for the dataset")
    details: DatasetDetails = Field(...)


class Defaults(BaseModel):
    embedding_strategy: str | None = Field(None, description="Default embedding strategy to use when not specified")
    retrieval_strategy: str | None = Field(None, description="Default retrieval strategy to use when not specified")
    parser: str | None = Field(None, description="Default parser to use when not specified")


class DeleteDataResponse(BaseModel):
    file_hash: str = Field(...)
    deleted_chunks: int = Field(..., description="Number of chunks deleted from vector store")


class DeleteDatabaseResponse(BaseModel):
    """Response model for database deletion."""
    message: str = Field(...)
    database: DatabaseDetailResponse = Field(...)
    collection_deleted: bool = Field(...)


class DeleteDatasetResponse(BaseModel):
    dataset: Dataset = Field(...)


class DeleteProjectResponse(BaseModel):
    project: Project = Field(..., description="The deleted project")


class DocumentPreviewRequest(BaseModel):
    """Request model for document preview."""
    dataset_id: str | None = Field(None, description="Dataset containing the file")
    file_hash: str | None = Field(None, description="Hash of the file to preview")
    file_content: str | None = Field(None, description="Base64-encoded file content")
    filename: str | None = Field(None, description="Filename for uploaded content")
    data_processing_strategy: str | None = Field(None, description="Data processing strategy to use. If not provided, uses the dataset's configured strategy or falls back to the first available strategy.")
    chunk_size: int | None = Field(None, description="Override chunk size")
    chunk_overlap: int | None = Field(None, description="Override chunk overlap")
    chunk_strategy: str | None = Field(None, description="Override chunk strategy")


class DocumentPreviewResponse(BaseModel):
    """Response model for document preview."""
    original_text: str = Field(...)
    chunks: list[ChunkPreviewInfo] = Field(...)
    filename: str = Field(...)
    size_bytes: int = Field(...)
    content_type: str | None = Field(None)
    parser_used: str = Field(...)
    chunk_strategy: str = Field(...)
    chunk_size: int = Field(...)
    chunk_overlap: int = Field(...)
    total_chunks: int = Field(...)
    avg_chunk_size: float = Field(...)
    total_size_with_overlaps: int = Field(...)
    avg_overlap_size: float = Field(0.0)
    warnings: list[str] | None = Field(None)


class DownloadModelRequest(BaseModel):
    provider: Provider = Field('universal')
    model_name: str = Field(...)


class DriftDetectRequest(BaseModel):
    """Request to detect drift in new data."""
    model: str = Field(..., description="Model identifier")
    data: list[list[float]] = Field(..., description="New data to check for drift")


class DriftFitRequest(BaseModel):
    """Request to fit a drift detector."""
    model: str | None = Field(None, description="Model identifier (auto-generated if not provided)")
    detector: str = Field('ks', description="Detector type: ks (numeric), mmd (multivariate), chi_squared (categorical)")
    reference_data: list[list[float]] = Field(..., description="Reference data to learn the baseline distribution")
    feature_names: list[str] | None = Field(None, description="Names for each feature column")
    params: dict[str, Any] | None = Field(None, description="Detector-specific parameters (e.g., p_val)")
    overwrite: bool = Field(True, description="Overwrite existing model if it exists")
    description: str | None = Field(None, description="Optional model description")


class DriftLoadRequest(BaseModel):
    """Request to load a drift model."""
    model: str = Field(..., description="Model identifier (supports -latest suffix)")


class EmbeddingRequest(BaseModel):
    """Request for text embeddings."""
    input: str | list[str] = Field(...)
    model: str = Field(...)
    encoding_format: str = Field('float')
    dimensions: int | None = Field(None)


class EmbeddingStrategyInfo(BaseModel):
    """Information about an embedding strategy."""
    name: str = Field(...)
    type_: str = Field(..., alias="type")
    priority: int = Field(...)
    is_default: bool = Field(...)


class Emotion(BaseModel):
    enabled: bool | None = Field(True, description="Enable speech emotion recognition. When enabled, analyzes audio to detect user emotional tone and includes it in the LLM context. This allows the model to respond appropriately to user mood.")
    model: str | None = Field('wav2vec2-lg-xlsr-en', description="Emotion recognition model ID: - wav2vec2-lg-xlsr-en: English emotion recognition (default, recommended) - wav2vec2-base-superb: Alternative model from SUPERB benchmark Custom HuggingFace model IDs are also supported.")
    confidence_threshold: float | None = Field(0.4, description="Minimum confidence threshold for emotion detection. Predictions below this threshold are reported as 'neutral'. Lower values increase sensitivity but may produce false positives.")


class EncoderConfig(BaseModel):
    max_length: int | None = Field(None, description="Maximum sequence length for tokenization. Auto-detected if not specified. ModernBERT supports up to 8,192 tokens, classic BERT supports 512. Must be a positive integer if specified.")
    use_flash_attention: bool | None = Field(True, description="Enable Flash Attention 2 for faster inference on CUDA devices. Requires flash_attn package and compatible GPU.")
    task: Task | None = Field('embedding', description="Task type for the encoder model: - embedding: Generate dense vector representations - classification: Sentiment, spam detection, intent routing - reranking: Cross-encoder document reranking - ner: Named entity recognition")


class ErrorResponse(BaseModel):
    error: str = Field(...)
    message: str = Field(...)
    details: str | None = Field(None)


class EventDetail(BaseModel):
    """Full event details including all sub-events."""
    event_id: str = Field(..., description="Unique event identifier")
    event_type: str = Field(..., description="Type of event")
    request_id: str = Field(..., description="Request identifier")
    timestamp: str = Field(..., description="Event start timestamp")
    namespace: str = Field(..., description="Project namespace")
    project: str = Field(..., description="Project name")
    config_hash: str = Field(..., description="Config hash at time of event")
    events: list[SubEvent] = Field(..., description="List of sub-events")
    status: str = Field(..., description="Event status")
    error: str | None = Field(None, description="Error message if failed")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")
    total_elapsed_time_ms: float | None = Field(None, description="Total event duration in milliseconds")
    time_to_first_token_ms: float | None = Field(None, description="Time to first token in milliseconds (for streaming inference)")


class EventSummary(BaseModel):
    """Summary of an event for list responses."""
    event_id: str = Field(..., description="Unique event identifier")
    event_type: str = Field(..., description="Type of event (inference, rag_processing, etc.)")
    request_id: str = Field(..., description="Request identifier")
    timestamp: str = Field(..., description="Event start timestamp")
    namespace: str = Field(..., description="Project namespace")
    project: str = Field(..., description="Project name")
    status: str = Field(..., description="Event status (completed, failed)")
    duration_ms: float | None = Field(None, description="Total event duration in milliseconds")
    config_hash: str = Field(..., description="Config hash at time of event")


class ExampleDataset(BaseModel):
    """Dataset defined by an example manifest, with computed file sizes."""
    example_id: str = Field(...)
    example_title: str | None = Field(None)
    name: str = Field(...)
    strategy: str | None = Field(None)
    database: str | None = Field(None)
    kind: str | None = Field(None)
    ingest: list[str] | None = Field(None)
    size_bytes: int | None = Field(None)
    size_human: str | None = Field(None)


class ExampleSummary(BaseModel):
    id: str = Field(...)
    slug: str | None = Field(None)
    title: str = Field(...)
    description: str | None = Field(None)
    primaryModel: str | None = Field(None)
    tags: list[str] | None = Field(None)
    dataset_count: int | None = Field(None)
    data_size_bytes: int | None = Field(None)
    data_size_human: str | None = Field(None)
    project_size_bytes: int | None = Field(None)
    project_size_human: str | None = Field(None)
    updated_at: str | None = Field(None)


class ExtraBody(BaseModel):
    n_ctx: int | None = Field(None, description="Context window size for GGUF models (Universal runtime only). If not specified, automatically computed based on available memory, model training context, and pattern-based defaults. See Universal runtime documentation for auto-detection behavior.")
    n_batch: int | None = Field(None, description="Batch size for prompt processing. Controls compute buffer memory usage. Lower values (e.g., 512) reduce memory significantly at slight speed cost. Critical for memory-constrained devices like Jetson (8GB shared memory).")
    n_gpu_layers: int | None = Field(None, description="Number of layers to offload to GPU. Use -1 to offload all layers. For models like Qwen3-1.7B with 29 layers, use 29 for full GPU offload.")
    n_threads: int | None = Field(None, description="Number of CPU threads to use. If not specified, auto-detected. For Jetson Orin Nano, use 6 (number of CPU cores).")
    flash_attn: bool | None = Field(None, description="Enable flash attention for faster inference. Requires GPU support. Recommended for Ampere+ GPUs (RTX 30xx, A100, Jetson Orin).")
    use_mmap: bool | None = Field(None, description="Memory-map model file for efficient memory usage. Default: true. Enables OS to swap model pages efficiently on memory-constrained devices.")
    use_mlock: bool | None = Field(None, description="Lock model in RAM to prevent swapping. Default: false. Set to false on memory-constrained devices (e.g., 8GB Jetson) to allow OS memory management.")
    cache_type_k: str | None = Field(None, description="KV cache key quantization type. Lower precision reduces memory usage. Common values: f32 (full), f16 (half), q8_0, q5_1, q5_0, q4_1, q4_0. Using q4_0 can reduce KV cache memory by ~4x vs f16.")
    cache_type_v: str | None = Field(None, description="KV cache value quantization type. Lower precision reduces memory usage. Common values: f32 (full), f16 (half), q8_0, q5_1, q5_0, q4_1, q4_0. Using q4_0 can reduce KV cache memory by ~4x vs f16.")


class Extractor(BaseModel):
    type_: str = Field(..., description="Extractor type", alias="type")
    config: dict[str, Any] | None = Field(None, description="Extractor configuration")
    required_for: list[str] | None = Field(None, description="Database names that require this extractor")
    condition: str | None = Field(None, description="Optional condition for when to use this extractor")
    file_include_patterns: list[str] | None = Field(None, description="Glob patterns for files to apply this extractor to")
    priority: int | None = Field(50, description="Extractor priority (lower = apply first)")


class FeatureImportanceRequest(BaseModel):
    """Request for global feature importance."""
    model_type: str = Field(..., description="Type of model: anomaly, classifier, catboost")
    model_id: str = Field(..., description="Model identifier")
    data: list[list[float]] = Field(..., description="Data to compute importance on")
    feature_names: list[str] | None = Field(None, description="Names for features")


class File(BaseModel):
    file: FileFile = Field(...)
    type_: str = Field(..., alias="type")


class FileFile(BaseModel):
    file_data: str | None = Field(None)
    file_id: str | None = Field(None)
    filename: str | None = Field(None)


class Function_Input(BaseModel):
    arguments: str = Field(...)
    name: str = Field(...)


class Function_Output(BaseModel):
    arguments: str = Field(...)
    name: str = Field(...)


class FunctionCall_Input(BaseModel):
    arguments: str = Field(...)
    name: str = Field(...)


class FunctionCall_Output(BaseModel):
    arguments: str = Field(...)
    name: str = Field(...)


class FunctionDefinition(BaseModel):
    name: str = Field(...)
    description: str | None = Field(None)
    parameters: dict[str, Any] | None = Field(None)
    strict: bool | None = Field(None)


class GGUFOption(BaseModel):
    filename: str = Field(...)
    quantization: str | None = Field(...)
    size_bytes: int = Field(...)
    size_human: str = Field(...)


class GGUFOptionsResponse(BaseModel):
    options: list[GGUFOption] = Field(...)


class GetProjectResponse(BaseModel):
    project: Project = Field(..., description="The project")


class GetTaskResponse(BaseModel):
    task_id: str = Field(..., description="The ID of the asynchronous task")
    state: str = Field(..., description="Current state of the task (e.g., PENDING, STARTED, SUCCESS, FAILURE)")
    meta: dict[str, Any] | None = Field(None, description="Progress metadata or intermediate results, if available")
    result: dict[str, Any] | str | None = Field(None, description="Result of the task if completed successfully")
    error: str | None = Field(None, description="Error message if the task failed")
    traceback: str | None = Field(None, description="Traceback information if the task failed")
    cancelled: bool = Field(False, description="Whether the task has been cancelled")


class ImageURL(BaseModel):
    url: str = Field(...)
    detail: Literal['auto', 'low', 'high'] | None = Field(None)


class ImportDataRequest(BaseModel):
    namespace: str = Field(...)
    project: str = Field(...)
    include_strategies: bool = Field(True)
    process: bool = Field(True)


class ImportDataResponse(BaseModel):
    project: str = Field(...)
    namespace: str = Field(...)
    datasets: list[str] = Field(...)
    task_ids: list[str] = Field(...)


class ImportDatasetRequest(BaseModel):
    namespace: str = Field(...)
    project: str = Field(...)
    dataset: str = Field(...)
    target_dataset: str | None = Field(None)
    include_strategies: bool = Field(True)
    process: bool = Field(True)


class ImportDatasetResponse(BaseModel):
    project: str = Field(...)
    namespace: str = Field(...)
    dataset: str = Field(...)
    file_count: int = Field(...)
    task_id: str | None = Field(None)


class ImportProjectRequest(BaseModel):
    namespace: str = Field(...)
    name: str = Field(...)
    process: bool = Field(True)


class ImportProjectResponse(BaseModel):
    project: str = Field(...)
    namespace: str = Field(...)
    datasets: list[str] = Field(...)
    task_ids: list[str] = Field(...)


class InputAudio(BaseModel):
    data: str = Field(...)
    format: Literal['wav', 'mp3'] = Field(...)


class ListDatasetsResponse(BaseModel):
    total: int = Field(...)
    datasets: list[Dataset | DatasetWithFileDetails] = Field(...)


class ListEventsResponse(BaseModel):
    """Response for list events endpoint."""
    total: int = Field(..., description="Total number of matching events")
    events: list[EventSummary] = Field(..., description="List of event summaries")
    limit: int = Field(..., description="Limit applied")
    offset: int = Field(..., description="Offset applied")


class ListModelsResponse(BaseModel):
    """Response for list models API endpoint."""
    total: int = Field(..., description="Total number of models")
    models: list[ModelResponse] = Field(..., description="List of models")


class ListProjectsResponse(BaseModel):
    total: int = Field(..., description="The total number of projects")
    projects: list[Project] = Field(..., description="The list of projects")


class LlamaFarmConfig_Input(BaseModel):
    version: Version = Field(..., description="Config version, must be \"v1\"")
    name: str = Field(..., description="Project name")
    namespace: str = Field(..., description="Project namespace")
    components: ComponentsDefinition_Input | None = Field(None, description="Reusable RAG components (embedding/retrieval strategies, parsers, defaults)")
    prompts: list[PromptSet] | None = Field([], description="List of named prompt sets")
    rag: RAGStrategyConfigurationSchema_Input | None = Field(None)
    datasets: list[Dataset] | None = Field([], description="List of dataset configurations")
    runtime: Runtime_Input = Field(...)
    mcp: Mcp_Input | None = Field(None, description="Model Context Protocol (MCP) client configuration")
    voice: Voice_Input | None = Field(None, description="Voice chat configuration for real-time speech interaction via WebSocket")


class LlamaFarmConfig_Output(BaseModel):
    version: Version = Field(..., description="Config version, must be \"v1\"")
    name: str = Field(..., description="Project name")
    namespace: str = Field(..., description="Project namespace")
    components: ComponentsDefinition_Output | None = Field(None, description="Reusable RAG components (embedding/retrieval strategies, parsers, defaults)")
    prompts: list[PromptSet] | None = Field([], description="List of named prompt sets")
    rag: RAGStrategyConfigurationSchema_Output | None = Field(None)
    datasets: list[Dataset] | None = Field([], description="List of dataset configurations")
    runtime: Runtime_Output = Field(...)
    mcp: Mcp_Output | None = Field(None, description="Model Context Protocol (MCP) client configuration")
    voice: Voice_Output | None = Field(None, description="Voice chat configuration for real-time speech interaction via WebSocket")


class Mcp_Input(BaseModel):
    servers: list[Server] | None = Field(None, description="List of MCP Servers available to the project during inference")


class Mcp_Output(BaseModel):
    servers: list[Server] | None = Field(None, description="List of MCP Servers available to the project during inference")


class MetadataFileContent(BaseModel):
    original_file_name: str = Field(...)
    resolved_file_name: str = Field(...)
    timestamp: float = Field(...)
    size: int = Field(...)
    mime_type: str = Field(...)
    hash: str = Field(...)
    chunk_count: int | None = Field(None)


class Model_Input(BaseModel):
    name: str = Field(..., description="Model identifier (unique name)")
    description: str | None = Field(None, description="Human-readable description of this model configuration")
    provider: Provider = Field(..., description="Runtime provider for this model")
    model: str = Field(..., description="Model name or ID")
    base_url: str | None = Field(None, description="Base URL for the provider")
    api_key: str | None = Field(None, description="API key for the provider")
    instructor_mode: str | None = Field(None, description="Instructor mode to use for structured output (e.g., tools, json, md_json)")
    model_api_parameters: dict[str, Any] | None = Field(None, description="Additional parameters passed directly to the API provider as request parameters. Common examples: temperature, top_p, max_tokens, frequency_penalty, etc.")
    extra_body: ExtraBody | None = Field(None, description="Provider-specific parameters passed in the request body's extra_body field. These parameters are sent directly to the underlying provider API. Note: For GGUF quantization, specify it in the model name using the format \"model_id:quantization\" (e.g., \"unsloth/Qwen3-4B-GGUF:Q4_K_M\"). The properties below document common GGUF/llama.cpp parameters, but any additional parameters will be passed through to the provider.")
    encoder_config: EncoderConfig | None = Field(None, description="Configuration for BERT-style encoder models (Universal runtime only). Used for embeddings, classification, reranking, and NER endpoints.")
    prompts: list[str] | None = Field([], description="List of prompt set names to use for this model (merged in order)")
    mcp_servers: list[str] | None = Field(None, description="List of MCP server names to use for this model (omit to use all servers, empty list for none)")
    tool_call_strategy: ToolCallStrategy | None = Field('native_api', description="Strategy to use for tool calls. `native_api` uses native tool calling through the client library (e.g. setting the `tools` parameter on the chat completions OpenAI request); `prompt_based` uses system prompting to inject tool definitions and instructions to guide any model to use tools. This is universal, but may not be as effective as native tool calling. Use `prompt_based` for models that do not support native tool calling.")
    tools: list[Tool] | None = Field([], description="List of tools to use for this model")
    keep_loaded: bool | None = Field(False, description="Keep this model loaded in memory to prevent cache eviction. When true, the runtime will not purge this model from cache even under memory pressure. Useful for models that need fast response times (e.g., voice chat LLMs).")
    rag_enabled: bool | None = Field(None, description="Default RAG behavior for this model. When true, RAG is enabled by default for chat requests using this model. When false, RAG is disabled by default. Can be overridden per-request via the rag_enabled API parameter. If not set, falls back to project-level behavior (RAG enabled when databases exist).")
    target_database: str | None = Field(None, description="Default RAG database for this model. When set, chat requests using this model will use this database instead of the project's default_database. Can be overridden per-request via the database API parameter. Must match a database name defined in the rag.databases section.")


class Model_Output(BaseModel):
    name: str = Field(..., description="Model identifier (unique name)")
    description: str | None = Field(None, description="Human-readable description of this model configuration")
    provider: Provider = Field(..., description="Runtime provider for this model")
    model: str = Field(..., description="Model name or ID")
    base_url: str | None = Field(None, description="Base URL for the provider")
    api_key: str | None = Field(None, description="API key for the provider")
    instructor_mode: str | None = Field(None, description="Instructor mode to use for structured output (e.g., tools, json, md_json)")
    model_api_parameters: dict[str, Any] | None = Field(None, description="Additional parameters passed directly to the API provider as request parameters. Common examples: temperature, top_p, max_tokens, frequency_penalty, etc.")
    extra_body: ExtraBody | None = Field(None, description="Provider-specific parameters passed in the request body's extra_body field. These parameters are sent directly to the underlying provider API. Note: For GGUF quantization, specify it in the model name using the format \"model_id:quantization\" (e.g., \"unsloth/Qwen3-4B-GGUF:Q4_K_M\"). The properties below document common GGUF/llama.cpp parameters, but any additional parameters will be passed through to the provider.")
    encoder_config: EncoderConfig | None = Field(None, description="Configuration for BERT-style encoder models (Universal runtime only). Used for embeddings, classification, reranking, and NER endpoints.")
    prompts: list[str] | None = Field([], description="List of prompt set names to use for this model (merged in order)")
    mcp_servers: list[str] | None = Field(None, description="List of MCP server names to use for this model (omit to use all servers, empty list for none)")
    tool_call_strategy: ToolCallStrategy | None = Field('native_api', description="Strategy to use for tool calls. `native_api` uses native tool calling through the client library (e.g. setting the `tools` parameter on the chat completions OpenAI request); `prompt_based` uses system prompting to inject tool definitions and instructions to guide any model to use tools. This is universal, but may not be as effective as native tool calling. Use `prompt_based` for models that do not support native tool calling.")
    tools: list[Tool] | None = Field([], description="List of tools to use for this model")
    keep_loaded: bool | None = Field(False, description="Keep this model loaded in memory to prevent cache eviction. When true, the runtime will not purge this model from cache even under memory pressure. Useful for models that need fast response times (e.g., voice chat LLMs).")
    rag_enabled: bool | None = Field(None, description="Default RAG behavior for this model. When true, RAG is enabled by default for chat requests using this model. When false, RAG is disabled by default. Can be overridden per-request via the rag_enabled API parameter. If not set, falls back to project-level behavior (RAG enabled when databases exist).")
    target_database: str | None = Field(None, description="Default RAG database for this model. When set, chat requests using this model will use this database instead of the project's default_database. Can be overridden per-request via the database API parameter. Must match a database name defined in the rag.databases section.")


# Enum: Model1
Model1 = Literal['tiny', 'base', 'small', 'medium', 'large-v3', 'distil-large-v3-turbo']

class ModelResponse(BaseModel):
    """Response for model API endpoint."""
    name: str = Field(..., description="Model identifier (unique name)")
    description: str | None = Field(None, description="Human-readable description of this model configuration")
    provider: Provider = Field(..., description="Runtime provider for this model")
    model: str = Field(..., description="Model name or ID")
    base_url: str | None = Field(None, description="Base URL for the provider")
    api_key: str | None = Field(None, description="API key for the provider")
    instructor_mode: str | None = Field(None, description="Instructor mode to use for structured output (e.g., tools, json, md_json)")
    model_api_parameters: dict[str, Any] | None = Field(None, description="Additional parameters passed directly to the API provider as request parameters. Common examples: temperature, top_p, max_tokens, frequency_penalty, etc.")
    extra_body: ExtraBody | None = Field(None, description="Provider-specific parameters passed in the request body's extra_body field. These parameters are sent directly to the underlying provider API. Note: For GGUF quantization, specify it in the model name using the format \"model_id:quantization\" (e.g., \"unsloth/Qwen3-4B-GGUF:Q4_K_M\"). The properties below document common GGUF/llama.cpp parameters, but any additional parameters will be passed through to the provider.")
    encoder_config: EncoderConfig | None = Field(None, description="Configuration for BERT-style encoder models (Universal runtime only). Used for embeddings, classification, reranking, and NER endpoints.")
    prompts: list[str] | None = Field([], description="List of prompt set names to use for this model (merged in order)")
    mcp_servers: list[str] | None = Field(None, description="List of MCP server names to use for this model (omit to use all servers, empty list for none)")
    tool_call_strategy: ToolCallStrategy | None = Field('native_api', description="Strategy to use for tool calls. `native_api` uses native tool calling through the client library (e.g. setting the `tools` parameter on the chat completions OpenAI request); `prompt_based` uses system prompting to inject tool definitions and instructions to guide any model to use tools. This is universal, but may not be as effective as native tool calling. Use `prompt_based` for models that do not support native tool calling.")
    tools: list[Tool] | None = Field([], description="List of tools to use for this model")
    keep_loaded: bool | None = Field(False, description="Keep this model loaded in memory to prevent cache eviction. When true, the runtime will not purge this model from cache even under memory pressure. Useful for models that need fast response times (e.g., voice chat LLMs).")
    rag_enabled: bool | None = Field(None, description="Default RAG behavior for this model. When true, RAG is enabled by default for chat requests using this model. When false, RAG is disabled by default. Can be overridden per-request via the rag_enabled API parameter. If not set, falls back to project-level behavior (RAG enabled when databases exist).")
    target_database: str | None = Field(None, description="Default RAG database for this model. When set, chat requests using this model will use this database instead of the project's default_database. Can be overridden per-request via the database API parameter. Must match a database name defined in the rag.databases section.")
    default: bool = Field(False, description="Whether this model is the default model in the runtime config")


class NERRequest(BaseModel):
    """Request for named entity recognition."""
    input: str | list[str] = Field(...)
    model: str = Field(...)


class NamedEmbeddingStrategy(BaseModel):
    name: str = Field(..., description="Strategy identifier")
    type_: Type3 = Field(..., description="Embedding type", alias="type")
    config: dict[str, Any] | None = Field(None, description="Embedder configuration")
    condition: str | None = Field(None, description="Optional condition for when to use this embedding (e.g., doc.type == 'code')")
    priority: int | None = Field(0, description="Priority for automatic selection")


class NamedParserDefinition(BaseModel):
    name: str = Field(..., description="Parser identifier")
    type_: str = Field(..., description="Parser type identifier (e.g., PDFParser_PyPDF2, TextParser_LlamaIndex)", alias="type")
    config: dict[str, Any] | None = Field(None, description="Parser configuration")
    file_extensions: list[str] | None = Field(None, description="File extensions this parser handles (e.g., [\".pdf\", \".PDF\"])")
    file_include_patterns: list[str] | None = Field(None, description="Glob patterns for files to include (e.g., [\"*.pdf\", \"report_*.pdf\"])")
    priority: int | None = Field(50, description="Parser priority (lower = try first)")
    mime_types: list[str] | None = Field(None, description="MIME types this parser handles (e.g., [\"application/pdf\"])")
    fallback_parser: str | None = Field(None, description="Parser to use if this one fails")


class NamedRetrievalStrategy(BaseModel):
    name: str = Field(..., description="Strategy identifier")
    type_: Type4 = Field(..., description="Retrieval type", alias="type")
    config: dict[str, Any] | None = Field(None, description="Retrieval configuration")
    default: bool | None = Field(False, description="Whether this is the default strategy")


class Parser(BaseModel):
    type_: str | None = Field(None, description="Parser type identifier (e.g., PDFParser_PyPDF2, TextParser_LlamaIndex)", alias="type")
    config: dict[str, Any] | None = Field(None, description="Parser configuration")
    file_extensions: list[str] | None = Field(None, description="File extensions this parser handles (e.g., [\".pdf\", \".PDF\"])")
    file_include_patterns: list[str] | None = Field(None, description="Glob patterns for files to include (e.g., [\"*.pdf\", \"report_*.pdf\"])")
    priority: int | None = Field(50, description="Parser priority (lower = try first)")
    mime_types: list[str] | None = Field(None, description="MIME types this parser handles (e.g., [\"application/pdf\"])")
    fallback_parser: str | None = Field(None, description="Parser to use if this one fails")


class PolarsBufferAppendRequest(BaseModel):
    """Request to append data to a Polars buffer."""
    buffer_id: str = Field(..., description="Buffer identifier")
    data: dict[str, Any] | list[dict[str, Any]] = Field(..., description="Single record or batch of records to append")


class PolarsBufferCreateRequest(BaseModel):
    """Request to create a named Polars buffer."""
    buffer_id: str = Field(..., description="Unique buffer identifier")
    window_size: int = Field(1000, description="Maximum records to keep (sliding window)")


class PolarsBufferDataResponse(BaseModel):
    """Response containing buffer data."""
    object: str = Field('polars_data')
    buffer_id: str = Field(...)
    rows: int = Field(...)
    columns: list[str] = Field(...)
    data: list[dict[str, Any]] = Field(...)


class PolarsBufferFeaturesRequest(BaseModel):
    """Request to compute features from a Polars buffer."""
    buffer_id: str = Field(..., description="Buffer identifier")
    rolling_windows: list[int] | None = Field(None, description="Rolling window sizes (default: [5, 10, 20])")
    include_rolling_stats: list[Literal['mean', 'std', 'min', 'max']] | None = Field(None, description="Which rolling stats to compute (default: all)")
    include_lags: bool = Field(True, description="Include lag features")
    lag_periods: list[int] | None = Field(None, description="Lag periods (default: [1, 2, 3])")
    tail: int | None = Field(None, description="Return only last N rows (optional)")


class PolarsBufferStats(BaseModel):
    """Statistics about a Polars buffer."""
    buffer_id: str = Field(...)
    size: int = Field(...)
    window_size: int = Field(...)
    columns: list[str] = Field(...)
    numeric_columns: list[str] = Field(...)
    memory_bytes: int = Field(...)
    append_count: int = Field(...)
    avg_append_ms: float = Field(...)


class PolarsBuffersListResponse(BaseModel):
    """List of active Polars buffers."""
    object: str = Field('list')
    data: list[PolarsBufferStats] = Field(...)
    total: int = Field(...)


class Project(BaseModel):
    namespace: str = Field(..., description="The namespace of the project")
    name: str = Field(..., description="The name of the project")
    config: LlamaFarmConfig_Output | dict[str, Any] = Field(..., description="The configuration of the project")
    validation_error: str | None = Field(None, description="Validation error message if config has issues")
    last_modified: str | None = Field(None, description="Last modified timestamp of the project config")


class PromptMessage(BaseModel):
    role: str = Field(..., description="Message role (e.g., \"system\", \"user\", \"assistant\", \"tool\")")
    content: str = Field(..., description="Message content")
    tool_call_id: str | None = Field(None, description="Tool call ID")


class PromptSet(BaseModel):
    name: str = Field(..., description="Unique prompt set identifier")
    messages: list[PromptMessage] = Field(..., description="List of messages in this prompt set")


class PromptTokensDetails(BaseModel):
    audio_tokens: int | None = Field(None)
    cached_tokens: int | None = Field(None)


# Enum: Provider
Provider = Literal['openai', 'ollama', 'lemonade', 'universal']

class QueryResponse(BaseModel):
    """RAG query response model."""
    query: str = Field(...)
    results: list[QueryResult] = Field(...)
    total_results: int = Field(...)
    processing_time_ms: float | None = Field(None)
    retrieval_strategy_used: str = Field(...)
    database_used: str = Field(...)


class QueryResult(BaseModel):
    """Single search result."""
    content: str = Field(...)
    score: float = Field(...)
    metadata: dict[str, Any] = Field(...)
    chunk_id: str | None = Field(None)
    document_id: str | None = Field(None)


class RAGDocumentResponse(BaseModel):
    """Response model for a single RAG document."""
    id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Document filename")
    chunk_count: int = Field(..., description="Number of chunks for this document")
    size_bytes: int = Field(0, description="Document size in bytes")
    parser_used: str = Field('unknown', description="Parser used to process")
    date_ingested: str = Field('', description="Date document was ingested")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")


class RAGHealthResponse(BaseModel):
    """RAG system health response matching Go CLI RAGHealth struct."""
    status: str = Field(...)
    database: str = Field(...)
    components: dict[str, ComponentHealth] = Field(...)
    last_check: str = Field(...)
    issues: list[str] | None = Field(None)


class RAGQueryRequest(BaseModel):
    """RAG query request model. Crafted so AI agents and humans can construct precise retrieval requests. See examples for common usage patterns across semantic, keyword, and hybrid search."""
    query: str = Field(..., description="Natural-language question or statement used to retrieve relevant chunks from the RAG database.")
    database: str | None = Field(None, description="Target database name (from `rag.databases[].name` in `llamafarm.yaml`). If omitted, the server uses the first configured database.")
    data_processing_strategy: str | None = Field(None, description="Optional name from `rag.data_processing_strategies[].name` indicating how the data was parsed/chunked.")
    retrieval_strategy: str | None = Field(None, description="Retrieval algorithm to use. Common values: `semantic` (vector), `bm25` (keyword), `hybrid` (blend), `mmr` (diversified). If omitted, server default applies.")
    top_k: int = Field(5, description="Maximum number of results to return. Higher values increase recall at the cost of speed.")
    score_threshold: float | None = Field(None, description="Minimum similarity score (0–1) required to include a result. Higher -> stricter matches.")
    metadata_filters: dict[str, Any] | None = Field(None, description="Optional exact/range filters on metadata. Examples: {`document_type`: `letter`} or {`date`: {`gte`: `2024-01-01`, `lte`: `2024-12-31`}}.")
    distance_metric: str | None = Field(None, description="Vector distance metric for semantic search. Typical values: `cosine`, `euclidean`, `dot`.")
    hybrid_alpha: float | None = Field(None, description="Blend factor for `hybrid` retrieval. 0 = keyword-only, 1 = semantic-only. Suggested: 0.5.")
    rerank_model: str | None = Field(None, description="Optional reranker (cross-encoder) to reorder top results, e.g., `bge-reranker-v2-m3`.")
    query_expansion: bool = Field(False, description="Enable query expansion (e.g., LLM rewrites/PRF) to improve recall on sparse queries.")
    max_tokens: int | None = Field(None, description="Optional token cap for downstream model responses (if used). Not all backends apply this.")


class RAGStatsResponse(BaseModel):
    """RAG database statistics response matching Go CLI RAGStats struct."""
    database: str = Field(...)
    vector_count: int = Field(...)
    document_count: int = Field(...)
    chunk_count: int = Field(...)
    collection_size_bytes: int = Field(...)
    index_size_bytes: int = Field(...)
    embedding_dimension: int = Field(...)
    distance_metric: str = Field(...)
    last_updated: str = Field(...)
    metadata: dict[str, Any] | None = Field(None)


class RAGStrategyConfigurationSchema_Input(BaseModel):
    components: ComponentsDefinition_Input | None = Field(None, description="Reusable component definitions")
    default_database: str | None = Field(None, description="Name of the default database to use for RAG queries. If not specified, uses the first database in the databases array.")
    databases: list[Database_Input] | None = Field(None, description="Independent database definitions with their own embedding/retrieval strategies")
    data_processing_strategies: list[DataProcessingStrategyDefinition] | None = Field(None, description="Processing pipelines for parsers and extractors. If empty or not specified, the built-in 'universal_rag' strategy is used automatically.")


class RAGStrategyConfigurationSchema_Output(BaseModel):
    components: ComponentsDefinition_Output | None = Field(None, description="Reusable component definitions")
    default_database: str | None = Field(None, description="Name of the default database to use for RAG queries. If not specified, uses the first database in the databases array.")
    databases: list[Database_Output] | None = Field(None, description="Independent database definitions with their own embedding/retrieval strategies")
    data_processing_strategies: list[DataProcessingStrategyDefinition] | None = Field(None, description="Processing pipelines for parsers and extractors. If empty or not specified, the built-in 'universal_rag' strategy is used automatically.")


class RerankRequest(BaseModel):
    """Request to rerank documents by relevance."""
    query: str = Field(...)
    documents: list[str] = Field(...)
    model: str = Field(...)
    top_n: int | None = Field(None)
    return_documents: bool = Field(True)


class RetrievalStrategyInfo(BaseModel):
    name: str = Field(...)
    type_: str = Field(..., alias="type")
    is_default: bool = Field(...)


class Runtime_Input(BaseModel):
    default_model: str | None = Field(None, description="Name of the default model to use (references a model name in the models list). If not specified, uses first model.")
    models: list[Model_Input] | None = Field(None, description="List of model configurations for multi-model support")


class Runtime_Output(BaseModel):
    default_model: str | None = Field(None, description="Name of the default model to use (references a model name in the models list). If not specified, uses first model.")
    models: list[Model_Output] | None = Field(None, description="List of model configurations for multi-model support")


class SHAPExplainRequest(BaseModel):
    """Request for SHAP explanation."""
    model_type: str = Field(..., description="Type of model: anomaly, classifier, catboost")
    model_id: str = Field(..., description="Model identifier")
    data: list[list[float]] = Field(..., description="Data points to explain")
    feature_names: list[str] | None = Field(None, description="Names for features (improves readability)")
    top_k: int = Field(5, description="Number of top contributing features to return")
    generate_narrative: bool = Field(False, description="Generate human-readable explanation narrative")


class Server(BaseModel):
    name: str = Field(..., description="MCP server identifier")
    transport: Transport = Field(..., description="Connection transport to the MCP server")
    command: str | None = Field(None, description="Command/binary to launch the MCP server (stdio)")
    args: list[str] | None = Field(None, description="Optional args for the stdio command")
    env: dict[str, str] | None = Field(None, description="Environment variables for the stdio command")
    base_url: str | None = Field(None, description="Base URL of the MCP server (http)")
    headers: dict[str, str] | None = Field(None, description="HTTP headers for the MCP server")


class SpeechRequest(BaseModel):
    """OpenAI-compatible speech synthesis request."""
    model: str = Field('kokoro', description="TTS model to use. Currently supports 'kokoro'.")
    input: str = Field(..., description="The text to synthesize into speech. Maximum 4096 characters.")
    voice: str = Field('af_heart', description="Voice ID to use for synthesis.")
    response_format: Literal['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm'] = Field('mp3', description="Audio output format.")
    speed: float = Field(1.0, description="Speed of generated audio. 0.25 to 4.0.")


class Stt(BaseModel):
    model: Model1 | None = Field('base', description="Whisper model size for transcription: - tiny (39M, fastest, lower accuracy) - base (74M, fast, good accuracy, default) - small (244M, medium speed, better accuracy) - medium (769M, slower, high accuracy) - large-v3 (1.5B, slowest, highest accuracy) - distil-large-v3-turbo (~800M, fast & accurate, recommended)")
    language: str | None = Field('en', description="Language code for transcription (e.g., \"en\", \"es\", \"fr\")")
    keep_loaded: bool | None = Field(False, description="Keep STT model loaded in memory to prevent cache eviction. When true, the runtime will not purge this model from cache even under memory pressure. Reduces latency for voice transcription.")


class SubEvent(BaseModel):
    """Individual sub-event within an event."""
    timestamp: str = Field(..., description="Sub-event timestamp")
    event_name: str = Field(..., description="Sub-event name")
    duration_ms: float = Field(..., description="Duration from event start in milliseconds")
    data: dict[str, Any] = Field(..., description="Sub-event data")


# Enum: Task
Task = Literal['embedding', 'classification', 'reranking', 'ner']

class TimeseriesBackendInfo(BaseModel):
    """Information about a timeseries backend."""
    name: str = Field(..., description="Backend identifier")
    description: str = Field(..., description="Human-readable description")
    requires_training: bool = Field(..., description="Whether this backend requires training data")
    supports_confidence_intervals: bool = Field(..., description="Whether this backend can produce confidence intervals")
    speed: Literal['fast', 'medium', 'slow'] = Field(..., description="Relative execution speed")


class TimeseriesBackendsResponse(BaseModel):
    """Response listing available backends."""
    backends: list[TimeseriesBackendInfo] = Field(..., description="List of available backends")


class TimeseriesDataPoint(BaseModel):
    """A single time-series data point."""
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    value: float = Field(..., description="Numeric value at this timestamp")


class TimeseriesDeleteResponse(BaseModel):
    """Response from deleting a model."""
    deleted: bool = Field(..., description="Whether the model was deleted")
    model: str = Field(..., description="Model name that was requested")
    message: str = Field(..., description="Status message")


class TimeseriesFitRequest(BaseModel):
    """Request to train a time-series forecaster. Training is only required for classical backends (arima, exponential_smoothing, theta). Zero-shot backends (chronos, chronos-bolt) don't need training. The model is auto-saved after training. If model name is not provided, a unique name is auto-generated (e.g., \"timeseries-a1b2c3d4\")."""
    model: str | None = Field(None, description="Model name (auto-generated if not provided)")
    backend: Literal['arima', 'exponential_smoothing', 'theta', 'chronos', 'chronos-bolt'] = Field('arima', description="Forecasting algorithm to use")
    data: list[TimeseriesDataPoint] | list[dict[str, Any]] = Field(..., description="Training data as list of {timestamp, value} objects")
    frequency: str | None = Field(None, description="Time frequency (D, H, M, etc.). Auto-detected if not provided.")
    overwrite: bool = Field(True, description="If True, overwrite existing model. If False, version with timestamp.")
    description: str | None = Field(None, description="Optional model description (saved to metadata)")


class TimeseriesFitResponse(BaseModel):
    """Response from training a forecaster."""
    model: str = Field(..., description="Model name (generated if not provided)")
    backend: str = Field(..., description="Backend used for training")
    saved_path: str = Field(..., description="Path where model was saved")
    training_time_ms: float = Field(..., description="Training time in milliseconds")
    samples_fitted: int = Field(..., description="Number of data points used")
    description: str | None = Field(None, description="Model description if provided")


class TimeseriesLoadRequest(BaseModel):
    """Request to load a saved model. Supports the '-latest' suffix to load the most recent version of a model (e.g., 'my-model-latest')."""
    model: str = Field(..., description="Model name (supports '-latest' suffix)")
    backend: str | None = Field(None, description="Backend hint for file matching")


class TimeseriesModelInfo(BaseModel):
    """Information about a saved model."""
    name: str = Field(..., description="Model name")
    filename: str = Field(..., description="Filename on disk")
    backend: str = Field(..., description="Backend used")
    path: str = Field(..., description="Full path to model file")
    size_bytes: int = Field(..., description="File size in bytes")
    created: str = Field(..., description="Creation timestamp (ISO 8601)")
    description: str | None = Field(None, description="Model description")


class TimeseriesModelsResponse(BaseModel):
    """Response listing saved models."""
    models: list[TimeseriesModelInfo] = Field(..., description="List of saved models")
    total: int = Field(..., description="Total number of models")


class TimeseriesPredictRequest(BaseModel):
    """Request to generate forecasts. For classical backends, the model must be fitted first. For zero-shot backends (chronos, chronos-bolt), provide historical data. Supports the '-latest' suffix in model name to use the most recent version."""
    model: str = Field(..., description="Model name (supports '-latest' suffix to use most recent version)")
    horizon: int = Field(..., description="Number of periods to forecast")
    confidence_level: float = Field(0.95, description="Confidence level for prediction intervals")
    data: list[TimeseriesDataPoint] | list[dict[str, Any]] | None = Field(None, description="Historical data (required for zero-shot backends)")


class TimeseriesPredictResponse(BaseModel):
    """Response from generating forecasts."""
    model_id: str = Field(..., description="Model identifier")
    backend: str = Field(..., description="Backend used for prediction")
    predictions: list[TimeseriesPrediction] = Field(..., description="List of forecast predictions")
    fit_time_ms: float | None = Field(None, description="Training time if model was just fitted")
    predict_time_ms: float = Field(..., description="Prediction time in milliseconds")


class TimeseriesPrediction(BaseModel):
    """A single forecast prediction with optional confidence intervals."""
    timestamp: str = Field(..., description="Predicted timestamp")
    value: float = Field(..., description="Point forecast value")
    lower: float | None = Field(None, description="Lower confidence bound")
    upper: float | None = Field(None, description="Upper confidence bound")


class Tool(BaseModel):
    type_: Type6 = Field(..., description="Type of tool", alias="type")
    name: str = Field(..., description="Name of the tool")
    description: str = Field(..., description="Description of the tool")
    parameters: dict[str, Any] = Field(..., description="Parameters of the tool")


# Enum: ToolCallStrategy
ToolCallStrategy = Literal['native_api', 'prompt_based']

class TopLogprob(BaseModel):
    token: str = Field(...)
    bytes: list[int] | None = Field(None)
    logprob: float = Field(...)


# Enum: Transport
Transport = Literal['stdio', 'http', 'sse']

class Tts(BaseModel):
    model: str | None = Field('kokoro', description="TTS model/backend ID: - kokoro: High-quality neural TTS with built-in voices (default) - chatterbox-turbo: Fast voice cloning TTS (350M params, sub-200ms) Requires voice_profiles configuration for voice cloning. - pocket-tts: Lightweight CPU TTS from Kyutai (100M params, ~6x realtime) Fast and efficient, no GPU required.")
    voice: str | None = Field('af_heart', description="Voice identifier for synthesis. For Kokoro, use built-in voice IDs: - af_heart (American Female, default) - af_bella, af_nicole, af_sarah, af_sky (American Female) - am_adam, am_michael (American Male) - bf_emma, bf_isabella (British Female) - bm_george, bm_lewis (British Male) For Pocket TTS, use built-in voice IDs: - alba (default), marius, javert, jean - fantine, cosette, eponine, azelma For Chatterbox Turbo, use a voice profile name defined in voice_profiles.")
    speed: float | None = Field(0.95, description="Speech speed multiplier (0.5-2.0). For more natural-sounding speech, try 0.9-0.95 (slightly slower than default). - 0.8-0.9: Slower, deliberate (good for complex topics) - 0.9-0.95: Natural conversational pace (recommended) - 1.0: Default speed - 1.1-1.2: Faster, energetic")
    keep_loaded: bool | None = Field(False, description="Keep TTS model loaded in memory to prevent cache eviction. When true, the runtime will not purge this model from cache even under memory pressure. Reduces latency for voice responses.")
    voice_profiles: dict[str, VoiceProfiles] | None = Field(None, description="Named voice profiles for voice cloning (Chatterbox Turbo only). Each profile maps a name to a reference audio file (~10s of speech).")
    temperature: float | None = Field(0.8, description="Chatterbox Turbo temperature (ignored for Kokoro). Controls randomness in generation. Higher = more varied output. - 0.5: More deterministic - 0.8: Balanced (default) - 1.2: More creative/varied")
    top_k: int | None = Field(1000, description="Chatterbox Turbo top-k sampling (ignored for Kokoro). Number of highest probability tokens to consider.")
    top_p: float | None = Field(0.95, description="Chatterbox Turbo nucleus sampling threshold (ignored for Kokoro). Cumulative probability cutoff for token selection.")
    repetition_penalty: float | None = Field(1.2, description="Chatterbox Turbo repetition penalty (ignored for Kokoro). Penalty applied to repeating tokens. Higher = less repetition.")


class TurnDetection(BaseModel):
    enabled: bool | None = Field(True, description="Enable smart end-of-turn detection using linguistic analysis. When enabled, the system analyzes partial transcriptions to detect thinking pauses vs actual end of utterance, preventing premature LLM responses. When disabled, uses fixed silence threshold only.")
    base_silence_duration: float | None = Field(0.4, description="Base silence duration for complete utterances (seconds). Used when the transcription appears linguistically complete (ends with punctuation or a complete phrase like \"yes\", \"thanks\").")
    thinking_silence_duration: float | None = Field(1.2, description="Extended silence duration for incomplete utterances (seconds). Used when linguistic analysis suggests the user is mid-thought (trailing conjunctions like \"and\", \"but\", prepositions like \"to\", \"with\", or filler words like \"um\", \"uh\").")
    max_silence_duration: float | None = Field(2.5, description="Maximum silence before forcing end-of-turn (seconds). Even if the utterance seems incomplete, processing starts after this timeout to ensure responsiveness.")


# Enum: Type3
Type3 = Literal['HuggingFaceEmbedder', 'OllamaEmbedder', 'OpenAIEmbedder', 'SentenceTransformerEmbedder', 'UniversalEmbedder']

# Enum: Type4
Type4 = Literal['VectorRetriever', 'HybridRetriever', 'BM25Retriever', 'RerankedRetriever', 'GraphRetriever', 'ElasticRetriever', 'BasicSimilarityStrategy', 'MetadataFilteredStrategy', 'MultiQueryStrategy', 'RerankedStrategy', 'HybridUniversalStrategy', 'CrossEncoderRerankedStrategy', 'MultiTurnRAGStrategy']

# Enum: Type6
Type6 = Literal['function']

class UpdateDatabaseRequest(BaseModel):
    """Request model for updating a database (partial update)."""
    config: dict[str, Any] | None = Field(None, description="Database-specific configuration")
    embedding_strategies: list[dict[str, Any]] | None = Field(None, description="Embedding strategies for this database")
    retrieval_strategies: list[dict[str, Any]] | None = Field(None, description="Retrieval strategies for this database")
    default_embedding_strategy: str | None = Field(None, description="Name of default embedding strategy")
    default_retrieval_strategy: str | None = Field(None, description="Name of default retrieval strategy")


class UpdateProjectRequest(BaseModel):
    config: LlamaFarmConfig_Input = Field(..., description="The full updated configuration of the project")


class UpdateProjectResponse(BaseModel):
    project: Project = Field(..., description="The updated project")


class ValidateDownloadRequest(BaseModel):
    model_name: str = Field(...)


# Enum: Version
Version = Literal['v1']

class Voice_Input(BaseModel):
    enabled: bool | None = Field(True, description="Enable or disable the voice chat endpoint")
    llm_model: str | None = Field(None, description="Reference to a model name in runtime.models[] to use for voice chat. The model's prompts array will be applied to voice conversations.")
    tts: Tts | None = Field(None, description="Text-to-speech configuration")
    stt: Stt | None = Field(None, description="Speech-to-text configuration")
    enable_thinking: bool | None = Field(False, description="Enable thinking/reasoning mode for the LLM. When false (default), the LLM is instructed to skip chain-of-thought reasoning and respond directly. This is recommended for voice chat since thinking output would be spoken aloud by TTS.")
    turn_detection: TurnDetection | None = Field(None, description="End-of-turn detection configuration. Controls how the system detects when a user has finished speaking vs when they're just pausing to think. This prevents the LLM from responding prematurely during thinking pauses.")
    emotion: Emotion | None = Field(None, description="Speech emotion recognition configuration. Analyzes user audio to detect emotional tone (angry, happy, sad, etc.) and provides context to the LLM.")


class Voice_Output(BaseModel):
    enabled: bool | None = Field(True, description="Enable or disable the voice chat endpoint")
    llm_model: str | None = Field(None, description="Reference to a model name in runtime.models[] to use for voice chat. The model's prompts array will be applied to voice conversations.")
    tts: Tts | None = Field(None, description="Text-to-speech configuration")
    stt: Stt | None = Field(None, description="Speech-to-text configuration")
    enable_thinking: bool | None = Field(False, description="Enable thinking/reasoning mode for the LLM. When false (default), the LLM is instructed to skip chain-of-thought reasoning and respond directly. This is recommended for voice chat since thinking output would be spoken aloud by TTS.")
    turn_detection: TurnDetection | None = Field(None, description="End-of-turn detection configuration. Controls how the system detects when a user has finished speaking vs when they're just pausing to think. This prevents the LLM from responding prematurely during thinking pauses.")
    emotion: Emotion | None = Field(None, description="Speech emotion recognition configuration. Analyzes user audio to detect emotional tone (angry, happy, sad, etc.) and provides context to the LLM.")


class VoiceInfo(BaseModel):
    """Information about an available TTS voice."""
    id: str = Field(..., description="Unique voice identifier.")
    name: str = Field(..., description="Human-readable voice name.")
    language: str = Field(..., description="Language code (e.g., 'en-US', 'en-GB').")
    model: str = Field(..., description="Model this voice belongs to.")
    preview_url: str | None = Field(None, description="URL to a preview audio sample (if available).")


class VoiceListResponse(BaseModel):
    """Response from the voices list endpoint."""
    object: str = Field('list')
    data: list[VoiceInfo] = Field(...)


class VoiceProfiles(BaseModel):
    audio_path: str = Field(..., description="Path to reference audio file for voice cloning. Can be relative to project directory or absolute.")
    description: str | None = Field(None, description="Human-readable description of this voice")


# Total generated types: 201
