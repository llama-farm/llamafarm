# API Reference

LlamaFarm provides a comprehensive REST API for managing projects, datasets, chat interactions, and RAG (Retrieval-Augmented Generation) operations. The API follows RESTful conventions and is compatible with OpenAI's chat completion format.

## Base URL

The API is served at: `http://localhost:8000`

All versioned endpoints use the `/v1` prefix:

```text
http://localhost:8000/v1
```

## Finding Your Namespace and Project Name

### Understanding Namespaces and Projects

LlamaFarm organizes your work into **namespaces** (organizational containers) and **projects** (individual configurations):

- **Namespace**: A top-level organizational unit (e.g., your username, team name, or organization)
- **Project Name**: The unique identifier for a specific LlamaFarm project within a namespace.

### From Your llamafarm.yaml

The easiest way to find your namespace and project name is to check your `llamafarm.yaml` configuration file:

```yaml
version: v1
name: my-project # This is your project name
namespace: my-org # This is your namespace
```

### From the File System

Projects are stored in:

```
~/.llamafarm/projects/{namespace}/{project_name}/
```

For example, if you see:

```
~/.llamafarm/projects/acme-corp/chatbot/
```

Then:

- Namespace: `acme-corp`
- Project name: `chatbot`

### Using the API

You can also list projects programmatically:

```bash
# List all projects in a namespace
curl http://localhost:8000/v1/projects/my-org
```

### Custom Data Directory

If you've set a custom data directory using the `LF_DATA_DIR` environment variable, check:

```
$LF_DATA_DIR/projects/{namespace}/{project_name}/
```

## Authentication

Currently, the API does not require authentication. This is designed for local development environments. For production deployments, implement authentication at the reverse proxy or load balancer level.

## Response Format

### Success Responses

Successful requests return JSON with appropriate HTTP status codes (200, 201, etc.):

```json
{
  "field": "value",
  ...
}
```

### Error Responses

Error responses follow a consistent format with appropriate HTTP status codes (400, 404, 500, etc.):

```json
{
  "detail": "Error message description"
}
```

Common HTTP status codes:

- `200 OK` - Request succeeded
- `201 Created` - Resource created successfully
- `400 Bad Request` - Invalid request parameters
- `404 Not Found` - Resource not found
- `422 Unprocessable Entity` - Validation error
- `500 Internal Server Error` - Server error

## API Endpoints Overview

### Projects

- `GET /v1/projects/{namespace}` - List projects
- `POST /v1/projects/{namespace}` - Create project
- `GET /v1/projects/{namespace}/{project}` - Get project details
- `PUT /v1/projects/{namespace}/{project}` - Update project configuration
- `DELETE /v1/projects/{namespace}/{project}` - Delete project

### Chat

- `POST /v1/projects/{namespace}/{project}/chat/completions` - Send chat message (OpenAI-compatible)
- `GET /v1/projects/{namespace}/{project}/chat/sessions/{session_id}/history` - Get chat history
- `DELETE /v1/projects/{namespace}/{project}/chat/sessions/{session_id}` - Delete chat session
- `DELETE /v1/projects/{namespace}/{project}/chat/sessions` - Delete all sessions
- `GET /v1/projects/{namespace}/{project}/models` - List available models

### Datasets

- `GET /v1/projects/{namespace}/{project}/datasets` - List datasets
- `POST /v1/projects/{namespace}/{project}/datasets` - Create dataset
- `DELETE /v1/projects/{namespace}/{project}/datasets/{dataset}` - Delete dataset
- `POST /v1/projects/{namespace}/{project}/datasets/{dataset}/data` - Upload file to dataset
- `POST /v1/projects/{namespace}/{project}/datasets/{dataset}/actions` - Trigger dataset actions (ingest/process) via Celery tasks
- `DELETE /v1/projects/{namespace}/{project}/datasets/{dataset}/data/{file_hash}` - Remove file from dataset

### RAG (Retrieval-Augmented Generation)

- `POST /v1/projects/{namespace}/{project}/rag/query` - Query RAG system
- `GET /v1/projects/{namespace}/{project}/rag/health` - Check RAG health
- `GET /v1/projects/{namespace}/{project}/rag/stats` - Get RAG statistics
- `GET /v1/projects/{namespace}/{project}/rag/databases` - List databases
- `GET /v1/projects/{namespace}/{project}/rag/databases/{database}` - Get database details
- `GET /v1/projects/{namespace}/{project}/rag/databases/{database}/documents` - List documents in database
- `POST /v1/projects/{namespace}/{project}/rag/databases` - Create database
- `PATCH /v1/projects/{namespace}/{project}/rag/databases/{database}` - Update database
- `DELETE /v1/projects/{namespace}/{project}/rag/databases/{database}` - Delete database

### Tasks

- `GET /v1/projects/{namespace}/{project}/tasks/{task_id}` - Get async task status
- `DELETE /v1/projects/{namespace}/{project}/tasks/{task_id}` - Cancel running task

### Event Logs

- `GET /v1/projects/{namespace}/{project}/event_logs` - List event logs
- `GET /v1/projects/{namespace}/{project}/event_logs/{event_id}` - Get event details

### Examples

- `GET /v1/examples` - List available examples
- `GET /v1/examples/{example_id}/datasets` - List example datasets
- `POST /v1/examples/{example_id}/import-project` - Import example as new project
- `POST /v1/examples/{example_id}/import-data` - Import example data into existing project
- `POST /v1/examples/{example_id}/import-dataset` - Import specific dataset from example

### Models Cache

- `GET /v1/models` - List cached models
- `POST /v1/models/download` - Download/cache a model
- `POST /v1/models/validate-download` - Check disk space before download
- `GET /v1/models/{model_id}/quantizations` - List GGUF quantization options
- `DELETE /v1/models/{model_name}` - Delete cached model

### Vision (OCR & Document Extraction)

- `POST /v1/vision/ocr` - OCR text extraction (accepts file upload or base64)
- `POST /v1/vision/documents/extract` - Document extraction/VQA (accepts file upload or base64)

### ML (Custom Classifiers & Anomaly Detection)

- `POST /v1/ml/classifier/fit` - Train custom text classifier (SetFit few-shot)
- `POST /v1/ml/classifier/predict` - Classify texts using trained model
- `POST /v1/ml/classifier/save` - Save trained classifier to disk
- `POST /v1/ml/classifier/load` - Load classifier from disk
- `GET /v1/ml/classifier/models` - List saved classifiers
- `DELETE /v1/ml/classifier/models/{name}` - Delete saved classifier
- `POST /v1/ml/anomaly/fit` - Train anomaly detector
- `POST /v1/ml/anomaly/score` - Score data points for anomalies
- `POST /v1/ml/anomaly/detect` - Detect anomalies (returns only anomalous points)
- `POST /v1/ml/anomaly/save` - Save trained anomaly model
- `POST /v1/ml/anomaly/load` - Load anomaly model from disk
- `GET /v1/ml/anomaly/models` - List saved anomaly models
- `DELETE /v1/ml/anomaly/models/{filename}` - Delete saved anomaly model

### Health

- `GET /health` - Overall health check
- `GET /health/liveness` - Liveness probe

### System

- `GET /` - Basic hello endpoint
- `GET /info` - System information
- `GET /v1/system/version-check` - Check for CLI updates
- `GET /v1/system/disk` - Get disk space information

---

## Projects API

### List Projects

List all projects in a namespace.

**Endpoint:** `GET /v1/projects/{namespace}`

**Parameters:**

- `namespace` (path, required): The namespace to list projects from

**Response:**

```json
{
  "total": 2,
  "projects": [
    {
      "namespace": "my-org",
      "name": "chatbot",
      "config": {
        "version": "v1",
        "name": "chatbot",
        "namespace": "my-org",
        "runtime": { ... },
        ...
      }
    }
  ]
}
```

**Example:**

```bash
curl http://localhost:8000/v1/projects/my-org
```

### Create Project

Create a new project in a namespace.

**Endpoint:** `POST /v1/projects/{namespace}`

**Parameters:**

- `namespace` (path, required): The namespace to create the project in

**Request Body:**

```json
{
  "name": "my-new-project",
  "config_template": "server" // Optional: server, rag, or custom template name
}
```

**Response:**

```json
{
  "project": {
    "namespace": "my-org",
    "name": "my-new-project",
    "config": { ... }
  }
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/v1/projects/my-org \
  -H "Content-Type: application/json" \
  -d '{"name": "chatbot", "config_template": "server"}'
```

### Get Project

Get details of a specific project.

**Endpoint:** `GET /v1/projects/{namespace}/{project}`

**Parameters:**

- `namespace` (path, required): Project namespace
- `project` (path, required): Project name

**Response:**

```json
{
  "project": {
    "namespace": "my-org",
    "name": "chatbot",
    "config": {
      "version": "v1",
      "name": "chatbot",
      "namespace": "my-org",
      "runtime": {
        "provider": "ollama",
        "model": "llama3.2:3b"
      },
      ...
    }
  }
}
```

**Example:**

```bash
curl http://localhost:8000/v1/projects/my-org/chatbot
```

### Update Project

Update a project's configuration.

**Endpoint:** `PUT /v1/projects/{namespace}/{project}`

**Parameters:**

- `namespace` (path, required): Project namespace
- `project` (path, required): Project name

**Request Body:**

```json
{
  "config": {
    "version": "v1",
    "name": "chatbot",
    "namespace": "my-org",
    "runtime": {
      "provider": "ollama",
      "model": "llama3.2:3b"
    },
    ...
  }
}
```

**Response:**

```json
{
  "project": {
    "namespace": "my-org",
    "name": "chatbot",
    "config": { ... }
  }
}
```

**Example:**

```bash
curl -X PUT http://localhost:8000/v1/projects/my-org/chatbot \
  -H "Content-Type: application/json" \
  -d @updated-config.json
```

### Delete Project

Delete a project (currently returns project info; actual deletion not implemented).

**Endpoint:** `DELETE /v1/projects/{namespace}/{project}`

**Parameters:**

- `namespace` (path, required): Project namespace
- `project` (path, required): Project name

**Response:**

```json
{
  "project": {
    "namespace": "my-org",
    "name": "chatbot",
    "config": { ... }
  }
}
```

---

## Chat API

### Send Chat Message (OpenAI-Compatible)

Send a chat message to the LLM. This endpoint is compatible with OpenAI's chat completion API.

**Endpoint:** `POST /v1/projects/{namespace}/{project}/chat/completions`

**Headers:**

- `X-Session-ID` (optional): Session ID for stateful conversations. If not provided, a new session is created.
- `X-No-Session` (optional): Set to any value for stateless mode (no conversation history)

**Request Body:**

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is LlamaFarm?"
    }
  ],
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 1000,
  "rag_enabled": true,
  "database": "main_db",
  "rag_top_k": 5,
  "rag_score_threshold": 0.7
}
```

**Request Fields:**

- `messages` (required): Array of chat messages with `role` and `content`
- `model` (optional): Select which model to use (OpenAI-compatible, added in PR #263 multi-model support)
- `stream` (optional): Enable streaming responses (Server-Sent Events)
- `temperature` (optional): Sampling temperature (0.0-2.0)
- `max_tokens` (optional): Maximum tokens to generate for the **answer** (thinking tokens are separate)
- `top_p` (optional): Nucleus sampling parameter
- `top_k` (optional): Top-k sampling parameter
- `rag_enabled` (optional): Enable/disable RAG (uses config default if not specified)
- `database` (optional): Database to use for RAG queries
- `rag_top_k` (optional): Number of RAG results to retrieve
- `rag_score_threshold` (optional): Minimum similarity score for RAG results
- `rag_queries` (optional): Array of custom queries for RAG retrieval, overriding the user message. Can be a single query `["my query"]` or multiple queries `["query1", "query2"]` - results from multiple queries are executed concurrently, merged, and deduplicated
- `think` (optional): Enable thinking/reasoning mode for supported models like Qwen3 (default: `false`)
- `thinking_budget` (optional): Maximum tokens for thinking process when `think: true` (default: `1024`)

**Response (Non-Streaming):**

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "llama3.2:3b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "LlamaFarm is a framework for building AI applications..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 100,
    "total_tokens": 120
  }
}
```

**Response Headers:**

- `X-Session-ID`: The session ID (only in stateful mode)

**Streaming Response:**

When `stream: true`, the response is sent as Server-Sent Events:

```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"llama3.2:3b","choices":[{"index":0,"delta":{"role":"assistant","content":"Llama"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"llama3.2:3b","choices":[{"index":0,"delta":{"content":"Farm"},"finish_reason":null}]}

data: [DONE]
```

**Example (Non-Streaming):**

```bash
curl -X POST http://localhost:8000/v1/projects/my-org/chatbot/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

**Example (Streaming):**

```bash
curl -X POST http://localhost:8000/v1/projects/my-org/chatbot/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": true
  }'
```

**Example (Stateless):**

```bash
curl -X POST http://localhost:8000/v1/projects/my-org/chatbot/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-No-Session: true" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

**Example (With RAG):**

```bash
curl -X POST http://localhost:8000/v1/projects/my-org/chatbot/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What are the FDA regulations?"}
    ],
    "rag_enabled": true,
    "database": "fda_db",
    "rag_top_k": 10
  }'
```

**Example (With Thinking/Reasoning):**

For models that support chain-of-thought reasoning (like Qwen3), enable thinking mode to see the model's reasoning process:

```bash
curl -X POST http://localhost:8000/v1/projects/my-org/chatbot/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is 15% of 85?"}
    ],
    "think": true,
    "thinking_budget": 512,
    "max_tokens": 200
  }'
```

**Response with Thinking:**

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "Qwen3-1.7B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "15% of 85 is **12.75**."
      },
      "finish_reason": "stop"
    }
  ],
  "thinking": {
    "content": "To find 15% of 85, I need to multiply 85 by 0.15. Let me calculate: 85 × 0.15 = 12.75.",
    "tokens": null
  }
}
```

**Token Allocation with Thinking:**

- `max_tokens`: Controls the **answer** length (default: 512)
- `thinking_budget`: Controls the **thinking** length (default: 1024 when enabled)
- Total generation = `thinking_budget` + `max_tokens`

This ensures your answer isn't cut short by the thinking process.

**Example (Custom RAG Query):**

Override the default RAG query (which uses the user message) with a custom search query. This is useful when the user's question is conversational but you want specific technical retrieval:

```bash
curl -X POST http://localhost:8000/v1/projects/my-org/chatbot/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Can you summarize the key findings?"}
    ],
    "rag_enabled": true,
    "database": "research_db",
    "rag_queries": ["clinical trial results primary endpoints efficacy safety"]
  }'
```

**Example (Multiple Custom RAG Queries):**

Execute multiple search queries concurrently and merge the results. This is useful for comparative analysis or comprehensive retrieval:

```bash
curl -X POST http://localhost:8000/v1/projects/my-org/chatbot/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Compare the two approaches"}
    ],
    "rag_enabled": true,
    "database": "research_db",
    "rag_queries": [
      "machine learning neural network methodology",
      "traditional statistical analysis regression"
    ],
    "rag_top_k": 10
  }'
```

Results from multiple queries are automatically executed concurrently, merged, deduplicated by content, sorted by relevance score, and limited to `rag_top_k` total results.

### Get Chat History

Retrieve conversation history for a session.

**Endpoint:** `GET /v1/projects/{namespace}/{project}/chat/sessions/{session_id}/history`

**Parameters:**

- `namespace` (path, required): Project namespace
- `project` (path, required): Project name
- `session_id` (path, required): Session ID

**Response:**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hello!"
    },
    {
      "role": "assistant",
      "content": "Hi! How can I help you today?"
    }
  ]
}
```

**Example:**

```bash
curl http://localhost:8000/v1/projects/my-org/chatbot/chat/sessions/abc-123/history
```

### Delete Chat Session

Delete a specific chat session and its history.

**Endpoint:** `DELETE /v1/projects/{namespace}/{project}/chat/sessions/{session_id}`

**Parameters:**

- `namespace` (path, required): Project namespace
- `project` (path, required): Project name
- `session_id` (path, required): Session ID

**Response:**

```json
{
  "message": "Session abc-123 deleted"
}
```

**Example:**

```bash
curl -X DELETE http://localhost:8000/v1/projects/my-org/chatbot/chat/sessions/abc-123
```

### Delete All Chat Sessions

Delete all chat sessions for a project.

**Endpoint:** `DELETE /v1/projects/{namespace}/{project}/chat/sessions`

**Parameters:**

- `namespace` (path, required): Project namespace
- `project` (path, required): Project name

**Response:**

```json
{
  "message": "Deleted 5 session(s)",
  "count": 5
}
```

**Example:**

```bash
curl -X DELETE http://localhost:8000/v1/projects/my-org/chatbot/chat/sessions
```

### List Available Models

List all configured models for a project.

**Endpoint:** `GET /v1/projects/{namespace}/{project}/models`

**Parameters:**

- `namespace` (path, required): Project namespace
- `project` (path, required): Project name

**Response:**

```json
{
  "total": 2,
  "models": [
    {
      "name": "fast-model",
      "provider": "ollama",
      "model": "llama3.2:3b",
      "base_url": "http://localhost:11434/v1",
      "default": true,
      "description": "Fast model for quick responses"
    },
    {
      "name": "smart-model",
      "provider": "ollama",
      "model": "llama3.2:70b",
      "base_url": "http://localhost:11434/v1",
      "default": false,
      "description": "Larger model for complex tasks"
    }
  ]
}
```

**Example:**

```bash
curl http://localhost:8000/v1/projects/my-org/chatbot/models
```

---

## Datasets API

### List Datasets

List all datasets in a project.

**Endpoint:** `GET /v1/projects/{namespace}/{project}/datasets`

**Parameters:**

- `namespace` (path, required): Project namespace
- `project` (path, required): Project name
- `include_extra_details` (query, optional): Include detailed file information (default: true)

**Response:**

```json
{
  "total": 2,
  "datasets": [
    {
      "name": "research_papers",
      "database": "main_db",
      "data_processing_strategy": "universal_processor",
      "files": ["abc123", "def456"],
      "details": {
        "total_files": 2,
        "total_size_bytes": 1048576,
        "file_details": [
          {
            "hash": "abc123",
            "original_filename": "paper1.pdf",
            "size": 524288,
            "timestamp": 1677652288.0
          }
        ]
      }
    }
  ]
}
```

**Example:**

```bash
curl http://localhost:8000/v1/projects/my-org/chatbot/datasets
```

### Get Available Strategies

Get available data processing strategies and databases for a project.

**Endpoint:** `GET /v1/projects/{namespace}/{project}/datasets/strategies`

**Parameters:**

- `namespace` (path, required): Project namespace
- `project` (path, required): Project name

**Response:**

```json
{
  "data_processing_strategies": ["universal_processor", "custom_strategy"],
  "databases": ["main_db", "research_db"]
}
```

**Example:**

```bash
curl http://localhost:8000/v1/projects/my-org/chatbot/datasets/strategies
```

### Create Dataset

Create a new dataset.

**Endpoint:** `POST /v1/projects/{namespace}/{project}/datasets`

**Parameters:**

- `namespace` (path, required): Project namespace
- `project` (path, required): Project name

**Request Body:**

```json
{
  "name": "research_papers",
  "data_processing_strategy": "universal_processor",
  "database": "main_db"
}
```

**Response:**

```json
{
  "dataset": {
    "name": "research_papers",
    "database": "main_db",
    "data_processing_strategy": "universal_processor",
    "files": []
  }
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/v1/projects/my-org/chatbot/datasets \
  -H "Content-Type: application/json" \
  -d '{
    "name": "research_papers",
    "data_processing_strategy": "universal_processor",
    "database": "main_db"
  }'
```

### Delete Dataset

Delete a dataset.

**Endpoint:** `DELETE /v1/projects/{namespace}/{project}/datasets/{dataset}`

**Parameters:**

- `namespace` (path, required): Project namespace
- `project` (path, required): Project name
- `dataset` (path, required): Dataset name

**Response:**

```json
{
  "dataset": {
    "name": "research_papers",
    "database": "main_db",
    "data_processing_strategy": "universal_processor",
    "files": []
  }
}
```

**Example:**

```bash
curl -X DELETE http://localhost:8000/v1/projects/my-org/chatbot/datasets/research_papers
```

### Upload File to Dataset

Upload a file to a dataset (stores the file but does not process it).

**Endpoint:** `POST /v1/projects/{namespace}/{project}/datasets/{dataset}/data`

**Parameters:**

- `namespace` (path, required): Project namespace
- `project` (path, required): Project name
- `dataset` (path, required): Dataset name

**Request:**

- Content-Type: `multipart/form-data`
- Body: File upload with field name `file`

**Response:**

```json
{
  "filename": "paper1.pdf",
  "hash": "abc123def456",
  "processed": false
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/v1/projects/my-org/chatbot/datasets/research_papers/data \
  -F "file=@paper1.pdf"
```

### Process Dataset

Processing is now driven exclusively through the dataset actions endpoint, which queues Celery tasks and returns a task ID you can poll later.

**Endpoint:** `POST /v1/projects/{namespace}/{project}/datasets/{dataset}/actions`

**Parameters:**

- `namespace` (path, required): Project namespace
- `project` (path, required): Project name
- `dataset` (path, required): Dataset name
- `action_type` (body, required): `"process"` (alias `"ingest"`)

**Request Body:**

```json
{
  "action_type": "process"
}
```

**Response:**

```json
{
  "message": "Accepted",
  "task_uri": "http://localhost:8000/v1/projects/my-org/chatbot/tasks/8f6f9c2a",
  "task_id": "8f6f9c2a"
}
```

Use `task_uri`/`task_id` with `GET /v1/projects/{namespace}/{project}/tasks/{task_id}` to monitor progress. When the Celery task finishes, the `result` payload matches the historical `ProcessDatasetResponse` structure (processed/skipped/failed counts plus per-file details).

**Example:**

```bash
curl -X POST http://localhost:8000/v1/projects/my-org/chatbot/datasets/research_papers/actions \
  -H "Content-Type: application/json" \
  -d '{"action_type":"process"}'
```

### Remove File from Dataset

Remove a file from a dataset.

**Endpoint:** `DELETE /v1/projects/{namespace}/{project}/datasets/{dataset}/data/{file_hash}`

**Parameters:**

- `namespace` (path, required): Project namespace
- `project` (path, required): Project name
- `dataset` (path, required): Dataset name
- `file_hash` (path, required): Hash of the file to remove
- `remove_from_disk` (query, optional): Also delete the file from disk (default: false)

**Response:**

```json
{
  "file_hash": "abc123"
}
```

**Example:**

```bash
curl -X DELETE http://localhost:8000/v1/projects/my-org/chatbot/datasets/research_papers/data/abc123
```

**Example (Remove from disk):**

```bash
curl -X DELETE "http://localhost:8000/v1/projects/my-org/chatbot/datasets/research_papers/data/abc123?remove_from_disk=true"
```

---

## RAG API

### Query RAG System

Perform a semantic search query against a RAG database.

**Endpoint:** `POST /v1/projects/{namespace}/{project}/rag/query`

**Parameters:**

- `namespace` (path, required): Project namespace
- `project` (path, required): Project name

**Request Body:**

```json
{
  "query": "What are the clinical trial requirements?",
  "database": "fda_db",
  "top_k": 5,
  "score_threshold": 0.7,
  "retrieval_strategy": "hybrid",
  "metadata_filters": {
    "document_type": "regulation"
  }
}
```

**Request Fields:**

- `query` (required): The search query text
- `database` (optional): Database name (uses default if not specified)
- `top_k` (optional): Number of results to return (default: 5)
- `score_threshold` (optional): Minimum similarity score
- `retrieval_strategy` (optional): Strategy to use for retrieval
- `metadata_filters` (optional): Filter results by metadata
- `distance_metric` (optional): Distance metric to use
- `hybrid_alpha` (optional): Alpha parameter for hybrid search
- `rerank_model` (optional): Model to use for reranking
- `query_expansion` (optional): Enable query expansion
- `max_tokens` (optional): Maximum tokens in results

**Response:**

```json
{
  "query": "What are the clinical trial requirements?",
  "results": [
    {
      "content": "Clinical trials must follow FDA regulations...",
      "score": 0.92,
      "metadata": {
        "document_id": "fda_21cfr312",
        "page": 5,
        "document_type": "regulation"
      },
      "chunk_id": "chunk_123",
      "document_id": "fda_21cfr312"
    }
  ],
  "total_results": 5,
  "processing_time_ms": 45.2,
  "retrieval_strategy_used": "hybrid",
  "database_used": "fda_db"
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/v1/projects/my-org/chatbot/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the clinical trial requirements?",
    "database": "fda_db",
    "top_k": 5
  }'
```

### List RAG Databases

List all configured RAG databases and their associated strategies for a project.

**Endpoint:** `GET /v1/projects/{namespace}/{project}/rag/databases`

**Parameters:**

- `namespace` (path, required): Project namespace
- `project` (path, required): Project name

**Response:**

```json
{
  "databases": [
    {
      "name": "main_db",
      "type": "ChromaStore",
      "is_default": true,
      "embedding_strategies": [
        {
          "name": "default_embeddings",
          "type": "OllamaEmbedder",
          "priority": 0,
          "is_default": true
        }
      ],
      "retrieval_strategies": [
        {
          "name": "basic_search",
          "type": "BasicSimilarityStrategy",
          "is_default": true
        }
      ]
    }
  ],
  "default_database": "main_db"
}
```

**Example:**

```bash
curl http://localhost:8000/v1/projects/my-org/chatbot/rag/databases
```

### Get Database Details

Get detailed information about a specific RAG database including its configuration and dependent datasets.

**Endpoint:** `GET /v1/projects/{namespace}/{project}/rag/databases/{database_name}`

**Parameters:**

- `namespace` (path, required): Project namespace
- `project` (path, required): Project name
- `database_name` (path, required): Name of the database

**Response:**

```json
{
  "name": "main_db",
  "type": "ChromaStore",
  "config": {
    "collection_name": "documents",
    "distance_function": "cosine"
  },
  "embedding_strategies": [
    {
      "name": "default_embeddings",
      "type": "OllamaEmbedder",
      "config": {
        "model": "nomic-embed-text",
        "dimension": 768
      },
      "priority": 0
    }
  ],
  "retrieval_strategies": [
    {
      "name": "basic_search",
      "type": "BasicSimilarityStrategy",
      "config": { "top_k": 10 },
      "default": true
    }
  ],
  "default_embedding_strategy": "default_embeddings",
  "default_retrieval_strategy": "basic_search",
  "dependent_datasets": ["research_papers", "documentation"]
}
```

**Example:**

```bash
curl http://localhost:8000/v1/projects/my-org/chatbot/rag/databases/main_db
```

### Create Database

Create a new RAG database in the project configuration.

**Endpoint:** `POST /v1/projects/{namespace}/{project}/rag/databases`

**Parameters:**

- `namespace` (path, required): Project namespace
- `project` (path, required): Project name

**Request Body:**

```json
{
  "name": "new_database",
  "type": "ChromaStore",
  "config": {
    "collection_name": "my_collection",
    "distance_function": "cosine"
  },
  "embedding_strategies": [
    {
      "name": "embeddings",
      "type": "OllamaEmbedder",
      "config": {
        "model": "nomic-embed-text",
        "dimension": 768
      }
    }
  ],
  "retrieval_strategies": [
    {
      "name": "basic_search",
      "type": "BasicSimilarityStrategy",
      "config": { "top_k": 10 },
      "default": true
    }
  ]
}
```

**Response (201 Created):**

```json
{
  "database": {
    "name": "new_database",
    "type": "ChromaStore",
    "is_default": false,
    "embedding_strategies": [...],
    "retrieval_strategies": [...]
  }
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/v1/projects/my-org/chatbot/rag/databases \
  -H "Content-Type: application/json" \
  -d '{
    "name": "new_database",
    "type": "ChromaStore",
    "embedding_strategies": [
      {"name": "embeddings", "type": "OllamaEmbedder", "config": {"model": "nomic-embed-text"}}
    ],
    "retrieval_strategies": [
      {"name": "basic", "type": "BasicSimilarityStrategy", "config": {}, "default": true}
    ]
  }'
```

### Update Database

Update a RAG database's mutable fields. Note: `name` and `type` are immutable.

**Endpoint:** `PATCH /v1/projects/{namespace}/{project}/rag/databases/{database_name}`

**Parameters:**

- `namespace` (path, required): Project namespace
- `project` (path, required): Project name
- `database_name` (path, required): Name of the database

**Request Body (all fields optional):**

```json
{
  "config": {
    "distance_function": "euclidean"
  },
  "embedding_strategies": [...],
  "retrieval_strategies": [...],
  "default_embedding_strategy": "new_default",
  "default_retrieval_strategy": "reranked_search"
}
```

**Response:**

```json
{
  "name": "main_db",
  "type": "ChromaStore",
  "config": {...},
  "embedding_strategies": [...],
  "retrieval_strategies": [...],
  "default_embedding_strategy": "new_default",
  "default_retrieval_strategy": "reranked_search",
  "dependent_datasets": []
}
```

**Example - Add a reranking strategy:**

```bash
curl -X PATCH http://localhost:8000/v1/projects/my-org/chatbot/rag/databases/main_db \
  -H "Content-Type: application/json" \
  -d '{
    "retrieval_strategies": [
      {"name": "basic_search", "type": "BasicSimilarityStrategy", "config": {"top_k": 10}},
      {"name": "reranked_search", "type": "CrossEncoderRerankedStrategy", "config": {"model_name": "reranker", "initial_k": 30}}
    ],
    "default_retrieval_strategy": "reranked_search"
  }'
```

### Delete Database

Delete a RAG database from the project. Fails if any datasets depend on this database.

**Endpoint:** `DELETE /v1/projects/{namespace}/{project}/rag/databases/{database_name}`

**Parameters:**

- `namespace` (path, required): Project namespace
- `project` (path, required): Project name
- `database_name` (path, required): Name of the database
- `delete_collection` (query, optional): Whether to delete the underlying vector store collection. Set to `false` to only remove from config. Default: `true`

**Response (200 OK):**

```json
{
  "message": "Database 'old_db' deleted successfully",
  "database": {
    "name": "old_db",
    "type": "ChromaStore",
    ...
  },
  "collection_deleted": true
}
```

**Error Response (409 Conflict - has dependent datasets):**

```json
{
  "detail": "Cannot delete database 'main_db': 2 dataset(s) depend on it. Delete or reassign these datasets first: ['dataset1', 'dataset2']"
}
```

**Example:**

```bash
# Delete database and its collection
curl -X DELETE http://localhost:8000/v1/projects/my-org/chatbot/rag/databases/old_db

# Only remove from config, keep the vector store data
curl -X DELETE "http://localhost:8000/v1/projects/my-org/chatbot/rag/databases/old_db?delete_collection=false"
```

### Check RAG Health

Get health status of the RAG system and databases.

**Endpoint:** `GET /v1/projects/{namespace}/{project}/rag/health`

**Parameters:**

- `namespace` (path, required): Project namespace
- `project` (path, required): Project name
- `database` (query, optional): Specific database to check (uses default if not specified)

**Response:**

```json
{
  "status": "healthy",
  "database": "main_db",
  "components": {
    "vector_store": {
      "name": "vector_store",
      "status": "healthy",
      "latency": 15.2,
      "message": "Vector store operational"
    },
    "embeddings": {
      "name": "embeddings",
      "status": "healthy",
      "latency": 8.5,
      "message": "Embedding service operational"
    }
  },
  "last_check": "2024-01-15T10:30:00Z",
  "issues": null
}
```

**Response Fields:**

- `status`: Overall health status (`healthy`, `degraded`, `unhealthy`)
- `database`: Database that was checked
- `components`: Individual component health checks
- `last_check`: Timestamp of the health check
- `issues`: Array of issues if any problems detected

**Component Health:**

- `name`: Component identifier
- `status`: Component status (`healthy`, `degraded`, `unhealthy`)
- `latency`: Response time in milliseconds
- `message`: Optional status message

**Example:**

```bash
curl http://localhost:8000/v1/projects/my-org/chatbot/rag/health
```

**Example (Specific database):**

```bash
curl "http://localhost:8000/v1/projects/my-org/chatbot/rag/health?database=main_db"
```

### Get RAG Statistics

Get statistics for a RAG database including vector counts and storage usage.

**Endpoint:** `GET /v1/projects/{namespace}/{project}/rag/stats`

**Parameters:**

- `namespace` (path, required): Project namespace
- `project` (path, required): Project name
- `database` (query, optional): Specific database to get stats for (uses default if not specified)

**Response:**

```json
{
  "database": "main_db",
  "vector_count": 1250,
  "storage_bytes": 52428800,
  "storage_human": "50 MB",
  "embedding_dimension": 768,
  "collection_name": "documents",
  "last_updated": "2024-01-15T10:30:00Z"
}
```

**Example:**

```bash
curl http://localhost:8000/v1/projects/my-org/chatbot/rag/stats
```

**Example (Specific database):**

```bash
curl "http://localhost:8000/v1/projects/my-org/chatbot/rag/stats?database=research_db"
```

### List Documents in Database

List all documents stored in a RAG database with their metadata.

**Endpoint:** `GET /v1/projects/{namespace}/{project}/rag/databases/{database_name}/documents`

**Parameters:**

- `namespace` (path, required): Project namespace
- `project` (path, required): Project name
- `database_name` (path, required): Name of the database
- `limit` (query, optional): Maximum documents to return (1-1000, default: 50)

**Response:**

```json
[
  {
    "id": "abc123def456",
    "filename": "clinical_trial_report.pdf",
    "chunk_count": 45,
    "size_bytes": 1048576,
    "parser_used": "PDFParser_LlamaIndex",
    "date_ingested": "2024-01-15T10:30:00Z",
    "metadata": {
      "document_type": "report",
      "source_dataset": "research_papers"
    }
  },
  {
    "id": "789xyz",
    "filename": "fda_guidelines.pdf",
    "chunk_count": 128,
    "size_bytes": 2097152,
    "parser_used": "PDFParser_PyPDF2",
    "date_ingested": "2024-01-14T15:20:00Z",
    "metadata": null
  }
]
```

**Example:**

```bash
curl http://localhost:8000/v1/projects/my-org/chatbot/rag/databases/main_db/documents
```

**Example (With limit):**

```bash
curl "http://localhost:8000/v1/projects/my-org/chatbot/rag/databases/main_db/documents?limit=100"
```

---

## Tasks API

### Get Task Status

Get the status of an asynchronous task.

**Endpoint:** `GET /v1/projects/{namespace}/{project}/tasks/{task_id}`

**Parameters:**

- `namespace` (path, required): Project namespace
- `project` (path, required): Project name
- `task_id` (path, required): Task ID returned from async operations

**Response:**

```json
{
  "task_id": "task-123-abc",
  "state": "SUCCESS",
  "meta": {
    "current": 2,
    "total": 2
  },
  "result": {
    "processed_files": 2,
    "failed_files": 0
  },
  "error": null,
  "traceback": null
}
```

**Task States:**

- `PENDING` - Task is queued
- `STARTED` - Task is running
- `SUCCESS` - Task completed successfully
- `FAILURE` - Task failed with error
- `RETRY` - Task is being retried

**Example:**

```bash
curl http://localhost:8000/v1/projects/my-org/chatbot/tasks/task-123-abc
```

### Cancel Task

Cancel a running task and revert any files that were successfully processed.

**Endpoint:** `DELETE /v1/projects/{namespace}/{project}/tasks/{task_id}`

**Parameters:**
- `namespace` (path, required): Project namespace
- `project` (path, required): Project name
- `task_id` (path, required): Task ID to cancel

**Description:**

This endpoint is primarily used for cancelling dataset processing operations. When a task is cancelled:
1. All pending Celery tasks are revoked (prevented from starting)
2. Running tasks are gracefully stopped (current work finishes)
3. The task is marked as cancelled in the backend
4. Any files that were successfully processed have their chunks removed from the vector store

**Response:**
```json
{
  "message": "Task cancelled and 3 file(s) reverted",
  "task_id": "task-123-abc",
  "cancelled": true,
  "pending_tasks_cancelled": 5,
  "running_tasks_at_cancel": 2,
  "files_reverted": 3,
  "files_failed_to_revert": 0,
  "errors": null,
  "already_completed": false,
  "already_cancelled": false
}
```

**Response Fields:**
- `message` - Human-readable status message
- `task_id` - The ID of the cancelled task
- `cancelled` - Whether the task was successfully cancelled
- `pending_tasks_cancelled` - Number of queued tasks that were stopped
- `running_tasks_at_cancel` - Number of tasks that were running when cancelled
- `files_reverted` - Number of files whose chunks were successfully removed
- `files_failed_to_revert` - Number of files that failed to clean up
- `errors` - Array of cleanup errors (if any)
- `already_completed` - True if task had already completed before cancellation
- `already_cancelled` - True if task was already cancelled

**Edge Cases:**

**Task Already Completed:**
```json
{
  "message": "Task already success",
  "task_id": "task-123-abc",
  "cancelled": false,
  "already_completed": true,
  ...
}
```

**Task Already Cancelled:**
```json
{
  "message": "Task already cancelled",
  "task_id": "task-123-abc",
  "cancelled": true,
  "already_cancelled": true,
  "files_reverted": 3,
  ...
}
```

**Cleanup Failures:**
```json
{
  "message": "Task cancelled with cleanup issues: 3 reverted, 1 failed",
  "task_id": "task-123-abc",
  "cancelled": true,
  "files_reverted": 3,
  "files_failed_to_revert": 1,
  "errors": [
    {
      "file_hash": "abc123def456",
      "error": "Vector store connection timeout"
    }
  ],
  ...
}
```

**Example:**
```bash
# Cancel a running dataset processing task
curl -X DELETE http://localhost:8000/v1/projects/my-org/chatbot/tasks/task-123-abc
```

**HTTP Status Codes:**
- `200 OK` - Task cancellation succeeded (or task already completed/cancelled)
- `404 Not Found` - Task not found or not a cancellable task type
- `500 Internal Server Error` - Server error during cancellation

**Notes:**
- Only group tasks (dataset processing) can be cancelled
- **Security:** Tasks can only be cancelled by the namespace/project they belong to
- Cancellation is idempotent (safe to call multiple times)
- Cleanup failures don't prevent cancellation from succeeding
- Successfully processed files are automatically reverted
- Manual cleanup is available via `POST /v1/projects/{namespace}/{project}/datasets/{dataset}/cleanup/{file_hash}` if automatic cleanup fails

---

## Examples API

### List Examples

List all available example projects.

**Endpoint:** `GET /v1/examples`

**Response:**

```json
{
  "examples": [
    {
      "id": "fda_rag",
      "slug": "fda_rag",
      "title": "FDA RAG Example",
      "description": "Example using FDA warning letters",
      "primaryModel": "llama3.2:3b",
      "tags": ["rag", "healthcare"],
      "dataset_count": 1,
      "data_size_bytes": 2097152,
      "data_size_human": "2.0MB",
      "project_size_bytes": 4096,
      "project_size_human": "4.0KB",
      "updated_at": "2024-01-15T10:30:00"
    }
  ]
}
```

**Example:**

```bash
curl http://localhost:8000/v1/examples
```

### Import Example as Project

Import an example as a new project.

**Endpoint:** `POST /v1/examples/{example_id}/import-project`

**Parameters:**

- `example_id` (path, required): Example ID to import

**Request Body:**

```json
{
  "namespace": "my-org",
  "name": "my-fda-project",
  "process": true
}
```

**Response:**

```json
{
  "project": "my-fda-project",
  "namespace": "my-org",
  "datasets": ["fda_letters"],
  "task_ids": ["task-123-abc"]
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/v1/examples/fda_rag/import-project \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "my-org",
    "name": "my-fda-project",
    "process": true
  }'
```

### Import Example Data

Import example data into an existing project.

**Endpoint:** `POST /v1/examples/{example_id}/import-data`

**Parameters:**

- `example_id` (path, required): Example ID to import data from

**Request Body:**

```json
{
  "namespace": "my-org",
  "project": "my-project",
  "include_strategies": true,
  "process": true
}
```

**Response:**

```json
{
  "project": "my-project",
  "namespace": "my-org",
  "datasets": ["fda_letters"],
  "task_ids": ["task-456-def"]
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/v1/examples/fda_rag/import-data \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "my-org",
    "project": "my-project",
    "include_strategies": true,
    "process": true
  }'
```

---

## Health API

### Overall Health Check

Check overall system health.

**Endpoint:** `GET /health`

**Response:**

```json
{
  "status": "healthy",
  "services": {
    "api": "healthy",
    "celery": "healthy",
    "redis": "healthy"
  }
}
```

**Example:**

```bash
curl http://localhost:8000/health
```

### Liveness Probe

Simple liveness check for container orchestration.

**Endpoint:** `GET /health/liveness`

**Response:**

```json
{
  "status": "alive"
}
```

**Example:**

```bash
curl http://localhost:8000/health/liveness
```

---

## System Info

### Root Endpoint

Basic hello endpoint.

**Endpoint:** `GET /`

**Response:**

```json
{
  "message": "Hello, World!"
}
```

### System Information

Get system version and configuration info.

**Endpoint:** `GET /info`

**Response:**

```json
{
  "version": "0.1.0",
  "data_directory": "/Users/username/.llamafarm"
}
```

**Example:**

```bash
curl http://localhost:8000/info
```

### Check for CLI Updates

Check if a newer version of the CLI is available.

**Endpoint:** `GET /v1/system/version-check`

**Response:**

```json
{
  "current_version": "0.0.17",
  "latest_version": "0.0.18",
  "name": "v0.0.18",
  "release_notes": "### Features\n- New feature X\n- Improved Y",
  "release_url": "https://github.com/llama-farm/llamafarm/releases/tag/v0.0.18",
  "published_at": "2024-01-15T10:30:00Z",
  "from_cache": false,
  "install": {
    "mac_linux": "curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash",
    "windows": "winget install LlamaFarm.CLI"
  }
}
```

**Example:**

```bash
curl http://localhost:8000/v1/system/version-check
```

### Get Disk Space

Get disk space information for the HuggingFace cache and system disk.

**Endpoint:** `GET /v1/system/disk`

**Response:**

```json
{
  "cache": {
    "total_bytes": 500000000000,
    "used_bytes": 200000000000,
    "free_bytes": 300000000000,
    "path": "/Users/username/.cache/huggingface",
    "percent_free": 60.0
  },
  "system": {
    "total_bytes": 1000000000000,
    "used_bytes": 400000000000,
    "free_bytes": 600000000000,
    "path": "/",
    "percent_free": 60.0
  }
}
```

**Response Fields:**

- `cache`: Disk info for the HuggingFace cache directory (where models are stored)
- `system`: Disk info for the system root directory
- Each contains: `total_bytes`, `used_bytes`, `free_bytes`, `path`, `percent_free`

**Example:**

```bash
curl http://localhost:8000/v1/system/disk
```

---

## Event Logs API

The Event Logs API provides observability into project operations including inference calls, RAG processing, and other events.

### List Event Logs

List event logs for a project with optional filtering.

**Endpoint:** `GET /v1/projects/{namespace}/{project}/event_logs`

**Parameters:**

- `namespace` (path, required): Project namespace
- `project` (path, required): Project name
- `type` (query, optional): Filter by event type (e.g., "inference", "rag_processing")
- `start_time` (query, optional): Filter events after this timestamp (ISO 8601 format)
- `end_time` (query, optional): Filter events before this timestamp (ISO 8601 format)
- `limit` (query, optional): Maximum number of events to return (1-100, default: 10)
- `offset` (query, optional): Number of events to skip for pagination

**Response:**

```json
{
  "total": 42,
  "events": [
    {
      "event_id": "evt_20240115_103000_inference_abc123",
      "type": "inference",
      "timestamp": "2024-01-15T10:30:00Z",
      "summary": {
        "model": "llama3.2:3b",
        "tokens": 150,
        "duration_ms": 1200
      }
    }
  ],
  "limit": 10,
  "offset": 0
}
```

**Example:**

```bash
# List recent events
curl http://localhost:8000/v1/projects/my-org/chatbot/event_logs

# Filter by type with pagination
curl "http://localhost:8000/v1/projects/my-org/chatbot/event_logs?type=inference&limit=20"

# Filter by time range
curl "http://localhost:8000/v1/projects/my-org/chatbot/event_logs?start_time=2024-01-15T00:00:00Z"
```

### Get Event Details

Get full details of a specific event including all sub-events.

**Endpoint:** `GET /v1/projects/{namespace}/{project}/event_logs/{event_id}`

**Parameters:**

- `namespace` (path, required): Project namespace
- `project` (path, required): Project name
- `event_id` (path, required): Event ID

**Response:**

```json
{
  "event_id": "evt_20240115_103000_inference_abc123",
  "type": "inference",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "model": "llama3.2:3b",
    "messages": [...],
    "response": {...},
    "tokens": {
      "prompt": 50,
      "completion": 100,
      "total": 150
    },
    "duration_ms": 1200
  },
  "sub_events": [...]
}
```

**Example:**

```bash
curl http://localhost:8000/v1/projects/my-org/chatbot/event_logs/evt_20240115_103000_inference_abc123
```

---

## Models Cache API

The Models Cache API allows you to manage locally cached models (primarily HuggingFace models used by Universal Runtime).

### List Cached Models

List all models cached on disk.

**Endpoint:** `GET /v1/models`

**Parameters:**

- `provider` (query, optional): Model provider (default: "universal")

**Response:**

```json
{
  "data": [
    {
      "model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
      "size_bytes": 90000000,
      "size_human": "90MB",
      "last_modified": "2024-01-15T10:30:00Z",
      "revisions": ["main"]
    }
  ]
}
```

**Example:**

```bash
curl http://localhost:8000/v1/models
```

### Download/Cache Model

Download and cache a model. Returns a streaming response with progress events.

**Endpoint:** `POST /v1/models/download`

**Request Body:**

```json
{
  "provider": "universal",
  "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"
}
```

**Response:** Server-Sent Events stream with progress updates:

```
data: {"event": "progress", "downloaded": 45000000, "total": 90000000, "percent": 50}

data: {"event": "complete", "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
```

**Example:**

```bash
curl -X POST http://localhost:8000/v1/models/download \
  -H "Content-Type: application/json" \
  -d '{"model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"}'
```

### Validate Download

Check if there's sufficient disk space for a model download before starting.

**Endpoint:** `POST /v1/models/validate-download`

**Request Body:**

```json
{
  "model_name": "unsloth/Qwen3-1.7B-GGUF"
}
```

**Response:**

```json
{
  "can_download": true,
  "warning": false,
  "message": "Sufficient disk space available",
  "available_bytes": 107374182400,
  "required_bytes": 1073741824,
  "cache_info": {
    "total_bytes": 500000000000,
    "used_bytes": 200000000000,
    "free_bytes": 300000000000,
    "path": "/Users/username/.cache/huggingface",
    "percent_free": 60.0
  },
  "system_info": {
    "total_bytes": 1000000000000,
    "used_bytes": 400000000000,
    "free_bytes": 600000000000,
    "path": "/",
    "percent_free": 60.0
  }
}
```

**Response Fields:**

- `can_download`: Whether download can proceed (false if critically low space)
- `warning`: Whether space is low but download can proceed
- `message`: Human-readable status message
- `available_bytes`: Available disk space in bytes
- `required_bytes`: Estimated space required for download
- `cache_info`: Disk info for HuggingFace cache location
- `system_info`: Disk info for system root

**Example:**

```bash
curl -X POST http://localhost:8000/v1/models/validate-download \
  -H "Content-Type: application/json" \
  -d '{"model_name": "unsloth/Qwen3-1.7B-GGUF"}'
```

### Get GGUF Quantization Options

List all available GGUF quantization options for a model with file sizes.

**Endpoint:** `GET /v1/models/{model_id}/quantizations`

**Parameters:**

- `model_id` (path, required): HuggingFace model identifier (e.g., "unsloth/Qwen3-1.7B-GGUF")

**Response:**

```json
{
  "options": [
    {
      "filename": "Qwen3-1.7B-Q4_K_M.gguf",
      "quantization": "Q4_K_M",
      "size_bytes": 1073741824,
      "size_human": "1.0 GB"
    },
    {
      "filename": "Qwen3-1.7B-Q8_0.gguf",
      "quantization": "Q8_0",
      "size_bytes": 1879048192,
      "size_human": "1.75 GB"
    },
    {
      "filename": "Qwen3-1.7B-F16.gguf",
      "quantization": "F16",
      "size_bytes": 3489660928,
      "size_human": "3.25 GB"
    }
  ]
}
```

**Example:**

```bash
curl http://localhost:8000/v1/models/unsloth/Qwen3-1.7B-GGUF/quantizations
```

### Delete Cached Model

Delete a cached model from disk.

**Endpoint:** `DELETE /v1/models/{model_name}`

**Parameters:**

- `model_name` (path, required): The model identifier to delete
- `provider` (query, optional): Model provider (default: "universal")

**Response:**

```json
{
  "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "revisions_deleted": 1,
  "size_freed": 90000000,
  "path": "/Users/username/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2"
}
```

**Example:**

```bash
curl -X DELETE "http://localhost:8000/v1/models/cross-encoder/ms-marco-MiniLM-L-6-v2"
```

---

## Multi-Model Support

As of PR #263, LlamaFarm supports multiple models per project with OpenAI-compatible `model` field in chat requests.

### Configuration

In your `llamafarm.yaml`:

```yaml
runtime:
  models:
    - name: fast-model
      provider: ollama
      model: llama3.2:3b
    - name: smart-model
      provider: ollama
      model: llama3.2:70b
  default_model: fast-model
```

### Using Different Models

The `model` field in chat completions requests is OpenAI-compatible and allows you to select which configured model to use:

```bash
curl -X POST http://localhost:8000/v1/projects/my-org/chatbot/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "smart-model",
    "messages": [
      {"role": "user", "content": "Explain quantum computing"}
    ]
  }'
```

If `model` is not specified, the `default_model` from your configuration is used. This makes LlamaFarm a drop-in replacement for OpenAI's API with the added benefit of model selection.

### Get Available Models

```bash
curl http://localhost:8000/v1/projects/my-org/chatbot/models
```

---

## Rate Limiting and Performance

### Concurrent Requests

The API handles concurrent requests efficiently:

- Chat sessions are thread-safe with internal locking
- Dataset processing can run asynchronously with Celery
- Multiple chat sessions can be active simultaneously

### Session Management

- Sessions expire after 30 minutes of inactivity
- Use `X-No-Session` header for stateless requests
- Session cleanup happens automatically

### Async Processing

For long-running operations (dataset processing):

1. POST to the dataset `actions` endpoint (`{"action_type":"process"}`) to queue a Celery task
2. Poll the task endpoint to check status
3. Retrieve final results when `state` is `SUCCESS`

---

## MCP (Model Context Protocol) Compatible Endpoints

LlamaFarm's API is compatible with the Model Context Protocol (MCP), allowing AI agents to interact with LlamaFarm programmatically. The following endpoints are tagged with `mcp` for easy discovery by MCP clients:

### MCP-Compatible Operations

**Project Management:**

- `GET /v1/projects/{namespace}` - List projects (operation ID: `projects_list`)
- `POST /v1/projects/{namespace}` - Create project (operation ID: `project_create`)
- `GET /v1/projects/{namespace}/{project}` - Get project (operation ID: `project_get`)
- `PUT /v1/projects/{namespace}/{project}` - Update project (operation ID: `project_update`)
- `DELETE /v1/projects/{namespace}/{project}` - Delete project (operation ID: `project_delete`)

**Model Management:**

- `GET /v1/projects/{namespace}/{project}/models` - List models (operation ID: `models_list`)

**Dataset Operations:**

- `GET /v1/projects/{namespace}/{project}/datasets` - List datasets (operation ID: `dataset_list`)
- `GET /v1/projects/{namespace}/{project}/datasets/strategies` - List strategies (operation ID: `dataset_strategies_list`)
- `POST /v1/projects/{namespace}/{project}/datasets` - Create dataset (operation ID: `dataset_create`)
- `DELETE /v1/projects/{namespace}/{project}/datasets/{dataset}` - Delete dataset (operation ID: `dataset_delete`)
- `POST /v1/projects/{namespace}/{project}/datasets/{dataset}/actions` - Dataset actions (operation ID: `dataset_actions`)
- `POST /v1/projects/{namespace}/{project}/datasets/{dataset}/data` - Upload data (operation ID: `dataset_data_upload`)

**RAG Operations:**

- `POST /v1/projects/{namespace}/{project}/rag/query` - Query RAG (operation ID: `rag_query`)
- `POST /v1/projects/{namespace}/{project}/rag/databases` - Create database (operation ID: `database_create`)
- `GET /v1/projects/{namespace}/{project}/rag/databases/{database}` - Get database (operation ID: `database_get`)
- `PATCH /v1/projects/{namespace}/{project}/rag/databases/{database}` - Update database (operation ID: `database_update`)
- `DELETE /v1/projects/{namespace}/{project}/rag/databases/{database}` - Delete database (operation ID: `database_delete`)

**Task Management:**

- `GET /v1/projects/{namespace}/{project}/tasks/{task_id}` - Get task status (operation ID: `task_get`)

### Using LlamaFarm with MCP Servers

You can configure LlamaFarm projects to expose tools through MCP servers, giving AI agents access to filesystems, databases, APIs, and custom business logic. See the [MCP documentation](../mcp/index.md) for configuration examples.

**Example: AI Agent with LlamaFarm MCP Tools**

```python
# AI agent can use LlamaFarm endpoints as tools
# through MCP protocol
{
  "tool": "llamafarm_rag_query",
  "arguments": {
    "query": "What are the clinical trial requirements?",
    "database": "fda_db",
    "top_k": 5
  }
}
```

---

## Best Practices

### Error Handling

Always check HTTP status codes and handle errors:

```bash
response=$(curl -s -w "\n%{http_code}" http://localhost:8000/v1/projects/my-org/chatbot)
http_code=$(echo "$response" | tail -n 1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" -eq 200 ]; then
  echo "Success: $body"
else
  echo "Error ($http_code): $body"
fi
```

### Using RAG Effectively

1. **Create datasets first**: Upload and process files before querying
2. **Use appropriate top_k**: Start with 5-10 results
3. **Set score thresholds**: Filter low-quality results (e.g., 0.7)
4. **Test queries**: Use the RAG query endpoint to test retrieval before chat

### Chat Sessions

1. **Stateful conversations**: Reuse session IDs for context
2. **Stateless queries**: Use `X-No-Session` for one-off questions
3. **Clean up**: Delete old sessions to free resources
4. **History access**: Use history endpoint to debug conversations

### Dataset Processing

1. **Upload first, process later**: Separate ingestion from processing
2. **Use async for large datasets**: Enable async processing for >10 files
3. **Monitor task status**: Poll task endpoint for progress
4. **Handle duplicates**: System automatically skips duplicate files

---

## API Clients

### Python Example

```python
import requests

class LlamaFarmClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def chat(self, namespace, project, message, session_id=None):
        url = f"{self.base_url}/v1/projects/{namespace}/{project}/chat/completions"
        headers = {}
        if session_id:
            headers["X-Session-ID"] = session_id

        response = self.session.post(
            url,
            headers=headers,
            json={
                "messages": [{"role": "user", "content": message}]
            }
        )
        response.raise_for_status()
        return response.json()

    def query_rag(self, namespace, project, query, database=None, top_k=5):
        url = f"{self.base_url}/v1/projects/{namespace}/{project}/rag/query"
        response = self.session.post(
            url,
            json={
                "query": query,
                "database": database,
                "top_k": top_k
            }
        )
        response.raise_for_status()
        return response.json()

# Usage
client = LlamaFarmClient()
result = client.chat("my-org", "chatbot", "Hello!")
print(result["choices"][0]["message"]["content"])
```

### JavaScript/TypeScript Example

```typescript
interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

interface ChatRequest {
  messages: ChatMessage[];
  stream?: boolean;
  rag_enabled?: boolean;
  database?: string;
}

class LlamaFarmClient {
  constructor(private baseUrl: string = "http://localhost:8000") {}

  async chat(
    namespace: string,
    project: string,
    messages: ChatMessage[],
    sessionId?: string
  ): Promise<any> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };

    if (sessionId) {
      headers["X-Session-ID"] = sessionId;
    }

    const response = await fetch(
      `${this.baseUrl}/v1/projects/${namespace}/${project}/chat/completions`,
      {
        method: "POST",
        headers,
        body: JSON.stringify({ messages }),
      }
    );

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return response.json();
  }

  async queryRAG(
    namespace: string,
    project: string,
    query: string,
    database?: string,
    topK: number = 5
  ): Promise<any> {
    const response = await fetch(
      `${this.baseUrl}/v1/projects/${namespace}/${project}/rag/query`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          database,
          top_k: topK,
        }),
      }
    );

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return response.json();
  }
}

// Usage
const client = new LlamaFarmClient();
const result = await client.chat("my-org", "chatbot", [
  { role: "user", content: "Hello!" },
]);
console.log(result.choices[0].message.content);
```

---

## Troubleshooting

### Common Issues

**Problem:** `404 Not Found` when accessing project

- **Solution**: Verify namespace and project name are correct. List projects to confirm.

**Problem:** Chat returns empty or error

- **Solution**: Check that the model is configured correctly and Ollama is running.

**Problem:** RAG query returns no results

- **Solution**: Ensure dataset is processed and database exists. Check RAG health endpoint.

**Problem:** Dataset processing stuck

- **Solution**: Check Celery worker status. Use async processing and poll task endpoint.

**Problem:** Session not persisting

- **Solution**: Ensure you're passing `X-Session-ID` header and not using `X-No-Session`.

### Debugging Tips

1. **Check logs**: Server logs are written to stdout
2. **Verify configuration**: Use `GET /v1/projects/{namespace}/{project}` to inspect config
3. **Test health**: Use `/health` endpoint to verify services are running
4. **Inspect tasks**: For async operations, poll task endpoint for detailed error info
5. **Use curl verbose**: Add `-v` flag to curl for detailed request/response info

---

## Vision API (OCR & Document Extraction)

The Vision API provides OCR and document extraction capabilities through the main LlamaFarm API server. These endpoints proxy to the Universal Runtime, handling file uploads and base64 image conversion automatically.

**Base URL:** `http://localhost:8000/v1/vision`

### OCR Endpoint

Extract text from images and PDFs using multiple OCR backends.

**Endpoint:** `POST /v1/vision/ocr`

**Content-Type:** `multipart/form-data`

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file` | file | No* | - | PDF or image file to process |
| `images` | string | No* | - | Base64-encoded images as JSON array |
| `model` | string | No | `surya` | OCR backend: `surya`, `easyocr`, `paddleocr`, `tesseract` |
| `languages` | string | No | `en` | Comma-separated language codes (e.g., `en,fr`) |
| `return_boxes` | boolean | No | `false` | Return bounding boxes for detected text |

*Either `file` or `images` must be provided.

**Supported File Types:** PDF, PNG, JPG, JPEG, GIF, WebP, BMP, TIFF

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "index": 0,
      "text": "Extracted text from the document...",
      "confidence": 0.95
    }
  ],
  "model": "surya",
  "usage": {"images_processed": 1}
}
```

**Example (File Upload):**

```bash
curl -X POST http://localhost:8000/v1/vision/ocr \
  -F "file=@document.pdf" \
  -F "model=easyocr" \
  -F "languages=en"
```

**Example (Base64 Images):**

```bash
curl -X POST http://localhost:8000/v1/vision/ocr \
  -F 'images=["data:image/png;base64,iVBORw0KGgo..."]' \
  -F "model=surya" \
  -F "languages=en"
```

### Document Extraction Endpoint

Extract structured data from documents using vision-language models.

**Endpoint:** `POST /v1/vision/documents/extract`

**Content-Type:** `multipart/form-data`

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file` | file | No* | - | PDF or image file to process |
| `images` | string | No* | - | Base64-encoded images as JSON array |
| `model` | string | Yes | - | HuggingFace model ID (e.g., `naver-clova-ix/donut-base-finetuned-docvqa`) |
| `prompts` | string | No | - | Comma-separated prompts for VQA task |
| `task` | string | No | `extraction` | Task type: `extraction`, `vqa`, `classification` |

*Either `file` or `images` must be provided.

**Supported Models:**

| Model | Description |
|-------|-------------|
| `naver-clova-ix/donut-base-finetuned-cord-v2` | Receipt/invoice extraction |
| `naver-clova-ix/donut-base-finetuned-docvqa` | Document Q&A |
| `microsoft/layoutlmv3-base-finetuned-docvqa` | Document Q&A with layout |

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "index": 0,
      "confidence": 0.9,
      "text": "<s_docvqa><s_question>What is the total?</s_question><s_answer>$15.99</s_answer>",
      "fields": [
        {"key": "question", "value": "What is the total?", "confidence": 0.9},
        {"key": "answer", "value": "$15.99", "confidence": 0.9}
      ]
    }
  ],
  "model": "naver-clova-ix/donut-base-finetuned-docvqa",
  "task": "vqa",
  "usage": {"documents_processed": 1}
}
```

**Example (Document VQA with File Upload):**

```bash
curl -X POST http://localhost:8000/v1/vision/documents/extract \
  -F "file=@receipt.pdf" \
  -F "model=naver-clova-ix/donut-base-finetuned-docvqa" \
  -F "prompts=What is the store name?,What is the total amount?" \
  -F "task=vqa"
```

**Example (Extraction with Base64):**

```bash
curl -X POST http://localhost:8000/v1/vision/documents/extract \
  -F 'images=["data:image/png;base64,iVBORw0KGgo..."]' \
  -F "model=naver-clova-ix/donut-base-finetuned-cord-v2" \
  -F "task=extraction"
```

---

## ML API (Custom Classifiers & Anomaly Detection)

The ML API provides custom text classification and anomaly detection capabilities through the main LlamaFarm API server. These endpoints proxy to the Universal Runtime with automatic model versioning support.

**Base URL:** `http://localhost:8000/v1/ml`

:::tip Model Versioning
When `overwrite: false` (default), models are saved with timestamps like `my-model_20251215_155054`. Use the `-latest` suffix (e.g., `my-model-latest`) to automatically resolve to the newest version.

**Endpoints supporting `-latest` resolution:**
- `/v1/ml/classifier/predict`, `/v1/ml/classifier/load`
- `/v1/ml/anomaly/score`, `/v1/ml/anomaly/detect`, `/v1/ml/anomaly/load`

Note: The `fit` endpoints do NOT support `-latest` since they create new models.
:::

### Custom Text Classification (SetFit)

Train custom classifiers with as few as 8-16 examples per class using SetFit (Sentence Transformer Fine-tuning).

#### Fit Classifier

Train a new text classifier.

**Endpoint:** `POST /v1/ml/classifier/fit`

**Request Body:**

```json
{
  "model": "intent-classifier",
  "base_model": "sentence-transformers/all-MiniLM-L6-v2",
  "training_data": [
    {"text": "I need to book a flight to NYC", "label": "booking"},
    {"text": "Reserve a hotel room for me", "label": "booking"},
    {"text": "Cancel my reservation please", "label": "cancellation"},
    {"text": "I want to cancel my booking", "label": "cancellation"},
    {"text": "What time does the flight leave?", "label": "inquiry"},
    {"text": "How much does it cost?", "label": "inquiry"}
  ],
  "num_iterations": 20,
  "batch_size": 16,
  "overwrite": false
}
```

**Request Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | Yes | - | Base name for the classifier |
| `base_model` | string | No | `all-MiniLM-L6-v2` | Sentence transformer to fine-tune |
| `training_data` | array | Yes | - | List of `{"text", "label"}` objects |
| `num_iterations` | int | No | 20 | Contrastive learning iterations |
| `batch_size` | int | No | 16 | Training batch size |
| `overwrite` | bool | No | false | If false, version with timestamp |

:::note Description Field
To add a description to a model, use the `/v1/ml/classifier/save` endpoint after fitting. The `description` parameter in fit requests is not persisted.
:::

**Response:**

```json
{
  "object": "fit_result",
  "model": "intent-classifier-test_20260102_202450",
  "base_model": "sentence-transformers/all-MiniLM-L6-v2",
  "samples_fitted": 40,
  "num_classes": 4,
  "labels": ["booking", "cancellation", "complaint", "inquiry"],
  "training_time_ms": 6751.66,
  "status": "fitted",
  "auto_saved": true,
  "saved_path": "~/.llamafarm/models/classifier/intent-classifier-test_20260102_202450",
  "base_name": "intent-classifier-test",
  "versioned_name": "intent-classifier-test_20260102_202450",
  "overwrite": false
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/v1/ml/classifier/fit \
  -H "Content-Type: application/json" \
  -d '{
    "model": "intent-classifier",
    "training_data": [
      {"text": "I need to book a flight to NYC", "label": "booking"},
      {"text": "Reserve a hotel room for me", "label": "booking"},
      {"text": "Cancel my reservation please", "label": "cancellation"},
      {"text": "I want to cancel my booking", "label": "cancellation"},
      {"text": "What time does the flight leave?", "label": "inquiry"},
      {"text": "How much does it cost?", "label": "inquiry"},
      {"text": "I am very unhappy with the service", "label": "complaint"},
      {"text": "This is unacceptable quality", "label": "complaint"}
    ],
    "num_iterations": 20
  }'
```

#### Predict with Classifier

Classify texts using a trained model.

**Endpoint:** `POST /v1/ml/classifier/predict`

**Request Body:**

```json
{
  "model": "intent-classifier-latest",
  "texts": [
    "Book me a flight to Paris tomorrow",
    "Cancel my upcoming trip",
    "What are the check-in times?",
    "This is absolutely terrible service"
  ]
}
```

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "text": "Book me a flight to Paris tomorrow",
      "label": "booking",
      "score": 0.66,
      "all_scores": {"booking": 0.66, "cancellation": 0.11, "complaint": 0.11, "inquiry": 0.13}
    },
    {
      "text": "Cancel my upcoming trip",
      "label": "cancellation",
      "score": 0.79,
      "all_scores": {"booking": 0.09, "cancellation": 0.79, "complaint": 0.07, "inquiry": 0.05}
    },
    {
      "text": "What are the check-in times?",
      "label": "inquiry",
      "score": 0.68,
      "all_scores": {"booking": 0.15, "cancellation": 0.06, "complaint": 0.11, "inquiry": 0.68}
    },
    {
      "text": "This is absolutely terrible service",
      "label": "complaint",
      "score": 0.77,
      "all_scores": {"booking": 0.06, "cancellation": 0.08, "complaint": 0.77, "inquiry": 0.10}
    }
  ],
  "total_count": 4,
  "model": "intent-classifier-test_20260102_202450"
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/v1/ml/classifier/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "intent-classifier-latest",
    "texts": [
      "Book me a flight to Paris tomorrow",
      "Cancel my upcoming trip",
      "What are the check-in times?",
      "This is absolutely terrible service"
    ]
  }'
```

#### Save Classifier

Save a trained classifier to disk for production use.

**Endpoint:** `POST /v1/ml/classifier/save`

**Request Body:**

```json
{
  "model": "intent-classifier_20251215_155054"
}
```

**Response:**

```json
{
  "object": "save_result",
  "model": "intent-classifier_20251215_155054",
  "path": "~/.llamafarm/models/classifier/intent-classifier_20251215_155054",
  "status": "saved"
}
```

#### Load Classifier

Load a previously saved classifier.

**Endpoint:** `POST /v1/ml/classifier/load`

**Request Body:**

```json
{
  "model": "intent-classifier-latest"
}
```

**Response:**

```json
{
  "object": "load_result",
  "model": "intent-classifier_20251215_155054",
  "status": "loaded"
}
```

#### List Classifier Models

**Endpoint:** `GET /v1/ml/classifier/models`

**Response:**

```json
{
  "object": "list",
  "data": [
    {"name": "intent-classifier_20251215_155054", "labels": ["booking", "cancellation", "inquiry"]}
  ]
}
```

#### Delete Classifier Model

**Endpoint:** `DELETE /v1/ml/classifier/models/{model_name}`

The `model_name` is the directory name (e.g., `intent-classifier_20251215_155054`). SetFit classifiers are stored as directories under `~/.llamafarm/models/classifier/`.

---

### Anomaly Detection

Train anomaly detectors on normal data and detect outliers in new data.

#### Fit Anomaly Detector

Train an anomaly detection model.

**Endpoint:** `POST /v1/ml/anomaly/fit`

**Request Body (Numeric Data):**

```json
{
  "model": "sensor-detector",
  "backend": "isolation_forest",
  "data": [
    [22.1, 1024], [23.5, 1100], [21.8, 980],
    [24.2, 1050], [22.7, 1080], [23.1, 990]
  ],
  "contamination": 0.1,
  "overwrite": false
}
```

**Request Body (Mixed Data with Schema):**

```json
{
  "model": "api-monitor",
  "backend": "isolation_forest",
  "data": [
    {"response_time_ms": 100, "bytes": 1024, "method": "GET", "user_agent": "Mozilla/5.0"},
    {"response_time_ms": 105, "bytes": 1100, "method": "POST", "user_agent": "Chrome/90.0"}
  ],
  "schema": {
    "response_time_ms": "numeric",
    "bytes": "numeric",
    "method": "label",
    "user_agent": "hash"
  },
  "contamination": 0.1
}
```

**Request Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | No | "default" | Model identifier |
| `backend` | string | No | "isolation_forest" | Algorithm: `isolation_forest`, `one_class_svm`, `local_outlier_factor`, `autoencoder` |
| `data` | array | Yes | - | Training data (numeric arrays or dicts) |
| `schema` | object | No | - | Feature encoding schema (required for dict data) |
| `contamination` | float | No | 0.1 | Expected proportion of anomalies (must be >0 and ≤0.5) |
| `overwrite` | bool | No | false | If false, version with timestamp |

**Response:**

```json
{
  "object": "fit_result",
  "model": "sensor_anomaly_detector_20260102_202438",
  "backend": "isolation_forest",
  "samples_fitted": 30,
  "training_time_ms": 85.77,
  "model_params": {
    "backend": "isolation_forest",
    "contamination": 0.05,
    "threshold": 0.894,
    "input_dim": 1
  },
  "status": "fitted",
  "base_name": "sensor_anomaly_detector",
  "versioned_name": "sensor_anomaly_detector_20260102_202438",
  "overwrite": false
}
```

**Example (Temperature Sensor Data):**

```bash
# Train on normal temperature readings (20-25°C range)
curl -X POST http://localhost:8000/v1/ml/anomaly/fit \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sensor_anomaly_detector",
    "backend": "isolation_forest",
    "data": [
      [22.1], [23.5], [21.8], [24.2], [22.7],
      [23.1], [21.5], [24.8], [22.3], [23.9],
      [21.2], [24.5], [22.8], [23.2], [21.9],
      [24.1], [22.5], [23.7], [21.6], [24.3]
    ],
    "contamination": 0.05
  }'
```

#### Score Anomalies

Score all data points for anomalies.

**Endpoint:** `POST /v1/ml/anomaly/score`

**Request Body:**

```json
{
  "model": "sensor-detector-latest",
  "backend": "isolation_forest",
  "data": [[22.0], [23.5], [0.0], [100.0], [21.5]],
  "threshold": 0.5
}
```

**Response:**

```json
{
  "object": "list",
  "data": [
    {"index": 0, "score": 0.23, "is_anomaly": false, "raw_score": 0.12},
    {"index": 1, "score": 0.21, "is_anomaly": false, "raw_score": 0.10},
    {"index": 2, "score": 0.89, "is_anomaly": true, "raw_score": -0.45},
    {"index": 3, "score": 0.95, "is_anomaly": true, "raw_score": -0.52},
    {"index": 4, "score": 0.22, "is_anomaly": false, "raw_score": 0.11}
  ],
  "summary": {
    "total_points": 5,
    "anomaly_count": 2,
    "anomaly_rate": 0.4,
    "threshold": 0.5
  }
}
```

#### Detect Anomalies

Detect anomalies (returns only anomalous points).

**Endpoint:** `POST /v1/ml/anomaly/detect`

Same request format as `/score`, but response only includes anomalous points.

**Example (Detecting Temperature Anomalies):**

```bash
# Test with mix of normal and anomalous readings
curl -X POST http://localhost:8000/v1/ml/anomaly/detect \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sensor_anomaly_detector-latest",
    "backend": "isolation_forest",
    "data": [[22.0], [23.5], [0.0], [21.5], [100.0], [24.0], [-10.0], [22.8], [35.0], [23.2]],
    "threshold": 0.5
  }'
```

**Response:**

```json
{
  "object": "list",
  "data": [
    {"index": 2, "score": 0.61, "raw_score": 0.65},
    {"index": 4, "score": 0.60, "raw_score": 0.64},
    {"index": 6, "score": 0.61, "raw_score": 0.65},
    {"index": 8, "score": 0.60, "raw_score": 0.64}
  ],
  "total_count": 4,
  "model": "sensor_anomaly_detector_20260102_202438",
  "backend": "isolation_forest",
  "summary": {
    "anomalies_detected": 4,
    "threshold": 0.5
  }
}
```

The anomalies detected are:
- Index 2: 0.0°C (freezing - way below normal)
- Index 4: 100.0°C (boiling - way above normal)
- Index 6: -10.0°C (sub-freezing)
- Index 8: 35.0°C (elevated temperature)

#### Save Anomaly Model

**Endpoint:** `POST /v1/ml/anomaly/save`

**Request Body:**

```json
{
  "model": "sensor-detector_20251215_160000",
  "backend": "isolation_forest"
}
```

#### Load Anomaly Model

**Endpoint:** `POST /v1/ml/anomaly/load`

**Request Body:**

```json
{
  "model": "sensor-detector-latest",
  "backend": "isolation_forest"
}
```

#### List Anomaly Models

**Endpoint:** `GET /v1/ml/anomaly/models`

#### Delete Anomaly Model

**Endpoint:** `DELETE /v1/ml/anomaly/models/{filename}`

The `filename` includes the file extension (e.g., `sensor-detector_isolation_forest.pkl`). Anomaly models are stored as `.pkl` files under `~/.llamafarm/models/anomaly/`.

---

## Universal Runtime API

The Universal Runtime is a separate service (port 11540) that provides specialized ML endpoints for document processing, text analysis, embeddings, and anomaly detection.

**Base URL:** `http://localhost:11540`

:::tip Using Vision APIs
For OCR and document extraction, you can use either:
- **LlamaFarm API** (`/v1/vision/*`) - Accepts file uploads directly, converts PDFs to images automatically
- **Universal Runtime** (`/v1/ocr`, `/v1/documents/extract`) - Accepts base64 images or file IDs
:::

### Starting the Universal Runtime

```bash
nx start universal-runtime
```

### Universal Runtime Endpoints

| Category | Endpoint | Description |
|----------|----------|-------------|
| **Health** | `GET /health` | Runtime health and loaded models |
| **Models** | `GET /v1/models` | List currently loaded models |
| **Chat** | `POST /v1/chat/completions` | OpenAI-compatible chat completions |
| **Embeddings** | `POST /v1/embeddings` | Generate text embeddings |
| **Files** | `POST /v1/files` | Upload files for processing |
| **Files** | `GET /v1/files` | List uploaded files |
| **Files** | `GET /v1/files/{id}` | Get file metadata |
| **Files** | `GET /v1/files/{id}/images` | Get file as base64 images |
| **Files** | `DELETE /v1/files/{id}` | Delete uploaded file |
| **OCR** | `POST /v1/ocr` | Extract text from images (base64) |
| **Documents** | `POST /v1/documents/extract` | Extract structured data (base64) |
| **Classification** | `POST /v1/classify` | Classify text using pre-trained models (sentiment, etc.) |
| **Custom Classifier** | `POST /v1/classifier/fit` | Train custom classifier (SetFit few-shot) |
| **Custom Classifier** | `POST /v1/classifier/predict` | Classify with trained custom model |
| **Custom Classifier** | `POST /v1/classifier/save` | Save trained classifier |
| **Custom Classifier** | `POST /v1/classifier/load` | Load saved classifier |
| **Custom Classifier** | `GET /v1/classifier/models` | List saved classifiers |
| **Custom Classifier** | `DELETE /v1/classifier/models/{name}` | Delete saved classifier |
| **NER** | `POST /v1/ner` | Named entity recognition |
| **Reranking** | `POST /v1/rerank` | Rerank documents by relevance |
| **Anomaly** | `POST /v1/anomaly/fit` | Train anomaly detector |
| **Anomaly** | `POST /v1/anomaly/score` | Score data for anomalies |
| **Anomaly** | `POST /v1/anomaly/detect` | Detect anomalies (filtered) |
| **Anomaly** | `POST /v1/anomaly/save` | Save trained model |
| **Anomaly** | `POST /v1/anomaly/load` | Load saved model |
| **Anomaly** | `GET /v1/anomaly/models` | List saved models |
| **Anomaly** | `DELETE /v1/anomaly/models/{filename}` | Delete saved model |

:::info Classification Endpoints
- **`/v1/classify`** (Universal Runtime only) - Use pre-trained HuggingFace models for sentiment, spam detection, etc. This endpoint is NOT proxied through the main LlamaFarm server.
- **`/v1/ml/classifier/*`** (LlamaFarm Server) - Train and use custom classifiers with your own categories via SetFit. Available at `http://localhost:8000`.
:::

### Quick Examples

**OCR:**
```bash
curl -X POST http://localhost:11540/v1/ocr \
  -H "Content-Type: application/json" \
  -d '{"model": "surya", "images": ["base64..."], "languages": ["en"]}'
```

**Embeddings:**
```bash
curl -X POST http://localhost:11540/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "sentence-transformers/all-MiniLM-L6-v2", "input": "Hello world"}'
```

**Custom Classification (SetFit):**
```bash
# Train a custom classifier with few examples
curl -X POST http://localhost:11540/v1/classifier/fit \
  -H "Content-Type: application/json" \
  -d '{
    "model": "intent-classifier",
    "training_data": [
      {"text": "Book a flight", "label": "booking"},
      {"text": "Cancel my order", "label": "cancellation"}
    ]
  }'

# Classify new texts
curl -X POST http://localhost:11540/v1/classifier/predict \
  -H "Content-Type: application/json" \
  -d '{"model": "intent-classifier", "texts": ["I need to reserve a room"]}'
```

**Anomaly Detection:**
```bash
# Train
curl -X POST http://localhost:11540/v1/anomaly/fit \
  -H "Content-Type: application/json" \
  -d '{"model": "my-detector", "backend": "isolation_forest", "data": [[1,2],[3,4]]}'

# Detect
curl -X POST http://localhost:11540/v1/anomaly/detect \
  -H "Content-Type: application/json" \
  -d '{"model": "my-detector", "data": [[1,2],[100,200]]}'
```

For complete documentation, see:
- [Specialized ML Models](../models/specialized-ml.md) - OCR, documents, classification, NER, reranking
- [Anomaly Detection Guide](../models/anomaly-detection.md) - Complete anomaly detection documentation
- [Models & Runtime](../models/index.md) - Runtime configuration

---

## Next Steps

- Learn about [Configuration](../configuration/index.md)
- Explore [RAG concepts](../rag/index.md)
- Review [Examples](../examples/index.md)
- Check [Troubleshooting](../troubleshooting/index.md)
