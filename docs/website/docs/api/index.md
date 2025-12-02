# API Reference

LlamaFarm provides a comprehensive REST API for managing projects, datasets, chat interactions, and RAG (Retrieval-Augmented Generation) operations. The API follows RESTful conventions and is compatible with OpenAI's chat completion format.

## Base URL

The API is served at: `http://localhost:8000`

All versioned endpoints use the `/v1` prefix:
```
http://localhost:8000/v1
```

## Finding Your Namespace and Project Name

### Understanding Namespaces and Projects

LlamaFarm organizes your work into **namespaces** (organizational containers) and **projects** (individual configurations):

- **Namespace**: A top-level organizational unit (e.g., your username, team name, or organization)
- **Project Name**: The unique identifier for a specific LlamaFarm project within a namespace

### From Your llamafarm.yaml

The easiest way to find your namespace and project name is to check your `llamafarm.yaml` configuration file:

```yaml
version: v1
name: my-project        # This is your project name
namespace: my-org       # This is your namespace
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
- `GET /v1/projects/{namespace}/{project}/rag/databases` - List databases
- `GET /v1/projects/{namespace}/{project}/rag/databases/{database}` - Get database details
- `POST /v1/projects/{namespace}/{project}/rag/databases` - Create database
- `PATCH /v1/projects/{namespace}/{project}/rag/databases/{database}` - Update database
- `DELETE /v1/projects/{namespace}/{project}/rag/databases/{database}` - Delete database

### Tasks
- `GET /v1/projects/{namespace}/{project}/tasks/{task_id}` - Get async task status

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
- `DELETE /v1/models/{model_name}` - Delete cached model

### Health
- `GET /health` - Overall health check
- `GET /health/liveness` - Liveness probe

### System
- `GET /` - Basic hello endpoint
- `GET /info` - System information
- `GET /v1/system/version-check` - Check for CLI updates

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
  "config_template": "server"  // Optional: server, rag, or custom template name
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
- `max_tokens` (optional): Maximum tokens to generate
- `top_p` (optional): Nucleus sampling parameter
- `top_k` (optional): Top-k sampling parameter
- `rag_enabled` (optional): Enable/disable RAG (uses config default if not specified)
- `database` (optional): Database to use for RAG queries
- `rag_top_k` (optional): Number of RAG results to retrieve
- `rag_score_threshold` (optional): Minimum similarity score for RAG results

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
      "config": {"top_k": 10},
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
      "config": {"top_k": 10},
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

You can configure LlamaFarm projects to expose tools through MCP servers, giving AI agents access to filesystems, databases, APIs, and custom business logic. See the MCP section in the [Introduction](../intro.md#the-power-of-mcp-model-context-protocol) for configuration examples.

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
  { role: "user", content: "Hello!" }
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

## Next Steps

- Learn about [Configuration](../configuration/index.md)
- Explore [RAG concepts](../rag/index.md)
- Review [Examples](../examples/index.md)
- Check [Troubleshooting](../troubleshooting/index.md)
