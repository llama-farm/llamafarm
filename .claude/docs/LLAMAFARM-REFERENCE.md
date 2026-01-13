# LlamaFarm Reference Guide for Claude Code Agents

This reference is used by Claude Code agents when working with LlamaFarm projects.

---

## Documentation Links

Use these links to look up detailed information:

### Core Documentation
| Topic | URL |
|-------|-----|
| **Main Docs** | https://docs.llamafarm.dev/docs/intro |
| **Configuration (llamafarm.yaml)** | https://docs.llamafarm.dev/docs/configuration |
| **CLI Reference** | https://docs.llamafarm.dev/docs/cli |
| **GitHub Repo** | https://github.com/llama-farm/llamafarm |

### API Reference
| Topic | URL |
|-------|-----|
| **Full API Reference** | https://docs.llamafarm.dev/docs/api |
| **Chat API (OpenAI-compatible)** | https://docs.llamafarm.dev/docs/api#send-chat-message-openai-compatible |
| **Datasets API** | https://docs.llamafarm.dev/docs/api#datasets-api |
| **Tasks API** | https://docs.llamafarm.dev/docs/api#tasks-api |
| **Vision/OCR API** | https://docs.llamafarm.dev/docs/api#vision-api-ocr--document-extraction |
| **MCP Integration** | https://docs.llamafarm.dev/docs/api#mcp-resources |

### RAG (Retrieval-Augmented Generation)
| Topic | URL |
|-------|-----|
| **RAG Overview** | https://docs.llamafarm.dev/docs/rag |
| **Databases & Vector Stores** | https://docs.llamafarm.dev/docs/rag/databases |
| **Retrieval Strategies** | https://docs.llamafarm.dev/docs/rag/retrieval-strategies |
| **Advanced Retrieval (Reranking)** | https://docs.llamafarm.dev/docs/rag/advanced-retrieval |
| **Data Processing & Parsers** | https://docs.llamafarm.dev/docs/rag/data-processing |
| **Extractors** | https://docs.llamafarm.dev/docs/rag/extractors |

### ML Models & Specialized Features
| Topic | URL |
|-------|-----|
| **Models & Runtime** | https://docs.llamafarm.dev/docs/models |
| **Anomaly Detection** | https://docs.llamafarm.dev/docs/models/anomaly-detection |
| **Text Classification (SetFit)** | https://docs.llamafarm.dev/docs/models/classification |
| **Embeddings** | https://docs.llamafarm.dev/docs/models/embeddings |
| **Specialized ML** | https://docs.llamafarm.dev/docs/models/specialized-ml |
| **Semantic Router** | See [Router API](#semantic-router-api) section below |

### Local Example YAML Files
| File | Description |
|------|-------------|
| `.claude/docs/llamafarm-simple.yaml` | Minimal working config with RAG |
| `.claude/docs/llamafarm_advanced.yaml` | Production config with multi-DB, reranking, tools |
| `.claude/docs/llamafarm-router-simple.yaml` | Basic 3-route semantic router |
| `.claude/docs/llamafarm-router-healthcare.yaml` | Healthcare domain router with multiple providers |
| `.claude/docs/llamafarm-router-complexity.yaml` | Complexity-based routing example |

---

## Quick Start Commands

### Initialize a New Project
```bash
lf init my-project              # Creates llamafarm.yaml
lf init --namespace my-org      # With namespace
```

### Start Services
```bash
lf start                        # Full startup (server + RAG + Designer UI)
lf services start               # Start all services
lf services start server        # Port 8000 - FastAPI
lf services start rag           # RAG worker (Celery)
lf services start universal-runtime  # Port 11540 - ML models
```

### Development Mode (nx commands)
```bash
nx start server                 # FastAPI server (port 8000)
nx start rag                    # RAG worker
nx start universal-runtime      # ML runtime (port 11540)
nx reset                        # Reset nx cache (when things break)
```

### Check Status
```bash
lf services status              # Human readable
lf services status --json       # Machine readable
```

### Stop Services
```bash
lf services stop
# Or kill specific ports:
lsof -ti:8000 | xargs kill -9   # Kill server
lsof -ti:11540 | xargs kill -9  # Kill runtime
```

---

## Service Ports

| Service | Port | Purpose |
|---------|------|---------|
| Server (FastAPI) | 8000 | REST API, Designer UI |
| Designer UI | 7724 | Web interface (or 3123 with Docker) |
| Universal Runtime | 11540 | ML models, embeddings, OCR |
| Ollama | 11434 | Local LLM inference |
| Lemonade | 11534 | Quantized model runtime |

---

## llamafarm.yaml Schema

### Key Concept: Models + Prompts
**Models usually have one or more prompts associated with them.** The prompt defines the system behavior, and the model executes it. Configure both in llamafarm.yaml.

### File Structure Overview
```yaml
version: v1              # Always v1
name: my-project         # Project name
namespace: default       # Organization namespace

runtime: { ... }         # REQUIRED: Models configuration
prompts: [ ... ]         # REQUIRED: System prompts
rag: { ... }             # OPTIONAL: RAG configuration (databases, strategies)
datasets: [ ... ]        # OPTIONAL: Document collections (TOP LEVEL, not inside rag!)
mcp: { ... }             # OPTIONAL: MCP server integrations
```

### Environment Variables in Config
```yaml
runtime:
  models:
    - name: openai
      provider: openai
      model: gpt-4
      api_key: ${OPENAI_API_KEY}           # From environment
      api_key: ${OPENAI_API_KEY:-sk-xxx}   # With default
```

---

## ⚠️ CRITICAL: RAG Configuration (Common Mistakes)

**This section covers the most common configuration errors. Read carefully!**

### Structure Overview

The RAG section has THREE main parts that must be configured correctly:

```yaml
rag:
  default_database: main_database   # Optional: which database to query by default

  databases:                        # Vector stores for embeddings
    - name: main_database
      type: ChromaStore
      # ... database config including embedding_strategies and retrieval_strategies

  data_processing_strategies:       # How to parse/chunk documents
    - name: universal_processor
      # ... parsers and extractors

# IMPORTANT: datasets is TOP-LEVEL, NOT inside rag!
datasets:                           # Document collections linking to databases
  - name: my_docs
    database: main_database
    data_processing_strategy: universal_processor
```

### ⚠️ Common Mistake #1: datasets inside rag

**WRONG:**
```yaml
rag:
  databases: [...]
  data_processing_strategies: [...]
  datasets:  # ❌ WRONG - datasets should NOT be inside rag
    - name: my_docs
```

**CORRECT:**
```yaml
rag:
  databases: [...]
  data_processing_strategies: [...]

datasets:  # ✅ CORRECT - datasets at TOP LEVEL
  - name: my_docs
    database: main_database
    data_processing_strategy: universal_processor
```

### ⚠️ Common Mistake #2: Missing embedding_strategies in database

**WRONG:**
```yaml
rag:
  databases:
    - name: main_database
      type: ChromaStore
      default_embedding_strategy: default_embeddings  # ❌ References non-existent strategy
```

**CORRECT:**
```yaml
rag:
  databases:
    - name: main_database
      type: ChromaStore
      default_embedding_strategy: default_embeddings
      embedding_strategies:  # ✅ Must define the strategy being referenced
        - name: default_embeddings
          type: UniversalEmbedder
          priority: 0
          config:
            model: sentence-transformers/all-MiniLM-L6-v2
            dimension: 384
            batch_size: 16
            timeout: 60
            auto_pull: true
```

### ⚠️ Common Mistake #3: Missing retrieval_strategies in database

Every database needs at least one retrieval strategy:

```yaml
rag:
  databases:
    - name: main_database
      type: ChromaStore
      default_retrieval_strategy: basic_search
      retrieval_strategies:  # ✅ Must define retrieval strategies
        - name: basic_search
          type: BasicSimilarityStrategy
          config:
            distance_metric: cosine
            top_k: 10
          default: true
```

### Complete Database Configuration

```yaml
rag:
  databases:
    - name: main_database
      type: ChromaStore
      config:
        persist_directory: "./data/chroma_db"
        distance_function: cosine
        collection_name: documents
      default_embedding_strategy: default_embeddings
      default_retrieval_strategy: basic_search

      # Embedding strategies (how to vectorize text)
      embedding_strategies:
        - name: default_embeddings
          type: UniversalEmbedder
          priority: 0
          config:
            model: sentence-transformers/all-MiniLM-L6-v2
            dimension: 384
            batch_size: 16
            timeout: 60
            auto_pull: true

      # Retrieval strategies (how to search)
      retrieval_strategies:
        - name: basic_search
          type: BasicSimilarityStrategy
          config:
            distance_metric: cosine
            top_k: 10
          default: true

        - name: reranked_search
          type: CrossEncoderRerankedStrategy
          config:
            model_name: cross-encoder/ms-marco-MiniLM-L-6-v2
            initial_k: 30
            final_k: 5
            relevance_threshold: 0.0
            timeout: 60
          default: false
```

### Data Processing Strategies

Define how documents are parsed and chunked:

```yaml
rag:
  data_processing_strategies:
    - name: universal_processor
      description: "Handles PDFs, Word docs, Markdown, CSV, Excel, text files"
      parsers:
        # PDF - primary parser
        - type: PDFParser_LlamaIndex
          file_include_patterns: ["*.pdf", "*.PDF"]
          priority: 10
          config:
            chunk_strategy: semantic
            chunk_size: 300
            chunk_overlap: 50
            extract_metadata: true
            extract_tables: true

        # PDF - fallback parser
        - type: PDFParser_PyPDF2
          file_include_patterns: ["*.pdf", "*.PDF"]
          priority: 50  # Higher = lower priority (fallback)
          config:
            chunk_size: 300
            chunk_overlap: 50
            chunk_strategy: paragraphs
            extract_metadata: true
            combine_pages: false  # CRITICAL: Must be false to enable chunking

        # Word documents
        - type: DocxParser_LlamaIndex
          file_include_patterns: ["*.docx", "*.DOCX"]
          priority: 10
          config:
            chunk_size: 500
            chunk_overlap: 100
            extract_metadata: true
            extract_tables: true

        # Markdown
        - type: MarkdownParser_Python
          file_include_patterns: ["*.md", "*.markdown", "README*"]
          priority: 10
          config:
            chunk_size: 400
            chunk_strategy: sections
            extract_metadata: true
            extract_code_blocks: true

        # Plain text (catch-all fallback)
        - type: TextParser_Python
          file_include_patterns: ["*.txt", "*.log", "*.json", "*.yaml"]
          priority: 50
          config:
            encoding: utf-8
            chunk_size: 500
            chunk_overlap: 100
            chunk_strategy: sentences
            clean_text: true
            extract_metadata: true

      extractors:
        - type: EntityExtractor
          file_include_patterns: ["*"]
          priority: 20
          config:
            entity_types: [PERSON, ORG, GPE, DATE, PRODUCT, MONEY]
            use_fallback: true

        - type: KeywordExtractor
          file_include_patterns: ["*"]
          priority: 30
          config:
            algorithm: yake
            max_keywords: 10
            min_keyword_length: 3
```

### Dataset Configuration (TOP LEVEL!)

```yaml
# datasets is at the TOP LEVEL of llamafarm.yaml, NOT inside rag!
datasets:
  - name: research_docs
    database: main_database              # Must match a database name in rag.databases
    data_processing_strategy: universal_processor  # Must match a strategy in rag.data_processing_strategies

  - name: proposals
    database: proposals_db
    data_processing_strategy: proposal_processor
```

---

## Example YAML Configurations

### Simple Configuration (Minimal RAG)

See `.claude/docs/llamafarm-simple.yaml` for a complete working example.

Key points:
- Single database with basic search
- Universal processor handles all file types
- Single default dataset

### Advanced Configuration (Multi-Database with Reranking)

See `.claude/docs/llamafarm_advanced.yaml` for a complete production example.

Key points:
- Multiple databases (proposals_db, company_docs)
- Multiple models with tools
- Reranking for better search quality
- Multiple data processing strategies
- Rich prompt library

---

## OpenAI-Compatible Chat API (Full Parameters)

### Endpoint
```
POST /v1/projects/{namespace}/{project}/chat/completions
```

### ⚠️ CRITICAL: Enabling RAG in Chat Requests

**To use RAG (document retrieval) in chat, you MUST include these parameters:**

```json
{
  "messages": [{"role": "user", "content": "Your question here"}],
  "model": "smart",
  "rag_enabled": true,           // ← REQUIRED to enable RAG
  "database": "main_database"    // ← Must match database name in llamafarm.yaml
}
```

**Common mistake:** Forgetting `rag_enabled: true` - without this, no document retrieval happens!

### All Parameters

```json
{
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello"}
  ],

  "model": "smart",
  "stream": false,

  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,

  "rag_enabled": true,
  "database": "main_db",
  "rag_top_k": 5,
  "rag_score_threshold": 0.7,
  "rag_queries": ["custom query 1", "custom query 2"],

  "think": true,
  "thinking_budget": 1000,

  "tools": [...],
  "tool_choice": "auto"
}
```

### Parameter Reference

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `messages` | array | **Yes** | Chat messages with role and content |
| `model` | string | No | Model name from llamafarm.yaml config (uses default if omitted) |
| `stream` | boolean | No | Enable SSE streaming responses |
| `temperature` | number | No | Sampling temperature (0.0-2.0) |
| `top_p` | number | No | Nucleus sampling threshold |
| `top_k` | number | No | Top-k sampling |

### RAG Parameters (Automatic Context Retrieval)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rag_enabled` | boolean | `false` | **Must be true to enable RAG retrieval** |
| `database` | string | Default DB | Vector database to query (must exist in llamafarm.yaml) |
| `rag_top_k` | int | 5 | Number of documents to retrieve |
| `rag_score_threshold` | float | 0.0 | Minimum similarity score (0-1). Documents below this are filtered out |
| `rag_queries` | array | null | Custom queries to use instead of user message. Supports multiple concurrent queries with automatic deduplication |

**How RAG works when enabled:**
1. Takes user message (or `rag_queries` if provided) as search query
2. Queries the specified vector database using configured retrieval strategy
3. Retrieves top_k documents above score_threshold
4. Injects retrieved content as context before sending to LLM
5. LLM generates response using both context and user message

### Example: Simple Chat with RAG

```bash
curl -X POST "http://localhost:8000/v1/projects/default/myproject/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What are the safety protocols?"}],
    "rag_enabled": true,
    "database": "safety_docs",
    "rag_top_k": 5
  }'
```

### Example: Chat with Custom RAG Queries

Use `rag_queries` to search for specific terms different from the user message:

```bash
curl -X POST "http://localhost:8000/v1/projects/default/myproject/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Summarize the key points"}],
    "rag_enabled": true,
    "database": "documents",
    "rag_queries": ["safety procedures", "emergency protocols", "compliance requirements"],
    "rag_top_k": 10
  }'
```

### Example: Streaming with RAG

```bash
curl -X POST "http://localhost:8000/v1/projects/default/myproject/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain the architecture"}],
    "stream": true,
    "rag_enabled": true,
    "database": "tech_docs"
  }'
```

### Thinking/Reasoning Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `think` | boolean | false | Enable chain-of-thought reasoning (Qwen3, Claude, etc.) |
| `thinking_budget` | int | 1024 | Maximum tokens for reasoning steps (separate from response) |

### Session Headers

| Header | Description |
|--------|-------------|
| `X-Session-ID` | Pass an ID to maintain conversation history across requests |
| `X-No-Session` | Set to any value for stateless mode (no conversation memory) |

**Session behavior:**
- Sessions auto-expire after 30 minutes of inactivity
- Response includes `X-Session-ID` header in stateful mode
- Without either header, a new session is created automatically

### Python Example: Chat with RAG

```python
import httpx

async def chat_with_rag(question: str, database: str = "documents"):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/v1/projects/default/myproject/chat/completions",
            json={
                "messages": [{"role": "user", "content": question}],
                "rag_enabled": True,  # ← Don't forget this!
                "database": database,
                "rag_top_k": 5,
                "rag_score_threshold": 0.5
            },
            headers={"X-Session-ID": "my-session-123"}
        )
        return response.json()

# Usage
result = await chat_with_rag("What are the project requirements?", "requirements_db")
print(result["choices"][0]["message"]["content"])
```

---

## Inline Tool Definitions in llamafarm.yaml (RECOMMENDED)

**The preferred way to define tools is in llamafarm.yaml, not per-request.** Tools are defined under the model configuration using the `tools` array.

### Defining Tools in YAML Config

```yaml
runtime:
  default_model: triage_agent
  models:
    - name: triage_agent
      provider: ollama
      model: qwen3:8b
      base_url: http://127.0.0.1:11434
      tool_call_strategy: native_api  # REQUIRED for tool calling
      tools:
        - type: function
          name: send_medevac
          description: Request emergency medical evacuation for a soldier
          parameters:
            type: object
            required:
              - soldier_id
              - lat
              - lon
              - condition
              - priority
            properties:
              soldier_id:
                type: string
                description: Unique identifier for the soldier
              lat:
                type: number
                description: GPS latitude of soldier's location
              lon:
                type: number
                description: GPS longitude of soldier's location
              condition:
                type: string
                description: Medical condition summary
              priority:
                type: string
                enum: ["routine", "priority", "urgent", "immediate"]
                description: Evacuation priority level

        - type: function
          name: send_supplies
          description: Request supply delivery to a location
          parameters:
            type: object
            required:
              - soldier_id
              - lat
              - lon
              - supply_type
            properties:
              soldier_id:
                type: string
                description: Soldier requesting supplies
              lat:
                type: number
                description: GPS latitude for delivery
              lon:
                type: number
                description: GPS longitude for delivery
              supply_type:
                type: string
                enum: ["ammo", "water", "medical", "food", "batteries"]
              quantity:
                type: integer
                description: Number of units needed

        - type: function
          name: radio_query
          description: Send a follow-up question to a soldier via radio
          parameters:
            type: object
            required:
              - soldier_id
              - question
            properties:
              soldier_id:
                type: string
                description: Soldier to query
              question:
                type: string
                description: Question to ask

        - type: function
          name: ignore
          description: Acknowledge situation but take no action
          parameters:
            type: object
            required:
              - reason
            properties:
              reason:
                type: string
                description: Why no action is needed
```

### Tool Schema Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Must be `"function"` |
| `name` | string | Yes | Unique identifier (e.g., `send_medevac`) |
| `description` | string | Yes | Human-readable description for the model |
| `parameters` | object | Yes | JSON Schema defining input parameters |
| `parameters.type` | string | Yes | Usually `"object"` |
| `parameters.required` | array | No | List of required parameter names |
| `parameters.properties` | object | Yes | Parameter definitions |

### Parameter Types

| Type | Description | Example |
|------|-------------|---------|
| `string` | Text values | `"soldier_id": {"type": "string"}` |
| `integer` | Whole numbers | `"quantity": {"type": "integer"}` |
| `number` | Decimal numbers | `"lat": {"type": "number"}` |
| `boolean` | True/false | `"urgent": {"type": "boolean"}` |
| `array` | Lists | `"items": {"type": "array", "items": {"type": "string"}}` |
| `enum` | Fixed choices | `"priority": {"type": "string", "enum": ["low", "high"]}` |

### Combining with MCP Servers

You can use both inline tools AND MCP servers on the same model:

```yaml
mcp:
  servers:
    - name: filesystem
      transport: stdio
      command: npx
      args: ['-y', '@modelcontextprotocol/server-filesystem', '/data']

runtime:
  models:
    - name: agent
      provider: ollama
      model: qwen3:8b
      mcp_servers:
        - filesystem           # MCP server tools
      tools:                   # PLUS inline tools
        - type: function
          name: custom_action
          description: Custom action defined inline
          parameters:
            type: object
            properties:
              action:
                type: string
```

### Tool Execution Flow

1. Model generates response with `tool_calls` containing name and arguments
2. Your application receives the tool call
3. Execute the tool in your code (call API, run script, etc.)
4. Send results back to the conversation as `role: "tool"` message
5. Model continues with the tool result
6. Maximum 10 iterations per conversation to prevent infinite loops

### Example: Complete Agent Configuration

```yaml
version: v1
name: tactical-monitor
namespace: tactical

runtime:
  default_model: triage_agent
  models:
    - name: triage_agent
      provider: ollama
      model: qwen3:8b
      base_url: http://127.0.0.1:11434
      tool_call_strategy: native_api
      tools:
        - type: function
          name: send_medevac
          description: Request emergency medical evacuation
          parameters:
            type: object
            required: [soldier_id, lat, lon, condition, priority]
            properties:
              soldier_id: {type: string}
              lat: {type: number}
              lon: {type: number}
              condition: {type: string}
              priority: {type: string, enum: [routine, priority, urgent, immediate]}

        - type: function
          name: send_supplies
          description: Request supply delivery
          parameters:
            type: object
            required: [soldier_id, lat, lon, supply_type]
            properties:
              soldier_id: {type: string}
              lat: {type: number}
              lon: {type: number}
              supply_type: {type: string, enum: [ammo, water, medical, food]}
              quantity: {type: integer}

        - type: function
          name: radio_query
          description: Send follow-up question via radio
          parameters:
            type: object
            required: [soldier_id, question]
            properties:
              soldier_id: {type: string}
              question: {type: string}

        - type: function
          name: ignore
          description: Acknowledge but take no action
          parameters:
            type: object
            required: [reason]
            properties:
              reason: {type: string}

prompts:
  - name: triage_system
    messages:
      - role: system
        content: |
          You are a military medical triage AI. Analyze biometric data and radio
          communications to make decisions. You have access to the following tools:

          - send_medevac: Request medical evacuation (use for life-threatening conditions)
          - send_supplies: Request supply delivery
          - radio_query: Ask follow-up questions to soldiers
          - ignore: Acknowledge situation, no action needed

          Always explain your reasoning before calling a tool.
```

---

## Inline Tool Calling in API Requests (Alternative)

You can also define tools per-request if you need dynamic tool definitions:

```json
{
  "messages": [{"role": "user", "content": "What's the weather in NYC?"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "City name"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

### Tool Choice Options

| Value | Behavior |
|-------|----------|
| `"auto"` | Model decides whether to call tools |
| `"none"` | Never call tools |
| `{"type": "function", "function": {"name": "..."}}` | Force specific tool |

### Handling Tool Calls

1. Model returns `tool_calls` in response
2. Execute the tool in your application
3. Send results back with `role: "tool"`

```json
{
  "messages": [
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "call_123", "content": "{\"temp\": 72}"}
  ]
}
```

---

## Datasets API

### Workflow: Upload First, Process Later

Files are stored immediately but require explicit processing to be vectorized.

### List Datasets
```bash
curl "http://localhost:8000/v1/projects/{namespace}/{project}/datasets"
```

### Create Dataset
```bash
curl -X POST "http://localhost:8000/v1/projects/{namespace}/{project}/datasets" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "research_docs",
    "data_processing_strategy": "default",
    "database": "main_db"
  }'
```

### Upload Files
```bash
curl -X POST "http://localhost:8000/v1/projects/{namespace}/{project}/datasets/{dataset}/data" \
  -F "files=@document1.pdf" \
  -F "files=@document2.txt"
```

### Process Dataset (Vectorize)
```bash
curl -X POST "http://localhost:8000/v1/projects/{namespace}/{project}/datasets/{dataset}/actions" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "process"}'
```

Returns a `task_id` for tracking.

### Delete Dataset
```bash
curl -X DELETE "http://localhost:8000/v1/projects/{namespace}/{project}/datasets/{dataset}"
```

### Delete Single File
```bash
curl -X DELETE "http://localhost:8000/v1/projects/{namespace}/{project}/datasets/{dataset}/data/{file_hash}"
```

---

## Tasks API (Async Operations)

### Check Task Status
```bash
curl "http://localhost:8000/v1/projects/{namespace}/{project}/tasks/{task_id}"
```

### Response
```json
{
  "task_id": "abc-123",
  "state": "STARTED",
  "meta": {
    "current": 5,
    "total": 10
  },
  "result": null,
  "error": null
}
```

### Task States

| State | Description |
|-------|-------------|
| `PENDING` | Queued, not started |
| `STARTED` | Currently processing |
| `SUCCESS` | Completed successfully |
| `FAILURE` | Failed with error |
| `RETRY` | Retrying after failure |

### Cancel Task
```bash
curl -X DELETE "http://localhost:8000/v1/projects/{namespace}/{project}/tasks/{task_id}"
```

Cancellation also removes processed chunks from vector store.

---

## Retrieval Strategies (Advanced RAG)

### BasicSimilarityStrategy
Simple vector search.

```yaml
retrieval_strategies:
  - name: basic
    type: BasicSimilarityStrategy
    config:
      top_k: 10
      distance_metric: cosine  # cosine, euclidean, manhattan, dot
      score_threshold: 0.5
```

### MetadataFilteredStrategy
Filter by document metadata before/after retrieval.

```yaml
retrieval_strategies:
  - name: filtered
    type: MetadataFilteredStrategy
    config:
      top_k: 10
      filter_mode: pre  # pre or post
      filters:
        department: "engineering"
        year: {"$gte": 2023}
```

### CrossEncoderRerankedStrategy
Two-stage: vector search + neural reranking. **10-100x faster than LLM reranking.**

```yaml
retrieval_strategies:
  - name: reranked
    type: CrossEncoderRerankedStrategy
    config:
      model_name: ms-marco-MiniLM-L-6-v2
      initial_k: 30      # Candidates from vector search
      final_k: 10        # Results after reranking
      relevance_threshold: 0.5
      timeout: 60
```

### Recommended Reranking Models

| Model | Size | Speed | Notes |
|-------|------|-------|-------|
| `ms-marco-MiniLM-L-6-v2` | 90MB | 300-500 docs/sec | **Recommended default** |
| `bge-reranker-v2-m3` | 560MB | 100-200 docs/sec | Multilingual, highest accuracy |
| `bge-reranker-base` | 280MB | 150-300 docs/sec | Good balance |

### MultiQueryStrategy
Generate query variations to improve recall.

```yaml
retrieval_strategies:
  - name: multi_query
    type: MultiQueryStrategy
    config:
      num_queries: 3
      aggregation_method: reciprocal_rank  # max, mean, weighted, reciprocal_rank
```

### HybridUniversalStrategy
Combine 2-5 strategies with fusion.

```yaml
retrieval_strategies:
  - name: hybrid
    type: HybridUniversalStrategy
    config:
      strategies:
        - basic_search
        - filtered_search
      fusion_method: weighted_average  # weighted_average, rank_fusion, score_fusion
      weights: [0.6, 0.4]
```

### MultiTurnRAGStrategy
For complex multi-part queries.

```yaml
retrieval_strategies:
  - name: multi_turn
    type: MultiTurnRAGStrategy
    config:
      complexity_threshold: 50  # Characters
      enable_reranking: true
      sub_query_top_k: 5
```

---

## Extractors (Document Processing)

Extractors enrich document chunks with metadata during processing.

### Available Extractors

| Extractor | Purpose | Output |
|-----------|---------|--------|
| `EntityExtractor` | NER (people, orgs, dates) | PERSON, ORG, GPE, DATE, EMAIL, PHONE |
| `KeywordExtractor` | Key terms | Keywords via RAKE, YAKE, TF-IDF, TextRank |
| `DateTimeExtractor` | Parse dates/times | Normalized datetime objects |
| `HeadingExtractor` | Document outline | Hierarchical headings |
| `LinkExtractor` | URLs, emails | Validated links |
| `PatternExtractor` | Regex patterns | Custom matches (SSN, IP, etc.) |
| `ContentStatisticsExtractor` | Readability | Scores, word counts |
| `SummaryExtractor` | Summaries | Extractive summaries |
| `TableExtractor` | Tables | CSV, dict, markdown formats |

### Configuration Example

```yaml
data_processing_strategies:
  - name: enriched
    type: DefaultDataProcessor
    extractors:
      - type: EntityExtractor
        config:
          entity_types: [PERSON, ORG, DATE]
      - type: KeywordExtractor
        config:
          algorithm: rake
          max_keywords: 10
      - type: DateTimeExtractor
        config:
          timezone: UTC
```

### Using Extracted Metadata for Filtering

After extraction, query with metadata filters:

```bash
curl -X POST ".../rag/query" \
  -d '{
    "query": "safety procedures",
    "filters": {
      "entities.ORG": "OSHA",
      "keywords": {"$contains": "safety"}
    }
  }'
```

---

## ML API - Anomaly Detection

**All ML endpoints are on the main LlamaFarm server (port 8000), NOT Universal Runtime.**

### Model Versioning

Models support automatic versioning:
- **Default behavior**: Models saved with timestamps (e.g., `my-model_20251215_155054`) when `overwrite: false`
- **Latest resolution**: Use `-latest` suffix to access newest version (e.g., `sensor-detector-latest`)

### Endpoints Overview

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/ml/anomaly/fit` | POST | Train detector on normal data |
| `/v1/ml/anomaly/score` | POST | Score all points (returns is_anomaly for each) |
| `/v1/ml/anomaly/detect` | POST | Return only anomalies |
| `/v1/ml/anomaly/save` | POST | Persist model to disk |
| `/v1/ml/anomaly/load` | POST | Restore model from storage |
| `/v1/ml/anomaly/models` | GET | List saved models |
| `/v1/ml/anomaly/models/{filename}` | DELETE | Remove a model |

### Complete Workflow Example

```bash
# Step 1: Train on normal data
curl -X POST "http://localhost:8000/v1/ml/anomaly/fit" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "api-detector",
    "backend": "isolation_forest",
    "data": [
      {"response_time_ms": 100, "bytes": 1024, "method": "GET"},
      {"response_time_ms": 105, "bytes": 1100, "method": "POST"},
      {"response_time_ms": 98, "bytes": 980, "method": "GET"}
    ],
    "schema": {
      "response_time_ms": "numeric",
      "bytes": "numeric",
      "method": "label"
    },
    "contamination": 0.1,
    "overwrite": false
  }'

# Response includes versioned model name:
# {"model": "api-detector_20251215_155054", ...}

# Step 2: Detect anomalies (use -latest to get newest version)
curl -X POST "http://localhost:8000/v1/ml/anomaly/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "api-detector-latest",
    "data": [
      {"response_time_ms": 100, "bytes": 1024, "method": "GET"},
      {"response_time_ms": 9000, "bytes": 500000, "method": "GET"}
    ]
  }'

# Step 3: Save the trained model
curl -X POST "http://localhost:8000/v1/ml/anomaly/save" \
  -H "Content-Type: application/json" \
  -d '{"model": "api-detector-latest"}'

# Step 4: Load model (after restart)
curl -X POST "http://localhost:8000/v1/ml/anomaly/load" \
  -H "Content-Type: application/json" \
  -d '{"model": "api-detector-latest"}'

# Step 5: List all saved models
curl "http://localhost:8000/v1/ml/anomaly/models"

# Step 6: Score all points (get is_anomaly for each)
curl -X POST "http://localhost:8000/v1/ml/anomaly/score" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "api-detector-latest",
    "data": [...]
  }'

# Step 7: Delete model
curl -X DELETE "http://localhost:8000/v1/ml/anomaly/models/api-detector_isolation_forest.joblib"
```

### Response Formats

**Training Response:**
```json
{
  "object": "fit_result",
  "model": "api-detector",
  "backend": "isolation_forest",
  "samples_fitted": 3,
  "training_time_ms": 45.2,
  "status": "fitted"
}
```

**Score Response:**
```json
{
  "object": "list",
  "data": [
    {"index": 0, "score": 0.12, "is_anomaly": false, "raw_score": -0.15},
    {"index": 1, "score": 0.89, "is_anomaly": true, "raw_score": 0.72}
  ],
  "summary": {
    "total_points": 2,
    "anomaly_count": 1,
    "anomaly_rate": 0.5,
    "threshold": 0.5
  }
}
```

### Supported Backends
| Backend | Best For |
|---------|----------|
| `isolation_forest` | General purpose, fast, high-dimensional |
| `one_class_svm` | Tight clusters, clear boundaries |
| `local_outlier_factor` | Varying densities, local patterns |
| `autoencoder` | Complex patterns, neural approach |

### Schema Encoding Types
| Type | Use Case |
|------|----------|
| `numeric` | Direct numeric values (response times, metrics) |
| `hash` | High-cardinality strings (IPs, user agents) |
| `label` | Category-to-integer (HTTP methods, status codes) |
| `onehot` | Low-cardinality enumerations |
| `binary` | Boolean values (0/1) |
| `frequency` | Occurrence-based encoding (rare vs common) |

---

## ML API - Text Classification (SetFit)

**All ML endpoints are on the main LlamaFarm server (port 8000), NOT Universal Runtime.**

Train custom classifiers with as few as 8-16 examples per class using SetFit.

### Model Versioning

Same versioning as anomaly detection:
- **Default behavior**: Models saved with timestamps when `overwrite: false`
- **Latest resolution**: Use `-latest` suffix (e.g., `intent-classifier-latest`)

### Endpoints Overview

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/ml/classifier/fit` | POST | Train classifier on labeled examples |
| `/v1/ml/classifier/predict` | POST | Classify texts using trained model |
| `/v1/ml/classifier/save` | POST | Persist model to disk |
| `/v1/ml/classifier/load` | POST | Restore model from storage |
| `/v1/ml/classifier/models` | GET | List all saved classifiers |
| `/v1/ml/classifier/models/{name}` | DELETE | Remove a saved classifier |

### Complete Workflow Example

```bash
# Step 1: Train classifier
curl -X POST "http://localhost:8000/v1/ml/classifier/fit" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "intent-classifier",
    "base_model": "sentence-transformers/all-MiniLM-L6-v2",
    "training_data": [
      {"text": "I need to book a flight to NYC", "label": "booking"},
      {"text": "Reserve a hotel room for next week", "label": "booking"},
      {"text": "Can I get a table for two tonight?", "label": "booking"},
      {"text": "Cancel my reservation please", "label": "cancellation"},
      {"text": "I want to cancel my booking", "label": "cancellation"},
      {"text": "Please remove my appointment", "label": "cancellation"},
      {"text": "What is the weather like?", "label": "other"},
      {"text": "Tell me a joke", "label": "other"}
    ],
    "num_iterations": 20,
    "batch_size": 16,
    "overwrite": false
  }'

# Response includes versioned model and training stats:
# {
#   "model": "intent-classifier_20251215_155054",
#   "sample_count": 8,
#   "labels": ["booking", "cancellation", "other"],
#   "training_time_ms": 2500,
#   "auto_save_path": "/path/to/models/..."
# }

# Step 2: Make predictions (use -latest for newest version)
curl -X POST "http://localhost:8000/v1/ml/classifier/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "intent-classifier-latest",
    "texts": [
      "I want to book a car for tomorrow",
      "Please cancel everything",
      "How are you doing?"
    ]
  }'

# Response includes all class scores:
# {
#   "predictions": [
#     {"text": "...", "label": "booking", "score": 0.92, "all_scores": {"booking": 0.92, "cancellation": 0.05, "other": 0.03}},
#     ...
#   ]
# }

# Step 3: Save for production
curl -X POST "http://localhost:8000/v1/ml/classifier/save" \
  -H "Content-Type: application/json" \
  -d '{"model": "intent-classifier-latest"}'

# Step 4: Reload after restart
curl -X POST "http://localhost:8000/v1/ml/classifier/load" \
  -H "Content-Type: application/json" \
  -d '{"model": "intent-classifier-latest"}'

# Step 5: List all classifiers
curl "http://localhost:8000/v1/ml/classifier/models"

# Step 6: Delete a classifier
curl -X DELETE "http://localhost:8000/v1/ml/classifier/models/intent-classifier"
```

### Response Formats

**Training Response:**
```json
{
  "object": "fit_result",
  "model": "intent-classifier",
  "base_model": "sentence-transformers/all-MiniLM-L6-v2",
  "samples_fitted": 8,
  "num_classes": 3,
  "labels": ["booking", "cancellation", "other"],
  "training_time_ms": 1234.56,
  "status": "fitted"
}
```

**Prediction Response:**
```json
{
  "object": "list",
  "data": [
    {
      "text": "I want to book a car for tomorrow",
      "label": "booking",
      "score": 0.94,
      "all_scores": {
        "booking": 0.94,
        "cancellation": 0.03,
        "other": 0.03
      }
    }
  ],
  "model": "intent-classifier"
}
```

### Recommended Base Models
| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `all-MiniLM-L6-v2` | 80MB | Fast | Good |
| `all-mpnet-base-v2` | 420MB | Medium | Better |
| `BAAI/bge-small-en-v1.5` | 130MB | Fast | Good |
| `BAAI/bge-base-en-v1.5` | 440MB | Medium | Better |

---

## Named Entity Recognition (NER)

```bash
curl -X POST "http://localhost:11540/v1/ner" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "John Smith works at Acme Corp in New York"
  }'
```

### Entity Types
PERSON, ORG, GPE (places), DATE, TIME, MONEY, EMAIL, PHONE, URL, VERSION, PRODUCT

---

## Reranking API

```bash
curl -X POST "http://localhost:11540/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "search query",
    "documents": ["doc1", "doc2", "doc3"],
    "model": "ms-marco-MiniLM-L-6-v2",
    "top_k": 3
  }'
```

---

## Lightweight Database Stack (DuckDB-Centric)

**Philosophy**: One embedded database handles 90% of use cases. No servers to manage.

### Database Selection Guide

| Data Type | Recommended | Extension | Install |
|-----------|-------------|-----------|---------|
| Relational/OLAP | **DuckDB** | (built-in) | `pip install duckdb` |
| Vector/Embeddings | **DuckDB** | `vss` | `INSTALL vss; LOAD vss;` |
| Spatial/Geo | **DuckDB** | `spatial` | `INSTALL spatial; LOAD spatial;` |
| Time-series | **DuckDB** | (built-in) | Native timestamps, window functions |
| Graph | **DuckDB** | `duckpgq` | `INSTALL duckpgq; LOAD duckpgq;` |
| Full-text search | **DuckDB** | `fts` | `INSTALL fts; LOAD fts;` |
| Simple cache | **SQLite** | (in-memory) | `sqlite3 :memory:` |

### Why DuckDB?

- **Zero dependencies**: Single file, in-process
- **Blazing fast**: Columnar storage, vectorized execution
- **Multi-database joins**: Attach SQLite, PostgreSQL, MySQL, Parquet, CSV
- **Extensions**: Vector search, spatial, graph, and more
- **Cross-platform**: macOS, Linux, Windows, WASM

### DuckDB Setup

```python
import duckdb

# Create database (or :memory: for temp)
conn = duckdb.connect('myapp.duckdb')

# Install extensions
conn.execute("INSTALL vss; LOAD vss;")      # Vector similarity
conn.execute("INSTALL spatial; LOAD spatial;")  # Geo/spatial
conn.execute("INSTALL duckpgq; LOAD duckpgq;")  # Graph queries
conn.execute("INSTALL fts; LOAD fts;")      # Full-text search
```

### Vector Search with DuckDB

```python
import duckdb

conn = duckdb.connect()
conn.execute("INSTALL vss; LOAD vss;")

# Create table with embeddings
conn.execute("""
    CREATE TABLE documents (
        id INTEGER PRIMARY KEY,
        content TEXT,
        embedding FLOAT[384]  -- Dimension matches your model
    )
""")

# Create vector index
conn.execute("""
    CREATE INDEX doc_idx ON documents
    USING HNSW (embedding) WITH (metric = 'cosine')
""")

# Query similar documents
results = conn.execute("""
    SELECT id, content, array_distance(embedding, ?::FLOAT[384]) as distance
    FROM documents
    ORDER BY distance
    LIMIT 5
""", [query_embedding]).fetchall()
```

### Graph Queries with DuckPGQ

```python
# Using SQL/PGQ syntax for graph queries
conn.execute("INSTALL duckpgq; LOAD duckpgq;")

# Define graph schema
conn.execute("""
    CREATE PROPERTY GRAPH social_network
    VERTEX TABLES (users)
    EDGE TABLES (
        friendships SOURCE KEY (user_id) REFERENCES users (id)
                    DESTINATION KEY (friend_id) REFERENCES users (id)
    )
""")

# Find paths between users (SQL/PGQ syntax)
conn.execute("""
    SELECT p.name AS person, f.name AS friend
    FROM GRAPH_TABLE (social_network
        MATCH (p:users)-[r:friendships]->(f:users)
        WHERE p.id = 1
        COLUMNS (p.name, f.name)
    )
""")

# Or use recursive CTEs with USING KEY for graph algorithms
conn.execute("""
    WITH RECURSIVE paths AS USING KEY (target) (
        SELECT source AS target, 0 AS distance
        FROM edges WHERE source = 1
        UNION ALL
        SELECT e.target, p.distance + 1
        FROM paths p JOIN edges e ON p.target = e.source
        WHERE p.distance < 5
    )
    SELECT * FROM paths
""")
```

### Spatial/Geo Queries

```python
conn.execute("INSTALL spatial; LOAD spatial;")

# Create table with geometry
conn.execute("""
    CREATE TABLE locations (
        id INTEGER,
        name TEXT,
        geom GEOMETRY
    )
""")

# Insert points
conn.execute("""
    INSERT INTO locations VALUES
    (1, 'HQ', ST_Point(-122.4194, 37.7749)),
    (2, 'Base', ST_Point(-122.4089, 37.7837))
""")

# Find nearby locations
conn.execute("""
    SELECT name, ST_Distance(geom, ST_Point(-122.42, 37.78)) as dist
    FROM locations
    ORDER BY dist
    LIMIT 5
""")
```

### Time-Series Queries

```python
# DuckDB handles time-series natively with window functions
conn.execute("""
    CREATE TABLE sensor_data (
        ts TIMESTAMP,
        sensor_id TEXT,
        value DOUBLE
    )
""")

# Rolling average, lag, lead
conn.execute("""
    SELECT
        ts,
        sensor_id,
        value,
        AVG(value) OVER (
            PARTITION BY sensor_id
            ORDER BY ts
            ROWS BETWEEN 10 PRECEDING AND CURRENT ROW
        ) as rolling_avg,
        LAG(value, 1) OVER (PARTITION BY sensor_id ORDER BY ts) as prev_value
    FROM sensor_data
    WHERE ts > NOW() - INTERVAL '1 hour'
""")
```

### Multi-Database Joins

```python
# Attach multiple databases and join across them
conn.execute("ATTACH 'sqlite_data.db' AS sqlite_db (TYPE SQLITE)")
conn.execute("ATTACH 'postgresql://user:pass@host/db' AS pg_db (TYPE POSTGRES)")

# Join data from SQLite, PostgreSQL, and DuckDB
conn.execute("""
    SELECT
        d.id,
        s.name,
        p.metadata
    FROM main.documents d
    JOIN sqlite_db.users s ON d.user_id = s.id
    JOIN pg_db.extra_info p ON d.id = p.doc_id
""")

# Also works with Parquet, CSV, JSON files directly
conn.execute("""
    SELECT * FROM 'data/*.parquet'
    JOIN 'users.csv' USING (id)
""")
```

### Hybrid Architecture (Simplified)

```
┌─────────────────────────────────────────────────────┐
│                    Your App                          │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌───────────────┐ ┌─────────┐ ┌───────────────┐
│    DuckDB     │ │LlamaFarm│ │ External APIs │
│ (all storage) │ │(AI/ML)  │ │  (optional)   │
│               │ │         │ │               │
│ • Relational  │ │ • LLM   │ │ • PostgreSQL  │
│ • Vector      │ │ • RAG   │ │ • S3/GCS      │
│ • Graph       │ │ • ML    │ │ • REST APIs   │
│ • Geo         │ │ • OCR   │ │               │
│ • Time-series │ │         │ │               │
└───────────────┘ └─────────┘ └───────────────┘
```

### Example: Biometrics with DuckDB + LlamaFarm

```python
import duckdb
import httpx

# Initialize DuckDB with extensions
conn = duckdb.connect('monitoring.duckdb')
conn.execute("INSTALL vss; LOAD vss;")

# Create tables
conn.execute("""
    CREATE TABLE IF NOT EXISTS vitals (
        ts TIMESTAMP DEFAULT NOW(),
        soldier_id TEXT,
        heart_rate DOUBLE,
        blood_pressure DOUBLE,
        temperature DOUBLE,
        location GEOMETRY
    )
""")

# Query recent vitals with rolling stats
def get_recent_vitals(soldier_id: str, minutes: int = 5):
    return conn.execute("""
        SELECT
            ts, heart_rate, blood_pressure, temperature,
            AVG(heart_rate) OVER (ORDER BY ts ROWS 10 PRECEDING) as hr_avg,
            STDDEV(heart_rate) OVER (ORDER BY ts ROWS 10 PRECEDING) as hr_std
        FROM vitals
        WHERE soldier_id = ?
        AND ts > NOW() - INTERVAL ? MINUTE
        ORDER BY ts DESC
    """, [soldier_id, minutes]).fetchall()

# Anomaly detection via LlamaFarm
async def detect_anomalies(data: list):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:11540/v1/anomaly/detect",
            json={"data": data, "model_id": "vitals_detector"}
        )
        return resp.json()

# LLM decision via LlamaFarm
async def get_recommendation(context: str):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:8000/v1/projects/military/monitor/chat/completions",
            json={
                "messages": [{"role": "user", "content": context}],
                "rag_enabled": True,
                "database": "medical_protocols"
            }
        )
        return resp.json()
```

### When to Use External Databases

Only reach for external DBs when you need:

| Need | External Option | Why Not DuckDB |
|------|-----------------|----------------|
| High write concurrency | PostgreSQL, MySQL | DuckDB is OLAP-optimized |
| Distributed clusters | CockroachDB, TiDB | DuckDB is single-node |
| Real-time streaming | Kafka, Redis Streams | DuckDB is batch-oriented |
| Enterprise features | Oracle, SQL Server | Compliance requirements |

### Resources

- [DuckDB Documentation](https://duckdb.org/docs/)
- [DuckDB Vector Search (vss)](https://duckdb.org/docs/extensions/vss)
- [DuckDB Spatial Extension](https://duckdb.org/docs/extensions/spatial)
- [DuckPGQ Graph Extension](https://github.com/cwida/duckpgq-extension)
- [DuckDB USING KEY for Graph Queries](https://duckdb.org/2025/05/23/using-key)

---

## Development Workflow

### Python (Server/RAG)
```bash
cd server && uv sync && uv run uvicorn server.main:app --reload
cd server && uv run pytest -q              # Run tests
cd rag && uv sync && uv run python cli.py test
```

### Go (CLI)
```bash
cd cli && go build -o lf
./lf --help
```

### Pre-commit Hooks (RUFF)
```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files

# RUFF specifically
ruff check --fix .
ruff format .
```

### Common Issues & Fixes
```bash
# Port already in use
lsof -ti:8000 | xargs kill -9

# Nx cache issues
nx reset

# Python dependency issues
cd server && uv sync --refresh

# Model not loading
# Check universal-runtime logs, ensure model downloaded
```

---

## Project Types

### 1. Project USING LlamaFarm
```
my-app/
├── llamafarm.yaml          # AI configuration
├── src/                    # Application code
│   └── ...
└── tests/
```

Workflow:
1. `lf init` to create llamafarm.yaml
2. `lf start` to run services
3. Call LlamaFarm APIs from your app

### 2. Contributing TO LlamaFarm
```
llamafarm/                  # Cloned repo
├── server/                 # Python FastAPI
├── cli/                    # Go CLI
├── rag/                    # RAG system
└── ...
```

Workflow:
1. `nx start server` (or rag, universal-runtime)
2. Make changes
3. `nx reset` if caching issues
4. Run tests: `cd server && uv run pytest`
5. Pre-commit: `ruff check --fix . && ruff format .`

---

## Example: Biometric Monitoring System

### Architecture
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ TimescaleDB  │     │  LlamaFarm   │     │    Redis     │
│ (vitals ts)  │     │ (AI/ML/RAG)  │     │  (cache)     │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       └────────────────────┼────────────────────┘
                            │
                   ┌────────┴────────┐
                   │  Monitor App    │
                   │  - Anomaly Det  │
                   │  - Classifier   │
                   │  - LLM Agent    │
                   └─────────────────┘
```

### llamafarm.yaml
```yaml
version: v1
name: soldier-monitor
namespace: military

runtime:
  default_model: analyst
  models:
    - name: analyst
      provider: ollama
      model: qwen3:8b
      base_url: http://127.0.0.1:11434

    - name: fast-classifier
      provider: universal
      model: sentence-transformers/all-MiniLM-L6-v2
      base_url: http://127.0.0.1:11540

rag:
  databases:
    - name: radio_logs
      type: ChromaStore
      default_embedding_strategy: default_embeddings
      default_retrieval_strategy: semantic_search

    - name: medical_protocols
      type: ChromaStore
      default_embedding_strategy: default_embeddings
      default_retrieval_strategy: reranked

  embedding_strategies:
    - name: default_embeddings
      type: UniversalEmbedder
      config:
        model: sentence-transformers/all-MiniLM-L6-v2
        base_url: http://127.0.0.1:11540/v1

  retrieval_strategies:
    - name: semantic_search
      type: BasicSimilarityStrategy
      config:
        top_k: 10

    - name: reranked
      type: CrossEncoderRerankedStrategy
      config:
        model_name: ms-marco-MiniLM-L-6-v2
        initial_k: 30
        final_k: 5

prompts:
  - name: triage_agent
    messages:
      - role: system
        content: |
          You are a military medical triage assistant.
          Analyze biometric data and radio communications.
          Recommend: CONTINUE_MONITORING, REQUEST_LOGISTICS, MEDEVAC_IMMEDIATE
          Always explain your reasoning.

  - name: radio_analyst
    messages:
      - role: system
        content: |
          You analyze radio communications for urgency and intent.
          Classify messages and extract key information.
```

### Usage Pattern
1. Store time-series biometrics in TimescaleDB
2. Train anomaly detector on normal biometrics (LlamaFarm)
3. Train classifier on radio message types (LlamaFarm)
4. Stream data → detect anomalies
5. Classify flagged communications
6. Query RAG for medical protocols
7. LLM makes final recommendation

---

## Health Checks

```bash
# Server health
curl http://localhost:8000/health

# RAG health
curl "http://localhost:8000/v1/projects/{ns}/{proj}/rag/health"

# Universal runtime
curl http://localhost:11540/health
```

---

## Useful Environment Variables

```bash
export LF_RUNTIME_PORT=11540
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export LLAMAFARM_HOME=/path/to/llamafarm
export LLAMAFARM_CONFIG=/path/to/llamafarm.yaml
```

---

## Semantic Router (Manual Training Required)

The semantic router enables sub-millisecond routing of LLM requests based on topic similarity or query complexity. It routes queries to the most appropriate model without invoking an LLM for the routing decision.

**⚠️ IMPORTANT: This version of LlamaFarm does NOT have built-in router API endpoints. You must train routers manually using the classifier API and embeddings.**

### Alternative: Use Text Classifier for Routing

Since there's no `/v1/ml/router/*` API, use the text classifier to achieve similar functionality:

```bash
# Train a classifier that acts as a router
curl -X POST "http://localhost:8000/v1/ml/classifier/fit" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "message-router",
    "training_data": [
      {"text": "what is my bill", "label": "billing"},
      {"text": "payment options", "label": "billing"},
      {"text": "help with my account", "label": "support"},
      {"text": "technical issue", "label": "support"},
      {"text": "general question", "label": "general"}
    ]
  }'

# Then use predict to route messages
curl -X POST "http://localhost:8000/v1/ml/classifier/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "message-router-latest",
    "texts": ["I need to pay my invoice"]
  }'
# Returns: {"label": "billing", "score": 0.95, ...}
# Your code then routes to the appropriate model based on label
```

### Project-Specific Storage

Routers are stored per-project at:
```
{project_dir}/lf_data/routers/{router_name}/
  config.json       # Router configuration
  embeddings.npz    # Pre-computed embeddings
```

This enables:
- Multiple projects with different routers
- Auto-save after training
- Auto-load on first request (survives server restarts)

### Configuration Syntax

The router appears as a model in `runtime.models` with `provider: router`:

```yaml
runtime:
  default_model: smart_router
  models:
    - name: smart_router
      provider: router
      embedder_model: sentence-transformers/all-MiniLM-L6-v2  # Required
      default_model: general_assistant  # Fallback model
      similarity_threshold: 0.75  # Minimum similarity to match (0-1)

      # Optional complexity routing
      complexity_classifier: query_complexity  # SetFit model name
      complexity_threshold: 0.7  # Confidence threshold
      complex_model: powerful_model  # Target for complex queries

      routes:
        - name: billing
          target_model: billing_specialist
          description: "Billing and payment questions"
          utterances:
            - "what is my bill"
            - "payment options"
            - "invoice question"

        - name: support
          target_model: tech_support
          # Can reference a dataset instead of inline utterances
          dataset: support_utterances
```

### Embedder Model Options

| Model | Dimensions | Speed | Notes |
|-------|-----------|-------|-------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Very Fast | **Default**, good balance |
| `BAAI/bge-small-en-v1.5` | 384 | Fast | Good accuracy |
| `BAAI/bge-base-en-v1.5` | 768 | Medium | Better accuracy |
| `BAAI/bge-large-en-v1.5` | 1024 | Slower | Best accuracy |
| `BAAI/bge-m3` | 1024 | Slower | Multilingual support |

### Train Router

```bash
curl -X POST "http://localhost:8000/v1/ml/router/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "customer_router",
    "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
    "default_model": "general_assistant",
    "similarity_threshold": 0.75,
    "namespace": "default",
    "project_id": "my_project",
    "routes": [
      {
        "name": "billing",
        "target_model": "billing_specialist",
        "description": "Billing and payment questions",
        "utterances": [
          "what is my bill",
          "payment options",
          "invoice question"
        ]
      },
      {
        "name": "support",
        "target_model": "tech_support",
        "description": "Technical support and troubleshooting",
        "utterances": [
          "help with login",
          "password reset",
          "technical problem"
        ]
      }
    ]
  }'
```

Response includes storage location:
```json
{
  "model": "customer_router",
  "status": "trained",
  "num_routes": 2,
  "namespace": "default",
  "project_id": "my_project",
  "storage_path": "/path/to/project/lf_data/routers/customer_router"
}
```

### Route a Query

```bash
curl -X POST "http://localhost:8000/v1/ml/router/route" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "customer_router",
    "query": "How can I pay my bill?",
    "namespace": "default",
    "project_id": "my_project"
  }'
```

Response includes routing metadata:
```json
{
  "target_model": "billing_specialist",
  "route_name": "billing",
  "similarity_score": 0.89,
  "matched_utterance": "payment options",
  "router_name": "customer_router",
  "namespace": "default",
  "project_id": "my_project",
  "complexity_label": null,
  "complexity_score": null
}
```

The routing decision is also logged for observability.

### Generate Synthetic Training Data

Use an LLM to generate diverse training utterances from a route description.

**Complexity Options:**
- `simple` - Short, direct questions (5-10 words)
- `complex` - Detailed, multi-part questions (15-30 words)
- `mixed` - A mix of simple and complex (default)

**Single Route Generation:**
```bash
curl -X POST "http://localhost:8000/v1/ml/router/generate-data" \
  -H "Content-Type: application/json" \
  -d '{
    "route_description": "Questions about medical billing, insurance claims, and coding",
    "count": 20,
    "complexity": "mixed",
    "model": "unsloth/Qwen3-1.7B-GGUF:Q4_K_M"
  }'
```

Response:
```json
{
  "utterances": [
    "What is the CPT code for a follow-up visit?",
    "How do I submit a prior authorization?",
    "Why was my claim denied?",
    "I need help understanding the complex billing statement I received..."
  ],
  "count": 20,
  "complexity": "mixed"
}
```

**Batch Generation for Multiple Routes:**
```bash
curl -X POST "http://localhost:8000/v1/ml/router/generate-data" \
  -H "Content-Type: application/json" \
  -d '{
    "routes": [
      {"route_name": "billing", "description": "billing inquiries", "count": 10},
      {"route_name": "support", "description": "tech support", "count": 10, "complexity": "complex"}
    ],
    "complexity": "mixed"
  }'
```

### List Saved Routers

**List Global Routers (legacy):**
```bash
curl "http://localhost:8000/v1/ml/router/models"
```

**List Project Routers:**
```bash
curl -X POST "http://localhost:8000/v1/ml/router/models/list" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "default",
    "project_id": "my_project"
  }'
```

Response:
```json
{
  "object": "list",
  "data": [
    {
      "name": "customer_router",
      "path": "/path/to/project/lf_data/routers/customer_router",
      "has_embeddings": true,
      "config": {
        "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
        "default_model": "general_assistant",
        "similarity_threshold": 0.75,
        "routes": [...]
      }
    }
  ],
  "total": 1,
  "namespace": "default",
  "project_id": "my_project"
}
```

### Use Router via Chat API

When the router is configured in `llamafarm.yaml`, use it like any other model:

```bash
curl -X POST "http://localhost:8000/v1/projects/default/my_project/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "smart_router",
    "messages": [{"role": "user", "content": "What is my bill this month?"}]
  }'
```

The router will:
1. Extract the query from messages
2. Compute semantic similarity with route utterances
3. Route to the matching target model (or default if no match)
4. Execute the request on the target model
5. Return the response transparently

The routing decision is logged with metadata for debugging and analytics.

### Complexity Routing

For routing based on query complexity (simple → fast model, complex → powerful model):

1. **Train a complexity classifier** using the classifier API:

```bash
curl -X POST "http://localhost:11540/v1/classifier/fit" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "query_complexity",
    "texts": [
      "What time is it?",
      "Hello",
      "Design a distributed system architecture..."
    ],
    "labels": ["simple", "simple", "complex"]
  }'
```

2. **Configure the router** to use the classifier:

```yaml
- name: smart_router
  provider: router
  embedder_model: sentence-transformers/all-MiniLM-L6-v2
  default_model: fast_local_model
  complexity_classifier: query_complexity
  complexity_threshold: 0.7
  complex_model: powerful_cloud_model
  routes: []  # Can be empty for complexity-only routing
```

3. **Route queries** - complex queries go to `powerful_cloud_model`, simple queries go to `fast_local_model`.

### Best Practices

1. **Use diverse utterances**: 10-20 varied examples per route work best
2. **Keep similarity threshold between 0.7-0.85**: Too low = false positives, too high = missed matches
3. **Use the generate-data endpoint**: Creates diverse, realistic training examples
4. **Test your router**: Use the `/route` endpoint to verify routing before production
5. **Consider complexity routing**: For cost optimization, route simple queries to smaller models
6. **Update routes incrementally**: Retrain when you find mis-routed queries

### Designer UI

Create and manage routers in the Designer UI:

1. Go to **Models** → **Trained models**
2. Click **Create** under "Semantic router models"
3. Configure:
   - Router name
   - Embedder model (sentence-transformers options)
   - Default fallback model
   - Similarity threshold
4. Add routes with:
   - Route name
   - Target model
   - Description (for synthetic data generation)
   - Utterances (or click "Generate examples")
5. Click **Train Router**
6. Test routing with sample queries

---

## MCP (Model Context Protocol) Integration

LlamaFarm supports MCP servers for external tool access, giving AI models access to external tools, APIs, and data sources through a standardized protocol.

### MCP Configuration in llamafarm.yaml

```yaml
mcp:
  servers:
    # STDIO Transport (Local Process)
    - name: filesystem
      transport: stdio
      command: npx
      args: ['-y', '@modelcontextprotocol/server-filesystem', '/path/to/dir']
      env:
        CUSTOM_VAR: "value"

    # HTTP Transport (Remote Server)
    - name: remote_tools
      transport: http
      base_url: https://api.example.com/mcp
      headers:
        Authorization: "Bearer ${env:API_KEY}"

    # SSE Transport (Server-Sent Events)
    - name: sse_server
      transport: sse
      base_url: http://localhost:8080/sse

runtime:
  models:
    - name: assistant
      provider: ollama
      model: qwen3:8b
      mcp_servers: [filesystem, remote_tools]  # List specific servers
      # mcp_servers: []  # Empty = no MCP access
      # (omit mcp_servers entirely to allow all configured servers)
```

### Transport Types

| Transport | Use Case | Configuration |
|-----------|----------|---------------|
| **stdio** | Local tools, custom Python servers | `command`, `args`, `env` |
| **http** | Remote/cloud MCP servers | `base_url`, `headers` |
| **sse** | Real-time streaming connections | `base_url` only |

### Per-Model Access Control

Control which models can access which MCP servers:

```yaml
runtime:
  models:
    # Specific access
    - name: file_assistant
      mcp_servers: [filesystem]  # Only filesystem tools

    # No MCP access
    - name: basic_chat
      mcp_servers: []  # Explicitly deny MCP

    # Full access (omit mcp_servers)
    - name: power_user
      # No mcp_servers field = access all configured servers
```

### Combining MCP with Inline Tools

Models can use both MCP servers AND inline tools:

```yaml
mcp:
  servers:
    - name: filesystem
      transport: stdio
      command: npx
      args: ['-y', '@modelcontextprotocol/server-filesystem', '/data']

runtime:
  models:
    - name: hybrid_agent
      provider: ollama
      model: qwen3:8b
      mcp_servers: [filesystem]  # External MCP tools
      tool_call_strategy: native_api
      tools:  # PLUS inline tools
        - type: function
          name: custom_action
          description: Custom action defined inline
          parameters:
            type: object
            properties:
              action: {type: string}
```

### Session Management

LlamaFarm manages MCP sessions automatically:
- **Connection pooling**: Reuses connections for efficiency
- **Tool list caching**: 5-minute cache to reduce overhead
- **Graceful shutdown**: Properly closes connections
- **Long-running timeout**: 1-hour timeout for persistent connections

### Building Custom MCP Servers

#### Using FastMCP (Recommended)

```python
# my_mcp_server.py
from fastmcp import FastMCP

mcp = FastMCP("My Tools")

@mcp.tool
def calculate_risk(value: float, threshold: float) -> dict:
    """Calculate risk level based on value and threshold"""
    risk = "high" if value > threshold else "normal"
    return {"risk": risk, "value": value, "threshold": threshold}

@mcp.tool
def send_alert(level: str, message: str) -> dict:
    """Send an alert notification"""
    # Implement alert logic
    return {"status": "sent", "level": level, "message": message}

if __name__ == "__main__":
    mcp.run(transport="sse", host="127.0.0.1", port=8080)
```

#### Using Official MCP Python SDK

```python
# mcp_server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

server = Server("my-tools")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="analyze",
            description="Analyze data",
            inputSchema={
                "type": "object",
                "properties": {"data": {"type": "string"}},
                "required": ["data"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "analyze":
        return [TextContent(type="text", text=f"Analyzed: {arguments['data']}")]

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Connecting LlamaFarm to Custom MCP Server

```yaml
mcp:
  servers:
    # For SSE server (FastMCP with transport="sse")
    - name: my_tools
      transport: sse
      base_url: http://127.0.0.1:8080/sse

    # For stdio server (Python script)
    - name: my_stdio_tools
      transport: stdio
      command: python
      args: ['mcp_server.py']
      env:
        PYTHONPATH: "/path/to/project"

runtime:
  models:
    - name: agent
      provider: ollama
      model: qwen3:8b
      mcp_servers: [my_tools]
```

### Dynamic Tool Server Pattern

For applications needing runtime tool updates, use a registry-based approach:

```python
# dynamic_mcp_server.py
from fastmcp import FastMCP
import json
from pathlib import Path

mcp = FastMCP("Dynamic Tools")
TOOLS_DIR = Path("./tools")

def load_tool_configs():
    """Load tool configurations from JSON files"""
    tools = {}
    for f in TOOLS_DIR.glob("*.json"):
        config = json.loads(f.read_text())
        tools[config["name"]] = config
    return tools

# Register a dispatcher tool that routes to dynamic tools
@mcp.tool
def execute_dynamic_tool(tool_name: str, parameters: dict) -> dict:
    """Execute a dynamically loaded tool by name"""
    tools = load_tool_configs()
    if tool_name not in tools:
        return {"error": f"Tool {tool_name} not found"}

    tool_config = tools[tool_name]
    # Execute based on tool type
    if tool_config["type"] == "http":
        # Make HTTP request
        pass
    elif tool_config["type"] == "python":
        # Execute Python function
        pass
    return {"status": "executed", "tool": tool_name}

@mcp.tool
def list_available_tools() -> list:
    """List all dynamically available tools"""
    return list(load_tool_configs().keys())

if __name__ == "__main__":
    mcp.run(transport="sse", host="127.0.0.1", port=8080)
```

### MCP Resources

- **Official Python SDK**: https://github.com/modelcontextprotocol/python-sdk
- **FastMCP Framework**: https://github.com/jlowin/fastmcp
- **MCP Specification**: https://modelcontextprotocol.io/specification/2025-11-25
- **PyPI Package**: https://pypi.org/project/mcp/

---

## Datasets API (Programmatic RAG Management)

### Dataset Workflow

1. **Create** dataset → 2. **Upload** files → 3. **Process** (vectorize) → 4. **Query** via chat

### Create Dataset

```bash
curl -X POST "http://localhost:8000/v1/projects/{namespace}/{project}/datasets" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "knowledge_base",
    "data_processing_strategy": "default",
    "database": "main_db"
  }'
```

### Upload Files

```bash
curl -X POST "http://localhost:8000/v1/projects/{namespace}/{project}/datasets/{dataset}/data" \
  -F "files=@document1.pdf" \
  -F "files=@document2.txt" \
  -F "files=@notes.md"
```

Response includes file hashes for tracking:
```json
{
  "uploaded": [
    {"filename": "document1.pdf", "hash": "abc123..."},
    {"filename": "document2.txt", "hash": "def456..."}
  ]
}
```

### Process Dataset (Vectorize)

```bash
curl -X POST "http://localhost:8000/v1/projects/{namespace}/{project}/datasets/{dataset}/actions" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "process"}'
```

Returns task ID for monitoring:
```json
{
  "task_id": "task-uuid-here",
  "status": "queued"
}
```

### Monitor Processing

```bash
curl "http://localhost:8000/v1/projects/{namespace}/{project}/tasks/{task_id}"
```

### List Datasets

```bash
curl "http://localhost:8000/v1/projects/{namespace}/{project}/datasets"
```

### Delete Dataset or File

```bash
# Delete entire dataset
curl -X DELETE "http://localhost:8000/v1/projects/{namespace}/{project}/datasets/{dataset}"

# Delete single file by hash
curl -X DELETE "http://localhost:8000/v1/projects/{namespace}/{project}/datasets/{dataset}/data/{file_hash}"
```

### Python Example: Programmatic RAG Setup

```python
import httpx

BASE_URL = "http://localhost:8000/v1/projects/default/myproject"

async def setup_knowledge_base(files: list[str]):
    async with httpx.AsyncClient() as client:
        # 1. Create dataset
        await client.post(f"{BASE_URL}/datasets", json={
            "name": "knowledge",
            "data_processing_strategy": "default",
            "database": "main_db"
        })

        # 2. Upload files
        for filepath in files:
            with open(filepath, "rb") as f:
                await client.post(
                    f"{BASE_URL}/datasets/knowledge/data",
                    files={"files": f}
                )

        # 3. Process (vectorize)
        resp = await client.post(
            f"{BASE_URL}/datasets/knowledge/actions",
            json={"action_type": "process"}
        )
        task_id = resp.json()["task_id"]

        # 4. Wait for completion
        while True:
            status = await client.get(f"{BASE_URL}/tasks/{task_id}")
            if status.json()["state"] == "SUCCESS":
                break
            await asyncio.sleep(1)

        return "Knowledge base ready"
```

---

## Chat with Automatic RAG

### Enabling RAG in Chat Requests

**CRITICAL**: Set `rag_enabled: true` to activate document retrieval:

```bash
curl -X POST "http://localhost:8000/v1/projects/default/myproject/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What are the safety protocols?"}],
    "rag_enabled": true,
    "database": "knowledge_base",
    "rag_top_k": 5,
    "rag_score_threshold": 0.5
  }'
```

### RAG Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rag_enabled` | boolean | false | **Must be true** to enable retrieval |
| `database` | string | default | Vector database to query |
| `rag_top_k` | int | 5 | Number of documents to retrieve |
| `rag_score_threshold` | float | 0.0 | Minimum similarity (0-1) |
| `rag_queries` | array | null | Custom queries instead of user message |

### Custom RAG Queries

Search for specific terms different from the user message:

```json
{
  "messages": [{"role": "user", "content": "Summarize the key points"}],
  "rag_enabled": true,
  "database": "documents",
  "rag_queries": ["safety procedures", "emergency protocols", "compliance"],
  "rag_top_k": 10
}
```

When multiple queries are provided:
1. All queries execute concurrently
2. Results are merged and deduplicated
3. Sorted by relevance score
4. Limited to `rag_top_k` total results

### When to Enable/Disable RAG

| Use Case | rag_enabled |
|----------|-------------|
| Question about uploaded documents | `true` |
| General chat/conversation | `false` |
| Tool-calling agent (context from tools) | `false` |
| Knowledge-grounded responses | `true` |
| Creative writing | `false` |

### Streaming with RAG

```bash
curl -X POST "http://localhost:8000/v1/projects/default/myproject/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain the architecture"}],
    "stream": true,
    "rag_enabled": true,
    "database": "tech_docs"
  }'
```
