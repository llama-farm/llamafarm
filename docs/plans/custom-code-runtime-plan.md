# Custom Code Runtime: Implementation Plan

## Executive Summary

Add a "Custom Code Runtime" that allows users to extend LlamaFarm with their own Python and TypeScript code. Users point to files in their `llamafarm.yaml`, and the system automatically:
1. Pulls in the code during `lf sync`
2. Installs required dependencies from a curated allowlist
3. Exposes the code as **agents**, **tools**, and **pre/post model hooks**

**Python is prioritized** for initial implementation due to the existing Python stack. TypeScript support follows as Phase 2.

---

## 5 Killer Use Cases

### 1. Anomaly-Triggered Alert Agent
**Scenario**: An agent that automatically responds when the Universal Runtime's anomaly detector flags suspicious data.

```yaml
# llamafarm.yaml
custom_code:
  agents:
    - name: anomaly_responder
      file: ./agents/anomaly_responder.py
      trigger:
        type: anomaly_detected
        model: api-traffic-detector
        threshold: 0.8
```

```python
# agents/anomaly_responder.py
from llamafarm.custom import CustomAgent, AnomalyEvent

class AnomalyResponderAgent(CustomAgent):
    """Triggered when anomaly detector flags suspicious API traffic."""

    async def on_anomaly(self, event: AnomalyEvent):
        # Get anomaly details
        score = event.score
        data_point = event.data

        # Call Slack MCP tool to alert team
        await self.call_tool("slack_send_message", {
            "channel": "#security-alerts",
            "message": f"🚨 Anomaly detected (score: {score:.2f})\n```{data_point}```"
        })

        # Run LLM analysis
        analysis = await self.chat(
            f"Analyze this suspicious API traffic pattern: {data_point}"
        )

        # Log to security DB
        await self.call_tool("postgres_insert", {
            "table": "security_incidents",
            "data": {"score": score, "analysis": analysis, "raw": data_point}
        })

        return {"action": "logged", "severity": "high" if score > 0.9 else "medium"}
```

### 2. Multi-Tool Orchestration Agent
**Scenario**: A research agent that combines web search, document parsing, and vector storage.

```yaml
custom_code:
  agents:
    - name: research_assistant
      file: ./agents/research_agent.py
      tools:
        - web_search      # MCP tool
        - pdf_extract     # Universal runtime
        - vector_store    # RAG pipeline
```

```python
# agents/research_agent.py
from llamafarm.custom import CustomAgent

class ResearchAssistant(CustomAgent):
    """Multi-step research with tool orchestration."""

    async def research(self, topic: str, depth: int = 3):
        # Step 1: Web search for sources
        search_results = await self.call_tool("web_search", {"query": topic, "limit": 10})

        # Step 2: Download and parse PDFs in parallel
        pdf_tasks = []
        for result in search_results:
            if result["url"].endswith(".pdf"):
                pdf_tasks.append(self.call_tool("pdf_extract", {"url": result["url"]}))

        documents = await asyncio.gather(*pdf_tasks)

        # Step 3: Store in vector DB for RAG
        for doc in documents:
            await self.call_tool("vector_store", {
                "text": doc["content"],
                "metadata": {"source": doc["url"], "topic": topic}
            })

        # Step 4: LLM synthesis
        synthesis = await self.chat(
            f"Based on the {len(documents)} documents I've collected about '{topic}', "
            f"provide a comprehensive summary with citations."
        )

        return {"summary": synthesis, "sources": [d["url"] for d in documents]}
```

### 3. Pre/Post Model Hooks for Content Moderation
**Scenario**: Guardrails that run before and after every LLM call.

```yaml
custom_code:
  hooks:
    - name: content_guardrails
      file: ./hooks/content_moderation.py
      timing: [pre_inference, post_inference]
      models: ["*"]  # Apply to all models
```

```python
# hooks/content_moderation.py
from llamafarm.custom import PreInferenceHook, PostInferenceHook
from llamafarm.custom.exceptions import BlockedContentError

class ContentModerationHooks:

    # Curated blocklist patterns
    BLOCKED_PATTERNS = ["jailbreak", "ignore previous", "pretend you are"]

    @PreInferenceHook
    async def check_input(self, messages: list, model: str, context: dict):
        """Block malicious prompts before they reach the model."""
        user_message = messages[-1]["content"] if messages else ""

        # Pattern check
        for pattern in self.BLOCKED_PATTERNS:
            if pattern.lower() in user_message.lower():
                raise BlockedContentError(f"Blocked pattern detected: {pattern}")

        # Optional: Call classifier model for advanced detection
        if context.get("strict_mode"):
            result = await self.call_classifier("prompt-injection-detector", user_message)
            if result["label"] == "malicious" and result["score"] > 0.85:
                raise BlockedContentError("Prompt injection detected")

        return messages  # Return modified or original messages

    @PostInferenceHook
    async def check_output(self, response: str, model: str, context: dict):
        """Validate model output before returning to user."""

        # Check for PII leakage using NER
        entities = await self.call_ner("dslim/bert-base-NER", response)

        pii_types = {"PER", "PHONE", "EMAIL", "SSN"}
        leaked_pii = [e for e in entities if e["label"] in pii_types]

        if leaked_pii:
            # Redact PII
            redacted = response
            for entity in leaked_pii:
                redacted = redacted.replace(entity["text"], f"[REDACTED-{entity['label']}]")
            return redacted

        return response
```

### 4. Custom Tool with External API Integration
**Scenario**: A tool that connects to an internal company API not covered by MCP servers.

```yaml
custom_code:
  tools:
    - name: salesforce_query
      file: ./tools/salesforce.py
      description: "Query Salesforce CRM for customer data"
      requires:
        - simple-salesforce  # From approved dependency list
```

```python
# tools/salesforce.py
from llamafarm.custom import CustomTool, ToolInput, ToolOutput
from simple_salesforce import Salesforce

class SalesforceQueryTool(CustomTool):
    """Query Salesforce CRM."""

    name = "salesforce_query"
    description = "Query Salesforce CRM for customer records, opportunities, and cases"

    class Input(ToolInput):
        object_type: str  # Account, Contact, Opportunity, Case
        query: str        # SOQL query or natural language
        limit: int = 10

    class Output(ToolOutput):
        records: list[dict]
        total_count: int

    def __init__(self):
        self.sf = Salesforce(
            username=os.environ["SF_USERNAME"],
            password=os.environ["SF_PASSWORD"],
            security_token=os.environ["SF_TOKEN"]
        )

    async def run(self, input: Input) -> Output:
        # Convert natural language to SOQL if needed
        if not input.query.upper().startswith("SELECT"):
            soql = await self.context.chat(
                f"Convert this to a Salesforce SOQL query for {input.object_type}: {input.query}"
            )
        else:
            soql = input.query

        # Execute query
        result = self.sf.query(soql)

        return self.Output(
            records=result["records"][:input.limit],
            total_count=result["totalSize"]
        )
```

### 5. Scheduled Data Pipeline Agent
**Scenario**: An agent that runs on a schedule to process and analyze data.

```yaml
custom_code:
  agents:
    - name: daily_report_generator
      file: ./agents/daily_reports.py
      schedule:
        cron: "0 6 * * *"  # Every day at 6 AM
      timeout: 300  # 5 minute max
```

```python
# agents/daily_reports.py
from llamafarm.custom import CustomAgent
import pandas as pd

class DailyReportGenerator(CustomAgent):
    """Generate daily business reports from multiple data sources."""

    async def run_scheduled(self):
        # Fetch data from multiple sources
        sales_data = await self.call_tool("postgres_query", {
            "query": "SELECT * FROM sales WHERE date = CURRENT_DATE - 1"
        })

        support_tickets = await self.call_tool("zendesk_search", {
            "query": "created>=yesterday status:solved"
        })

        # Process with pandas
        df_sales = pd.DataFrame(sales_data)
        daily_revenue = df_sales["amount"].sum()
        top_products = df_sales.groupby("product")["amount"].sum().nlargest(5)

        # LLM analysis
        analysis = await self.chat(f"""
        Analyze yesterday's business metrics:
        - Total Revenue: ${daily_revenue:,.2f}
        - Top Products: {top_products.to_dict()}
        - Support Tickets Resolved: {len(support_tickets)}

        Provide executive summary with key insights and recommendations.
        """)

        # Generate and send report
        await self.call_tool("email_send", {
            "to": "leadership@company.com",
            "subject": f"Daily Business Report - {date.today()}",
            "body": analysis,
            "attachments": [{"name": "sales.csv", "data": df_sales.to_csv()}]
        })

        return {"status": "sent", "revenue": daily_revenue}
```

---

## Architecture Design

### System Overview

```
                                    ┌─────────────────────────────────────────┐
                                    │           llamafarm.yaml                │
                                    │  custom_code:                           │
                                    │    agents: [...]                        │
                                    │    tools: [...]                         │
                                    │    hooks: [...]                         │
                                    └─────────────┬───────────────────────────┘
                                                  │
                                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              lf sync / lf start                              │
│  1. Parse custom_code config                                                 │
│  2. Validate files exist and are within project directory                    │
│  3. Check dependencies against allowlist                                     │
│  4. Install dependencies (uv pip install)                                    │
│  5. Load and validate code modules                                           │
└──────────────────────────────────────────────────────────────────────────────┘
                                                  │
                    ┌─────────────────────────────┼─────────────────────────────┐
                    ▼                             ▼                             ▼
        ┌───────────────────┐         ┌───────────────────┐         ┌───────────────────┐
        │   Custom Agents   │         │   Custom Tools    │         │   Custom Hooks    │
        │                   │         │                   │         │                   │
        │ • Registered with │         │ • Exposed via     │         │ • Pre/Post        │
        │   agent factory   │         │   tool factory    │         │   inference       │
        │ • Can call tools  │         │ • OpenAI-compat   │         │ • Anomaly events  │
        │ • Can use LLM     │         │   function schema │         │ • Scheduled       │
        │ • Trigger-based   │         │ • Sandboxed exec  │         │                   │
        └───────────────────┘         └───────────────────┘         └───────────────────┘
                    │                             │                             │
                    └─────────────────────────────┼─────────────────────────────┘
                                                  │
                                                  ▼
                              ┌───────────────────────────────────────┐
                              │         Custom Code Runtime           │
                              │                                       │
                              │  • FastAPI server (port 11550)        │
                              │  • Sandboxed execution environment    │
                              │  • Resource limits (CPU, memory, time)│
                              │  • Access to Universal Runtime APIs   │
                              │  • MCP tool bridge                    │
                              └───────────────────────────────────────┘
                                                  │
                    ┌─────────────────────────────┼─────────────────────────────┐
                    ▼                             ▼                             ▼
        ┌───────────────────┐         ┌───────────────────┐         ┌───────────────────┐
        │ Universal Runtime │         │    MCP Servers    │         │   RAG Pipeline    │
        │  (port 11540)     │         │   (configured)    │         │                   │
        │                   │         │                   │         │                   │
        │ • Chat/Embeddings │         │ • Slack, GitHub   │         │ • Vector stores   │
        │ • Anomaly detect  │         │ • Web search      │         │ • Document parse  │
        │ • Classification  │         │ • Database tools  │         │                   │
        └───────────────────┘         └───────────────────┘         └───────────────────┘
```

### Directory Structure

```
runtimes/custom-code/
├── project.json              # Nx configuration
├── pyproject.toml            # Python dependencies
├── server.py                 # FastAPI entrypoint
├── core/
│   ├── __init__.py
│   ├── loader.py             # Code loading & validation
│   ├── sandbox.py            # Sandboxed execution
│   ├── dependencies.py       # Dependency management
│   └── registry.py           # Agent/tool/hook registry
├── base/
│   ├── __init__.py
│   ├── agent.py              # CustomAgent base class
│   ├── tool.py               # CustomTool base class
│   ├── hook.py               # Hook base classes
│   └── context.py            # Execution context
├── routers/
│   ├── __init__.py
│   ├── agents.py             # /v1/custom/agents/*
│   ├── tools.py              # /v1/custom/tools/*
│   └── hooks.py              # /v1/custom/hooks/*
├── bridges/
│   ├── __init__.py
│   ├── universal.py          # Bridge to Universal Runtime
│   ├── mcp.py                # Bridge to MCP tools
│   └── rag.py                # Bridge to RAG pipeline
└── tests/
    └── ...
```

---

## Configuration Schema

Add to `config/schema.yaml`:

```yaml
custom_code:
  type: object
  description: Custom Python/TypeScript code extensions
  properties:
    enabled:
      type: boolean
      default: true
      description: Enable custom code execution

    base_path:
      type: string
      default: "./"
      description: Base path for resolving custom code files (relative to project root)

    agents:
      type: array
      description: Custom agent definitions
      items:
        type: object
        required: [name, file]
        properties:
          name:
            type: string
            pattern: "^[a-z][a-z0-9_]*$"
            description: Unique agent identifier
          file:
            type: string
            description: Path to Python/TypeScript file (relative to base_path)
          description:
            type: string
            description: Human-readable description
          trigger:
            type: object
            description: Event-based trigger configuration
            properties:
              type:
                type: string
                enum: [anomaly_detected, schedule, webhook, manual]
              model:
                type: string
                description: Anomaly model name (for anomaly_detected trigger)
              threshold:
                type: number
                minimum: 0
                maximum: 1
              cron:
                type: string
                description: Cron expression (for schedule trigger)
              webhook_path:
                type: string
                description: Webhook endpoint path (for webhook trigger)
          tools:
            type: array
            items:
              type: string
            description: List of tool names this agent can access
          timeout:
            type: integer
            default: 60
            minimum: 1
            maximum: 600
            description: Maximum execution time in seconds
          requires:
            type: array
            items:
              type: string
            description: Required dependencies (must be in allowlist)

    tools:
      type: array
      description: Custom tool definitions
      items:
        type: object
        required: [name, file]
        properties:
          name:
            type: string
            pattern: "^[a-z][a-z0-9_]*$"
          file:
            type: string
          description:
            type: string
          requires:
            type: array
            items:
              type: string
          timeout:
            type: integer
            default: 30
            minimum: 1
            maximum: 300

    hooks:
      type: array
      description: Pre/post inference hooks
      items:
        type: object
        required: [name, file, timing]
        properties:
          name:
            type: string
            pattern: "^[a-z][a-z0-9_]*$"
          file:
            type: string
          timing:
            type: array
            items:
              type: string
              enum: [pre_inference, post_inference, on_error, on_anomaly]
          models:
            type: array
            items:
              type: string
            description: Model names to apply hook to ("*" for all)
          priority:
            type: integer
            default: 100
            description: Hook execution order (lower = first)
          requires:
            type: array
            items:
              type: string
```

---

## Curated Dependency Allowlist

The runtime only allows installation of vetted, commonly-used packages:

```python
# runtimes/custom-code/core/dependencies.py

ALLOWED_DEPENDENCIES = {
    # Data Processing
    "pandas": ">=2.0.0",
    "numpy": ">=1.24.0",
    "polars": ">=0.20.0",

    # HTTP & APIs
    "httpx": ">=0.26.0",
    "requests": ">=2.31.0",
    "aiohttp": ">=3.9.0",

    # Database Clients
    "asyncpg": ">=0.29.0",
    "aiosqlite": ">=0.19.0",
    "redis": ">=5.0.0",
    "pymongo": ">=4.6.0",

    # Cloud SDKs
    "boto3": ">=1.34.0",
    "google-cloud-storage": ">=2.14.0",
    "azure-storage-blob": ">=12.19.0",

    # CRM & Business Tools
    "simple-salesforce": ">=1.12.0",
    "hubspot-api-client": ">=8.0.0",
    "stripe": ">=7.0.0",

    # Data Validation
    "pydantic": ">=2.5.0",
    "jsonschema": ">=4.21.0",

    # Parsing & Extraction
    "beautifulsoup4": ">=4.12.0",
    "lxml": ">=5.1.0",
    "pdfplumber": ">=0.10.0",
    "python-docx": ">=1.1.0",

    # Scheduling
    "apscheduler": ">=3.10.0",

    # Observability
    "structlog": ">=24.1.0",
    "opentelemetry-api": ">=1.22.0",

    # Utilities
    "python-dateutil": ">=2.8.0",
    "pytz": ">=2024.1",
    "humanize": ">=4.9.0",
    "tenacity": ">=8.2.0",  # Retry logic
    "cachetools": ">=5.3.0",
}

# Explicitly blocked (security risks)
BLOCKED_DEPENDENCIES = {
    "subprocess32",
    "os-sys",
    "pyinstaller",
    "cx_freeze",
    "eval",
    "exec",
}
```

---

## Security & Guardrails

### 1. Code Validation
```python
# runtimes/custom-code/core/loader.py

class CodeValidator:
    """Validate custom code before loading."""

    FORBIDDEN_IMPORTS = {
        "os.system", "subprocess", "eval", "exec", "compile",
        "__import__", "importlib.import_module",
    }

    FORBIDDEN_PATTERNS = [
        r"os\.system\s*\(",
        r"subprocess\.",
        r"eval\s*\(",
        r"exec\s*\(",
        r"__import__\s*\(",
        r"open\s*\([^)]*,\s*['\"]w",  # Writing files
        r"socket\.",                   # Raw sockets
    ]

    def validate_file(self, file_path: Path) -> ValidationResult:
        content = file_path.read_text()

        # Check for forbidden patterns
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, content):
                return ValidationResult(valid=False, error=f"Forbidden pattern: {pattern}")

        # AST analysis for deeper checks
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.FORBIDDEN_IMPORTS:
                            return ValidationResult(valid=False, error=f"Forbidden import: {alias.name}")
        except SyntaxError as e:
            return ValidationResult(valid=False, error=f"Syntax error: {e}")

        return ValidationResult(valid=True)
```

### 2. Sandboxed Execution
```python
# runtimes/custom-code/core/sandbox.py

class Sandbox:
    """Sandboxed execution environment for custom code."""

    def __init__(
        self,
        max_memory_mb: int = 512,
        max_cpu_seconds: int = 30,
        allowed_paths: list[Path] = None,
    ):
        self.max_memory = max_memory_mb * 1024 * 1024
        self.max_cpu = max_cpu_seconds
        self.allowed_paths = allowed_paths or []

    async def execute(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        timeout: float = 30.0,
    ) -> Any:
        """Execute function with resource limits."""

        # Set resource limits
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (self.max_memory, self.max_memory))

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **(kwargs or {})),
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            raise ExecutionTimeoutError(f"Execution exceeded {timeout}s limit")
        except MemoryError:
            raise ExecutionMemoryError(f"Execution exceeded {self.max_memory // 1024 // 1024}MB limit")
        finally:
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
```

### 3. Path Traversal Prevention
```python
def validate_file_path(file_path: str, project_root: Path) -> Path:
    """Ensure file path is within project directory."""
    resolved = (project_root / file_path).resolve()

    if not resolved.is_relative_to(project_root):
        raise SecurityError(f"Path traversal detected: {file_path}")

    if not resolved.exists():
        raise FileNotFoundError(f"Custom code file not found: {file_path}")

    if not resolved.suffix in {".py", ".ts"}:
        raise ValueError(f"Unsupported file type: {resolved.suffix}")

    return resolved
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)
- [ ] Create `runtimes/custom-code/` directory structure
- [ ] Implement `project.json` and `pyproject.toml`
- [ ] Add `custom_code` to `config/schema.yaml`
- [ ] Regenerate types with `generate_types.py`
- [ ] Create `CustomCodeProvider` in runtime service
- [ ] Implement code loader with validation
- [ ] Implement sandboxed execution environment
- [ ] Create dependency validator with allowlist

### Phase 2: Custom Tools (Week 2-3)
- [ ] Implement `CustomTool` base class
- [ ] Create tool registry and factory
- [ ] Bridge to MCP tool system
- [ ] Add `/v1/custom/tools/*` API endpoints
- [ ] Test with Salesforce example

### Phase 3: Custom Agents (Week 3-4)
- [ ] Implement `CustomAgent` base class
- [ ] Create agent registry
- [ ] Implement trigger system (manual, webhook, schedule)
- [ ] Bridge to Universal Runtime for LLM calls
- [ ] Add `/v1/custom/agents/*` API endpoints
- [ ] Test with Research Assistant example

### Phase 4: Hooks & Events (Week 4-5)
- [ ] Implement hook base classes
- [ ] Create hook registry with priority ordering
- [ ] Integrate pre/post inference hooks into chat orchestrator
- [ ] Implement anomaly event system
- [ ] Connect to Universal Runtime anomaly detection
- [ ] Test with Content Moderation example

### Phase 5: CLI & Sync Integration (Week 5)
- [ ] Update `lf sync` to process custom code
- [ ] Implement dependency installation
- [ ] Add `lf custom validate` command
- [ ] Add `lf custom list` command
- [ ] Add `lf custom run <agent>` command

### Phase 6: TypeScript Support (Future)
- [ ] Add Deno/Bun runtime option
- [ ] Create TypeScript base classes
- [ ] Implement Node.js dependency allowlist
- [ ] Bridge TypeScript tools to Python runtime

---

## API Endpoints

### Custom Tools API
```
POST   /v1/custom/tools/{name}/invoke    # Invoke a custom tool
GET    /v1/custom/tools                  # List all custom tools
GET    /v1/custom/tools/{name}           # Get tool details & schema
```

### Custom Agents API
```
POST   /v1/custom/agents/{name}/run      # Run agent manually
POST   /v1/custom/agents/{name}/trigger  # Trigger agent with event
GET    /v1/custom/agents                 # List all custom agents
GET    /v1/custom/agents/{name}          # Get agent details
GET    /v1/custom/agents/{name}/history  # Get agent run history
```

### Custom Hooks API
```
GET    /v1/custom/hooks                  # List all hooks
GET    /v1/custom/hooks/{name}           # Get hook details
POST   /v1/custom/hooks/{name}/test      # Test hook with sample input
```

---

## Example llamafarm.yaml

```yaml
version: v1
name: my-ai-project
namespace: production

runtime:
  default_model: llama-3.2
  models:
    - name: llama-3.2
      provider: universal
      model: unsloth/Llama-3.2-3B-Instruct-GGUF:Q4_K_M
      base_url: http://127.0.0.1:11540

custom_code:
  enabled: true
  base_path: "./custom/"

  agents:
    - name: anomaly_responder
      file: agents/anomaly_responder.py
      description: "Responds to anomaly detection events"
      trigger:
        type: anomaly_detected
        model: api-traffic-detector
        threshold: 0.8
      tools:
        - slack_send_message
        - postgres_insert
      requires:
        - httpx
        - structlog

    - name: research_assistant
      file: agents/research_agent.py
      description: "Multi-step research with tool orchestration"
      trigger:
        type: manual
      tools:
        - web_search
        - pdf_extract
        - vector_store
      timeout: 120
      requires:
        - pandas
        - beautifulsoup4

    - name: daily_reports
      file: agents/daily_reports.py
      trigger:
        type: schedule
        cron: "0 6 * * *"
      timeout: 300
      requires:
        - pandas
        - httpx

  tools:
    - name: salesforce_query
      file: tools/salesforce.py
      description: "Query Salesforce CRM for customer data"
      requires:
        - simple-salesforce

    - name: internal_api
      file: tools/internal_api.py
      description: "Call internal company APIs"
      requires:
        - httpx

  hooks:
    - name: content_guardrails
      file: hooks/content_moderation.py
      timing: [pre_inference, post_inference]
      models: ["*"]
      priority: 10  # Run first
      requires:
        - httpx

    - name: audit_logger
      file: hooks/audit.py
      timing: [post_inference]
      models: ["*"]
      priority: 100
      requires:
        - structlog

mcp:
  servers:
    - name: slack
      transport: stdio
      command: npx
      args: ["-y", "@anthropic/mcp-server-slack"]

    - name: postgres
      transport: http
      base_url: http://localhost:3001/mcp
```

---

## Success Metrics

1. **Developer Experience**
   - Time to create first custom tool: < 15 minutes
   - Time to deploy custom agent: < 5 minutes
   - Documentation completeness: 100% of base classes documented

2. **Security**
   - Zero arbitrary code execution vulnerabilities
   - All dependencies vetted and version-pinned
   - Memory/CPU limits enforced

3. **Performance**
   - Custom tool invocation overhead: < 50ms
   - Hook execution overhead: < 10ms per hook
   - Agent startup time: < 2 seconds

4. **Reliability**
   - Custom code failures don't crash main server
   - Graceful degradation when custom code unavailable
   - Automatic retry for transient failures
