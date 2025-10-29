## LlamaFarm Server – Guide for AI Agents and Contributors

This document standardizes how to work within the `server/` FastAPI application so that human and AI contributors produce consistent, predictable code. It encodes architectural patterns, naming, logging, testing, and common pitfalls. Inspiration: “The renaissance of written coding conventions” – consistency enables AI to be a force multiplier rather than a chaos amplifier. See: `https://www.brokenrobot.xyz/blog/the-renaissance-of-written-coding-conventions/?utm_source=tldrnewsletter`.

### High-level Architecture
- App factory: `api/main.py` builds the FastAPI app, registers routers under `/v1`, and sets middleware (structlog, correlation ID, error pass-through, CORS). Root exposes `/health/*` and simple info endpoints.
- Entrypoint: `server/main.py` configures logging, prepares data dirs, seeds a default project, mounts MCP HTTP (`/mcp`), and runs Uvicorn in dev.
- Routers (HTTP): `api/routers/` includes:
  - `projects/` – project CRUD-ish helpers and chat endpoints
  - `datasets/` – dataset listing/strategies
  - `rag/` – RAG health and query APIs
  - `system/` – upgrades and maintenance
  - `examples.py` – examples metadata
  - `health.py` – liveness & health summary
  - Note: the `inference` router is deprecated and omitted from app registration; do not extend it.
- Services: `services/` contains application logic (project/model resolution, runtime/provider selection, health checks, docs context injection, project chat orchestration, MCP integration).
- Agents: `agents/base/*` defines the agent abstraction (`LFAgent`), history, system prompt generator, context providers, and provider clients. `agents/chat_orchestrator.py` implements tool-calling loops, streaming, MCP execution, and optional session persistence.
- Core: `core/` includes `settings.py`, structured logging helpers, MCP registry, and version metadata.

### Request Flow Convention
- Keep routers thin. The canonical flow is:
  1) Router parses/validates request, resolves project/model/session
  2) Delegate to a service in `services/*`
  3) Service constructs/uses an `LFAgent` (orchestrator) and optional RAG context provider
  4) Agent delegates to the provider-specific client for chat/stream/tool-calling
  5) Router adapts the response shape (streaming SSE or JSON completion)
- Avoid embedding business logic in routers; prefer testable services.

### Runtime/Provider Abstraction
- Always obtain the runtime client via `services/runtime_service.RuntimeService.get_provider(model_config).get_client()`.
- Do not create raw OpenAI/Ollama SDK clients in routers or services.
- Provider clients implement `agents/base/clients/client.py:LFAgentClient` and must expose:
  - `chat(messages)` → str
  - `stream_chat(messages)` → async generator of str
  - `stream_chat_with_tools(messages, tools)` → async generator of StreamEvent
- Clients normalize tool-calling behavior. Prefer native tool-calling if supported; otherwise, inject tool schemas into the system prompt and parse tool-call JSON safely.

### Agent Conventions
- System prompts: built by `LFAgentSystemPromptGenerator` and prepended before history.
- History: `LFAgentHistory` holds ordered `LFAgentChatMessage`s. Use `reset_history()` to clear.
- Tool-calling loop: `ChatOrchestratorAgent` iterates up to a fixed maximum (default 10). Each tool result is fed back with explicit guidance to produce a final answer and avoid duplicate calls.
- MCP tools: enabled via `enable_mcp()` if project config declares MCP servers. Loaded by `MCPToolFactory`; execution via `_execute_mcp_tool`.
- Persistence: optional. When enabled with a `session_id`, writes `project_dir/sessions/<id>/history.json` atomically. Keep responses and tool results concise to limit file size.

### RAG Context Injection
- Use `ProjectChatService` to resolve RAG parameters and populate a `RAGContextProvider` with `ChunkItem`s from retrieval results.
- The agent’s system prompt generator integrates context providers; avoid manually appending RAG text to user prompts.

### Logging & Error Handling
- Use `FastAPIStructLogger(__name__)` per module. Include contextual fields when available: `namespace`, `project_id`, `session_id`, `model`, `provider`.
- Do not log secrets or full tool arguments; truncate values and include previews.
- Let global exception handlers shape responses. Router-level try/except should raise `HTTPException` with clear messages; avoid leaking stack traces to clients.

### Sessions & Caching
- Stateful chat endpoints maintain an in-memory session cache in the router with TTL and can enable on-disk history via the agent. Ensure expired sessions are cleaned and do not leave unbounded session directories.
- Stateless mode should bypass persistence, constructing a fresh agent per request.

### Naming & Structure
- Modules: prefer nouns for services (`project_service.py`, `model_service.py`) and clear `providers/*` for runtime providers.
- Functions: verbs/verb-phrases; keep signatures explicit.
- Types: use Pydantic models for request/response schemas in routers; internal dataclasses or Pydantic models for service-level structured data.

### Common Mistakes to Avoid
- Skipping `RuntimeService`: Do not instantiate provider SDKs directly; always go through the provider abstraction.
- Business logic in routers: Move orchestration to `services/*` and agents.
- Tool schema bloat: Keep tool parameter schemas minimal; overly nested schemas harm model reliability and latency.
- Mixed tool-calling protocols: Let the client handle detection (native vs JSON-in-tags). Do not implement tool parsing in routers/services.
- Duplicate chat stacks: Do not reintroduce the deprecated `inference` router patterns. Use `projects/*/chat` with `ProjectChatService` + `ChatOrchestratorAgent`.
- Leaky logging: Do not log entire payloads or secrets; use previews and structured fields.

### Checklists
- Adding a new router endpoint
  - Define Pydantic models for request/response under the router module
  - Delegate to a service; keep router slim
  - Integrate with existing session handling if interactive/chat
  - Add structured logging and error handling
  - Update docs if user-facing behavior changes

- Adding a service or extending chat behavior
  - Keep public functions small with clear inputs/outputs
  - Use `RuntimeService` → provider client → agent flow
  - Reuse `RAGContextProvider` where applicable
  - Add focused tests (unit-level; no network by default)

- Adding a runtime provider
  - Implement `providers/<name>_provider.py` extending `RuntimeProvider`
  - Return a concrete `LFAgentClient` implementation
  - Implement `check_health()` and default base_url/api_key resolution
  - Update docs and tests

- Adding an MCP tool
  - Implement tool in the MCP server (outside this package) or configure existing servers
  - Ensure tool input schema is compact and validated
  - Verify tool discovery via `enable_mcp()` and basic execution via the orchestrator

### Testing Guidance
- Prefer unit tests around services and the agent client interface. Mock provider clients where possible.
- Validate `StreamEvent` sequences for both content-only and tool-calling paths.
- Add regression tests for session handling and history persistence edge cases.

### Deprecations
- `api/routers/inference/*` is deprecated and omitted from `api/main.py`. Do not depend on or extend it.

### References
- App factory: `api/main.py`
- Entrypoint: `server/main.py`
- Routers: `api/routers/*`
- Services: `services/*`
- Agents: `agents/base/*`, `agents/chat_orchestrator.py`
- Core: `core/settings.py`, `core/logging.py`, `core/mcp_registry.py`
- RAG integration: `services/project_chat_service.py`, RAG package in repo root


