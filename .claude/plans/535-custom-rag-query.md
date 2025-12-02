# Plan: Add Custom RAG Query Support to Chat/Completions

**Issue:** #535
**Status:** Draft

## Overview

Add the ability to pass custom RAG queries to the `/chat/completions` endpoint, allowing users to override the default behavior of using the full chat message for retrieval.

## Current Behavior

1. User sends `POST /{namespace}/{project_id}/chat/completions` with messages
2. Latest user message is extracted (`projects.py:494-502`)
3. This message is passed directly to `_perform_rag_search()` as the query
4. RAG results are injected into system prompt via `RAGContextProvider`

**Key files:**
- `server/api/routers/projects/projects.py` - ChatRequest model and endpoint
- `server/services/project_chat_service.py` - RAG search execution
- `server/services/rag_service.py` - RAG service interface

## Implementation Plan

### Step 1: Update ChatRequest Model

**File:** `server/api/routers/projects/projects.py`

Add new optional fields to `ChatRequest`:

```python
class ChatRequest(BaseModel):
    # ... existing fields ...

    # New RAG query override fields
    rag_query: str | None = Field(
        default=None,
        description="Custom query for RAG retrieval. Overrides using the user message."
    )
    rag_queries: list[str] | None = Field(
        default=None,
        description="Multiple custom queries for RAG retrieval. Results are merged."
    )
```

### Step 2: Update RAGParameters Dataclass

**File:** `server/services/project_chat_service.py`

Add custom query fields to `RAGParameters`:

```python
@dataclass
class RAGParameters:
    rag_enabled: bool
    database: str | None = None
    retrieval_strategy: str | None = None
    rag_top_k: int | None = None
    rag_score_threshold: float | None = None
    # New fields
    custom_query: str | None = None
    custom_queries: list[str] | None = None
```

### Step 3: Update `_resolve_rag_parameters()`

**File:** `server/services/project_chat_service.py`

Pass through the custom query parameters:

```python
def _resolve_rag_parameters(self, request, project_config) -> RAGParameters:
    # ... existing resolution logic ...

    return RAGParameters(
        rag_enabled=rag_enabled,
        database=database,
        retrieval_strategy=retrieval_strategy,
        rag_top_k=top_k,
        rag_score_threshold=score_threshold,
        custom_query=request.rag_query,
        custom_queries=request.rag_queries,
    )
```

### Step 4: Update `_perform_rag_search()`

**File:** `server/services/project_chat_service.py`

Modify to use custom query if provided:

```python
async def _perform_rag_search(
    self,
    message: str,
    rag_params: RAGParameters,
    project_dir: str,
) -> list[Any]:
    # Determine which query/queries to use
    if rag_params.custom_queries:
        # Multiple custom queries - execute each and merge results
        all_results = []
        seen_chunk_ids = set()

        for query in rag_params.custom_queries:
            results = await self._execute_single_rag_search(
                query=query,
                rag_params=rag_params,
                project_dir=project_dir,
            )
            # Deduplicate by chunk_id
            for result in results:
                chunk_id = result.get("chunk_id") or result.get("content", "")[:100]
                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    all_results.append(result)

        # Sort by score and limit to top_k
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results[:rag_params.rag_top_k or 5]

    elif rag_params.custom_query:
        # Single custom query
        query = rag_params.custom_query
    else:
        # Default: use the user message
        query = message

    return await self._execute_single_rag_search(
        query=query,
        rag_params=rag_params,
        project_dir=project_dir,
    )

async def _execute_single_rag_search(
    self,
    query: str,
    rag_params: RAGParameters,
    project_dir: str,
) -> list[Any]:
    """Execute a single RAG search with the given query."""
    # Move existing search logic here
    results = search_with_rag(
        project_dir=project_dir,
        database=rag_params.database,
        message=query,
        top_k=rag_params.rag_top_k,
        retrieval_strategy=rag_params.retrieval_strategy,
        score_threshold=rag_params.rag_score_threshold,
    )
    return self._normalize_rag_results(results)
```

### Step 5: Update Endpoint to Pass Parameters

**File:** `server/api/routers/projects/projects.py`

The existing flow should work since we're adding optional fields to the request model. Verify the fields are accessible in the service layer.

### Step 6: Update Documentation

**File:** `docs/website/docs/api/index.md`

Add documentation for the new parameters:

```markdown
### Custom RAG Queries

Override the default RAG query (user message) with custom queries:

#### Single Custom Query
```json
{
  "messages": [{"role": "user", "content": "Summarize the findings"}],
  "rag_enabled": true,
  "rag_query": "clinical trial results primary endpoints efficacy"
}
```

#### Multiple Custom Queries
```json
{
  "messages": [{"role": "user", "content": "Compare the approaches"}],
  "rag_enabled": true,
  "rag_queries": [
    "machine learning methodology",
    "traditional statistical analysis"
  ]
}
```

Results from multiple queries are merged and deduplicated.
```

### Step 7: Add Tests

**File:** `server/tests/test_chat_rag_query.py` (new file)

```python
import pytest
from server.api.routers.projects.projects import ChatRequest

def test_chat_request_with_custom_rag_query():
    request = ChatRequest(
        messages=[{"role": "user", "content": "Hello"}],
        rag_query="custom search query"
    )
    assert request.rag_query == "custom search query"
    assert request.rag_queries is None

def test_chat_request_with_multiple_rag_queries():
    request = ChatRequest(
        messages=[{"role": "user", "content": "Hello"}],
        rag_queries=["query1", "query2"]
    )
    assert request.rag_queries == ["query1", "query2"]
    assert request.rag_query is None

def test_rag_query_takes_precedence():
    # If both provided, rag_queries takes precedence
    request = ChatRequest(
        messages=[{"role": "user", "content": "Hello"}],
        rag_query="single",
        rag_queries=["multi1", "multi2"]
    )
    # rag_queries should be used
    assert request.rag_queries == ["multi1", "multi2"]
```

## Validation Rules

1. `rag_query` and `rag_queries` are mutually exclusive at the application level
   - If both provided, `rag_queries` takes precedence
2. `rag_queries` must be non-empty if provided
3. Each query in `rag_queries` must be non-empty string
4. Custom queries only apply when `rag_enabled=True`

## Backward Compatibility

- All new fields are optional with `None` defaults
- Existing requests without custom queries work exactly as before
- No breaking changes to the API contract

## Files to Modify

1. `server/api/routers/projects/projects.py` - Add request model fields
2. `server/services/project_chat_service.py` - Update RAGParameters and search logic
3. `docs/website/docs/api/index.md` - Document new parameters
4. `server/tests/test_chat_rag_query.py` - Add tests (new file)

## Estimated Effort

- Implementation: ~2-3 hours
- Testing: ~1 hour
- Documentation: ~30 minutes

## Open Questions

1. Should we add validation to prevent both `rag_query` and `rag_queries` being set?
   - **Recommendation:** Allow both, use `rag_queries` if both are provided

2. For multiple queries, how should we handle `top_k`?
   - **Recommendation:** Apply `top_k` to merged results, not per-query

3. Should we expose which query matched each result in the response?
   - **Recommendation:** Future enhancement, not for initial implementation
