---
title: lf chat
sidebar_position: 3
---

# `lf chat`

Send a single prompt to your project’s chat endpoint. By default, responses include RAG context defined in `llamafarm.yaml`.

## Synopsis

```
lf chat                        # Interactive TUI using project from llamafarm.yaml
lf chat [namespace/project]    # Interactive TUI for explicit project
lf chat [namespace/project] "message" [flags]  # One-off request
```

If you omit `namespace/project`, the CLI resolves them from `llamafarm.yaml`.

## Useful Flags

| Flag | Description |
| ---- | ----------- |
| `--file`, `-f` | Read prompt content from a file. |
| `--no-rag` | Skip retrieval—direct LLM call. |
| `--database` | Target a specific RAG database. |
| `--retrieval-strategy` | Override the retrieval strategy. |
| `--rag-top-k` | Adjust the number of results (default 5). |
| `--rag-score-threshold` | Minimum similarity score for results. |
| `--curl` | Print the sanitized `curl` request instead of executing. |

## Behaviour

- Automatically starts the server if needed.
- Filters client/error messages from the transcript before sending.
- Streams responses; exit code is non-zero if the API returns an error.
- Redacts authorization headers when using `--curl`.

## Examples

```bash
# Interactive project chat (auto-detect project)
lf chat

# Basic one-off chat (RAG enabled)
lf chat "Summarize the latest FDA letters."

# Explicit project with file input
lf chat company/legal -f prompt.txt

# Pure LLM request with curl preview
lf chat --no-rag --curl "Explain RAG in 2 sentences"

# Override strategy for targeted retrieval
lf chat --database main_db --retrieval-strategy hybrid_search "Find biologics references"
```

## Sessions

- Set `LLAMAFARM_SESSION_ID=abc123` to keep context between calls.
- `lf start` manages its own session history in `.llamafarm/projects/.../dev/context`.
- Delete session files to reset state or start a new namespace/project for isolation.

## See Also

- [`lf rag query`](./lf-rag.md)
- [Configuration Guide](../configuration/index.md)
