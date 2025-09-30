# Tests

Higher-level test scripts live here. Most verification now runs via project-specific commands (pytest, go test, nx build docs).

## When to Use These
- Smoke-testing example flows end-to-end.
- Validating README snippets if you modify them.
- Running bespoke integration checks that haven’t been codified in unit tests yet.

## Suggested Test Commands
```bash
# Server and RAG Python tests
cd server
uv run --group test python -m pytest

cd ../rag
uv run pytest tests/

# CLI tests
cd ../cli
go test ./...

# Docs build to catch broken links/sidebar issues
cd ..
nx build docs
```

Legacy shell scripts remain for reference but may be outdated—prefer the commands above or add new automated tests adjacent to the code you change.
