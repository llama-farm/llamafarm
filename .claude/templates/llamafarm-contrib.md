# LlamaFarm Contribution Template

Follow these steps when contributing to LlamaFarm itself:

## Phase 1: Environment Setup
1. Clone or update LlamaFarm repo
2. Install dependencies: `cd server && uv sync`
3. Reset nx if needed: `nx reset`

## Phase 2: Start Development Services
Use nx commands (NOT lf commands) for development:

```bash
# Start in separate terminals
nx start server          # Port 8000
nx start rag             # RAG worker
nx start universal-runtime  # Port 11540
```

### If ports are stuck:
```bash
lsof -ti:8000 | xargs kill -9
lsof -ti:11540 | xargs kill -9
```

## Phase 3: Create Test Configuration
1. Create a test directory outside llamafarm
2. Run `lf init test-project` there
3. Configure llamafarm.yaml to test new feature
4. Use this for integration testing

## Phase 4: Implementation
1. Implement changes in appropriate module
   - `server/` - API endpoints, services
   - `rag/` - RAG processing
   - `cli/` - CLI commands (Go)
2. Follow existing patterns
3. Add appropriate error handling
4. Include logging

## Phase 5: Linting (REQUIRED)
LlamaFarm uses RUFF. Run before committing:

```bash
cd server
ruff check --fix .
ruff format .
```

## Phase 6: Testing
```bash
# Run tests
cd server && uv run pytest -v

# Run specific test
cd server && uv run pytest tests/test_specific.py -v

# Run with coverage
cd server && uv run pytest --cov=server
```

## Phase 7: Integration Testing
1. Use your test llamafarm.yaml
2. Make API calls to test the feature
3. Verify expected behavior
4. Check edge cases

## Phase 8: Documentation
1. Update docstrings
2. Update API docs if endpoints changed
3. Add examples if helpful

## Phase 9: Commit
1. Ensure all tests pass
2. Ensure linters pass
3. Create meaningful commit message
4. **NO Claude attribution**

### Commit format:
```
feat(server): Add new endpoint for X

- Implemented Y
- Added tests for Z
- Updated documentation

Tests: All passing
```

## Common Issues

**nx cache problems:**
```bash
nx reset
```

**Python dependency issues:**
```bash
cd server && uv sync --refresh
```

**Model not loading:**
```bash
# Check universal-runtime logs
# Ensure model is downloaded
```

## Reference
See `.claude/docs/LLAMAFARM-REFERENCE.md` for API details.
