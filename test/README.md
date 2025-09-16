# LlamaFarm Test Suite

Comprehensive test suite for LlamaFarm components and integration.

## Directory Structure

```
test/
├── cli/                    # CLI command tests
│   ├── test_complete_cli_flow.py
│   ├── test_lf_run_commands.sh
│   ├── test_lf_run_quick.sh
│   └── verify_lf_run.sh
├── integration/            # Integration tests
│   ├── test_rag_integration.sh
│   ├── test_server_ingest.py
│   └── verify_cli_rag_integration.py
└── scripts/               # Script validation tests
    └── test_readme_examples.sh
```

## Running Tests

### Quick Tests
```bash
# Test basic CLI functionality
./test/cli/test_lf_run_quick.sh

# Verify lf run commands
./test/cli/verify_lf_run.sh
```

### Integration Tests
```bash
# Full RAG integration test
./test/integration/test_rag_integration.sh

# Test server ingestion
python test/integration/test_server_ingest.py

# Verify CLI-RAG integration
python test/integration/verify_cli_rag_integration.py
```

### Complete Test Suite
```bash
# Run all CLI tests
./test/cli/test_complete_cli_flow.py

# Test all lf run command variations
./test/cli/test_lf_run_commands.sh

# Validate README examples work
./test/scripts/test_readme_examples.sh
```

## Test Coverage

- **CLI Tests**: Verify all CLI commands work correctly
- **Integration Tests**: Test component interactions (RAG, models, server)
- **Script Tests**: Ensure documentation examples are accurate

## Prerequisites

Before running tests:
1. Build the CLI: `cd cli && go build -o ../lf main.go && cd ..`
2. Start the server: `nx start server`
3. Ensure Ollama is running: `ollama serve`
4. Pull required models: `ollama pull llama3.1:8b` and `ollama pull nomic-embed-text`