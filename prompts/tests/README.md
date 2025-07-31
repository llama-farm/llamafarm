# Tests

This directory contains the test suite for the LlamaFarm Prompts System.

## Running Tests

```bash
# Run all tests
uv run python -m pytest tests/ -v

# Run specific test file
uv run python -m pytest tests/test_prompt_system.py -v

# Run tests with coverage
uv run python -m pytest tests/ -v --cov=prompts --cov-report=term-missing
```

## Test Categories

- **System Tests**: Core prompt system functionality
- **Template Tests**: Template rendering and validation
- **Strategy Tests**: Strategy selection logic
- **CLI Tests**: Command-line interface functionality
- **Integration Tests**: End-to-end system testing

## Adding Tests

When adding new features:
1. Create corresponding test files
2. Follow existing test patterns
3. Include both unit and integration tests
4. Ensure good test coverage
5. Test error conditions and edge cases