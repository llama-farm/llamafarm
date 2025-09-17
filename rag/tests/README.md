# RAG Tests

**Streamlined test suite with 42 essential tests focusing on real functionality with minimal mocking.**

## Quick Start

### Run All Tests
```bash
# From the rag/ directory
pytest

# With coverage reporting
pytest --cov

# Verbose output
pytest -v

# Run tests in parallel (faster)
pytest -n auto
```

### Run Specific Test Categories
```bash
# Unit tests only (fast)
pytest -m "not slow and not integration"

# Integration tests
pytest -m integration

# Skip tests requiring external services
pytest -m "not ollama and not chromadb"

# Run only fast tests
pytest -m "not slow"
```

## Test Organization

### Core Components (`components/`)
- **`embedders/`** - Embedding model tests
  - `test_ollama_embedder.py` - Ollama embedding integration
- **`extractors/`** - Metadata extraction tests
  - `test_entity_extractor.py` - Named entity recognition
  - `test_statistics_extractor.py` - Content statistics
- **`parsers/`** - Document parsing tests
  - `test_text_parser.py` - Plain text parsing
- **`stores/`** - Vector database tests
  - `test_chroma_store.py` - ChromaDB integration

### System Tests
- **`test_retrieval_system.py`** - End-to-end retrieval workflows
- **`test_strategy_system.py`** - Strategy pattern implementations
- **`test_cli_comprehensive.py`** - CLI command testing
- **`test_enhanced_pipeline.py`** - Pipeline orchestration

### Integration Tests (`e2e/`)
- **`test_cli_integration.py`** - Full CLI workflow testing

## Test Configuration

Tests are configured via `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests", "."]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests", 
    "ollama: marks tests that require Ollama to be running",
    "chromadb: marks tests that require ChromaDB",
]
```

## Test Commands Reference

### Basic Commands
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov=parsers --cov=embedders --cov=stores --cov=utils

# Generate HTML coverage report
pytest --cov --cov-report=html
# View: open htmlcov/index.html

# Run specific test file
pytest tests/test_retrieval_system.py

# Run specific test function
pytest tests/test_retrieval_system.py::test_basic_similarity_strategy

# Run tests matching pattern
pytest -k "test_embed"
```

### Performance & Debugging
```bash
# Verbose output with test names
pytest -v

# Show local variables on failure
pytest -l

# Drop into debugger on failure
pytest --pdb

# Run failed tests from last run
pytest --lf

# Profile test performance
pytest --durations=10

# Run tests in parallel (requires pytest-xdist)
pip install pytest-xdist
pytest -n auto
```

### Coverage Analysis
```bash
# Basic coverage
pytest --cov

# Coverage with missing line numbers
pytest --cov --cov-report=term-missing

# Generate multiple report formats
pytest --cov --cov-report=html --cov-report=xml --cov-report=term

# Coverage for specific modules
pytest --cov=retrieval --cov=core tests/
```

### Filtering Tests
```bash
# Skip slow tests
pytest -m "not slow"

# Only integration tests
pytest -m integration

# Skip external service dependencies
pytest -m "not ollama and not chromadb"

# Custom marker combinations
pytest -m "integration and not slow"
```

## Test Markers

Use these markers to categorize tests:

```python
import pytest

@pytest.mark.slow
def test_large_document_processing():
    """Test that takes >5 seconds"""
    pass

@pytest.mark.integration
def test_end_to_end_workflow():
    """Test that requires multiple components"""
    pass

@pytest.mark.ollama
def test_ollama_embedding():
    """Test requiring Ollama service"""
    pass

@pytest.mark.chromadb
def test_vector_search():
    """Test requiring ChromaDB"""
    pass
```

## Test Development

### Writing New Tests

1. **Create test file**: `test_your_feature.py`
2. **Import fixtures**: Use existing fixtures from `conftest.py`
3. **Add markers**: Mark slow/integration/service-dependent tests
4. **Mock external dependencies**: Use `pytest-mock` or `unittest.mock`

```python
import pytest
from unittest.mock import Mock, patch

class TestNewFeature:
    def test_basic_functionality(self):
        """Test basic feature works"""
        assert True
    
    @pytest.mark.slow
    def test_performance_scenario(self):
        """Test performance with large data"""
        pass
    
    @patch('external_service.client')
    def test_with_mocked_service(self, mock_client):
        """Test with mocked external dependency"""
        mock_client.return_value = Mock()
        # Test implementation
```

### Test Data

Test data is located in `test_data/`:
- `test_doc1.txt` - Simple text document
- `test_doc2.txt` - Multi-paragraph text
- `test_doc3.md` - Markdown with headers
- `test_strategies.yaml` - Test configuration

## Continuous Integration

### Local CI Simulation
```bash
# Run the full CI test suite
pytest --cov --cov-report=xml -m "not slow"

# With linting (requires dev dependencies)
black --check .
isort --check-only .
mypy .
pytest --cov
```

### Pre-commit Testing
```bash
# Install pre-commit hooks
pre-commit install

# Run all hooks manually
pre-commit run --all-files

# Run specific hook
pre-commit run pytest
```

## Troubleshooting

### Common Issues

**Import/Module errors:**
```bash
# If tests fail with ModuleNotFoundError, run from rag/ directory
cd /path/to/rag/
pytest

# Some test files may have outdated import paths - these are being updated
# Skip problematic test files temporarily:
pytest --ignore=tests/test_parsers.py --ignore=tests/test_pdf_parser.py
```

**Tests hang or timeout:**
```bash
# Add timeout to pytest
pytest --timeout=300  # 5 minute timeout
```

**Import errors:**
```bash
# Ensure you're in the right directory
cd /path/to/rag/
pytest

# Or use absolute imports
python -m pytest tests/
```

**Service dependency failures:**
```bash
# Skip service-dependent tests
pytest -m "not ollama and not chromadb"

# Check if services are running
docker ps  # for containerized services
curl http://localhost:11434/api/tags  # for Ollama
```

**Coverage issues:**
```bash
# Check coverage configuration
pytest --cov-config=pyproject.toml --cov

# Debug coverage paths
pytest --cov --cov-report=term-missing -v
```

### Performance Optimization

**Speed up test runs:**
```bash
# Skip slow tests during development
pytest -m "not slow"

# Use parallel execution
pytest -n auto

# Run only changed tests (requires pytest-testmon)
pip install pytest-testmon
pytest --testmon
```

**Memory optimization:**
```bash
# Run with memory profiling
pip install pytest-memprof
pytest --memprof

# Limit test scope
pytest tests/unit/  # only unit tests
```

## Environment Setup

### Required Dependencies
```bash
# Core testing dependencies
pip install pytest pytest-cov pytest-asyncio pytest-mock

# Optional performance tools
pip install pytest-xdist pytest-timeout pytest-memprof

# Development tools
pip install black isort mypy pre-commit
```

### Service Dependencies

**Ollama (for embedding tests):**
```bash
# Install and start Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull nomic-embed-text
```

**ChromaDB (for vector store tests):**
```bash
# ChromaDB runs in-process, no separate setup needed
# For persistent testing, ensure ./data/chroma_db/ is writable
```

## Test Coverage Goals

- **Unit Tests**: >90% coverage for core modules
- **Integration Tests**: All major user workflows
- **Performance Tests**: Key operations under load
- **Error Handling**: Exception paths and edge cases

Current coverage can be viewed by running:
```bash
pytest --cov --cov-report=html
open htmlcov/index.html
```

## Contributing

When adding new tests:

1. Follow existing naming conventions (`test_*.py`)
2. Add appropriate markers for slow/integration tests
3. Mock external dependencies where possible
4. Include both positive and negative test cases
5. Update this README if adding new test categories

For questions or issues with testing, check the [test fixes summary](TEST_FIXES_SUMMARY.md) or create an issue.