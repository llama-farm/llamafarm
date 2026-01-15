# Pytest Testing Checklist

Testing patterns for LlamaFarm Python components.

---

## Category: Test Organization

### Test File Naming and Structure

**What to check**: Tests follow naming conventions and mirror source structure

**Pass criteria**:
- Test files named `test_*.py` or `*_test.py`
- Tests in `tests/` directory at component root
- Mirror source structure: `src/services/` -> `tests/services/`

**Example** (from rag/):
```
rag/
├── core/
│   └── base.py
├── tasks/
│   └── ingest_tasks.py
└── tests/
    ├── conftest.py
    ├── test_base.py
    └── test_ingest_tasks.py
```

**Severity**: Low

---

### Test Function Naming

**What to check**: Test function names describe behavior being tested

**Good pattern**:
```python
def test_process_document_returns_chunks():
    ...

def test_process_document_raises_on_empty_input():
    ...

def test_embedder_handles_connection_failure():
    ...
```

**Bad pattern**:
```python
def test_1():
    ...

def test_process():  # Too vague
    ...
```

**Severity**: Low

---

### conftest.py for Shared Fixtures

**What to check**: Shared fixtures in conftest.py

**Good pattern** (from rag/tests/conftest.py):
```python
"""Essential pytest configuration and fixtures."""

import sys
from pathlib import Path

import pytest

# Add parent directories to path
rag_dir = Path(__file__).parent.parent
sys.path.insert(0, str(rag_dir))

from core.base import Document


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for testing."""
    return [
        Document(
            id="doc1",
            content="This is a test document",
            metadata={"type": "technical"},
            embeddings=[0.1, 0.2, 0.3],
        ),
    ]
```

**Severity**: Medium

---

## Category: Fixtures

### Fixture Scope Selection

**What to check**: Use appropriate fixture scope

**Good pattern**:
```python
@pytest.fixture(scope="session")
def database_connection():
    """Expensive setup - share across all tests in session."""
    conn = create_connection()
    yield conn
    conn.close()

@pytest.fixture(scope="module")
def model_instance():
    """Share within test module."""
    return load_model()

@pytest.fixture  # scope="function" is default
def clean_state():
    """Fresh state for each test."""
    state = create_state()
    yield state
    state.cleanup()
```

**Pass criteria**:
- `scope="session"` for expensive resources (DB connections, model loading)
- `scope="module"` for module-level shared resources
- `scope="function"` (default) for test isolation

**Severity**: Medium

---

### Temporary Directory Fixtures

**What to check**: Use temp directories for file-based tests

**Good pattern** (from rag/tests/conftest.py):
```python
from collections.abc import Generator

@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def sample_csv_file(temp_dir: str) -> str:
    """Create a temporary CSV file with sample data."""
    csv_path = Path(temp_dir) / "test.csv"
    csv_path.write_text("name,value\ntest,123")
    return str(csv_path)
```

**Severity**: Medium

---

### Factory Fixtures

**What to check**: Use factories for customizable test data

**Good pattern**:
```python
@pytest.fixture
def make_document():
    """Factory for creating test documents."""
    def _make(
        content: str = "test content",
        id: str | None = None,
        **metadata
    ) -> Document:
        return Document(
            content=content,
            id=id or str(uuid.uuid4()),
            metadata=metadata
        )
    return _make

def test_process_with_metadata(make_document):
    doc = make_document(content="custom", type="article", author="test")
    result = process(doc)
    assert result.metadata["type"] == "article"
```

**Severity**: Low

---

### Autouse Fixtures Sparingly

**What to check**: Limit autouse to truly universal setup

**Search pattern**:
```bash
rg "@pytest.fixture\(.*autouse=True" --type py tests/
```

**Pass criteria**: autouse only for:
- Environment variable reset
- Logging configuration
- Global state cleanup

**Bad pattern**:
```python
@pytest.fixture(autouse=True)
def setup_database():  # Too heavy for autouse
    ...
```

**Severity**: Low

---

## Category: Async Testing

### pytest-asyncio Markers

**What to check**: Async tests are properly marked

**Good pattern**:
```python
import pytest

@pytest.mark.asyncio
async def test_async_operation():
    result = await some_async_function()
    assert result is not None

@pytest.mark.asyncio
async def test_model_loading():
    model = await load_model("test-model")
    assert model.model_id == "test-model"
```

**Search pattern** (find unmarked async tests):
```bash
rg "async def test_" --type py tests/ -B 2 | rg -v "@pytest.mark.asyncio"
```

**Pass criteria**: All async test functions have `@pytest.mark.asyncio`

**Severity**: High (tests fail without marker)

---

### Async Fixtures

**What to check**: Async fixtures properly defined

**Good pattern**:
```python
@pytest.fixture
async def async_client():
    """Create async HTTP client for testing."""
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
async def loaded_model():
    """Load model for testing."""
    model = LanguageModel("test-model", "cpu")
    await model.load()
    yield model
    await model.unload()
```

**Severity**: Medium

---

### pytest-asyncio Configuration

**What to check**: Configure asyncio mode in pyproject.toml

**Good pattern**:
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"  # or "strict"
asyncio_default_fixture_loop_scope = "function"
```

**Severity**: Medium

---

## Category: Mocking

### Mock at the Right Level

**What to check**: Mock where imported, not where defined

**Good pattern**:
```python
# If service.py does: from external import api_client
# Mock in service's namespace, not external's
@patch("myapp.service.api_client")
def test_service_calls_api(mock_client):
    mock_client.get.return_value = {"data": "test"}
    result = service.fetch_data()
    assert result == {"data": "test"}
```

**Bad pattern**:
```python
@patch("external.api_client")  # Wrong - mock at definition site
def test_service(mock_client):
    ...
```

**Severity**: High

---

### Mock Return Values and Side Effects

**What to check**: Set appropriate mock behaviors

**Good pattern**:
```python
from unittest.mock import Mock, AsyncMock

# Sync mock with return value
mock_client.get.return_value = Mock(
    status_code=200,
    json=Mock(return_value={"key": "value"})
)

# Async mock
mock_client.fetch = AsyncMock(return_value={"data": "test"})

# Side effects for multiple calls
mock_client.get.side_effect = [
    {"first": "call"},
    {"second": "call"},
    ConnectionError("Failed"),
]
```

**Severity**: Medium

---

### Avoid Over-Mocking

**What to check**: Don't mock implementation details

**Pass criteria**:
- Mock external dependencies (APIs, databases, file systems)
- Don't mock internal methods being tested
- Use real objects when practical
- Integration tests with minimal mocking

**Bad pattern**:
```python
def test_process_document(mocker):
    # Over-mocking - testing mock behavior, not real code
    mocker.patch.object(processor, '_validate')
    mocker.patch.object(processor, '_transform')
    mocker.patch.object(processor, '_save')
    processor.process(doc)
```

**Severity**: Medium

---

### pytest-mock for Convenience

**What to check**: Use pytest-mock's mocker fixture

**Good pattern**:
```python
def test_with_mocker(mocker):
    mock_api = mocker.patch("myapp.service.external_api")
    mock_api.return_value = {"status": "ok"}

    result = service.call_api()

    mock_api.assert_called_once()
    assert result["status"] == "ok"
```

**Severity**: Low

---

## Category: Parametrization

### Parametrized Tests

**What to check**: Use parametrize for multiple test cases

**Good pattern**:
```python
@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
    ("", ""),
    ("MiXeD", "MIXED"),
])
def test_uppercase(input, expected):
    assert uppercase(input) == expected

@pytest.mark.parametrize("invalid_input", [
    None,
    123,
    [],
    {},
])
def test_uppercase_rejects_invalid_types(invalid_input):
    with pytest.raises(TypeError):
        uppercase(invalid_input)
```

**Severity**: Low

---

### Parametrize IDs for Readability

**What to check**: Add IDs for readable test output

**Good pattern**:
```python
@pytest.mark.parametrize("input,expected", [
    pytest.param("hello", "HELLO", id="simple-word"),
    pytest.param("", "", id="empty-string"),
    pytest.param("a b c", "A B C", id="with-spaces"),
    pytest.param("123abc", "123ABC", id="mixed-chars"),
])
def test_uppercase(input, expected):
    assert uppercase(input) == expected
```

**Severity**: Low

---

## Category: Assertions

### Specific Assertions

**What to check**: Use specific assertion patterns

**Good pattern**:
```python
# Equality
assert result == expected
assert result != other

# Membership
assert item in collection
assert "key" in dictionary

# Type checking
assert isinstance(result, Document)

# Approximate equality for floats
assert result == pytest.approx(3.14159, rel=1e-3)
```

**Bad pattern**:
```python
assert result  # Not specific - what value is expected?
assert bool(result)  # Same problem
```

**Severity**: Low

---

### Exception Testing

**What to check**: Test expected exceptions with context

**Good pattern**:
```python
def test_raises_on_invalid_input():
    with pytest.raises(ValueError, match="must be positive"):
        process(-1)

def test_raises_specific_exception():
    with pytest.raises(ConfigError) as exc_info:
        load_config("nonexistent.yaml")
    assert "not found" in str(exc_info.value)
    assert exc_info.value.path == "nonexistent.yaml"
```

**Severity**: Medium

---

### Assert Message Context

**What to check**: Add context to assertions when helpful

**Good pattern**:
```python
def test_document_processing():
    result = process(documents)

    assert len(result.documents) == 10, f"Expected 10 docs, got {len(result.documents)}"
    assert result.errors == [], f"Unexpected errors: {result.errors}"
```

**Severity**: Low

---

## Category: Test Markers

### Custom Markers for Test Categories

**What to check**: Define and use custom markers

**Good pattern** (in conftest.py):
```python
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "slow: slow tests")
    config.addinivalue_line("markers", "gpu: tests requiring GPU")
```

**Usage**:
```python
@pytest.mark.integration
def test_database_connection():
    ...

@pytest.mark.slow
def test_full_pipeline():
    ...

# Run only fast tests
# pytest -m "not slow"

# Run integration tests
# pytest -m integration
```

**Severity**: Low

---

### Skip Markers

**What to check**: Use skip markers appropriately

**Good pattern**:
```python
import pytest

@pytest.mark.skip(reason="Feature not implemented yet")
def test_future_feature():
    ...

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)
def test_gpu_inference():
    ...

@pytest.mark.xfail(reason="Known bug, fix pending")
def test_known_issue():
    ...
```

**Severity**: Low

---

## Category: Test Coverage

### Coverage Configuration

**What to check**: Coverage properly configured

**Good pattern** (pyproject.toml):
```toml
[tool.pytest.ini_options]
addopts = "--cov=src --cov-report=term-missing --cov-report=html"
testpaths = ["tests"]

[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/__pycache__/*"]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
```

**Severity**: Low

---

### Critical Path Coverage

**What to check**: Core functionality is tested

**Pass criteria**:
- All public API functions have tests
- Error paths are tested
- Edge cases covered
- Integration points tested

**Severity**: High
