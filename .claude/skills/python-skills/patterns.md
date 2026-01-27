# Python Patterns Checklist

Idiomatic Python patterns for LlamaFarm components.

---

## Category: Code Organization

### Import Organization

**What to check**: Imports should be organized in groups: stdlib, third-party, local

**Search pattern**:
```bash
rg "^import |^from " --type py server/ | head -50
```

**Pass criteria**: Imports sorted by ruff (isort rules enabled)

**Severity**: Low

**Recommendation**: Run `ruff check --fix` to auto-sort imports

---

### Module Structure

**What to check**: Modules should have clear separation of concerns

**Pass criteria**:
- Models/types in dedicated `models/` or `types.py`
- Routes in dedicated `routers/` or `api/`
- Business logic in `services/` or `core/`
- Utilities in `utils/`
- Configuration in `core/settings.py`

**Severity**: Medium

---

## Category: Data Structures

### Dataclass vs Pydantic Selection

**What to check**: Use the right tool for the job

**Pass criteria**:
- `@dataclass` for simple internal data containers (e.g., `Document`, `ProcessingResult`)
- `pydantic.BaseModel` for API request/response models requiring validation
- `pydantic_settings.BaseSettings` for environment-based configuration

**Search pattern**:
```bash
rg "@dataclass|class.*\(BaseModel\)|class.*\(BaseSettings\)" --type py
```

**Severity**: Medium

**Example** (from rag/core/base.py):
```python
@dataclass
class Document:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
```

---

### Default Factory for Mutable Defaults

**What to check**: Never use mutable default arguments

**Bad pattern**:
```python
def process(items: list = []):  # WRONG - shared mutable default
def init(config: dict = {}):    # WRONG
```

**Good pattern**:
```python
def process(items: list | None = None):
    items = items if items is not None else []

# Or with dataclass
@dataclass
class Config:
    items: list[str] = field(default_factory=list)
```

**Search pattern**:
```bash
rg "def \w+\([^)]*=\s*\[\]|=\s*\{\}" --type py
```

**Pass criteria**: No mutable default arguments

**Severity**: High

---

### Field Default Factory in Dataclasses

**What to check**: Use `field(default_factory=...)` for mutable defaults

**Good pattern**:
```python
from dataclasses import dataclass, field

@dataclass
class ProcessingResult:
    documents: list[Document]
    errors: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
```

**Severity**: High

---

## Category: Modern Python Syntax

### Use Modern Type Syntax (PEP 604, 585)

**What to check**: Use Python 3.10+ type syntax

**Pass criteria**:
- `list[str]` not `List[str]`
- `dict[str, Any]` not `Dict[str, Any]`
- `str | None` not `Optional[str]`
- `tuple[int, ...]` not `Tuple[int, ...]`

**Search pattern**:
```bash
rg "from typing import.*(List|Dict|Tuple|Optional)" --type py
```

**Severity**: Low (handled by ruff UP rules)

**Recommendation**: Run `ruff check --fix --select UP` to auto-upgrade

---

### Use Context Managers for Resources

**What to check**: Resources should use context managers

**Pass criteria**:
- File operations use `with open()`
- Lock acquisition uses `async with lock:`
- Temporary resources use context managers

**Good pattern** (from config/helpers/loader.py):
```python
with open(file_path, encoding="utf-8") as f:
    return yaml_instance.load(f)
```

**Search pattern** (find potential issues):
```bash
rg "\.close\(\)|\.release\(\)" --type py
```

**Severity**: Medium

---

### Comprehensions over Loops

**What to check**: Prefer comprehensions for simple transformations

**Good pattern**:
```python
# List comprehension
texts = [doc.content for doc in documents]

# Dict comprehension
result = {k: _commented_map_to_dict(v) for k, v in obj.items()}

# Generator for large sequences
data = (process(item) for item in large_list)
```

**Bad pattern**:
```python
texts = []
for doc in documents:
    texts.append(doc.content)
```

**Severity**: Low

---

### Use suppress() for Expected Exceptions

**What to check**: Use contextlib.suppress for expected exceptions

**Good pattern** (from server/services/dataset_service.py):
```python
from contextlib import suppress

with suppress(FileNotFoundError):
    existing_file = DataService.get_data_file_metadata_by_hash(...)
```

**Bad pattern**:
```python
try:
    existing_file = DataService.get_data_file_metadata_by_hash(...)
except FileNotFoundError:
    pass
```

**Severity**: Low

---

## Category: Pydantic Patterns

### Pydantic v2 Validator Syntax

**What to check**: Use Pydantic v2 validator syntax

**Good pattern**:
```python
from pydantic import field_validator, model_validator

class Model(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def validate_model(self) -> "Model":
        return self
```

**Bad pattern** (Pydantic v1):
```python
@validator("name")  # DEPRECATED
@root_validator     # DEPRECATED
```

**Search pattern**:
```bash
rg "@validator|@root_validator" --type py
```

**Pass criteria**: Use `@field_validator` and `@model_validator`

**Severity**: Medium

**Known Technical Debt**: The RAG module (`rag/components/metadata/metadata_config.py`) still contains Pydantic v1 `@validator` decorators and `.dict()` calls that need migration to v2 patterns. This is tracked for future cleanup.

---

### Pydantic v2 Model Config

**What to check**: Use ConfigDict instead of inner Config class

**Good pattern**:
```python
from pydantic import BaseModel, ConfigDict

class Model(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        arbitrary_types_allowed=True
    )
```

**Bad pattern** (Pydantic v1):
```python
class Model(BaseModel):
    class Config:  # DEPRECATED
        arbitrary_types_allowed = True
```

**Severity**: Low

---

### Use model_dump() Instead of dict()

**What to check**: Use Pydantic v2 serialization methods

**Good pattern**:
```python
config_dict = config.model_dump(mode="json", exclude_none=True)
json_str = model.model_dump_json()
```

**Bad pattern**:
```python
config_dict = config.dict()  # DEPRECATED
```

**Search pattern**:
```bash
rg "\.dict\(\)" --type py | rg -v "model_dump"
```

**Severity**: Medium

---

## Category: String Formatting

### F-Strings for Interpolation

**What to check**: Use f-strings over .format() or %

**Search pattern**:
```bash
rg "\.format\(|%s|%d" --type py | rg -v "strftime"
```

**Pass criteria**: Use f-strings for string interpolation

**Severity**: Low

---

### Structured Logging with Extra Dict

**What to check**: Use structured logging with extra dict, not f-strings

**Good pattern**:
```python
logger.info(
    "Processing file",
    extra={"path": path, "size": size, "task_id": task_id}
)
```

**Bad pattern**:
```python
logger.info(f"Processing file {path} with size {size}")
```

**Search pattern**:
```bash
rg 'logger\.(info|error|warning|debug)\(f"' --type py
```

**Pass criteria**: Use extra dict for structured data

**Severity**: Medium

**Recommendation**: Structured logs are easier to parse, search, and aggregate

---

## Category: Class Patterns

### Abstract Base Classes

**What to check**: Use ABC for extensible component hierarchies

**Good pattern** (from rag/core/base.py):
```python
from abc import ABC, abstractmethod

class Component(ABC):
    @abstractmethod
    def process(self, documents: list[Document]) -> ProcessingResult:
        pass

class Parser(Component):
    @abstractmethod
    def parse(self, source: str) -> ProcessingResult:
        pass
```

**Severity**: Medium

---

### Classmethod for Factory/Service Methods

**What to check**: Use @classmethod for service-layer methods

**Good pattern** (from server/services/dataset_service.py):
```python
class DatasetService:
    @classmethod
    def list_datasets(cls, namespace: str, project: str) -> list[Dataset]:
        project_config = ProjectService.load_config(namespace, project)
        return project_config.datasets or []

    @classmethod
    async def add_file_to_dataset(cls, ...) -> tuple[bool, MetadataFileContent]:
        ...
```

**Severity**: Medium

---

### Property for Computed Attributes

**What to check**: Use @property for computed or derived values

**Good pattern**:
```python
class Model:
    @property
    def config_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition.from_datamodel_tool(t)
            for t in self._client._model_config.tools or []
        ]
```

**Severity**: Low
