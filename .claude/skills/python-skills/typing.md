# Type Hints and Pydantic Checklist

Type annotation patterns for LlamaFarm Python components.

---

## Category: Basic Type Hints

### Function Return Types

**What to check**: All public functions should have return type hints

**Good pattern**:
```python
def process_document(
    content: str,
    metadata: dict[str, Any] | None = None,
) -> ProcessingResult:
    ...

async def load_model(model_id: str) -> BaseModel:
    ...
```

**Search pattern**:
```bash
rg "^def [a-z_]+\([^)]*\):" --type py | rg -v " -> "
```

**Pass criteria**: All public functions have return type annotations

**Severity**: Medium

---

### None Return Type

**What to check**: Functions returning None should be explicit

**Good pattern**:
```python
def log_event(event: str) -> None:
    logger.info(event)

async def cleanup() -> None:
    await close_connections()
```

**Severity**: Low

---

## Category: Modern Type Syntax (Python 3.10+)

### Union Types with Pipe Operator

**What to check**: Use `|` syntax for unions (PEP 604)

**Good**: `str | None`, `int | float`
**Bad**: `Optional[str]`, `Union[int, float]`

**Search pattern**:
```bash
rg "Optional\[|Union\[" --type py
```

**Pass criteria**: Use `X | Y` syntax

**Severity**: Low

**Recommendation**: Run `ruff check --fix --select UP` to auto-upgrade

---

### Built-in Generic Types (PEP 585)

**What to check**: Use lowercase built-in generics

**Good**:
```python
items: list[str]
mapping: dict[str, int]
coordinates: tuple[float, float]
unique_ids: set[int]
```

**Bad**:
```python
from typing import List, Dict, Tuple, Set
items: List[str]  # DEPRECATED
```

**Search pattern**:
```bash
rg "from typing import.*(List|Dict|Tuple|Set)" --type py
```

**Pass criteria**: Use lowercase built-in generics

**Severity**: Low

---

### collections.abc for Abstract Types

**What to check**: Use collections.abc for abstract types

**Good pattern**:
```python
from collections.abc import Sequence, Mapping, Iterator, Callable, AsyncGenerator

def process(items: Sequence[str]) -> Iterator[str]:
    for item in items:
        yield item.upper()

async def stream_data() -> AsyncGenerator[bytes]:
    ...
```

**Bad pattern**:
```python
from typing import Sequence, Iterator  # DEPRECATED in 3.9+
```

**Severity**: Low

---

## Category: Variable Annotations

### Complex Variable Types

**What to check**: Complex variables should have type annotations

**Good pattern**:
```python
# Module-level
_models: dict[str, BaseModel] = {}
_cleanup_task: asyncio.Task[None] | None = None

# Inside functions
results: list[Document] = []
cache: dict[str, Model] = {}
```

**Severity**: Low

---

### Final for Constants

**What to check**: Use Final for constants

**Good pattern**:
```python
from typing import Final

MAX_RETRIES: Final = 3
DEFAULT_TIMEOUT: Final[float] = 30.0
API_VERSION: Final = "v1"
```

**Severity**: Low

---

## Category: Generic Types

### TypeVar for Generic Functions

**What to check**: Use TypeVar for type-preserving generics

**Good pattern**:
```python
from typing import TypeVar

T = TypeVar("T")

def first(items: Sequence[T]) -> T | None:
    return items[0] if items else None

# With bounds
ModelT = TypeVar("ModelT", bound=BaseModel)

def load(model_class: type[ModelT], data: dict) -> ModelT:
    return model_class(**data)
```

**Severity**: Medium

---

### Protocol for Structural Typing

**What to check**: Use Protocol for duck typing interfaces

**Good pattern** (from rag/core/base.py pattern):
```python
from typing import Protocol

class Processable(Protocol):
    def process(self, documents: list[Document]) -> ProcessingResult: ...

class Embeddable(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def get_embedding_dimension(self) -> int: ...

def run_processor(p: Processable) -> ProcessingResult:
    return p.process([])
```

**Search pattern**:
```bash
rg "class \w+\(Protocol\)" --type py
```

**Severity**: Medium

---

### TypedDict for Dict Structures

**What to check**: Use TypedDict for known dict structures

**Good pattern**:
```python
from typing import TypedDict, NotRequired

class DocumentMetadata(TypedDict):
    source: str
    page: int
    author: NotRequired[str]
    timestamp: NotRequired[float]

def process_doc(meta: DocumentMetadata) -> None:
    print(meta["source"])  # Type-safe access
```

**Severity**: Medium

---

## Category: Callable Types

### Callable Signatures

**What to check**: Use Callable for function parameters

**Good pattern**:
```python
from collections.abc import Callable

def apply(
    items: list[T],
    transform: Callable[[T], R],
) -> list[R]:
    return [transform(item) for item in items]

# For async callables
async def with_retry(
    func: Callable[[], Awaitable[T]],
    max_retries: int = 3,
) -> T:
    ...
```

**Severity**: Medium

---

### ParamSpec for Decorators

**What to check**: Use ParamSpec for type-safe decorators

**Good pattern**:
```python
from typing import ParamSpec, TypeVar
from functools import wraps

P = ParamSpec("P")
R = TypeVar("R")

def logged(func: Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        logger.info(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper
```

**Severity**: Low

---

## Category: Pydantic Models

### Pydantic v2 Field Syntax

**What to check**: Use Pydantic v2 Field patterns

**Good pattern**:
```python
from pydantic import BaseModel, Field, ConfigDict

class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] = "float"

    model_config = ConfigDict(str_strip_whitespace=True)
```

**Severity**: Medium

---

### Annotated with Field Constraints

**What to check**: Use Annotated for complex field constraints

**Good pattern**:
```python
from typing import Annotated
from pydantic import Field

class Model(BaseModel):
    name: Annotated[str, Field(min_length=1, max_length=100)]
    count: Annotated[int, Field(ge=0, le=1000)]
    ratio: Annotated[float, Field(ge=0.0, le=1.0)]
```

**Severity**: Medium

---

### Field Validators (Pydantic v2)

**What to check**: Use v2 validator decorators

**Good pattern**:
```python
from pydantic import field_validator, model_validator

class Config(BaseModel):
    url: str
    port: int

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

    @model_validator(mode="after")
    def validate_config(self) -> "Config":
        # Cross-field validation
        return self
```

**Bad pattern**:
```python
@validator("url")  # DEPRECATED - Pydantic v1
@root_validator  # DEPRECATED - Pydantic v1
```

**Search pattern**:
```bash
rg "@validator|@root_validator" --type py
```

**Severity**: Medium

---

### Generic Pydantic Models

**What to check**: Use Generic for reusable response models

**Good pattern**:
```python
from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")

class Response(BaseModel, Generic[T]):
    data: T
    success: bool = True
    error: str | None = None

class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    per_page: int

# Usage
Response[User](data=user, success=True)
PaginatedResponse[Document](items=docs, total=100, page=1, per_page=10)
```

**Severity**: Low

---

## Category: Type Narrowing

### Type Guards

**What to check**: Use TypeGuard for runtime type narrowing

**Good pattern**:
```python
from typing import TypeGuard

def is_string_list(val: list[Any]) -> TypeGuard[list[str]]:
    return all(isinstance(x, str) for x in val)

def process(items: list[Any]) -> None:
    if is_string_list(items):
        # items is now list[str]
        for s in items:
            print(s.upper())
```

**Severity**: Low

---

### isinstance Narrowing

**What to check**: Use isinstance for type narrowing

**Good pattern**:
```python
def process(value: str | int | None) -> str:
    if value is None:
        return "default"
    if isinstance(value, int):
        return str(value)
    # value is now str
    return value.upper()
```

**Severity**: Low

---

## Category: Async Types

### AsyncGenerator Type

**What to check**: Async generators have proper return types

**Good pattern** (from server/agents/base/agent.py):
```python
from collections.abc import AsyncGenerator

async def run_async_stream(
    self,
    messages: list[LFChatCompletionMessageParam] | None = None,
) -> AsyncGenerator[LFChatCompletionChunk]:
    async for chunk in self._client.stream_chat(messages=messages):
        yield chunk
```

**Severity**: Medium

---

### Awaitable and Coroutine Types

**What to check**: Use proper async type hints

**Good pattern**:
```python
from collections.abc import Awaitable, Coroutine

async def fetch_data() -> dict[str, Any]:
    ...

# Type for a coroutine parameter
def schedule(coro: Awaitable[T]) -> asyncio.Task[T]:
    return asyncio.create_task(coro)
```

**Severity**: Low

---

## Category: Type Aliases

### Type Aliases for Complex Types

**What to check**: Create type aliases for complex or repeated types

**Good pattern**:
```python
from typing import TypeAlias

# Simple alias
JsonDict: TypeAlias = dict[str, Any]
Embedding: TypeAlias = list[float]

# Complex alias
MessageHistory: TypeAlias = list[dict[str, str | list[dict[str, str]]]]
Handler: TypeAlias = Callable[[Request], Awaitable[Response]]
```

**Severity**: Low

---

### NewType for Distinct Types

**What to check**: Use NewType for distinct string/int types

**Good pattern**:
```python
from typing import NewType

UserId = NewType("UserId", str)
DocumentId = NewType("DocumentId", str)

def get_user(user_id: UserId) -> User:
    ...

# Usage
user_id = UserId("user-123")
get_user(user_id)  # OK
get_user("user-123")  # Type error - need explicit NewType wrapper
```

**Severity**: Low

---

## Category: Literal Types

### Literal for String Enums

**What to check**: Use Literal for fixed string values

**Good pattern**:
```python
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
Status = Literal["pending", "running", "completed", "failed"]

def set_log_level(level: LogLevel) -> None:
    ...

class Task(BaseModel):
    status: Status = "pending"
```

**Severity**: Medium
