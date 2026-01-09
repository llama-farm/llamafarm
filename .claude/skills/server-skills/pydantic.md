# Pydantic Patterns for LlamaFarm Server

Best practices and code review checklist for Pydantic v2 models, validation, and serialization.

## Overview

LlamaFarm uses Pydantic v2 throughout:

- **API Models** - Request/response schemas in route handlers
- **Settings** - Configuration via pydantic-settings
- **Domain Models** - Internal data structures with validation
- **Config Models** - LlamaFarmConfig from the config package

---

## Ideal Patterns

### API Request/Response Models

```python
from pydantic import BaseModel, Field, ConfigDict

class CreateProjectRequest(BaseModel):
    """Request model for creating a new project."""
    name: str = Field(..., description="The name of the project")
    config_template: str | None = Field(
        None,
        description="The config template to use for the project"
    )

class CreateProjectResponse(BaseModel):
    """Response model for project creation."""
    project: Project = Field(..., description="The created project")

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={"example": {"project": {"name": "my-project"}}}
    )
```

### Settings with pydantic-settings

```python
from pydantic_settings import BaseSettings
from pathlib import Path

default_data_dir = str(Path.home() / ".llamafarm")

class Settings(BaseSettings, env_file=".env"):
    """Application settings loaded from environment."""
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    LOG_JSON_FORMAT: bool = False

    lf_data_dir: str = default_data_dir
    celery_broker_url: str = ""

settings = Settings()  # Module-level singleton
```

### Model with Validation

```python
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

class Project(BaseModel):
    """Project domain model with validation."""
    namespace: str
    name: str
    config: LlamaFarmConfig
    validation_error: str | None = None
    last_modified: datetime | None = None

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("name cannot be empty")
        return v.strip()
```

### Union Types for Flexible Responses

```python
from pydantic import BaseModel
from config.datamodel import LlamaFarmConfig

class Project(BaseModel):
    """Project that can hold validated config or raw dict."""
    namespace: str
    name: str
    config: LlamaFarmConfig | dict = Field(
        ...,
        description="The configuration (validated model or raw dict)"
    )
    validation_error: str | None = Field(
        None,
        description="Validation error message if config has issues"
    )
```

### Serialization with model_dump

```python
# Serialize to dict
config_dict = config.model_dump(mode="json", exclude_none=True)

# Serialize for JSON response
response_data = model.model_dump(mode="json", exclude_none=True)

# Serialize specific fields
partial = model.model_dump(include={"name", "namespace"})
```

---

## Checklist

### 1. Use Pydantic v2 Patterns

**Description:** Use Pydantic v2 APIs, not deprecated v1 patterns.

**Search Pattern:**
```bash
grep -rn "class Config:" server/ | grep -v ".venv"
```

**Pass Criteria:** No `class Config:` inside models. Use `model_config = ConfigDict(...)`.

**Severity:** Medium

**Recommendation:** Migrate to v2:
```python
# v1 (deprecated)
class MyModel(BaseModel):
    class Config:
        str_strip_whitespace = True

# v2 (correct)
class MyModel(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
```

---

### 2. Field Descriptions Provided

**Description:** API-facing models should have Field descriptions for OpenAPI docs.

**Search Pattern:**
```bash
grep -rn "Field(\.\.\." server/api/routers/ | grep -v "description="
```

**Pass Criteria:** Required fields have descriptions.

**Severity:** Low

**Recommendation:** Add descriptions:
```python
name: str = Field(..., description="The name of the resource")
```

---

### 3. Use model_dump Not dict()

**Description:** Use `model_dump()` instead of deprecated `.dict()` method.

**Search Pattern:**
```bash
grep -rn "\.dict()" server/ | grep -v ".venv"
```

**Pass Criteria:** No `.dict()` calls on Pydantic models.

**Severity:** Medium

**Recommendation:** Replace with `model_dump()`:
```python
# v1 (deprecated)
data = model.dict(exclude_none=True)

# v2 (correct)
data = model.dump(mode="json", exclude_none=True)
```

---

### 4. Use model_validate Not parse_obj

**Description:** Use `model_validate()` instead of deprecated `parse_obj()`.

**Search Pattern:**
```bash
grep -rn "parse_obj\|parse_raw" server/ | grep -v ".venv"
```

**Pass Criteria:** No `parse_obj()` or `parse_raw()` calls.

**Severity:** Medium

**Recommendation:** Replace with v2 methods:
```python
# v1 (deprecated)
model = MyModel.parse_obj(data)

# v2 (correct)
model = MyModel.model_validate(data)
```

---

### 5. Validators Use @field_validator

**Description:** Use `@field_validator` decorator, not deprecated `@validator`.

**Search Pattern:**
```bash
grep -rn "@validator" server/ | grep -v ".venv"
```

**Pass Criteria:** No `@validator` decorators. Use `@field_validator`.

**Severity:** Medium

**Recommendation:**
```python
# v1 (deprecated)
@validator("name")
def validate_name(cls, v):
    return v.strip()

# v2 (correct)
@field_validator("name")
@classmethod
def validate_name(cls, v: str) -> str:
    return v.strip()
```

---

### 6. Model Serializers Use @model_serializer

**Description:** Use `@model_serializer` for custom serialization, not `__json__`.

**Search Pattern:**
```bash
grep -rn "def __json__" server/ | grep -v ".venv"
```

**Pass Criteria:** No `__json__` methods. Use `@model_serializer`.

**Severity:** Low

**Recommendation:**
```python
from pydantic import model_serializer

class MyModel(BaseModel):
    @model_serializer
    def serialize(self) -> dict:
        return {"custom": "serialization"}
```

---

### 7. Proper Optional Type Hints

**Description:** Use `T | None` syntax for optional fields, not `Optional[T]`.

**Search Pattern:**
```bash
grep -rn "Optional\[" server/ | grep -v ".venv" | grep -v "typing import"
```

**Pass Criteria:** Use modern `T | None` syntax.

**Severity:** Low

**Recommendation:**
```python
# Old style
from typing import Optional
field: Optional[str] = None

# Modern style (Python 3.10+)
field: str | None = None
```

---

### 8. ConfigDict Used for Model Configuration

**Description:** Use ConfigDict for model configuration, not class Config.

**Search Pattern:**
```bash
grep -rn "model_config\s*=" server/api/ | grep -v "ConfigDict"
```

**Pass Criteria:** `model_config` assigned from `ConfigDict()`.

**Severity:** Low

**Recommendation:**
```python
from pydantic import ConfigDict

class MyModel(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )
```

---

### 9. Exclude None in JSON Responses

**Description:** API responses should exclude None values for cleaner JSON.

**Search Pattern:**
```bash
grep -rn "model_dump(" server/api/ | grep -v "exclude_none"
```

**Pass Criteria:** API serialization uses `exclude_none=True`.

**Severity:** Low

**Recommendation:**
```python
# In response
return model.model_dump(mode="json", exclude_none=True)

# Or use custom response class
class NoNoneJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return super().render(jsonable_encoder(content, exclude_none=True))
```

---

### 10. Proper Error Extraction from ValidationError

**Description:** Extract structured errors from Pydantic ValidationError.

**Search Pattern:**
```bash
grep -rn "ValidationError" server/ | grep -v "import\|.venv"
```

**Pass Criteria:** ValidationError handling extracts structured error messages.

**Severity:** Medium

**Recommendation:**
```python
except ValidationError as e:
    if hasattr(e, "errors") and callable(e.errors):
        error_details = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err.get("loc", []))
            msg = err.get("msg", "validation error")
            error_details.append(f"{loc}: {msg}")
        validation_error_msg = "; ".join(error_details[:5])
```

---

## Anti-Patterns to Avoid

### 1. Mutable Default Arguments

```python
# BAD - mutable default
class Config(BaseModel):
    items: list = []

# GOOD - use Field with default_factory
class Config(BaseModel):
    items: list = Field(default_factory=list)
```

### 2. Mixing v1 and v2 APIs

```python
# BAD - mixed APIs
class MyModel(BaseModel):
    class Config:  # v1
        ...
    model_config = ConfigDict(...)  # v2

# GOOD - v2 only
class MyModel(BaseModel):
    model_config = ConfigDict(...)
```

### 3. Bare dict Instead of TypedDict or Model

```python
# BAD - no type safety
def process(data: dict) -> dict:
    return {"result": data["value"]}

# GOOD - typed models
class InputData(BaseModel):
    value: str

class OutputData(BaseModel):
    result: str

def process(data: InputData) -> OutputData:
    return OutputData(result=data.value)
```

### 4. Missing Field Constraints

```python
# BAD - no constraints
class User(BaseModel):
    age: int
    email: str

# GOOD - with constraints
class User(BaseModel):
    age: int = Field(..., ge=0, le=150)
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
```
