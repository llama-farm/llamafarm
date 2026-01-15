# Pydantic v2 Configuration Patterns Checklist

Pydantic patterns for LlamaFarm configuration models.

**Important**: The `datamodel.py` file is **auto-generated** from `schema.yaml` via `datamodel-codegen`. These patterns describe the **expected output** of generation and should be verified in the generated code. Custom validators belong in `validators.py`, not in the generated models.

**Workflow**:
1. Edit `schema.yaml` to change model structure
2. Run `nx run generate-types` to regenerate `datamodel.py`
3. Add custom validation logic in `validators.py` (function-based, not decorators)

---

## Category: Model Configuration

### Use ConfigDict Instead of Inner Config Class

**What to check**: All Pydantic models use `model_config = ConfigDict(...)` pattern

**Good pattern**:
```python
from pydantic import BaseModel, ConfigDict

class Database(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )
    name: str
    type: str
```

**Bad pattern**:
```python
class Database(BaseModel):
    class Config:  # DEPRECATED - Pydantic v1
        extra = "forbid"
```

**Search pattern**:
```bash
rg "class Config:" --type py config/
```

**Pass criteria**: No inner `Config` classes in Pydantic models

**Severity**: Medium

**Recommendation**: Use `model_config = ConfigDict(...)` for all configuration

---

### Use extra="forbid" for Strict Configuration

**What to check**: Configuration models reject unknown fields

**Good pattern**:
```python
class EmbeddingStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    type: Type1
    config: dict[str, Any] | None = None
```

**Why**: Catches typos and misconfigurations early

**Search pattern**:
```bash
rg 'extra="forbid"' --type py config/
```

**Pass criteria**: All strict configuration models use `extra="forbid"`

**Severity**: Medium

---

## Category: Constrained Types

### Use constr for String Patterns

**What to check**: String fields with patterns use `constr(pattern=...)`

**Good pattern**:
```python
from pydantic import constr

class PromptSet(BaseModel):
    name: constr(pattern=r"^[a-z][a-z0-9_]*$")
```

**Why**: Enforces naming conventions at validation time

**Search pattern**:
```bash
rg "constr\(pattern=" --type py config/
```

**Severity**: Medium

---

### Use constr for Length Constraints

**What to check**: String fields with length limits use `constr(min_length=..., max_length=...)`

**Good pattern**:
```python
name: constr(pattern=r"^[a-z][a-z0-9_]*$", min_length=1, max_length=50)
description: constr(min_length=10, max_length=500) | None = None
```

**Bad pattern**:
```python
name: str  # No length validation - could be empty or extremely long
```

**Severity**: Medium

---

### Use conint for Integer Ranges

**What to check**: Integer fields with bounds use `conint(ge=..., le=...)`

**Good pattern**:
```python
from pydantic import conint

class Parser(BaseModel):
    priority: conint(ge=0, le=1000) | None = Field(50, description="Parser priority")

class EncoderConfig(BaseModel):
    max_length: conint(ge=1) | None = None
```

**Search pattern**:
```bash
rg "conint\(" --type py config/
```

**Severity**: Medium

---

## Category: Field Definitions

### Use Field for Documentation

**What to check**: All fields have descriptions via `Field(description=...)`

**Good pattern**:
```python
class Model(BaseModel):
    name: str = Field(..., description="Model identifier (unique name)")
    provider: Provider = Field(..., description="Runtime provider for this model")
    base_url: str | None = Field(None, description="Base URL for the provider")
```

**Bad pattern**:
```python
class Model(BaseModel):
    name: str  # No description - unclear purpose
    provider: str
```

**Search pattern**:
```bash
rg 'Field\([^)]*description=' --type py config/
```

**Pass criteria**: All public fields have descriptions

**Severity**: Low

---

### Use Ellipsis for Required Fields

**What to check**: Required fields use `Field(...)` or have no default

**Good pattern**:
```python
class Database(BaseModel):
    name: constr(...) = Field(..., description="Unique database identifier")
    type: Type = Field(..., description="Database type")
    config: dict[str, Any] | None = Field(None, description="Optional config")
```

**Severity**: Low

---

### Use Field Examples for Schema Docs

**What to check**: Fields with complex formats include examples

**Good pattern**:
```python
class Parser(BaseModel):
    file_extensions: list[constr(pattern=r"^\.[a-zA-Z0-9]+$")] | None = Field(
        None,
        description='File extensions this parser handles',
        examples=[[".pdf", ".PDF"], [".csv", ".tsv"], [".md", ".markdown"]],
    )
```

**Search pattern**:
```bash
rg "examples=\[" --type py config/
```

**Severity**: Low

---

## Category: Nested Models

### Properly Type Nested Models

**What to check**: Nested configurations use typed Pydantic models

**Good pattern**:
```python
class Database(BaseModel):
    embedding_strategies: list[EmbeddingStrategy] | None = None
    retrieval_strategies: list[RetrievalStrategy] | None = None

class LlamaFarmConfig(BaseModel):
    rag: RAGStrategyConfigurationSchema | None = None
    runtime: Runtime
    mcp: Mcp | None = None
```

**Bad pattern**:
```python
class Database(BaseModel):
    embedding_strategies: list[dict] | None = None  # Loses type safety
```

**Severity**: High

---

### Use Optional with None Default for Optional Nested Models

**What to check**: Optional nested models have `None` as default

**Good pattern**:
```python
class LlamaFarmConfig(BaseModel):
    rag: RAGStrategyConfigurationSchema | None = Field(None, description="RAG configuration")
    mcp: Mcp | None = Field(None, description="MCP client configuration")
```

**Severity**: Medium

---

## Category: Enum Types

### Use Enum for Fixed Value Sets

**What to check**: Fields with fixed options use Python Enum

**Good pattern**:
```python
from enum import Enum

class Provider(Enum):
    openai = "openai"
    ollama = "ollama"
    lemonade = "lemonade"
    universal = "universal"

class Model(BaseModel):
    provider: Provider = Field(..., description="Runtime provider")
```

**Why**: Type-safe, self-documenting, and generates proper JSONSchema enum

**Search pattern**:
```bash
rg "class \w+\(Enum\)" --type py config/
```

**Severity**: Medium

---

### Enum Values Match Schema Strings

**What to check**: Enum member values are the schema string values

**Good pattern**:
```python
class Transport(Enum):
    stdio = "stdio"  # Value matches what appears in YAML
    http = "http"
    sse = "sse"
```

**Bad pattern**:
```python
class Transport(Enum):
    STDIO = "stdio"  # Member name doesn't match value - confusing
```

**Severity**: Low

---

## Category: Serialization

### Use model_dump for Dict Conversion

**What to check**: Use `model_dump()` instead of deprecated `dict()`

**Good pattern**:
```python
config_dict = config.model_dump(mode="json", exclude_none=True)
```

**Bad pattern**:
```python
config_dict = config.dict()  # DEPRECATED
```

**Search pattern**:
```bash
rg "\.dict\(\)" --type py config/
```

**Pass criteria**: No `.dict()` calls on Pydantic models

**Severity**: Medium

---

### Use mode="json" for JSON-Safe Output

**What to check**: When serializing for JSON/YAML, use `mode="json"`

**Good pattern**:
```python
# Converts Enums to strings, datetimes to ISO format, etc.
config_dict = config.model_dump(mode="json", exclude_none=True)
```

**Why**: Ensures output is JSON/YAML serializable

**Severity**: Medium

---

### Use exclude_none for Clean Output

**What to check**: Use `exclude_none=True` when optional fields should be omitted

**Good pattern**:
```python
config_dict = validated.model_dump(mode="json", exclude_none=True)
```

**Why**: Keeps configuration files clean, avoids `null` values

**Severity**: Low

---

## Category: Validation

### Use field_validator for Single Field Validation

**What to check**: Use `@field_validator` for field-specific validation

**Good pattern** (for custom validators in `validators.py`):
```python
from pydantic import field_validator

class Config(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v
```

**Bad pattern**:
```python
@validator("url")  # DEPRECATED - Pydantic v1
```

**Note**: In the config module, custom validation uses **function-based validators** in `validators.py` instead of decorator-based validators in models. This is because `datamodel.py` is generated and cannot contain custom decorators.

**Search pattern**:
```bash
rg "@validator\(" --type py config/
```

**Pass criteria**: Use `@field_validator`, not `@validator`

**Severity**: Medium

---

### Use model_validator for Cross-Field Validation

**What to check**: Use `@model_validator` for validations involving multiple fields

**Decorator pattern** (for non-generated models):
```python
from pydantic import model_validator

class Database(BaseModel):
    embedding_strategies: list[EmbeddingStrategy] | None = None
    default_embedding_strategy: str | None = None

    @model_validator(mode="after")
    def validate_default_strategy_exists(self) -> "Database":
        if self.default_embedding_strategy and self.embedding_strategies:
            names = [s.name for s in self.embedding_strategies]
            if self.default_embedding_strategy not in names:
                raise ValueError(f"default_embedding_strategy must reference an existing strategy")
        return self
```

**Function-based pattern** (used in config module's `validators.py`):
```python
def validate_llamafarm_config(config_dict: dict[str, Any]) -> None:
    """Validate constraints that JSONSchema cannot express."""
    # Cross-field validation on plain dicts before Pydantic construction
    if "default_embedding_strategy" in config_dict:
        strategies = config_dict.get("embedding_strategies", [])
        if config_dict["default_embedding_strategy"] not in [s["name"] for s in strategies]:
            raise ValueError("default_embedding_strategy must reference an existing strategy")
```

**Bad pattern**:
```python
@root_validator  # DEPRECATED - Pydantic v1
```

**Search pattern**:
```bash
rg "@root_validator" --type py config/
```

**Pass criteria**: Use `@model_validator` or function-based validators, not `@root_validator`

**Severity**: Medium

---

## Category: Generated Models

### Never Edit datamodel.py Directly

**What to check**: `datamodel.py` is generated and should not be manually edited

**Good pattern**:
- Edit `schema.yaml` to change model structure
- Run `nx run generate-types` to regenerate
- Add custom validators in `validators.py`

**Bad pattern**:
- Manually editing `datamodel.py` (changes will be overwritten)

**Pass criteria**: `datamodel.py` header shows it's generated

**Severity**: High

---

### Keep Validators Separate from Generated Code

**What to check**: Custom validation logic lives in `validators.py`

**Good pattern**:
```python
# validators.py - separate file for custom validation
def validate_llamafarm_config(config_dict: dict[str, Any]) -> None:
    """Validate constraints that JSONSchema cannot express."""
    # Uniqueness checks, cross-references, naming patterns
```

**Why**: Generated code can be regenerated without losing customizations

**Severity**: High
