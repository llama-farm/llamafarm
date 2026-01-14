# Security Best Practices Checklist

Security patterns for LlamaFarm Python components.

---

## Category: Injection Prevention

### No eval() or exec() with User Input

**What to check**: Never use eval or exec with untrusted data

**Search pattern**:
```bash
rg "\beval\(|\bexec\(" --type py
```

**Pass criteria**: No eval/exec calls, or only with hardcoded/trusted strings

**Severity**: Critical

**Why it matters**: eval/exec can execute arbitrary code, leading to complete system compromise

---

### No Shell Injection

**What to check**: Avoid shell=True with subprocess

**Bad pattern**:
```python
import subprocess

# DANGEROUS - allows shell injection
subprocess.run(f"process {user_input}", shell=True)
os.system(f"cat {filename}")  # Also dangerous
```

**Good pattern**:
```python
import subprocess

# Safe - no shell interpretation
subprocess.run(["process", user_input], shell=False)
subprocess.run(["cat", filename])
```

**Search pattern**:
```bash
rg "subprocess.*shell=True|os\.system\(" --type py
```

**Pass criteria**: No shell=True with variable input; no os.system()

**Severity**: Critical

---

### SQL Injection Prevention

**What to check**: Use parameterized queries, never string interpolation

**Bad pattern**:
```python
# DANGEROUS - SQL injection
query = f"SELECT * FROM users WHERE name = '{name}'"
cursor.execute(query)

query = "SELECT * FROM docs WHERE id = " + doc_id
```

**Good pattern**:
```python
# Safe - parameterized query
query = "SELECT * FROM users WHERE name = :name"
cursor.execute(query, {"name": name})

# With SQLAlchemy
stmt = select(User).where(User.name == name)
```

**Search pattern**:
```bash
rg "f\".*SELECT|f'.*SELECT|\".*SELECT.*\{|'.*SELECT.*\{" --type py
```

**Pass criteria**: No string interpolation in SQL queries

**Severity**: Critical

---

## Category: Path Traversal Prevention

### Validate File Paths

**What to check**: Validate paths to prevent directory traversal attacks

**Bad pattern**:
```python
def get_file(filename: str):
    # DANGEROUS - allows ../../../etc/passwd
    return open(f"/data/{filename}").read()
```

**Good pattern**:
```python
from pathlib import Path

BASE_DIR = Path("/data").resolve()

def get_file(filename: str) -> str:
    # Resolve the full path
    requested_path = (BASE_DIR / filename).resolve()

    # Verify it's within allowed directory
    if not requested_path.is_relative_to(BASE_DIR):
        raise ValueError(f"Invalid path: {filename}")

    if not requested_path.is_file():
        raise FileNotFoundError(f"File not found: {filename}")

    return requested_path.read_text()
```

**Search pattern**:
```bash
rg "open\(.*\+|Path\(.*\+" --type py
```

**Pass criteria**: All file paths validated against base directory

**Severity**: Critical

---

### Resolve Paths Before Use

**What to check**: Always resolve() paths before security checks

**Good pattern**:
```python
from pathlib import Path

def safe_path(base_dir: Path, user_path: str) -> Path:
    # Resolve to absolute path (resolves .., symlinks)
    full_path = (base_dir / user_path).resolve()

    # Check containment AFTER resolution
    if not full_path.is_relative_to(base_dir.resolve()):
        raise ValueError("Path traversal detected")

    return full_path
```

**Why resolve() matters**:
- Removes `..` components
- Follows symlinks
- Converts to absolute path

**Severity**: High

---

### Symlink Protection

**What to check**: Consider symlink attacks in path validation

**Good pattern**:
```python
def safe_read(base_dir: Path, filename: str) -> str:
    path = (base_dir / filename).resolve()

    # Check containment
    if not path.is_relative_to(base_dir.resolve()):
        raise ValueError("Path outside allowed directory")

    # Optionally reject symlinks
    if path.is_symlink():
        raise ValueError("Symlinks not allowed")

    return path.read_text()
```

**Severity**: Medium

---

## Category: Secrets Management

### No Hardcoded Secrets

**What to check**: No secrets in source code

**Search pattern**:
```bash
rg -i "(api_key|apikey|password|secret|token|credential)\s*=\s*['\"][^'\"]+['\"]" --type py
```

**Pass criteria**: All secrets loaded from environment variables or secret managers

**Severity**: Critical

---

### Environment Variables for Secrets

**What to check**: Use pydantic-settings for configuration

**Good pattern** (from server/core/settings.py):
```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Secrets from environment
    DATABASE_URL: str
    API_KEY: str
    JWT_SECRET: str

    # Optional secrets with defaults only for non-sensitive values
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

settings = Settings()
```

**Severity**: High

---

### No Secrets in Logs

**What to check**: Never log sensitive data

**Bad pattern**:
```python
logger.info(f"Connecting with token: {api_token}")
logger.debug(f"User password: {password}")
```

**Good pattern**:
```python
logger.info("Connecting to API", extra={"token_prefix": api_token[:8] + "..."})
logger.debug("User authentication attempt", extra={"user_id": user_id})
```

**Search pattern**:
```bash
rg -i "logger?\.(info|debug|warning|error).*password|token|secret|key" --type py
```

**Pass criteria**: No secrets in log statements

**Severity**: Critical

---

### Secrets in Error Messages

**What to check**: Don't expose secrets in exceptions

**Bad pattern**:
```python
raise AuthError(f"Invalid token: {token}")
```

**Good pattern**:
```python
raise AuthError("Invalid or expired authentication token")
```

**Severity**: High

---

## Category: Input Validation

### Pydantic for All External Input

**What to check**: Use Pydantic models for request validation

**Good pattern**:
```python
from pydantic import BaseModel, Field, EmailStr

class CreateUserRequest(BaseModel):
    email: EmailStr
    name: str = Field(min_length=1, max_length=100)
    age: int = Field(ge=0, le=150)

@app.post("/users")
async def create_user(request: CreateUserRequest):
    # request is already validated
    return await user_service.create(request)
```

**Severity**: High

---

### Validate at API Boundaries

**What to check**: Validate input as early as possible

**Good pattern**:
```python
@app.post("/documents")
async def create_document(
    request: DocumentRequest,  # Validated by Pydantic
    current_user: User = Depends(get_current_user),  # Authenticated
):
    # By this point, input is validated and user is authenticated
    return await document_service.create(request, current_user)
```

**Severity**: Medium

---

### Sanitize for Output Context

**What to check**: Escape output appropriately for context

**Good pattern**:
```python
import html

# For HTML output
safe_content = html.escape(user_content)

# For JSON (Pydantic handles this)
return response.model_dump_json()
```

**Severity**: Medium (for web responses)

---

## Category: Deserialization Safety

### Safe YAML Loading

**What to check**: Always use safe_load for YAML

**Bad pattern**:
```python
import yaml

# DANGEROUS - allows arbitrary Python object instantiation
data = yaml.load(user_data)  # Full loader
data = yaml.load(user_data, Loader=yaml.FullLoader)  # Still dangerous
```

**Good pattern**:
```python
import yaml

# Safe - only allows basic types
data = yaml.safe_load(user_data)

# Or use ruamel.yaml with safe settings
from ruamel.yaml import YAML
yaml = YAML(typ='safe')
```

**Search pattern**:
```bash
rg "yaml\.load\(" --type py | rg -v "safe_load"
```

**Pass criteria**: Only safe_load used for untrusted YAML

**Severity**: Critical

---

### Avoid Pickle with Untrusted Data

**What to check**: Never unpickle untrusted data

**Search pattern**:
```bash
rg "pickle\.load|pickle\.loads" --type py
```

**Pass criteria**: No pickle with user-provided data; use JSON instead

**Severity**: Critical

**Why it matters**: Pickle can execute arbitrary code during deserialization

---

### JSON is Safe (but validate structure)

**What to check**: JSON is safe to parse, but validate the structure

**Good pattern**:
```python
import json
from pydantic import BaseModel

# Parse JSON (safe)
data = json.loads(user_input)

# Validate structure with Pydantic
validated = MyModel.model_validate(data)
```

**Severity**: Low

---

## Category: Authentication & Authorization

### Token Validation

**What to check**: Properly validate authentication tokens

**Good pattern**:
```python
import jwt
from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=["HS256"]
        )
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = await get_user(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user
```

**Severity**: High

---

### Dependency Injection for Auth

**What to check**: Use FastAPI dependencies for authentication

**Good pattern**:
```python
from fastapi import Depends

@app.get("/protected")
async def protected_route(
    current_user: User = Depends(get_current_user)
):
    return {"user": current_user.email}

@app.delete("/admin/users/{user_id}")
async def delete_user(
    user_id: str,
    current_user: User = Depends(require_admin),  # Admin check
):
    return await user_service.delete(user_id)
```

**Pass criteria**: All protected routes use auth dependencies

**Severity**: High

---

### Authorization Checks

**What to check**: Verify user has permission for the action

**Good pattern**:
```python
async def get_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
):
    document = await document_service.get(document_id)
    if document is None:
        raise HTTPException(status_code=404)

    # Authorization check
    if document.owner_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")

    return document
```

**Severity**: High

---

## Category: Rate Limiting

### API Rate Limiting

**What to check**: Implement rate limiting for public endpoints

**Good pattern**:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/api/public")
@limiter.limit("10/minute")
async def public_endpoint(request: Request):
    ...

@app.post("/api/login")
@limiter.limit("5/minute")
async def login(request: Request):
    ...
```

**Severity**: Medium

---

### Resource Limits

**What to check**: Limit resource consumption

**Good pattern**:
```python
from pydantic import BaseModel, Field

class UploadRequest(BaseModel):
    # Limit file size
    content: str = Field(max_length=10_000_000)  # 10MB

class BatchRequest(BaseModel):
    # Limit batch size
    items: list[str] = Field(max_length=100)

# Timeout for long operations
result = await asyncio.wait_for(
    long_operation(),
    timeout=30.0
)
```

**Severity**: Medium

---

## Category: Dependency Security

### Pin Dependencies

**What to check**: Pin exact versions in production

**Good pattern** (pyproject.toml):
```toml
[project]
dependencies = [
    "fastapi>=0.100.0,<0.200.0",
    "pydantic>=2.0.0,<3.0.0",
]

# Or use uv.lock for exact pinning
```

**Severity**: Medium

---

### Audit Dependencies

**What to check**: Regularly check for vulnerabilities

**Commands**:
```bash
# With pip-audit
pip-audit

# With safety
safety check

# With uv
uv pip audit
```

**Severity**: Medium

---

## Category: Secure Defaults

### HTTPS in Production

**What to check**: Enforce HTTPS for production APIs

**Good pattern**:
```python
from fastapi import FastAPI
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

app = FastAPI()

if settings.ENVIRONMENT == "production":
    app.add_middleware(HTTPSRedirectMiddleware)
```

**Severity**: High

---

### Secure Headers

**What to check**: Set security headers

**Good pattern**:
```python
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response

app.add_middleware(SecurityHeadersMiddleware)
```

**Severity**: Medium

---

### CORS Configuration

**What to check**: Configure CORS appropriately

**Good pattern**:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,  # Not ["*"] in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
```

**Bad pattern**:
```python
# DANGEROUS in production
allow_origins=["*"]
```

**Severity**: Medium
