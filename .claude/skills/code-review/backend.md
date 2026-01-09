# Backend Review Checklist

Domain-specific review items for Python, FastAPI, and async codebases.

Add these categories to the summary table:
- Security (Backend)
- FastAPI Patterns
- Pydantic Validation
- Async Patterns
- Error Handling (Backend)
- Database/ORM

---

## Category: Security (Backend)

### SQL Injection

**Search patterns**:
```bash
# Raw SQL with string formatting
grep -rE "execute\(.*%|execute\(.*\.format\(|execute\(.*f['\"]" --include="*.py"

# String concatenation in queries
grep -rE "SELECT.*\+|INSERT.*\+|UPDATE.*\+|DELETE.*\+" --include="*.py"
```

**Pass criteria**: All queries use parameterized statements or ORM
**Severity**: Critical

---

### Path Traversal

**Search patterns**:
```bash
# Unsanitized file paths
grep -rE "open\(.*\+|Path\(.*\+|os\.path\.join\(.*request" --include="*.py"

# Direct user input in file operations
grep -rE "with open\(|\.read\(|\.write\(" --include="*.py" -B 3 | grep -E "request\.|params\.|query\."
```

**Pass criteria**: All file paths validated and sanitized
**Severity**: Critical

---

### SSRF (Server-Side Request Forgery)

**Search patterns**:
```bash
# HTTP requests with user-controlled URLs
grep -rE "requests\.(get|post|put|delete)\(.*request\.|httpx\.(get|post)\(.*request\." --include="*.py"

# aiohttp with user input
grep -rE "session\.(get|post)\(" --include="*.py" -B 5 | grep -E "request\.|params\."
```

**Pass criteria**: All external URLs validated against allowlist
**Severity**: High

---

### Insecure Deserialization

**Search patterns**:
```bash
# Pickle usage
grep -rE "pickle\.(load|loads)\(|cPickle\." --include="*.py"

# YAML unsafe load (must use SafeLoader or safe_load)
grep -rE "yaml\.load\(" --include="*.py" | grep -v "Loader=SafeLoader\|Loader=yaml\.SafeLoader\|safe_load"
```

**Pass criteria**: No pickle with untrusted data; YAML uses safe_load
**Severity**: Critical

---

### Weak Cryptography

**Search patterns**:
```bash
# MD5/SHA1 for passwords
grep -rE "md5\(|sha1\(" --include="*.py" | grep -iE "password|secret|token"

# Hardcoded crypto keys
grep -rE "key\s*=\s*['\"][^'\"]{8,}" --include="*.py"
```

**Pass criteria**: Use bcrypt/argon2 for passwords; no hardcoded keys
**Severity**: High

---

## Category: FastAPI Patterns

### Missing response_model

**Search patterns**:
```bash
# Routes without response_model
grep -rE "@(app|router)\.(get|post|put|patch|delete)\(" --include="*.py" | grep -v "response_model"
```

**Pass criteria**: All endpoints have explicit response_model
**Severity**: Medium

---

### Untyped Request Bodies

**Search patterns**:
```bash
# Route handlers with untyped parameters (missing Pydantic models)
grep -rE "@(app|router)\.(post|put|patch)\(" --include="*.py" -A 3 | grep -E "def\s+\w+\([^)]*:\s*(dict|Dict|Any)\b"

# Body() without type annotation
grep -rE "Body\(\.\.\.\)" --include="*.py"

# Parameters with generic dict type in route handlers
grep -rE "def\s+\w+\([^)]*:\s*dict\b" --include="*.py"
```

**Pass criteria**: All request bodies have Pydantic model types
**Severity**: High

---

### Missing Status Codes

**Search patterns**:
```bash
# POST without 201, DELETE without 204
grep -rE "@(app|router)\.post\(" --include="*.py" | grep -v "status_code"
grep -rE "@(app|router)\.delete\(" --include="*.py" | grep -v "status_code"
```

**Pass criteria**: Appropriate HTTP status codes for each operation
**Severity**: Low

---

### Sync Functions in Async Routes

**Search patterns**:
```bash
# Async route calling sync functions
grep -rE "async def.*:" --include="*.py" -A 20 | grep -E "time\.sleep|requests\.(get|post)|open\("
```

**Pass criteria**: Async routes use async I/O (httpx, aiofiles, asyncio.sleep)
**Severity**: High

---

### Missing Dependency Injection

**Search patterns**:
```bash
# Direct instantiation in routes
grep -rE "def\s+\w+\(" --include="*.py" -A 10 | grep -E "^\s+\w+\s*=\s*\w+Service\(|^\s+\w+\s*=\s*\w+Repository\("
```

**Pass criteria**: Services/repos injected via Depends()
**Severity**: Medium

---

### N+1 Query Patterns

**Search technique**:
- Look for loops that execute queries
- Check for missing eager loading

**Search patterns**:
```bash
# Queries inside loops
grep -rE "for\s+\w+\s+in" --include="*.py" -A 5 | grep -E "\.query\(|\.filter\(|\.get\("
```

**Pass criteria**: No queries inside loops; use eager loading
**Severity**: High

---

## Category: Pydantic Validation

### Missing Field Validators

**Search patterns**:
```bash
# Models without validators for sensitive fields
grep -rE "email:|password:|url:|phone:" --include="*.py" -B 5 -A 5 | grep -v "@validator\|@field_validator"
```

**Pass criteria**: Sensitive fields have validation
**Severity**: Medium

---

### Overly Permissive Models

**Search patterns**:
```bash
# Models that accept extra fields
grep -rE "extra\s*=\s*['\"]allow['\"]|Config:.*extra\s*=\s*'allow'" --include="*.py"

# Models with Any type
grep -rE ":\s*Any\b" --include="*.py"
```

**Pass criteria**: Models are strict; no arbitrary field acceptance
**Severity**: Medium

---

### Missing Field Constraints

**Search patterns**:
```bash
# String fields without max_length
grep -rE ":\s*str\s*$|:\s*str\s*=" --include="*.py" | grep -v "max_length\|Field\("

# Numeric fields without bounds
grep -rE ":\s*int\s*$|:\s*float\s*$" --include="*.py" | grep -v "ge=\|le=\|gt=\|lt=\|Field\("
```

**Pass criteria**: Fields have appropriate constraints
**Severity**: Low

---

### Untyped Optional Fields

**Search patterns**:
```bash
# Optional without explicit type
grep -rE "Optional\[Any\]|:\s*Optional\s*$" --include="*.py"

# None default without Optional
grep -rE "=\s*None\s*$" --include="*.py" | grep -v "Optional\|None\s*\|"
```

**Pass criteria**: Optional fields have explicit inner types
**Severity**: Medium

---

## Category: Async Patterns

### Blocking Calls in Async Functions

**Search patterns**:
```bash
# Blocking I/O in async
grep -rE "async def" --include="*.py" -A 30 | grep -E "time\.sleep\(|requests\.|open\(|\.read\(\)|\.write\("

# Blocking database calls
grep -rE "async def" --include="*.py" -A 30 | grep -E "\.execute\(|\.query\(" | grep -v "await"
```

**Pass criteria**: All I/O in async functions is awaited
**Severity**: Critical

---

### Missing Await Keywords

**Search patterns**:
```bash
# Find async function calls without await (two-step process):
# Step 1: Extract async function names defined in the codebase
grep -rE "async def (\w+)\(" --include="*.py" -ho | sed 's/async def //' | sed 's/($//' | sort -u

# Step 2: For each async function name, check for calls without await
# Example: if 'fetch_data' is async, search for calls not preceded by await
grep -rE "\bfetch_data\s*\(" --include="*.py" | grep -v "await\s\+fetch_data\|await fetch_data"

# Common async methods called without await
grep -rE "\.(read|write|execute|commit|send|recv|connect|close)\s*\(" --include="*.py" | grep -v "await"
```

**Pass criteria**: All coroutines are awaited
**Severity**: Critical

---

### Improper Task Handling

**Search patterns**:
```bash
# Fire-and-forget tasks
grep -rE "asyncio\.create_task\(" --include="*.py" | grep -v "await\|tasks\.append\|gather"

# Missing exception handling in tasks
grep -rE "create_task\(" --include="*.py" -A 10 | grep -v "try:\|except:"
```

**Pass criteria**: Background tasks are tracked and exceptions handled
**Severity**: High

---

### Missing Timeout Handling

**Search patterns**:
```bash
# HTTP requests without timeout
grep -rE "httpx\.(get|post|put|delete)\(|requests\.(get|post)" --include="*.py" | grep -v "timeout="

# aiohttp without timeout
grep -rE "session\.(get|post)\(" --include="*.py" | grep -v "timeout="
```

**Pass criteria**: All external calls have timeouts
**Severity**: High

---

## Category: Error Handling (Backend)

### Generic Exception Catches

**Search patterns**:
```bash
# Bare except or Exception catch
grep -rE "except\s*:|except\s+Exception:" --include="*.py"
```

**Pass criteria**: Catch specific exceptions, not bare Exception
**Severity**: Medium

---

### Missing HTTPException Usage

**Search patterns**:
```bash
# Routes that raise generic exceptions
grep -rE "raise\s+\w+Error\(|raise\s+Exception\(" --include="*.py" | grep -v "HTTPException"
```

**Pass criteria**: Routes raise HTTPException with proper status codes
**Severity**: Medium

---

### Swallowed Exceptions

**Search patterns**:
```bash
# Except with pass or continue
grep -rE "except.*:\s*$" --include="*.py" -A 1 | grep -E "pass$|continue$"

# Except with only logging
grep -rE "except.*:" --include="*.py" -A 2 | grep -E "logger\.(error|warning|info)" | grep -v "raise"
```

**Pass criteria**: Exceptions are handled or re-raised
**Severity**: High

---

### Missing Error Response Schemas

**Search patterns**:
```bash
# HTTPException without detail structure
grep -rE "HTTPException\(" --include="*.py" | grep -v "detail={"
```

**Pass criteria**: Error responses have consistent, typed structure
**Severity**: Low

---

## Category: Database/ORM

### Missing Session Management

**Search patterns**:
```bash
# Sessions not closed or context-managed
grep -rE "Session\(\)" --include="*.py" | grep -v "with\s|yield"

# Raw connections not closed
grep -rE "\.connect\(\)" --include="*.py" -A 10 | grep -v "\.close\(\)|with\s"
```

**Pass criteria**: All sessions/connections properly managed
**Severity**: High

---

### Transactions Not Committed/Rolled Back

**Search patterns**:
```bash
# Begin without commit/rollback
grep -rE "\.begin\(\)" --include="*.py" -A 20 | grep -v "\.commit\(\)|\.rollback\(\)"

# Multiple writes without transaction
grep -rE "\.add\(|\.delete\(" --include="*.py" -B 5 | grep -v "begin\|transaction"
```

**Pass criteria**: Transactions explicitly committed or rolled back
**Severity**: High

---

### Lazy Loading in Async Context

**Search patterns**:
```bash
# Accessing relationships in async code
grep -rE "async def" --include="*.py" -A 30 | grep -E "\.\w+\.\w+\s*$|\.\w+\[" | grep -v "await"
```

**Pass criteria**: Use eager loading or explicit queries in async
**Severity**: High

---

### Missing Database Indexes

**Search technique**:
- Review model definitions for commonly queried fields
- Check if foreign keys and filter fields have indexes

**Search patterns**:
```bash
# Filter/order fields without index
grep -rE "\.filter\(.*==|\.order_by\(" --include="*.py"
# Compare with index definitions in models
```

**Pass criteria**: Frequently queried fields have indexes
**Severity**: Medium

---

### Raw SQL Without Parameterization

**Search patterns**:
```bash
# text() without bind parameters
grep -rE "text\(['\"].*\{|text\(f['\"]" --include="*.py"

# execute with string formatting
grep -rE "\.execute\(.*%.*%" --include="*.py"
```

**Pass criteria**: All raw SQL uses bind parameters
**Severity**: Critical
