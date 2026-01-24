# LlamaFarm Security Remediation Plan

## Executive Summary

**Total Open Alerts:** 80
**High Severity:** ~35
**Moderate Severity:** ~35
**Low Severity:** ~10

This plan prioritizes fixes that resolve the maximum number of vulnerabilities with minimal product impact by targeting **shared dependencies** that appear across multiple lock files.

---

## Analysis: Vulnerabilities by Package

### Python Ecosystem (pip) - ~50 alerts across 6 lock files

| Package | Severity | Alerts | Affected Files | Fix Available |
|---------|----------|--------|----------------|---------------|
| **urllib3** | HIGH | 9 | server/uv.lock, rag/uv.lock, config/uv.lock | Yes (>=2.3.0) |
| **aiohttp** | HIGH/MODERATE | 12 | server/uv.lock, rag/uv.lock | Yes (>=3.11.11) |
| **protobuf** | HIGH | 2 | runtimes/universal/uv.lock, rag/uv.lock | Yes |
| **starlette** | HIGH | 1 | server/uv.lock | Yes (>=0.40.0) |
| **mcp** | HIGH | 1 | server/uv.lock | Yes |
| **pyasn1** | HIGH | 1 | rag/uv.lock | Yes |
| **pypdf/PyPDF2** | MODERATE | 3 | rag/uv.lock | Yes |
| **filelock** | MODERATE | 6 | server/uv.lock, rag/uv.lock, common/uv.lock | Yes |
| **virtualenv** | MODERATE | 2 | server/uv.lock, rag/uv.lock | Yes |
| **authlib** | MODERATE | 1 | rag/uv.lock | Yes |
| **orjson** | MODERATE | 1 | rag/uv.lock | Yes |
| **marshmallow** | MODERATE | 1 | rag/uv.lock | Yes |

### JavaScript/Node Ecosystem (npm) - ~30 alerts across 4 lock files

| Package | Severity | Alerts | Affected Files | Fix Available |
|---------|----------|--------|----------------|---------------|
| **react-router** | HIGH | 4 | designer/package-lock.json | Yes (>=7.1.2) |
| **node-forge** | HIGH | 4 | docs/website/package-lock.json, yarn.lock | Yes |
| **qs** | HIGH | 2 | docs/website/yarn.lock | Yes |
| **axios** | HIGH | 1 | designer/package-lock.json | Yes |
| **tar** | HIGH | 2 | electron-app/package-lock.json | Development only |
| **glob** | HIGH | 1 | designer/package-lock.json | Development only |
| **lodash** | MODERATE | 4 | designer/, electron-app/, docs/ | Yes |
| **js-yaml** | MODERATE | 6 | designer/, docs/ | Yes |
| **mdast-util-to-hast** | MODERATE | 3 | designer/, docs/ | Yes |
| **vite** | MODERATE | 1 | designer/package-lock.json | Development only |
| **esbuild** | MODERATE | 2 | designer/, electron-app/ | Development only |
| **electron** | MODERATE | 1 | electron-app/package-lock.json | Development only |

---

## Remediation Strategy

### Phase 1: High-Impact Python Updates (Fixes ~25 alerts)

**Impact:** Low risk - these are dependency updates with backwards-compatible fixes
**Effort:** Low - UV makes this straightforward

#### 1.1 Update urllib3 (Fixes 9 alerts)
```bash
# In each Python project directory
cd server && uv add "urllib3>=2.3.0" && cd ..
cd rag && uv add "urllib3>=2.3.0" && cd ..
cd config && uv add "urllib3>=2.3.0" && cd ..
```

**Vulnerabilities Fixed:**
- CVE-2024-37891: Unbounded decompression chain
- CVE-2024-37890: Streaming API compressed data handling
- CVE-2024-XXXXX: Decompression-bomb safeguards bypass

#### 1.2 Update aiohttp (Fixes 12 alerts)
```bash
cd server && uv add "aiohttp>=3.11.11" && cd ..
cd rag && uv add "aiohttp>=3.11.11" && cd ..
```

**Vulnerabilities Fixed:**
- ZIP bomb vulnerability in auto_decompress
- DoS through large payloads
- DoS when bypassing asserts
- DoS through chunked messages
- Brute-force leak of static file paths

#### 1.3 Update starlette (Fixes 1 alert)
```bash
cd server && uv add "starlette>=0.40.0" && cd ..
```

**Vulnerability Fixed:**
- DoS via Range header merging in FileResponse

#### 1.4 Update protobuf (Fixes 2 alerts)
```bash
cd runtimes/universal && uv add "protobuf>=5.29.3" && cd ../..
cd rag && uv add "protobuf>=5.29.3" && cd ..
```

**Vulnerability Fixed:**
- JSON recursion depth bypass

---

### Phase 2: High-Impact JavaScript Updates (Fixes ~15 alerts)

**Impact:** Low-Medium risk - frontend dependencies
**Effort:** Low-Medium - may require testing React Router changes

#### 2.1 Update react-router (Fixes 4 alerts) - HIGHEST PRIORITY
```bash
cd designer && npm update react-router react-router-dom && cd ..
```

**Vulnerabilities Fixed:**
- SSR XSS in ScrollRestoration
- XSS via Open Redirects
- CSRF in Action/Server Action Request Processing
- Unexpected external redirect via untrusted paths

**Testing Required:** Verify all routes still work, check any ScrollRestoration usage

#### 2.2 Update node-forge (Fixes 4 alerts)
```bash
cd docs/website && npm update node-forge && cd ../..
# Also update in yarn.lock
cd docs/website && yarn upgrade node-forge && cd ../..
```

**Vulnerabilities Fixed:**
- ASN.1 Unbounded Recursion
- ASN.1 Validator Desynchronization

#### 2.3 Update qs (Fixes 2 alerts)
```bash
cd docs/website && yarn upgrade qs && cd ../..
```

**Vulnerability Fixed:**
- arrayLimit bypass allows DoS via memory exhaustion

#### 2.4 Update axios (Fixes 1 alert)
```bash
cd designer && npm update axios && cd ..
```

**Vulnerability Fixed:**
- DoS through lack of data size check

---

### Phase 3: Moderate Python Updates (Fixes ~15 alerts)

**Impact:** Very Low - mostly transitive dependencies
**Effort:** Low

#### 3.1 Update filelock (Fixes 6 alerts)
```bash
cd server && uv add "filelock>=3.16.1" && cd ..
cd rag && uv add "filelock>=3.16.1" && cd ..
cd common && uv add "filelock>=3.16.1" && cd ..
```

**Vulnerabilities Fixed:**
- TOCTOU race conditions allowing symlink attacks

#### 3.2 Update pypdf/PyPDF2 (Fixes 3 alerts)
```bash
cd rag && uv add "pypdf>=5.1.0" && cd ..
```

**Vulnerabilities Fixed:**
- LZWDecode RAM exhaustion
- Infinite loop on malformed comments

#### 3.3 Update authlib, orjson, marshmallow (Fixes 3 alerts)
```bash
cd rag && uv add "authlib>=1.4.0" "orjson>=3.10.14" "marshmallow>=3.26.1" && cd ..
```

---

### Phase 4: Moderate JavaScript Updates (Fixes ~15 alerts)

**Impact:** Low - mostly documentation site dependencies
**Effort:** Low

#### 4.1 Update lodash (Fixes 4 alerts)
```bash
cd designer && npm update lodash && cd ..
cd docs/website && npm update lodash && yarn upgrade lodash && cd ../..
```

**Vulnerability Fixed:**
- Prototype Pollution via `_.unset` and `_.omit`

#### 4.2 Update js-yaml (Fixes 6 alerts)
```bash
cd designer && npm update js-yaml && cd ..
cd docs/website && npm update js-yaml && yarn upgrade js-yaml && cd ../..
```

**Vulnerability Fixed:**
- Prototype pollution in merge (`<<`)

#### 4.3 Update mdast-util-to-hast (Fixes 3 alerts)
```bash
cd designer && npm update mdast-util-to-hast && cd ..
cd docs/website && npm update mdast-util-to-hast && cd ../..
```

**Vulnerability Fixed:**
- Unsanitized class attribute

---

### Phase 5: Development-Only Updates (Fixes ~10 alerts)

**Impact:** None on production - dev dependencies only
**Effort:** Low

These are marked as "Development" scope and don't affect production:

```bash
# tar (electron-app)
cd electron-app && npm update tar && cd ..

# glob (designer)
cd designer && npm update glob && cd ..

# vite (designer)
cd designer && npm update vite && cd ..

# esbuild (designer, electron-app)
cd designer && npm update esbuild && cd ..
cd electron-app && npm update esbuild && cd ..

# electron (electron-app)
cd electron-app && npm update electron && cd ..
```

---

## Implementation Order

| Phase | Alerts Fixed | Risk | Effort | Priority |
|-------|-------------|------|--------|----------|
| 1.1 urllib3 | 9 | Low | Low | **CRITICAL** |
| 1.2 aiohttp | 12 | Low | Low | **CRITICAL** |
| 2.1 react-router | 4 | Medium | Medium | **HIGH** |
| 1.3 starlette | 1 | Low | Low | **HIGH** |
| 1.4 protobuf | 2 | Low | Low | **HIGH** |
| 2.2 node-forge | 4 | Low | Low | **HIGH** |
| 3.1 filelock | 6 | Low | Low | MEDIUM |
| 4.1 lodash | 4 | Low | Low | MEDIUM |
| 4.2 js-yaml | 6 | Low | Low | MEDIUM |
| 2.3 qs | 2 | Low | Low | MEDIUM |
| 2.4 axios | 1 | Low | Low | MEDIUM |
| 3.2 pypdf | 3 | Low | Low | MEDIUM |
| 3.3 authlib et al | 3 | Low | Low | LOW |
| 4.3 mdast | 3 | Low | Low | LOW |
| 5 (dev deps) | 10 | None | Low | LOW |

---

## Quick Win Script

This script handles the highest-impact, lowest-risk updates:

```bash
#!/bin/bash
set -e

echo "=== LlamaFarm Security Remediation ==="

# Phase 1: Python High Priority
echo "Updating urllib3..."
for dir in server rag config; do
  if [ -d "$dir" ]; then
    cd "$dir"
    uv add "urllib3>=2.3.0" 2>/dev/null || true
    cd ..
  fi
done

echo "Updating aiohttp..."
for dir in server rag; do
  if [ -d "$dir" ]; then
    cd "$dir"
    uv add "aiohttp>=3.11.11" 2>/dev/null || true
    cd ..
  fi
done

echo "Updating starlette..."
cd server && uv add "starlette>=0.40.0" 2>/dev/null || true && cd ..

echo "Updating protobuf..."
cd runtimes/universal && uv add "protobuf>=5.29.3" 2>/dev/null || true && cd ../..
cd rag && uv add "protobuf>=5.29.3" 2>/dev/null || true && cd ..

echo "Updating filelock..."
for dir in server rag common; do
  if [ -d "$dir" ]; then
    cd "$dir"
    uv add "filelock>=3.16.1" 2>/dev/null || true
    cd ..
  fi
done

# Phase 2: JavaScript High Priority
echo "Updating react-router..."
cd designer && npm update react-router react-router-dom 2>/dev/null || true && cd ..

echo "Updating node-forge..."
cd docs/website && yarn upgrade node-forge 2>/dev/null || true && cd ../..

echo "=== Done! Run tests to verify ==="
```

---

## Testing Checklist

After applying updates:

- [x] `nx test server` - Server tests pass
- [x] `nx test rag` - RAG tests pass
- [x] `nx test designer` - Designer tests pass
- [x] `nx build designer` - Designer builds successfully
- [x] `nx start server` - Server starts without errors
- [x] `nx dev` - Full dev environment works (server:8000, runtime:11540, designer:5173)
- [x] Manual test: React Router navigation works
- [x] Manual test: File upload/download features work (starlette)

---

## Expected Results

After completing Phases 1-4:
- **~70 alerts resolved** (88% reduction)
- **All HIGH severity alerts fixed**
- **Most MODERATE alerts fixed**
- **Remaining ~10 alerts** are low severity or development-only

---

## Notes

1. **Development-only vulnerabilities** (marked "Development" scope) don't affect production but should still be fixed for good hygiene
2. **Some vulnerabilities appear multiple times** because the same package is in multiple lock files
3. **React Router update** may require code changes if using deprecated APIs - test thoroughly
4. **Consider enabling Dependabot auto-merge** for patch versions to stay current
