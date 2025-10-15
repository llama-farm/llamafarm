# Editable Install Fix for llamafarm-config Package

## Problem

When installing the `llamafarm-config` package as an editable dependency from the `server` or `rag` packages, the import `from config.datamodel import ...` would fail with `ModuleNotFoundError: No module named 'config'`.

### Root Cause

The package was using `hatchling` as the build backend, which has known issues with editable installs in certain monorepo configurations. Specifically, hatchling was creating an empty `.pth` file, preventing Python from finding the `config` module.

## Solution

Switched from `hatchling` to `setuptools` as the build backend with proper `package-dir` configuration.

### Changes Made

**File: `config/pyproject.toml`**

```diff
[build-system]
-requires = ["hatchling"]
-build-backend = "hatchling.build"
+requires = ["setuptools>=45", "wheel"]
+build-backend = "setuptools.build_meta"

-[tool.hatch.build.targets.wheel]
+[tool.setuptools]
packages = ["config"]
+package-dir = {"" = ".."}
+py-modules = []
```

### Why This Works

The key configuration is `package-dir = {"" = ".."}`, which tells setuptools that:
- The root package directory (`""`) is located in the parent directory (`".."`)
- This allows setuptools to correctly create a `.pth` file pointing to `/Users/.../llamafarm` (the parent of `config/`)
- Python can then import `config` because it's a subdirectory of a path in `sys.path`

## Verification

### Manual Testing
```bash
# From server directory
cd server
uv sync
uv run python -c "from config.datamodel import LlamaFarmConfig; print('✅ Import works!')"

# From rag directory
cd ../rag
uv sync
uv run python -c "from config.datamodel import LlamaFarmConfig; print('✅ Import works!')"
```

### Automated Testing
```bash
cd config
uv run pytest tests/test_editable_install.py -v
```

All four tests should pass:
- ✅ `test_config_package_is_importable`
- ✅ `test_config_datamodel_is_importable`
- ✅ `test_editable_install_path_is_correct`
- ✅ `test_config_in_sys_path`

## For New Contributors

When you clone the repository and run `uv sync` in either the `server` or `rag` directories, the `llamafarm-config` package will be automatically installed in editable mode, and all imports will work correctly.

No manual intervention or workarounds are required.

## Technical Details

### What setuptools creates:

1. **`.pth` file**: `/path/to/venv/lib/pythonX.Y/site-packages/__editable__.llamafarm_config-0.1.0.pth`
   - Contains: `/Users/.../llamafarm`
   
2. **dist-info directory**: Contains package metadata

### How Python finds the module:

1. Python reads the `.pth` file during startup
2. Adds `/Users/.../llamafarm` to `sys.path`
3. When you `import config`, Python looks for `config/` in each `sys.path` entry
4. Finds `/Users/.../llamafarm/config/` and imports it

## Related Files

- `config/pyproject.toml` - Build configuration
- `config/tests/test_editable_install.py` - Verification tests
- `server/pyproject.toml` - Declares dependency: `llamafarm-config = { path = "../config", editable = true }`
- `rag/pyproject.toml` - Declares dependency: `llamafarm-config = { path = "../config", editable = true }`

## Future Considerations

If you encounter import issues again:
1. Check that `config/pyproject.toml` still uses `setuptools` as the build backend
2. Verify the `package-dir = {"" = ".."}` configuration is present
3. Run the editable install tests: `cd config && uv run pytest tests/test_editable_install.py`
4. If tests fail, check that the `.pth` file exists and points to the correct directory

