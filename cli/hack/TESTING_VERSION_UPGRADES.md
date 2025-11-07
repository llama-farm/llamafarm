# Testing Version Upgrade Service Shutdown

This guide explains how to test the automatic service shutdown during version upgrades in development.

## The Challenge

In dev mode, the CLI typically uses "dev" or "main" as the version, which skips the upgrade logic. To test the upgrade flow, we need to simulate version changes using environment variables.

## Quick Testing Methods

### Method 1: Using LF_VERSION_REF (Simplest)

The `LF_VERSION_REF` environment variable overrides the CLI version detection:

```bash
# Build the CLI
cd cli
go build -o lf main.go

# Terminal 1: Start services with "v1.0.0"
export LF_VERSION_REF=v1.0.0
export LF_DEBUG=1  # Enable debug logging
./lf start

# Services are now running...

# Terminal 2: Simulate upgrade to "v1.1.0"
export LF_VERSION_REF=v1.1.0
export LF_DEBUG=1
./lf start

# You should see in the debug output:
# "Source version mismatch (current: v1.0.0, target: v1.1.0) - stopping all services before upgrade"
# "All services stopped successfully"
```

### Method 2: Manual Version File Editing

You can directly manipulate the `.source_version` file:

```bash
# Start services normally
./lf start

# Check current version
cat ~/.llamafarm/.source_version

# Manually change to an old version
echo "v0.9.0" > ~/.llamafarm/.source_version

# Set target version and restart
export LF_VERSION_REF=v1.0.0
export LF_DEBUG=1
./lf start

# Check logs for service shutdown messages
```

### Method 3: Branch Switching (For CI/CD Testing)

Test with different branches:

```bash
# Start with main branch
export LF_VERSION_REF=main
./lf start

# Switch to a feature branch
export LF_VERSION_REF=feat-some-feature
./lf start

# Services should be stopped and source re-downloaded
```

## Automated Test Script

Run the automated test script:

```bash
cd cli
go build -o lf main.go
./hack/test-version-upgrade.sh ./lf
```

## What to Look For

When testing, you should see these debug messages:

1. **Version Detection:**
   ```
   Using version from LF_VERSION_REF: v1.1.0
   ```

2. **Version Mismatch Detection:**
   ```
   Source version mismatch (current: v1.0.0, target: v1.1.0) - stopping all services before upgrade
   ```

3. **Service Shutdown:**
   ```
   All services stopped successfully
   ```

4. **Download/Upgrade:**
   ```
   Downloading LlamaFarm source code (v1.1.0)...
   ```

## Testing Without Network Access

If you don't want to actually download versions:

```bash
# Use commit SHAs that don't exist to trigger the logic without downloads
export LF_VERSION_REF=0000000000000000000000000000000000000000
echo "1111111111111111111111111111111111111111" > ~/.llamafarm/.source_version

# Try to start - will detect mismatch and stop services (then fail on download)
LF_DEBUG=1 ./lf start 2>&1 | grep -E "version mismatch|stopping all services"
```

## Verifying Services Were Stopped

Check that services actually stopped:

```bash
# Before upgrade
./lf status
# Should show services running

# After triggering upgrade (with version mismatch)
./lf status
# Should show services stopped

# Check process manager logs
cat ~/.llamafarm/logs/*.log | grep -i stop
```

## Integration Test Example

Complete test scenario:

```bash
#!/bin/bash
set -e

cd cli
go build -o lf main.go

echo "=== Starting with v1.0.0 ==="
export LF_VERSION_REF=v1.0.0
./lf start &
LF_PID=$!
sleep 5

echo "=== Checking services are running ==="
./lf status

echo "=== Upgrading to v1.1.0 (should stop services) ==="
export LF_VERSION_REF=v1.1.0
LF_DEBUG=1 ./lf start 2>&1 | tee upgrade.log

echo "=== Verifying service shutdown message in logs ==="
if grep -q "stopping all services before upgrade" upgrade.log; then
    echo "✅ Services were stopped before upgrade"
else
    echo "❌ Service shutdown not detected"
    exit 1
fi

# Cleanup
./lf stop --all
kill $LF_PID 2>/dev/null || true
```

## Troubleshooting

**Issue:** Not seeing version mismatch messages
- **Solution:** Make sure `LF_DEBUG=1` is set and check that the `.source_version` file exists and differs from `LF_VERSION_REF`

**Issue:** Services don't stop
- **Solution:** Check that services are actually running before the upgrade attempt with `./lf status`

**Issue:** Version stays as "dev"
- **Solution:** Make sure `LF_VERSION_REF` is exported: `export LF_VERSION_REF=v1.0.0`

## Environment Variables Reference

- `LF_VERSION_REF` - Override CLI version (highest priority)
- `LF_DEBUG=1` - Enable debug logging to see version mismatch messages
- `GITHUB_TOKEN` - Required for downloading branch artifacts (optional for this test)

## Files Involved

- `~/.llamafarm/.source_version` - Tracks currently installed source version
- `~/.llamafarm/src/` - Source code directory
- `~/.llamafarm/logs/*.log` - Service logs showing shutdown
- `cli/cmd/orchestrator/source_manager.go` - Contains the upgrade logic
