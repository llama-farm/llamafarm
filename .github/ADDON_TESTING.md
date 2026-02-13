# Addon Testing Guide

This guide explains how to test addon functionality using snapshot releases and branch-based builds before merging to main or making an official release.

## Overview

The addon system allows testing changes to addon packages without creating an official release. This is done using:

1. **Branch-Based Testing**: Test changes from a feature branch before merging to main
2. **Snapshot Releases**: Pre-release versions with a `-snapshot` suffix
3. **Draft Releases**: GitHub releases marked as draft (not visible to public)
4. **Environment Variable Override**: `LF_ADDON_RELEASE_TAG` to specify which release to download from

## Testing Workflow

### Option 1: Branch-Based Testing (Development)

**Best for:** Testing changes in a feature branch before merging to main

This is the recommended workflow when developing addon features or fixes.

#### Step 1: Create Feature Branch

```bash
git checkout -b feat-update-stt-packages
# Make changes to tools/build_addon_wheels.py, cli/cmd/addons_registry.go, etc.
git commit -am "Update STT addon packages"
git push origin feat-update-stt-packages
```

#### Step 2: Trigger Branch Build

Via GitHub UI:
1. Go to **Actions** → **Test Addon from Branch**
2. Click **Run workflow**
3. Fill in:
   - **Branch**: `feat-update-stt-packages`
   - **Addon**: `stt` (or `all`)
   - **Platform**: `macos-arm64` (or `all` for comprehensive testing)
   - **Create release**: ✓ (creates snapshot release for easy testing)
4. Click **Run workflow**

Via GitHub CLI:
```bash
gh workflow run test-addon-branch.yml \
  -f branch=feat-update-stt-packages \
  -f addon=stt \
  -f platform=macos-arm64 \
  -f create_release=true
```

#### Step 3: Test the Build

The workflow creates a snapshot release with a tag like `v-snapshot-feat-update-stt-packages-20260202-123456`.

```bash
# Get the snapshot tag from workflow output or release page
export LF_ADDON_RELEASE_TAG=v-snapshot-feat-update-stt-packages-20260202-123456

# Install and test
lf addons install stt
lf services start universal-runtime

# Verify functionality
# ...
```

#### Step 4: Iterate

If you need to make changes:

```bash
# Make fixes in your branch
git commit -am "Fix package version"
git push

# Re-run the workflow with the same branch name
# It will create a new snapshot with updated timestamp
```

#### Step 5: Clean Up and Merge

Once testing is complete:

```bash
# Clean up test installations
unset LF_ADDON_RELEASE_TAG
lf addons uninstall stt

# Delete snapshot releases
gh release delete v-snapshot-feat-update-stt-packages-20260202-123456 --yes

# Merge to main
git checkout main
git merge feat-update-stt-packages
git push origin main
```

---

### Option 2: Using Snapshot Release Workflow (Pre-Release Testing)

This creates a complete test release with all addon wheels automatically built and uploaded.

#### Step 1: Trigger Snapshot Release

Via GitHub UI:
1. Go to **Actions** → **Create Snapshot Release**
2. Click **Run workflow**
3. Fill in:
   - **Snapshot version**: `0.0.27-snapshot.1` (increment for each test)
   - **Build addons**: ✓ (checked)
   - **Draft**: ✓ (checked for internal testing)
4. Click **Run workflow**

Via GitHub CLI:
```bash
gh workflow run snapshot-release.yml \
  -f snapshot_version=0.0.27-snapshot.1 \
  -f build_addons=true \
  -f draft=true
```

#### Step 2: Wait for Build to Complete

The workflow will:
1. Create a git tag `v0.0.27-snapshot.1`
2. Create a draft GitHub release
3. Build addon wheels for all platforms
4. Upload wheels to the release

Monitor progress:
```bash
gh run watch
```

#### Step 3: Test Addon Installation

Once the workflow completes:

```bash
# Point CLI to the snapshot release
export LF_ADDON_RELEASE_TAG=v0.0.27-snapshot.1

# Install addon (downloads from snapshot release)
lf addons install stt

# Verify installation
lf addons list

# Start the service
lf services start universal-runtime

# Test the addon functionality
# (e.g., make an API call that uses speech-to-text)
```

#### Step 4: Clean Up

After testing:

```bash
# Switch back to stable releases
unset LF_ADDON_RELEASE_TAG

# Uninstall test addon if needed
lf addons uninstall stt

# Delete the snapshot release (if no longer needed)
gh release delete v0.0.27-snapshot.1 --yes
git push origin :refs/tags/v0.0.27-snapshot.1
```

---

### Option 3: Manual Workflow Dispatch

For more control, you can manually trigger individual workflows.

#### Step 1: Build Addon Wheels

```bash
# Build specific addon for specific platform (for quick testing)
gh workflow run build-addon-wheels.yml \
  -f addon=stt \
  -f platform=macos-arm64

# Or build all addons for all platforms
gh workflow run build-addon-wheels.yml \
  -f addon=all \
  -f platform=all
```

#### Step 2: Download Artifacts

```bash
# List recent workflow runs
gh run list --workflow=build-addon-wheels.yml

# Download artifacts from a specific run
gh run download <run-id>
```

#### Step 3: Create Manual Release

```bash
# Create a draft release
gh release create v0.0.27-test \
  --draft \
  --prerelease \
  --title "Test Release v0.0.27" \
  --notes "Test release for addon development"

# Upload addon wheels
gh release upload v0.0.27-test \
  stt-wheels-macos-arm64.tar.gz \
  stt-wheels-macos-arm64.tar.gz.sha256

# Test installation
export LF_ADDON_RELEASE_TAG=v0.0.27-test
lf addons install stt
```

---

### Option 4: Local Testing (Rapid Iteration)

For rapid iteration during development, you can test locally without GitHub releases.

#### Step 1: Build Wheels Locally

```bash
# Build addon wheels locally
python tools/build_addon_wheels.py \
  --addon stt \
  --platform macos-arm64 \
  --output dist/addons

# Verify the tarball
ls -lh dist/addons/
tar -tzf dist/addons/stt-wheels-macos-arm64.tar.gz | head
```

#### Step 2: Manual Installation

```bash
# Create addon directory
mkdir -p ~/.llamafarm/addons/stt

# Extract wheels directly
tar -xzf dist/addons/stt-wheels-macos-arm64.tar.gz \
  -C ~/.llamafarm/addons/stt/

# Update state file manually
cat > ~/.llamafarm/addons.json <<EOF
{
  "version": "1",
  "installed_addons": {
    "stt": {
      "name": "stt",
      "version": "1.0.0",
      "component": "universal-runtime",
      "installed_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
      "platform": "macos-arm64"
    }
  }
}
EOF

# Test by starting the service
lf services start universal-runtime

# Check logs to verify addon is loaded
tail -f ~/.llamafarm/logs/universal-runtime.log
```

#### Step 3: Verify PYTHONPATH Injection

```bash
# Check that addon path is in PYTHONPATH
lf services status

# Or inspect the running process
ps aux | grep universal-runtime
```

---

## Testing Checklist

When testing a new addon or addon changes:

- [ ] Build completes successfully for all target platforms
- [ ] Tarball size is reasonable (check for bloat)
- [ ] SHA256 checksums are generated
- [ ] CLI can download from snapshot release
- [ ] Addon extracts correctly to `~/.llamafarm/addons/{name}/`
- [ ] State file is updated correctly
- [ ] Service starts with PYTHONPATH injection
- [ ] Addon packages are importable in Python
- [ ] Addon functionality works as expected
- [ ] Service can be stopped and restarted
- [ ] Addon can be uninstalled cleanly

---

## Common Issues

### "Download failed: 404"

The release or asset doesn't exist. Verify:
```bash
# Check if release exists
gh release view v0.0.27-snapshot.1

# List release assets
gh release view v0.0.27-snapshot.1 --json assets
```

### "No binary wheels available"

A package in the addon spec doesn't have binary wheels. Check:
```bash
# Test downloading locally
pip download --only-binary=:all: --dest /tmp package-name
```

If no wheels exist, either:
- Remove the package from the addon spec
- Build wheels manually and include them
- Document as a manual installation step

### PYTHONPATH Not Applied

Check service startup logs:
```bash
tail -f ~/.llamafarm/logs/universal-runtime.log | grep PYTHONPATH
```

Or inspect the state file:
```bash
cat ~/.llamafarm/addons.json | jq .
```

---

## Best Practices

1. **Use Sequential Snapshot Versions**: `0.0.27-snapshot.1`, `0.0.27-snapshot.2`, etc.
2. **Test on Multiple Platforms**: At least macOS (ARM/Intel) and Linux
3. **Clean Up Old Snapshots**: Delete draft releases after testing
4. **Document Breaking Changes**: Note any changes to addon structure in release notes
5. **Test Upgrade Path**: Install old version, then new version to verify upgrade works

---

## Promoting Snapshot to Stable

Once testing is complete:

```bash
# 1. Delete the snapshot release
gh release delete v0.0.27-snapshot.1 --yes

# 2. Delete the snapshot tag
git push origin :refs/tags/v0.0.27-snapshot.1

# 3. Create official release
gh release create v0.0.27 \
  --title "Release v0.0.27" \
  --notes-file CHANGELOG.md

# 4. The build-addon-wheels workflow will automatically run
#    and upload wheels to the official release
```
