# Testing the Desktop App

## Testing Unsigned DMGs on macOS

When testing unsigned DMG builds on macOS, you'll encounter this error:

> **"LlamaFarm" is damaged and can't be opened. You should move it to the Trash.**

**Don't panic!** The app isn't actually damaged - this is macOS Gatekeeper blocking unsigned apps.

### Why This Happens

1. **App is unsigned** - We build with `"identity": null` for testing
2. **Quarantine attribute** - macOS marks downloaded/built files as "from the internet"
3. **Gatekeeper protection** - macOS blocks unsigned apps by default

### Solution 1: Remove Quarantine (Recommended for Testing)

**Before installing:**
```bash
# Mount the DMG first, then:
xattr -cr "/Volumes/LlamaFarm/LlamaFarm.app"

# Now drag to Applications and open normally
```

**After installing:**
```bash
# If you already moved it to Applications:
xattr -cr "/Applications/LlamaFarm.app"

# Then open normally
```

### Solution 2: Right-Click to Open

1. **Don't** double-click the app
2. **Right-click** (or Control+click) on `LlamaFarm.app`
3. Select **"Open"** from the context menu
4. Click **"Open"** in the security dialog
5. macOS will remember this choice for future launches

### Solution 3: Enable Signing (For Distribution)

For real distribution, enable code signing:

1. **Uncomment the signed build** in `.github/workflows/release-desktop.yml`
2. **Comment out the unsigned build**
3. Create a release - the workflow will sign and notarize automatically

See [SIGNING.md](SIGNING.md) for detailed instructions.

## Local Development Testing

### Building Locally

```bash
cd electron-app

# Build unsigned DMG for local testing
npm run dist:mac
```

The DMG will be in `release/0.0.1/`.

### Testing the DMG

```bash
# Open the DMG
open release/0.0.1/LlamaFarm-0.0.1-arm64.dmg

# In a new terminal, remove quarantine before installing
xattr -cr "/Volumes/LlamaFarm/LlamaFarm.app"

# Now drag to Applications and test
```

### Verifying Installation

```bash
# Check if CLI is installed
/usr/local/bin/lf version

# Check if services start
lf services status
lf services start

# Check if Designer is accessible
curl -s http://localhost:14345/health
```

## Testing Signed Builds

Once code signing is enabled:

```bash
# Verify signature
codesign --verify --deep --strict "/Applications/LlamaFarm.app"

# Check notarization
spctl -a -vvv -t install "/Applications/LlamaFarm.app"
# Should show: "accepted" and "source=Notarized Developer ID"

# Check no quarantine issues
xattr -l "/Applications/LlamaFarm.app"
# Should NOT show: com.apple.quarantine
```

## Common Issues

### "Cannot be opened because the developer cannot be verified"

This means the app is unsigned. Use Solution 1 or 2 above.

### "App is damaged" on M1/M2/M3 Macs

Make sure you're using the **ARM64 DMG**:
- ✅ `LlamaFarm-0.0.1-arm64.dmg` - For Apple Silicon
- ❌ `LlamaFarm-0.0.1.dmg` - For Intel Macs

### Services won't start

```bash
# Check Docker is running
docker ps

# Check Ollama is installed
ollama --version

# Check port is available
lsof -i :14345
```

### CLI not found after install

The app installs CLI to its own userData directory. To use it system-wide:

```bash
# Find where app installed CLI
find ~/Library/Application\ Support/LlamaFarm -name "lf"

# Create symlink (if needed)
sudo ln -sf ~/Library/Application\ Support/LlamaFarm/bin/lf /usr/local/bin/lf
```

## Testing Checklist

Before releasing:

- [ ] Build DMG locally and test installation
- [ ] Remove quarantine and verify app opens
- [ ] Check CLI installs correctly
- [ ] Verify services start automatically
- [ ] Test Designer loads at localhost:14345
- [ ] Test auto-update check (production builds only)
- [ ] Test menu items (Help links, etc.)
- [ ] Test quit behavior (services stop cleanly)
- [ ] Verify no console errors in DevTools
- [ ] Test on both Intel and ARM64 Macs (if possible)

## Debugging

### View Console Logs

```bash
# Open Console.app and filter for "LlamaFarm"
# Or use command line:
log stream --predicate 'process == "LlamaFarm"' --level debug
```

### View App Logs

```bash
# Main app logs
cat ~/Library/Logs/LlamaFarm/main.log

# Services logs
lf services logs
```

### DevTools in Production

In development mode, DevTools open automatically. For production:

1. Open LlamaFarm
2. In Terminal: `defaults write com.llamafarm.desktop DevToolsEnabled -bool true`
3. Restart app
4. DevTools should open

## Testing Auto-Updates

Auto-updates only work in **production builds** (packaged and installed).

### Prerequisites

1. Create two GitHub releases (e.g., v0.0.1 and v0.0.2)
2. Upload DMGs to both releases
3. Install v0.0.1 DMG to Applications

### Test Update Flow

```bash
# Install v0.0.1
open release/0.0.1/LlamaFarm-0.0.1-arm64.dmg
# Drag to Applications

# Launch app
open /Applications/LlamaFarm.app

# Wait ~5 seconds - app should notify of v0.0.2 update
# Click "Restart Now" or "Later"
# If "Later", update installs on next quit
```

### Verify Update

```bash
# Check app version
/Applications/LlamaFarm.app/Contents/MacOS/LlamaFarm --version

# Or check in app: Help → About LlamaFarm
```

## Performance Testing

### Startup Time

```bash
time /Applications/LlamaFarm.app/Contents/MacOS/LlamaFarm
# Should open within 5-10 seconds (first launch slower)
```

### Memory Usage

```bash
# Monitor memory
top -pid $(pgrep -f LlamaFarm)

# Should be < 200MB for main process
```

### CPU Usage

```bash
# Should be low when idle (< 5%)
top -pid $(pgrep -f LlamaFarm) -stats pid,cpu,mem,command
```

## Additional Resources

- [Electron App Distribution](https://www.electronjs.org/docs/latest/tutorial/application-distribution)
- [macOS Code Signing](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution)
- [Testing Electron Apps](https://www.electronjs.org/docs/latest/tutorial/automated-testing)
