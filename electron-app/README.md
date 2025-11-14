# LlamaFarm Desktop

> 🦙 Native desktop application for LlamaFarm - Build powerful AI locally, extend anywhere.

LlamaFarm Desktop is an Electron-based application that packages the LlamaFarm Designer with automatic backend management. Download, install, and start building AI applications—no terminal required.

## Features

- ✨ **One-Click Setup**: Automatically installs the LlamaFarm CLI on first launch
- 🔄 **Auto-Updates**: Checks for new versions and prompts users to update (via GitHub releases)
- 🚀 **Service Management**: Automatically starts and stops backend services via `lf services`
- 🎨 **Beautiful UI**: Dark theme with seamless hidden title bar and draggable Designer header
- 💻 **Native Experience**: Professional menu system with Help links to docs and GitHub
- 📊 **Splash Screen**: Visual progress tracking during startup with status updates
- 🛡️ **Secure**: Sandboxed renderer process with context isolation

## Download

**macOS** (recommended to start with):
- [Download for macOS (Intel)](https://github.com/llama-farm/llamafarm/releases/latest/download/LlamaFarm-mac-x64.dmg)
- [Download for macOS (Apple Silicon)](https://github.com/llama-farm/llamafarm/releases/latest/download/LlamaFarm-mac-arm64.dmg)

**Windows** (coming soon):
- Download for Windows (installer)
- Download for Windows (portable)

**Linux** (coming soon):
- Download AppImage
- Download .deb package

## Quick Start

### For Users

1. **Download and Install**
   - Download the appropriate version for your OS
   - macOS: Open the DMG and drag LlamaFarm to Applications
   - Windows: Run the installer
   - Linux: Run the AppImage or install the .deb package

2. **First Launch**
   - Open LlamaFarm from your Applications folder
   - The app will automatically:
     - Download and install the LlamaFarm CLI
     - Check for CLI updates
     - Start backend services (server + RAG)
     - Open the Designer UI
   - Watch the splash screen for progress

3. **Start Building**
   - Use the Designer to create projects, upload data, and chat with your AI
   - Access help via the Help menu (links to docs, GitHub, website)
   - The app manages all backend services automatically

### For Developers

#### Prerequisites

- Node.js 18+
- npm or yarn
- Git
- LlamaFarm CLI installed (`lf` command available)

#### Development Setup

```bash
# Clone the repository
cd llamafarm/electron-app

# Install dependencies
npm install

# Make sure LlamaFarm services are running
# In another terminal from the llamafarm root:
lf services start
# Or use nx to start the Designer dev server:
nx start designer

# Start Electron in development mode
npm run dev
```

This will:
- Start the Electron app in dev mode
- Enable hot reload for main process changes
- Open DevTools automatically (in non-packaged mode)
- Load Designer from `http://localhost:8000`

**Important**: In development mode, make sure you have:
1. The LlamaFarm backend running (`lf services start` from anywhere, or `nx start designer` from root)
2. The Designer accessible at `http://localhost:8000`

**Note**: The app loads Designer from `http://localhost:8000` (NOT 127.0.0.1), as the Designer's API configuration depends on the hostname being `localhost`.

#### Building for Production

```bash
# Build TypeScript to dist/
npm run build

# Build and create DMG for macOS (creates both Intel and Apple Silicon)
npm run dist:mac

# Build for Windows
npm run dist:win

# Build for Linux
npm run dist:linux
```

Built applications will be in `release/{version}/`:
- DMG files (macOS)
- ZIP files with blockmap (for auto-updates)
- Installers for Windows/Linux

**Build Outputs** (macOS example):
- `LlamaFarm-{version}-arm64-mac.zip` - Apple Silicon app bundle
- `LlamaFarm-{version}-arm64-mac.zip.blockmap` - Update diff file
- `LlamaFarm-{version}-arm64.dmg` - Apple Silicon installer
- `LlamaFarm-{version}-arm64.dmg.blockmap` - Update diff file
- `LlamaFarm-{version}-mac.zip` - Intel app bundle
- `LlamaFarm-{version}-mac.zip.blockmap` - Update diff file
- `LlamaFarm-{version}.dmg` - Intel installer
- `LlamaFarm-{version}.dmg.blockmap` - Update diff file

## Architecture

```
electron-app/
├── src/
│   ├── main/                     # Main process (Node.js)
│   │   ├── index.ts              # Entry point & app lifecycle
│   │   ├── window-manager.ts     # Window creation & management
│   │   ├── menu-manager.ts       # Application menu
│   │   └── backend/
│   │       └── cli-installer.ts  # Auto-installs CLI from GitHub
│   ├── preload/                  # Preload scripts (IPC bridge)
│   │   └── index.ts              # Secure context bridge
│   └── renderer/                 # Renderer process (no custom UI)
│       └── (Designer loads from localhost:8000)
├── build/                        # Build resources
│   ├── icon.png                  # macOS app icon (1024x1024)
│   └── entitlements.mac.plist    # macOS entitlements
├── dist/                         # Compiled TypeScript output
└── release/                      # Built applications (gitignored)
```

### Main Process (`src/main/index.ts`)

The main process handles the complete application lifecycle:

1. **CLI Installation & Updates**
   - Downloads CLI from GitHub releases if not installed
   - Runs CLI upgrade on every launch to ensure latest version
   - Updates splash screen with progress

2. **Service Management**
   - Checks if services are running via `lf services status`
   - Starts services if needed via `lf services start` (includes server + RAG)
   - Stops services on quit via `lf services stop`

3. **Health Monitoring**
   - Polls `http://127.0.0.1:8000/health` endpoint
   - Waits up to 30 seconds for server to become ready
   - Shows error if server fails to start

4. **Window Management**
   - Creates splash screen with dark theme and progress bar
   - Creates main window with hidden title bar
   - Positions traffic lights at x:20, y:18
   - Injects CSS to make Designer header draggable with 80px left padding

5. **Auto-Updates**
   - Checks for updates from GitHub releases (production only)
   - Downloads updates automatically
   - Prompts user to restart when update is ready
   - Auto-installs on quit

6. **Application Menu**
   - Standard Edit, View, Window menus
   - Help menu with links to:
     - LlamaFarm Website (https://llamafarm.dev)
     - Documentation (https://docs.llamafarm.dev)
     - GitHub Repository
     - Issue Tracker

### Window Manager (`src/main/window-manager.ts`)

Creates and manages application windows:

**Splash Window:**
- 500x400 frameless window
- Dark theme matching Designer UI
- Embedded HTML with progress bar and status messages
- Automatically closes when main window is ready

**Main Window:**
- Hidden title bar (`titleBarStyle: 'hidden'`)
- Traffic lights positioned at `{ x: 20, y: 18 }`
- Loads Designer from `http://localhost:8000`
- Injects CSS to make header draggable
- Shows after 5 second timeout or when ready (whichever comes first)
- Opens DevTools in development mode

**UI Polish:**
```typescript
// Injected CSS makes Designer header draggable
header, [class*="header"], [class*="Header"] {
  -webkit-app-region: drag;
  padding-left: 80px !important; // Room for traffic lights
}
// Keep buttons/links clickable
header button, header a, header input, header select {
  -webkit-app-region: no-drag;
}
```

### CLI Installer (`src/main/backend/cli-installer.ts`)

Handles CLI lifecycle:
- Downloads CLI binary from GitHub releases
- Installs to system location (`/usr/local/bin/lf` on macOS)
- Checks if CLI is installed
- Upgrades CLI to latest version
- Reports progress via callbacks

### IPC Communication

Minimal IPC surface via preload script:

```typescript
// Get CLI installation info
const info = await window.llamafarm.cli.getInfo()
// Returns: { isInstalled: boolean, path: string | null }
```

The app primarily relies on CLI commands rather than custom IPC handlers.

## Configuration

### Build Configuration (`package.json`)

The `build` section controls electron-builder settings:

```json
{
  "build": {
    "appId": "com.llamafarm.desktop",
    "productName": "LlamaFarm",
    "publish": {
      "provider": "github",
      "owner": "llama-farm",
      "repo": "llamafarm"
    },
    "mac": {
      "category": "public.app-category.developer-tools",
      "icon": "build/icon.png",
      "target": [
        {"target": "dmg", "arch": ["x64", "arm64"]},
        {"target": "zip", "arch": ["x64", "arm64"]}
      ],
      "identity": null
    },
    "extraResources": [
      {
        "from": "../designer/dist",
        "to": "designer",
        "filter": ["**/*"]
      }
    ]
  }
}
```

**Key Settings:**
- `identity: null` - Disables code signing (set to certificate name for production)
- `publish.provider: github` - Enables auto-updates from GitHub releases
- `extraResources` - Bundles Designer build (production mode)

### Auto-Updater Configuration

Auto-updates are configured in `src/main/index.ts`:

```typescript
import { autoUpdater } from 'electron-updater'

autoUpdater.autoDownload = true        // Download updates automatically
autoUpdater.autoInstallOnAppQuit = true // Install when user quits
```

**Update Flow:**
1. App checks for updates 5 seconds after launch (production only)
2. If update available, downloads in background
3. When complete, shows dialog: "Restart Now" or "Later"
4. If "Restart Now", quits and installs immediately
5. If "Later", installs next time user quits

**For Updates to Work:**
- App must be built with electron-builder
- GitHub release must contain the built artifacts
- App must be installed (not running from DMG)

## Development Notes

### File Structure

- `src/main/` - Main process code (TypeScript)
  - `index.ts` - App lifecycle, startup sequence
  - `window-manager.ts` - Window creation & UI
  - `menu-manager.ts` - Application menu
  - `backend/cli-installer.ts` - CLI download & installation
- `src/preload/` - Preload scripts (secure IPC bridge)
- `build/` - Build resources (icons, entitlements)
  - `icon.png` - App icon (1024x1024 PNG)
  - `entitlements.mac.plist` - macOS entitlements for code signing
- `dist/` - Compiled TypeScript output (from `npm run build`)
- `release/` - Built applications (gitignored)

### Key Technologies

- **Electron 28**: Cross-platform desktop framework
- **electron-vite**: Fast Vite-based build tool with TypeScript support
- **electron-builder**: Application packaging and distribution
- **electron-updater**: Automatic updates from GitHub releases
- **TypeScript**: Type-safe development
- **Axios**: HTTP client for health checks

### Development Commands

```bash
# Development
npm run dev              # Start in dev mode with hot reload
npm run build            # Compile TypeScript to dist/
npm run preview          # Build and run (like production)

# Production builds
npm run pack:mac         # Build but don't package (for testing)
npm run dist:mac         # Build and create DMG
npm run dist:win         # Build for Windows
npm run dist:linux       # Build for Linux
```

### Startup Sequence

1. **App Ready** (`onReady()`)
   - Set app name to "LlamaFarm"
   - Force dark mode: `nativeTheme.themeSource = 'dark'`
   - Create application menu
   - Create splash screen

2. **Ensure CLI** (`ensureCLI()`)
   - Check if CLI installed (progress: 10%)
   - Install if missing (progress: 10-90%)
   - Always run upgrade (progress: 50-70%)
   - Start services (progress: 70-90%)

3. **Wait for Server** (`waitForServer()`)
   - Poll `http://127.0.0.1:8000/health` (progress: 80%)
   - Max 30 attempts (30 seconds)
   - Throws error if server doesn't start

4. **Create Main Window** (progress: 95%)
   - Load Designer from `http://localhost:8000`
   - Inject draggable header CSS
   - Show window when ready or after 5 seconds
   - Close splash screen

5. **Check for Updates** (background, production only)
   - Wait 5 seconds after startup
   - Check GitHub releases for newer version

6. **On Quit** (`onWillQuit()`)
   - Stop services via `lf services stop`
   - Cleanup windows
   - Exit cleanly

### Code Signing & Notarization (macOS)

For production distribution on macOS, you need to sign and notarize the app.

See [SIGNING.md](SIGNING.md) for detailed instructions on:
- Setting up Apple Developer account
- Creating certificates and API keys
- Configuring GitHub secrets
- Enabling signing in the workflow

## Troubleshooting

### CLI Installation Fails

**Problem**: CLI download fails or installation errors occur.

**Solutions**:
- Check internet connection
- Verify GitHub releases are accessible: https://github.com/llama-farm/llamafarm/releases
- Try manual installation: `curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash`
- Check splash screen error message for details

### Services Won't Start

**Problem**: Backend services fail to start or timeout.

**Solutions**:
- Ensure Docker is installed and running (required for LlamaFarm)
- Ensure Ollama is installed: https://ollama.com/download
- Check available ports (8000 must be free for server)
- Try running `lf services start` manually to see detailed errors
- Check Docker logs: `docker logs llamafarm-server`

### Designer Not Loading

**Problem**: White screen or Designer doesn't appear.

**Solutions**:
- Wait 30 seconds for initial startup (first time can be slow)
- Verify server is running: open http://localhost:8000 in a browser
- Check DevTools console: View → Toggle Developer Tools
- Try restarting the app
- Verify services are running: `lf services status`

**Common Cause**: Server not ready yet. The app waits up to 30 seconds for `http://127.0.0.1:8000/health` to respond.

### App Won't Quit

**Problem**: App hangs when trying to quit.

**Solutions**:
- Force quit: Cmd+Q (macOS) or Alt+F4 (Windows)
- Kill the process: `pkill -f LlamaFarm`
- Stop services manually: `lf services stop`
- Check for orphaned processes: `ps aux | grep lf`

### Auto-Updates Not Working

**Problem**: App doesn't check for or install updates.

**Solutions**:
- Auto-updates only work in production (packaged app)
- Check you're running the installed app, not from DMG
- Verify GitHub releases contain the proper artifacts
- Check DevTools console for update check errors
- Updates check happens 5 seconds after launch

### Build Errors

**Problem**: Build fails with errors.

**Common Issues**:
1. **Icon missing**: Ensure `build/icon.png` exists (1024x1024 PNG)
2. **Entitlements missing**: Ensure `build/entitlements.mac.plist` exists
3. **Code signing fails**: Set `"identity": null` in package.json to disable
4. **Designer not built**: Run `nx build designer` from root first

## Contributing

See the main [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

### Adding Features

1. **Main process features**: Add to `src/main/` modules
2. **IPC handlers**: Add to `src/main/index.ts` (`setupIPCHandlers`) and `src/preload/index.ts`
3. **Menu items**: Update `src/main/menu-manager.ts`
4. **UI changes**: Modify CSS injection in `src/main/window-manager.ts` or update Designer

### Testing Changes

```bash
# Quick test
npm run dev

# Full production test
npm run build
npm run preview

# Build and install locally
npm run dist:mac
# Then open release/{version}/LlamaFarm-{version}.dmg
```

### Reporting Issues

Report issues at: https://github.com/llama-farm/llamafarm/issues

**Include:**
- OS and version (macOS 13.0, Windows 11, etc.)
- App version (Help → About LlamaFarm)
- Steps to reproduce
- Screenshots or screen recordings
- DevTools console output (View → Toggle Developer Tools)
- Splash screen error message (if any)

## License

Apache 2.0 - See [LICENSE](../LICENSE) for details.

---

Built with ❤️ by the LlamaFarm community
