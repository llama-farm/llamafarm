# LlamaFarm Desktop

> 🦙 Native desktop application for LlamaFarm - Build powerful AI locally, extend anywhere.

LlamaFarm Desktop is an Electron-based application that packages the LlamaFarm Designer with automatic backend management. Download, install, and start building AI applications—no terminal required.

## Features

- ✨ **One-Click Setup**: Automatically installs the LlamaFarm CLI on first launch
- 🚀 **Auto Backend Management**: Starts, monitors, and recovers backend services automatically
- 💻 **Native Experience**: System tray integration, native notifications, and OS-specific optimizations
- 🔄 **Auto Recovery**: Automatically restarts crashed services with exponential backoff
- 📊 **Status Monitoring**: Real-time health monitoring with visual feedback
- 🎨 **Integrated Designer**: Full LlamaFarm Designer UI built-in
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
     - Start the backend services
     - Open the Designer UI

3. **Start Building**
   - Use the Designer to create projects, upload data, and chat with your AI
   - The status bar shows backend health
   - Click "Restart" if you need to restart services

### For Developers

#### Prerequisites

- Node.js 18+
- npm or yarn
- Git

#### Development Setup

```bash
# Clone the repository
cd llamafarm/electron-app

# Install dependencies
npm install

# Start in development mode
npm run dev
```

This will:
- Start the Electron app in dev mode
- Enable hot reload for renderer changes
- Open DevTools automatically

**Important**: In development mode, make sure you have:
1. The LlamaFarm backend running (`lf start` or `nx dev` from the root)
2. The Designer dev server running (usually on port 3000 or handled by `nx dev`)

#### Building for Production

```bash
# Build for macOS
npm run dist:mac

# Build for Windows
npm run dist:win

# Build for Linux
npm run dist:linux

# Build for all platforms (requires running on macOS for macOS builds)
npm run dist
```

Built applications will be in `release/{version}/`.

## Architecture

```
electron-app/
├── src/
│   ├── main/                    # Main process (Node.js)
│   │   ├── index.ts             # Entry point
│   │   ├── window-manager.ts    # Window management
│   │   └── backend/
│   │       ├── cli-installer.ts # Auto-installs CLI
│   │       ├── backend-manager.ts # Lifecycle management
│   │       └── health-checker.ts  # Health monitoring
│   ├── preload/                 # Preload scripts (IPC bridge)
│   │   └── index.ts
│   └── renderer/                # Renderer process (UI)
│       ├── splash.html          # Splash screen
│       └── index.html           # Main window (embeds Designer)
├── build/                       # Build resources (icons, etc.)
└── release/                     # Built applications
```

### Main Process

The main process handles:
- **CLI Installation**: Downloads and installs the `lf` CLI from GitHub releases
- **Backend Lifecycle**: Spawns and monitors `lf start` process
- **Health Monitoring**: Polls `/health` endpoint and updates UI
- **Auto Recovery**: Restarts crashed services with exponential backoff
- **Window Management**: Creates splash screen and main window

### Renderer Process

The renderer process:
- Shows a beautiful splash screen during startup
- Embeds the Designer UI in an iframe (or serves it directly)
- Displays backend status in a persistent status bar
- Provides restart/stop controls

### IPC Communication

Secure IPC bridge via preload script:
- `backend:status` - Get current backend status
- `backend:restart` - Restart backend services
- `backend:stop` - Stop backend services
- `cli:info` - Get CLI installation info

## Configuration

### Environment Variables

In development, you can create a `.env` file:

```bash
# Backend URL (default: http://localhost:8000)
BACKEND_URL=http://localhost:8000

# Designer URL in dev mode (default: http://localhost:3000)
DESIGNER_DEV_URL=http://localhost:3000

# Enable debug logging
DEBUG=true
```

### Build Configuration

Edit `electron-app/package.json` under the `build` section:

```json
{
  "build": {
    "appId": "com.llamafarm.desktop",
    "productName": "LlamaFarm",
    "mac": {
      "category": "public.app-category.developer-tools"
    }
  }
}
```

## Troubleshooting

### CLI Installation Fails

**Problem**: CLI download fails or installation errors occur.

**Solutions**:
- Check internet connection
- Verify GitHub releases are accessible
- Try manual installation: `curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash`
- Check logs in: `~/Library/Logs/LlamaFarm/` (macOS)

### Backend Won't Start

**Problem**: Backend services fail to start or crash immediately.

**Solutions**:
- Ensure Docker is installed and running (required for LlamaFarm)
- Ensure Ollama is installed: https://ollama.com/download
- Check available ports (8000, 7724 must be free)
- View logs in the app's userData directory
- Try running `lf start` manually to see detailed errors

### Designer Not Loading

**Problem**: White screen or "Designer Loading..." message persists.

**Solutions**:
- Wait 2-3 minutes for initial startup (first time can be slow)
- Check backend status (should show "Running" in status bar)
- Verify Designer is accessible at http://localhost:7724
- Try restarting the app
- Check DevTools console (View → Toggle Developer Tools)

### App Won't Quit

**Problem**: App hangs when trying to quit.

**Solutions**:
- Force quit: Cmd+Q (macOS) or Alt+F4 (Windows)
- Kill the process: `pkill -f LlamaFarm`
- Check for orphaned processes: `ps aux | grep lf`

## Development Notes

### File Structure

- `src/main/` - Main process code (TypeScript)
- `src/preload/` - Preload scripts (secure IPC bridge)
- `src/renderer/` - Renderer process (HTML/CSS/JS)
- `build/` - Build resources (icons, entitlements)
- `release/` - Built applications (gitignored)

### Key Technologies

- **Electron**: Cross-platform desktop framework
- **electron-vite**: Fast Vite-based build tool
- **electron-builder**: Application packaging and distribution
- **TypeScript**: Type-safe development
- **Axios**: HTTP client for health checks

### Testing

```bash
# Run in dev mode
npm run dev

# Build and preview
npm run build
npm run preview
```

### Code Signing (macOS)

For distribution, you'll need to sign the app:

1. Get an Apple Developer account
2. Create a Developer ID certificate
3. Update `build/entitlements.mac.plist`
4. Set environment variables:
   ```bash
   export APPLE_ID="your@email.com"
   export APPLE_ID_PASSWORD="app-specific-password"
   export CSC_LINK="path/to/certificate.p12"
   export CSC_KEY_PASSWORD="certificate-password"
   ```

## Contributing

See the main [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

### Adding Features

1. Main process features: Add to `src/main/`
2. IPC handlers: Add to `src/main/index.ts` and `src/preload/index.ts`
3. UI features: Modify renderer HTML or integrate with Designer

### Reporting Issues

Report issues at: https://github.com/llama-farm/llamafarm/issues

Include:
- OS and version
- App version (Help → About)
- Steps to reproduce
- Logs (Help → Show Logs)

## License

Apache 2.0 - See [LICENSE](../LICENSE) for details.

---

Built with ❤️ by the LlamaFarm community
