# Quick Start Guide - LlamaFarm Desktop

This guide will get you up and running with LlamaFarm Desktop in 5 minutes.

## For End Users

### Download & Install

1. **Download the app**

   - Go to https://github.com/llama-farm/llamafarm/releases/latest
   - Download the appropriate version:
     - macOS Intel: `LlamaFarm-{version}.dmg`
     - macOS Apple Silicon: `LlamaFarm-{version}-arm64.dmg`

2. **Install**

   - Open the downloaded DMG
   - Drag LlamaFarm to your Applications folder
   - Eject the DMG

3. **First Launch**
   - Open LlamaFarm from Applications
   - You'll see a splash screen saying "Checking for LlamaFarm CLI..."
   - **Wait 2-3 minutes** for initial setup:
     - CLI installation (~30 seconds)
     - Backend services startup (~2-3 minutes on first launch)
   - The Designer will open automatically when ready

### What Happens on First Launch?

The app automatically:

1. ✅ Downloads the LlamaFarm CLI (`lf`)
2. ✅ Installs it to your user directory
3. ✅ Downloads necessary Python dependencies
4. ✅ Starts the FastAPI server
5. ✅ Starts the RAG worker
6. ✅ Opens the Designer UI

**No terminal commands required!**

### Using the App

Once the app is running:

- **Status Bar**: Shows backend health at the top

  - Green = All systems running
  - Yellow = Some services degraded
  - Red = Error or stopped

- **Designer**: Full visual interface for:

  - Creating projects
  - Uploading datasets
  - Configuring models
  - Chatting with your AI

- **Restart Button**: Click to restart backend services

### Prerequisites

Before installing LlamaFarm Desktop, you need:

1. **Docker Desktop**

   - Download: https://www.docker.com/products/docker-desktop
   - Required for backend services
   - Must be running when you start the app

2. **Ollama** (recommended)

   - Download: https://ollama.com/download
   - For local LLM inference
   - Adjust context window in Settings → Advanced (recommend 100K tokens)

3. **macOS Requirements**
   - macOS 10.15 (Catalina) or later
   - 4GB RAM minimum (8GB+ recommended)
   - 5GB free disk space

## For Developers

### Development Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/llama-farm/llamafarm.git
   cd llamafarm/electron-app
   ```

2. **Install dependencies**

   ```bash
   npm install
   ```

3. **Build the Designer** (required first time)

   ```bash
   cd ../designer
   npm install
   npm run build
   cd ../electron-app
   ```

4. **Run in development mode**

   ```bash
   # Terminal 1: Start the backend
   cd ..
   lf start
   # or: nx dev

   # Terminal 2: Start Electron app
   cd electron-app
   npm run dev
   ```

### Development Mode Features

- ✅ Hot reload for renderer changes
- ✅ DevTools open automatically
- ✅ Detailed console logging
- ✅ Loads Designer from dev server

### Building

```bash
# Build TypeScript
npm run build

# Package for testing (doesn't create installer)
npm run pack:mac

# Create distributable DMG
npm run dist:mac
```

Output will be in `release/{version}/`.

### Project Structure

```
electron-app/
├── src/
│   ├── main/          # Main process (Node.js)
│   ├── preload/       # IPC bridge
│   └── renderer/      # UI (HTML/CSS/JS)
├── build/             # Build resources
├── package.json       # Dependencies & scripts
└── README.md         # Documentation
```

## Troubleshooting

### "CLI installation failed"

**Problem**: Network error or GitHub rate limit

**Solutions**:

- Check internet connection
- Wait a few minutes and restart the app
- Install manually: `curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash`

### "Backend won't start"

**Problem**: Docker not running or ports in use

**Solutions**:

- Ensure Docker Desktop is running
- Check that port 14345 is free
- Try restarting Docker
- View logs: Help → Show Logs

### "White screen / Designer not loading"

**Problem**: Backend not ready yet

**Solutions**:

- Wait 2-3 minutes (first launch can be slow)
- Check status bar (should say "Backend Running")
- Try restarting: Click "Restart" button
- Open DevTools: View → Toggle Developer Tools

### Getting Help

- **Documentation**: See README.md and DEVELOPMENT.md
- **Issues**: https://github.com/llama-farm/llamafarm/issues
- **Discord**: https://discord.gg/RrAUXTCVNF
- **Email**: support@llamafarm.ai (for commercial support)

## Tips & Best Practices

### Performance

- **First launch is slow**: Downloads and installs dependencies (~2-3 min)
- **Subsequent launches are fast**: ~30 seconds to fully ready
- **Close unused apps**: LlamaFarm uses significant RAM for AI models

### Resource Usage

- **RAM**: 2-4GB (depends on active models)
- **CPU**: Low when idle, high during processing
- **Disk**: ~5GB for app + models

### Updating

- **Auto-update**: The app automatically checks for updates on launch and prompts you when a new version is available
- **Manual update**: Download latest DMG and reinstall

### Data Location

- **App Data**: `~/Library/Application Support/LlamaFarm/`
- **CLI**: `~/Library/Application Support/LlamaFarm/bin/lf`
- **Projects**: `~/.llamafarm/projects/`
- **Logs**: `~/Library/Logs/LlamaFarm/`

## Next Steps

1. ✅ Install prerequisites (Docker, Ollama)
2. ✅ Download and install LlamaFarm Desktop
3. ✅ Complete first launch setup
4. 🚀 Start building AI applications!

**Learn More**:

- [Main Documentation](../../docs/website/docs/intro.md)
- [Configuration Guide](../../docs/website/docs/configuration/index.md)
- [RAG Guide](../../docs/website/docs/rag/index.md)
- [Examples](../../docs/website/docs/examples/index.md)
