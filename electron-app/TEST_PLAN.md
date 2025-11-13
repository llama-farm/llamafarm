# LlamaFarm Desktop - Test Plan

## Prerequisites ✓
- [x] Node.js 20+ installed
- [x] Docker Desktop running
- [x] LF CLI installed
- [x] Dependencies installed (`npm install`)

## Testing Scenarios

### Scenario 1: Development Mode Test
**Purpose**: Test the app with a live backend

1. **Terminal 1 - Start Backend**:
   ```bash
   cd /Users/robthelen/llamafarm-head/llamafarm
   lf start
   # OR
   nx dev
   ```

   Wait for:
   - ✅ Server started on http://localhost:8000
   - ✅ RAG worker started
   - ✅ Designer available on http://localhost:7724

2. **Terminal 2 - Start Electron App**:
   ```bash
   cd /Users/robthelen/llamafarm-head/llamafarm/electron-app
   npm run dev
   ```

3. **What to Test**:
   - [ ] Splash screen appears
   - [ ] "LlamaFarm CLI ready" message (CLI already installed)
   - [ ] "Starting LlamaFarm backend..." message
   - [ ] Backend status transitions: starting → running
   - [ ] Main window opens
   - [ ] Status bar shows "Backend Running" (green)
   - [ ] Designer UI loads inside the window
   - [ ] Can interact with Designer (create project, chat, etc.)

4. **Test Restart Functionality**:
   - [ ] Click "Restart" button in status bar
   - [ ] Backend stops and restarts
   - [ ] Status bar updates correctly
   - [ ] Designer reconnects automatically

5. **Test Quit**:
   - [ ] Cmd+Q (macOS) or close window
   - [ ] App shuts down gracefully
   - [ ] Backend process stops (check with `ps aux | grep lf`)

### Scenario 2: Fresh Install Simulation
**Purpose**: Test CLI auto-installation

1. **Remove CLI temporarily**:
   ```bash
   mkdir -p ~/Desktop/lf-backup
   cp /usr/local/bin/lf ~/Desktop/lf-backup/
   sudo rm /usr/local/bin/lf
   rm -rf ~/Library/Application\ Support/llamafarm-desktop/
   ```

2. **Start Electron App**:
   ```bash
   npm run dev
   ```

3. **What to Test**:
   - [ ] "Checking for LlamaFarm CLI..." message
   - [ ] "Downloading..." with progress bar
   - [ ] CLI downloads to user directory
   - [ ] Installation completes successfully
   - [ ] App continues to backend startup

4. **Restore CLI**:
   ```bash
   sudo cp ~/Desktop/lf-backup/lf /usr/local/bin/
   sudo chmod +x /usr/local/bin/lf
   ```

### Scenario 3: Backend Crash Recovery
**Purpose**: Test auto-recovery

1. **Start app normally** (`npm run dev`)

2. **Kill backend process**:
   ```bash
   # In another terminal
   pkill -f "lf start"
   ```

3. **What to Test**:
   - [ ] Status bar turns red/orange
   - [ ] "Backend crashed unexpectedly" message
   - [ ] Auto-restart initiates
   - [ ] "Restarting in Xs (attempt 1/3)" message
   - [ ] Backend successfully restarts
   - [ ] Status returns to green "Running"

### Scenario 4: Production Build Test
**Purpose**: Test packaged app

1. **Build Designer first**:
   ```bash
   cd /Users/robthelen/llamafarm-head/llamafarm/designer
   npm install
   npm run build
   ```

2. **Build Electron App**:
   ```bash
   cd /Users/robthelen/llamafarm-head/llamafarm/electron-app
   npm run dist:mac
   ```

3. **Install and Test**:
   - [ ] DMG created in `release/` directory
   - [ ] Open DMG
   - [ ] Drag to Applications
   - [ ] Launch from Applications
   - [ ] Full startup sequence works
   - [ ] Designer loads from bundled files (not dev server)

## Common Issues

### Issue: "EADDRINUSE: address already in use"
**Cause**: Port 8000 or 7724 already in use
**Fix**:
```bash
lsof -i :8000 -i :7724
kill -9 <PID>
```

### Issue: White screen in main window
**Cause**: Backend not ready or Designer not loading
**Fix**:
- Wait 2-3 minutes for first startup
- Check DevTools: View → Toggle Developer Tools
- Verify backend is running: `curl http://localhost:8000/health`

### Issue: "Cannot find module..."
**Cause**: Missing dependencies
**Fix**: `npm install` again

### Issue: App won't quit
**Cause**: Backend process not terminating
**Fix**:
```bash
pkill -9 -f "lf start"
pkill -9 -f "Electron"
```

## Success Criteria

✅ **All must pass**:
1. App installs CLI automatically if missing
2. App starts backend successfully
3. Health monitoring shows correct status
4. Auto-recovery works on crash
5. Graceful shutdown stops all processes
6. Designer UI loads and is functional
7. No console errors in DevTools
8. Built DMG installs and runs correctly

## Performance Benchmarks

- CLI download: < 1 minute
- Backend startup: 30s - 3 minutes (first time)
- Subsequent startups: < 30 seconds
- Memory usage: 150-300MB (Electron) + 2-4GB (Backend)
- CPU idle: < 5%

## Next Steps After Testing

If all tests pass:
1. Create GitHub Actions workflow for automated builds
2. Set up code signing for macOS (Apple Developer Account required)
3. Create release notes
4. Upload DMG to GitHub releases
5. Update main README with download links
