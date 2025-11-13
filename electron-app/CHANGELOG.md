# Changelog - LlamaFarm Desktop

All notable changes to the Electron desktop application will be documented in this file.

## [Unreleased]

### Added
- Initial Electron desktop application
- Automatic CLI installation for macOS, Windows, and Linux
- Backend lifecycle management (start, stop, restart, monitor)
- Health monitoring with visual feedback
- Auto-recovery on backend crashes (exponential backoff)
- Beautiful splash screen with progress indicators
- Integrated Designer UI
- Status bar showing backend health
- Secure IPC communication via preload script
- Graceful shutdown handling
- Development mode with hot reload
- macOS build configuration with code signing support
- Comprehensive documentation (README, DEVELOPMENT guide)

### Features
- **One-Click Setup**: No terminal required, everything automatic
- **Auto Backend Management**: Handles `lf start` lifecycle transparently
- **Health Monitoring**: Real-time status updates every 10 seconds
- **Auto Recovery**: Restarts crashed services (up to 3 attempts)
- **Native Feel**: Platform-specific behaviors and optimizations
- **Secure**: Sandboxed renderer with context isolation

### Platform Support
- ✅ macOS (Intel & Apple Silicon) - Primary target
- 🚧 Windows - Coming soon
- 🚧 Linux - Coming soon

## Roadmap

### v0.1.0 (MVP)
- [x] CLI auto-installation
- [x] Backend lifecycle management
- [x] Health monitoring
- [x] Auto-recovery
- [x] Splash screen
- [x] Designer integration
- [ ] macOS builds (DMG)
- [ ] Basic testing
- [ ] Initial release

### v0.2.0
- [ ] System tray integration
- [ ] Native notifications
- [ ] Log viewer
- [ ] Settings panel
- [ ] Windows support
- [ ] Auto-updater

### v0.3.0
- [ ] Linux support
- [ ] Multiple project support
- [ ] Resource monitoring
- [ ] Custom themes
- [ ] Plugin system

## Notes

- This changelog tracks the Electron app specifically
- See main [CHANGELOG.md](../CHANGELOG.md) for LlamaFarm platform changes
