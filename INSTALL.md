# LlamaFarm CLI Installation Guide

This guide covers multiple ways to install the LlamaFarm CLI (`lf`) on your system.

## Quick Install (Recommended)

The easiest way to install LlamaFarm CLI is using our installation script:

```bash
curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash
```

This will:
- Automatically detect your operating system and architecture
- Download the latest release
- Install the `lf` binary to `/usr/local/bin`
- Make it executable and accessible from your PATH

## Custom Installation Options

### Install Specific Version

```bash
VERSION=v1.0.0 curl -fsSL https://raw.githubusercontent.com/llamafarm/llamafarm/main/install.sh | bash
```

### Install to Custom Directory

```bash
curl -fsSL https://raw.githubusercontent.com/llamafarm/llamafarm/main/install.sh | bash -s -- --install-dir ~/.local/bin
```

### Combined Options

```bash
curl -fsSL https://raw.githubusercontent.com/llamafarm/llamafarm/main/install.sh | bash -s -- --version v1.0.0 --install-dir ~/.local/bin
```

## Manual Installation

### 1. Download from Releases

1. Visit the [releases page](https://github.com/llamafarm/llamafarm/releases)
2. Download the appropriate archive for your platform:
   - **Linux**: `llamafarm-cli_v1.0.0_linux_amd64.tar.gz`
   - **macOS Intel**: `llamafarm-cli_v1.0.0_darwin_amd64.tar.gz`
   - **macOS Apple Silicon**: `llamafarm-cli_v1.0.0_darwin_arm64.tar.gz`
   - **Windows**: `llamafarm-cli_v1.0.0_windows_amd64.zip`

3. Extract the archive:
   ```bash
   # Linux/macOS
   tar -xzf llamafarm-cli_v1.0.0_linux_amd64.tar.gz

   # Windows (using PowerShell)
   Expand-Archive llamafarm-cli_v1.0.0_windows_amd64.zip
   ```

4. Move the binary to your PATH:
   ```bash
   # Linux/macOS
   sudo mv lf /usr/local/bin/

   # Or to user directory
   mkdir -p ~/.local/bin
   mv lf ~/.local/bin/
   export PATH="$HOME/.local/bin:$PATH"
   ```

### 2. Using Package Managers

*Package manager support coming soon*

## Build from Source

If you prefer to build from source or need the latest development version:

### Prerequisites

- Go 1.19 or later
- Git

### Build Steps

```bash
# Clone the repository
git clone https://github.com/llamafarm/llamafarm.git
cd llamafarm/cli

# Install dependencies
go mod tidy

# Build the CLI
go build -o lf .

# Install globally (optional)
sudo mv lf /usr/local/bin/
```

### Development Build

For development, you can run the CLI directly:

```bash
go run main.go [command]
```

## Platform Support

The LlamaFarm CLI supports the following platforms:

| Operating System | Architecture | Status |
|-----------------|-------------|---------|
| Linux           | amd64       | ✅ Supported |
| Linux           | arm64       | ✅ Supported |
| Linux           | arm         | ✅ Supported |
| Linux           | 386         | ✅ Supported |
| macOS           | amd64 (Intel) | ✅ Supported |
| macOS           | arm64 (Apple Silicon) | ✅ Supported |
| Windows         | amd64       | ✅ Supported |
| Windows         | 386         | ✅ Supported |

## Verification

After installation, verify that the CLI is working correctly:

```bash
# Check version
lf version

# View available commands
lf help

# Test a specific command
lf designer help
```

## Troubleshooting

### Binary Not Found in PATH

If you get "command not found" after installation:

1. **Check if the binary exists:**
   ```bash
   ls -la /usr/local/bin/lf
   ```

2. **Check your PATH:**
   ```bash
   echo $PATH
   ```

3. **Add to PATH if needed:**
   ```bash
   # For bash/zsh
   echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc

   # For fish
   fish_add_path /usr/local/bin
   ```

### Permission Denied

If you get permission errors during installation:

1. **Install to user directory instead:**
   ```bash
   curl -fsSL https://raw.githubusercontent.com/llamafarm/llamafarm/main/install.sh | bash -s -- --install-dir ~/.local/bin
   ```

2. **Or use sudo for system-wide installation:**
   ```bash
   curl -fsSL https://raw.githubusercontent.com/llamafarm/llamafarm/main/install.sh | sudo bash
   ```

### Download Issues

If the download fails:

1. **Check your internet connection**
2. **Try manual download** from the releases page
3. **Check if the release exists** for your platform
4. **Use a specific version** if the latest is not available:
   ```bash
   VERSION=v1.0.0 curl -fsSL https://raw.githubusercontent.com/llamafarm/llamafarm/main/install.sh | bash
   ```

### Outdated Version

To update to the latest version:

```bash
# Simply run the install script again
curl -fsSL https://raw.githubusercontent.com/llamafarm/llamafarm/main/install.sh | bash
```

The script will overwrite the existing binary with the latest version.

## Uninstallation

To remove the LlamaFarm CLI:

```bash
# If installed to /usr/local/bin
sudo rm /usr/local/bin/lf

# If installed to ~/.local/bin
rm ~/.local/bin/lf

# Clean up any configuration files (optional)
rm -rf ~/.llamafarm
```

## Security

### Script Verification

For security-conscious users, you can inspect the installation script before running:

```bash
# Download and inspect the script
curl -fsSL https://raw.githubusercontent.com/llamafarm/llamafarm/main/install.sh -o install.sh
cat install.sh

# Run after inspection
chmod +x install.sh
./install.sh
```

### Checksums

All releases include SHA256 checksums. You can verify the integrity of downloaded files:

```bash
# Download the binary and checksum
curl -L -O https://github.com/llamafarm/llamafarm/releases/download/v1.0.0/llamafarm-cli_v1.0.0_linux_amd64.tar.gz
curl -L -O https://github.com/llamafarm/llamafarm/releases/download/v1.0.0/SHA256SUMS

# Verify checksum
sha256sum -c SHA256SUMS
```

## Getting Help

- **CLI Help**: `lf help`
- **Command-specific help**: `lf [command] --help`
- **Issues**: [GitHub Issues](https://github.com/llamafarm/llamafarm/issues)
- **Documentation**: [Project Documentation](https://github.com/llamafarm/llamafarm)

## Next Steps

Once installed, check out the [CLI documentation](cli/README.md) to learn about available commands and features.