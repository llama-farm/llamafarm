# LlamaFarm CLI

A command-line interface for managing and interacting with LlamaFarm.

## Prerequisites

- Go 1.19 or later
- Git

## Installation

### Option 1: Quick Install (Recommended)

Install the latest version with a single command:

```bash
curl -fsSL https://raw.githubusercontent.com/llamafarm/llamafarm/main/install.sh | bash
```

This will automatically detect your platform and install the appropriate binary to `/usr/local/bin`.

#### Custom Installation Options

```bash
# Install specific version
VERSION=v1.0.0 curl -fsSL https://raw.githubusercontent.com/llamafarm/llamafarm/main/install.sh | bash

# Install to custom directory
curl -fsSL https://raw.githubusercontent.com/llamafarm/llamafarm/main/install.sh | bash -s -- --install-dir ~/.local/bin

# Install specific version to custom directory
curl -fsSL https://raw.githubusercontent.com/llamafarm/llamafarm/main/install.sh | bash -s -- --version v1.0.0 --install-dir ~/.local/bin
```

### Option 2: Manual Download

1. Go to the [releases page](https://github.com/llamafarm/llamafarm/releases)
2. Download the appropriate binary for your platform
3. Extract the archive and place the `lf` binary in your PATH

### Option 3: Build from Source

1. Clone the repository and navigate to the CLI directory:
   ```bash
   cd cli
   ```

2. Install dependencies:
   ```bash
   go mod tidy
   ```

3. Build the CLI:
   ```bash
   go build -o lf
   ```

4. (Optional) Install globally:
   ```bash
   go install
   ```

### Option 4: Direct Run (Development)

You can also run the CLI directly without building:
```bash
go run main.go [command]
```

## Usage

### Basic Commands

Once installed, you can use the CLI with the following commands:

```bash
# Show help
lf help

# Show version
lf version

# Start the AI designer
lf designer start
```

### Available Commands

- `help` - Show help information for any command
- `version` - Print the version number
- `designer start` - Start the AI designer

### Command Structure

This CLI is built using [Cobra](https://github.com/spf13/cobra), which provides:

- Easy subcommand creation
- Automatic help generation
- Flag and argument parsing
- Shell completion support

## Development

### Adding New Commands

To add a new command:

1. Create a new file in the `cmd/` directory (e.g., `cmd/newcommand.go`)
2. Follow the pattern established in existing commands
3. Add the command to the root command in the `init()` function

Example structure:
```go
package cmd

import (
    "fmt"
    "github.com/spf13/cobra"
)

var newCmd = &cobra.Command{
    Use:   "new",
    Short: "A brief description",
    Long:  "A longer description",
    Run: func(cmd *cobra.Command, args []string) {
        fmt.Println("New command executed")
    },
}

func init() {
    rootCmd.AddCommand(newCmd)
}
```

### Project Structure

```
cli/
├── main.go          # Entry point
├── go.mod           # Go module definition
├── go.sum           # Dependency checksums
├── README.md        # This file
└── cmd/             # Command definitions
    ├── root.go      # Root command and CLI setup
    ├── version.go   # Version command
    └── hello.go     # Example hello command
```

### Building for Different Platforms

```bash
# Linux
GOOS=linux GOARCH=amd64 go build -o llamafarm-cli-linux

# Windows
GOOS=windows GOARCH=amd64 go build -o llamafarm-cli.exe

# macOS
GOOS=darwin GOARCH=amd64 go build -o llamafarm-cli-macos
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]