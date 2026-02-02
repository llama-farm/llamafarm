package orchestrator

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/llamafarm/cli/cmd/utils"
)

// serviceBinaryNames maps service names (as used in ServiceGraph) to their
// PyApp binary base names (without platform suffix or extension).
var serviceBinaryNames = map[string]string{
	"server":            "llamafarm-server",
	"rag":               "llamafarm-rag",
	"universal-runtime": "llamafarm-runtime",
}

// IsBinaryMode returns true when services should be launched from pre-built
// PyApp binaries instead of via uv + source.
func IsBinaryMode() bool {
	return os.Getenv("LF_DEPLOY_MODE") == "binary"
}

// GetBinDir returns the directory where service binaries are located.
// Priority: LF_BIN_DIR env var > {LF_DATA_DIR}/bin (i.e. ~/.llamafarm/bin/).
func GetBinDir() (string, error) {
	if dir := os.Getenv("LF_BIN_DIR"); dir != "" {
		return dir, nil
	}
	lfDir, err := utils.GetLFDataDir()
	if err != nil {
		return "", fmt.Errorf("could not determine bin directory: %w", err)
	}
	return filepath.Join(lfDir, "bin"), nil
}

// ResolveBinaryPath locates the binary for the given service name.
// It searches the bin directory for a platform-suffixed name first
// (e.g. llamafarm-server-macos-arm64), then falls back to the bare name
// (e.g. llamafarm-server). On Windows, .exe is appended.
func ResolveBinaryPath(serviceName string) (string, error) {
	baseName, ok := serviceBinaryNames[serviceName]
	if !ok {
		return "", fmt.Errorf("unknown service %q for binary mode", serviceName)
	}

	binDir, err := GetBinDir()
	if err != nil {
		return "", err
	}

	suffix := getPlatformSuffix()
	ext := ""
	if runtime.GOOS == "windows" {
		ext = ".exe"
	}

	// Try platform-specific name first (e.g. llamafarm-server-macos-arm64)
	candidates := []string{
		filepath.Join(binDir, baseName+"-"+suffix+ext),
		filepath.Join(binDir, baseName+ext),
	}

	for _, path := range candidates {
		if info, err := os.Stat(path); err == nil && !info.IsDir() {
			return path, nil
		}
	}

	return "", fmt.Errorf(
		"binary for service %q not found in %s (tried %v)",
		serviceName, binDir, candidates,
	)
}

// getPlatformSuffix returns the platform-architecture suffix used by the
// PyApp build system (e.g. "macos-arm64", "linux-x86_64", "windows-x86_64").
func getPlatformSuffix() string {
	osName := runtime.GOOS
	arch := runtime.GOARCH

	// Map Go OS names to PyApp conventions
	switch osName {
	case "darwin":
		osName = "macos"
	}

	// Map Go arch names to PyApp conventions
	switch arch {
	case "amd64":
		arch = "x86_64"
	}

	return osName + "-" + arch
}
