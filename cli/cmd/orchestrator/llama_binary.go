package orchestrator

import (
	"archive/zip"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/llamafarm/cli/cmd/utils"
)

// LlamaCppVersion is the pinned llama.cpp release version
const LlamaCppVersion = "b7376"

// BinaryInfo contains information about a platform-specific binary
type BinaryInfo struct {
	URL     string // Direct download URL from llama.cpp releases
	SHA256  string // Checksum for verification (empty = skip verification)
	LibPath string // Path to library inside archive (e.g., "lib/libllama.so")
	LibName string // Final library name (e.g., "libllama.so")
}

// LlamaBinarySpec defines llama.cpp binary download configuration
// All releases are .zip format from ggml-org/llama.cpp
var LlamaBinarySpec = map[HardwareCapability]BinaryInfo{
	HardwareCPU: {
		URL:     fmt.Sprintf("https://github.com/ggml-org/llama.cpp/releases/download/%s/llama-%s-bin-ubuntu-x64.zip", LlamaCppVersion, LlamaCppVersion),
		SHA256:  "", // TODO: Populate at release
		LibPath: "build/bin/libllama.so",
		LibName: "libllama.so",
	},
	HardwareCUDA: {
		// Note: Linux CUDA builds no longer available in recent releases
		// Falls back to Vulkan or CPU
		URL:     fmt.Sprintf("https://github.com/ggml-org/llama.cpp/releases/download/%s/llama-%s-bin-ubuntu-vulkan-x64.zip", LlamaCppVersion, LlamaCppVersion),
		SHA256:  "",
		LibPath: "build/bin/libllama.so",
		LibName: "libllama.so",
	},
	HardwareMetal: {
		URL:     fmt.Sprintf("https://github.com/ggml-org/llama.cpp/releases/download/%s/llama-%s-bin-macos-arm64.zip", LlamaCppVersion, LlamaCppVersion),
		SHA256:  "",
		LibPath: "build/bin/libllama.dylib",
		LibName: "libllama.dylib",
	},
	HardwareROCm: {
		URL:     fmt.Sprintf("https://github.com/ggml-org/llama.cpp/releases/download/%s/llama-%s-bin-ubuntu-vulkan-x64.zip", LlamaCppVersion, LlamaCppVersion),
		SHA256:  "",
		LibPath: "build/bin/libllama.so",
		LibName: "libllama.so",
	},
}

// WindowsBinarySpec for Windows platforms
var WindowsBinarySpec = map[HardwareCapability]BinaryInfo{
	HardwareCPU: {
		URL:     fmt.Sprintf("https://github.com/ggml-org/llama.cpp/releases/download/%s/llama-%s-bin-win-cpu-x64.zip", LlamaCppVersion, LlamaCppVersion),
		SHA256:  "",
		LibPath: "llama.dll", // Windows: library is in root
		LibName: "llama.dll",
	},
	HardwareCUDA: {
		URL:     fmt.Sprintf("https://github.com/ggml-org/llama.cpp/releases/download/%s/llama-%s-bin-win-cuda12.4-x64.zip", LlamaCppVersion, LlamaCppVersion),
		SHA256:  "",
		LibPath: "llama.dll",
		LibName: "llama.dll",
	},
}

// GetLlamaCacheDir returns the cache directory for llama.cpp binaries.
// This matches the paths used by the Python llamafarm-llama package.
func GetLlamaCacheDir() string {
	// Check for environment override
	if cacheDir := os.Getenv("LLAMAFARM_CACHE_DIR"); cacheDir != "" {
		return cacheDir
	}

	homeDir, _ := os.UserHomeDir()

	switch runtime.GOOS {
	case "darwin":
		return filepath.Join(homeDir, "Library", "Caches", "llamafarm-llama")
	case "windows":
		localAppData := os.Getenv("LOCALAPPDATA")
		if localAppData == "" {
			localAppData = homeDir
		}
		return filepath.Join(localAppData, "llamafarm-llama", "cache")
	default: // Linux and others
		xdgCache := os.Getenv("XDG_CACHE_HOME")
		if xdgCache == "" {
			xdgCache = filepath.Join(homeDir, ".cache")
		}
		return filepath.Join(xdgCache, "llamafarm-llama")
	}
}

// GetLlamaLibName returns the platform-specific library name
func GetLlamaLibName() string {
	switch runtime.GOOS {
	case "darwin":
		return "libllama.dylib"
	case "windows":
		return "llama.dll"
	default:
		return "libllama.so"
	}
}

// IsLlamaBinaryInstalled checks if llama.cpp binaries are already installed
func IsLlamaBinaryInstalled() bool {
	cacheDir := GetLlamaCacheDir()
	libPath := filepath.Join(cacheDir, LlamaCppVersion, GetLlamaLibName())
	_, err := os.Stat(libPath)
	return err == nil
}

// EnsureLlamaBinary downloads llama.cpp binaries if not already installed.
// Returns the path to the installed binaries.
func EnsureLlamaBinary() (string, error) {
	cacheDir := GetLlamaCacheDir()
	versionDir := filepath.Join(cacheDir, LlamaCppVersion)
	libPath := filepath.Join(versionDir, GetLlamaLibName())

	// Check if already installed
	if _, err := os.Stat(libPath); err == nil {
		utils.LogDebug(fmt.Sprintf("llama.cpp binaries already installed at %s", versionDir))
		return versionDir, nil
	}

	// Download and install
	utils.LogDebug(fmt.Sprintf("Installing llama.cpp %s to %s", LlamaCppVersion, versionDir))
	if err := InstallLlamaBinary(versionDir); err != nil {
		return "", err
	}

	return versionDir, nil
}

// GetBinaryInfo returns the binary info for the detected hardware
func GetBinaryInfo(hardware HardwareCapability) (BinaryInfo, error) {
	var spec map[HardwareCapability]BinaryInfo

	if runtime.GOOS == "windows" {
		spec = WindowsBinarySpec
	} else {
		spec = LlamaBinarySpec
	}

	info, ok := spec[hardware]
	if !ok {
		// Fall back to CPU
		info, ok = spec[HardwareCPU]
		if !ok {
			return BinaryInfo{}, fmt.Errorf("no binary available for hardware %s on %s", hardware, runtime.GOOS)
		}
		utils.LogDebug(fmt.Sprintf("No %s binary for %s, falling back to CPU", hardware, runtime.GOOS))
	}

	return info, nil
}

// InstallLlamaBinary downloads and installs the llama.cpp binary for detected hardware
func InstallLlamaBinary(destDir string) error {
	hardware := DetectHardware()
	utils.LogDebug(fmt.Sprintf("Detected hardware: %s", hardware))

	info, err := GetBinaryInfo(hardware)
	if err != nil {
		return err
	}

	utils.LogDebug(fmt.Sprintf("Installing llama.cpp %s for %s", LlamaCppVersion, hardware))
	utils.LogDebug(fmt.Sprintf("URL: %s", info.URL))

	// Create destination directory
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", destDir, err)
	}

	// Download to temp file
	tmpFile, err := os.CreateTemp("", "llama-*.archive")
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}
	defer os.Remove(tmpFile.Name())
	defer tmpFile.Close()

	utils.LogDebug(fmt.Sprintf("Downloading to %s", tmpFile.Name()))

	resp, err := http.Get(info.URL)
	if err != nil {
		return fmt.Errorf("failed to download %s: %w", info.URL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download failed with status %d", resp.StatusCode)
	}

	// Copy with progress
	written, err := io.Copy(tmpFile, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to download: %w", err)
	}
	utils.LogDebug(fmt.Sprintf("Downloaded %d bytes", written))

	// Verify checksum if available
	if info.SHA256 != "" {
		tmpFile.Seek(0, 0)
		hash := sha256.New()
		if _, err := io.Copy(hash, tmpFile); err != nil {
			return fmt.Errorf("failed to compute checksum: %w", err)
		}
		actual := hex.EncodeToString(hash.Sum(nil))
		if actual != info.SHA256 {
			return fmt.Errorf("checksum mismatch: expected %s, got %s", info.SHA256, actual)
		}
		utils.LogDebug("Checksum verified")
	}

	// Extract library (all llama.cpp releases are .zip format)
	destPath := filepath.Join(destDir, info.LibName)

	if strings.HasSuffix(info.URL, ".zip") {
		if err := extractZip(tmpFile.Name(), info.LibPath, destPath); err != nil {
			return fmt.Errorf("failed to extract: %w", err)
		}
	} else {
		return fmt.Errorf("unknown archive format: %s", info.URL)
	}

	utils.LogDebug(fmt.Sprintf("Installed llama.cpp to %s", destPath))
	return nil
}

// extractZip extracts a specific file from a zip archive
func extractZip(archivePath, srcPath, destPath string) error {
	r, err := zip.OpenReader(archivePath)
	if err != nil {
		return err
	}
	defer r.Close()

	srcName := filepath.Base(srcPath)

	for _, f := range r.File {
		// Check if this is the file we want
		if strings.HasSuffix(f.Name, srcName) || f.Name == srcPath {
			rc, err := f.Open()
			if err != nil {
				return err
			}
			defer rc.Close()

			destFile, err := os.Create(destPath)
			if err != nil {
				return err
			}
			defer destFile.Close()

			if _, err := io.Copy(destFile, rc); err != nil {
				return err
			}

			// Set executable permission on Unix
			if runtime.GOOS != "windows" {
				os.Chmod(destPath, 0755)
			}

			return nil
		}
	}

	return fmt.Errorf("file %s not found in archive", srcPath)
}
