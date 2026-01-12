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
		URL:     fmt.Sprintf("https://github.com/ggml-org/llama.cpp/releases/download/%s/llama-%s-bin-win-cuda-12.4-x64.zip", LlamaCppVersion, LlamaCppVersion),
		SHA256:  "",
		LibPath: "llama.dll",
		LibName: "llama.dll",
	},
}

// GetLlamaCacheDir returns the cache directory for llama.cpp binaries.
// This matches the paths used by the Python llamafarm-llama package.
func GetLlamaCacheDir() (string, error) {
	// Check for environment override
	if cacheDir := os.Getenv("LLAMAFARM_CACHE_DIR"); cacheDir != "" {
		return cacheDir, nil
	}

	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("failed to get user home directory: %w", err)
	}

	switch runtime.GOOS {
	case "darwin":
		return filepath.Join(homeDir, "Library", "Caches", "llamafarm-llama"), nil
	case "windows":
		localAppData := os.Getenv("LOCALAPPDATA")
		if localAppData == "" {
			localAppData = homeDir
		}
		return filepath.Join(localAppData, "llamafarm-llama", "cache"), nil
	default: // Linux and others
		xdgCache := os.Getenv("XDG_CACHE_HOME")
		if xdgCache == "" {
			xdgCache = filepath.Join(homeDir, ".cache")
		}
		return filepath.Join(xdgCache, "llamafarm-llama"), nil
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
	cacheDir, err := GetLlamaCacheDir()
	if err != nil {
		return false
	}
	libPath := filepath.Join(cacheDir, LlamaCppVersion, GetLlamaLibName())
	_, err = os.Stat(libPath)
	return err == nil
}

// EnsureLlamaBinary downloads llama.cpp binaries if not already installed.
// Returns the path to the installed binaries.
func EnsureLlamaBinary() (string, error) {
	cacheDir, err := GetLlamaCacheDir()
	if err != nil {
		return "", err
	}
	versionDir := filepath.Join(cacheDir, LlamaCppVersion)

	// Check if already installed
	if IsLlamaBinaryInstalled() {
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

		// Extract all dependencies (ggml libs, metal shaders, etc.)
		if err := extractDependencies(tmpFile.Name(), destDir); err != nil {
			utils.LogDebug(fmt.Sprintf("Warning: failed to extract some dependencies: %v", err))
		}
	} else {
		return fmt.Errorf("unknown archive format: %s", info.URL)
	}

	utils.LogDebug(fmt.Sprintf("Installed llama.cpp to %s", destPath))
	return nil
}

// extractDependencies extracts all library dependencies from a llama.cpp release
// This includes ggml libs (.dll/.dylib/.so), metal shaders, CUDA libs, etc.
func extractDependencies(archivePath, destDir string) error {
	r, err := zip.OpenReader(archivePath)
	if err != nil {
		return err
	}
	defer r.Close()

	// Determine which file patterns to look for based on platform
	mainLib := GetLlamaLibName()
	var patterns []string
	switch runtime.GOOS {
	case "windows":
		patterns = []string{".dll"}
	case "darwin":
		// macOS: version before extension (libggml.0.0.0.dylib)
		patterns = []string{".dylib", ".metal"}
	default: // Linux
		// Linux: versioned (libggml.so.0.0.0) and unversioned (libggml.so, ggml-cpu.so)
		patterns = []string{".so.", ".so"}
	}

	extractedCount := 0
	for _, f := range r.File {
		// Skip directories
		if f.FileInfo().IsDir() {
			continue
		}

		// Skip symlinks (we handle the actual files)
		if f.Mode()&os.ModeSymlink != 0 {
			continue
		}

		name := filepath.Base(f.Name)

		// Validate filename to prevent path traversal attacks
		if name == "" || name == "." || name == ".." ||
			strings.ContainsAny(name, "/\\") || filepath.IsAbs(name) {
			continue
		}

		nameLower := strings.ToLower(name)

		// Check if this is a dependency file we should extract
		shouldExtract := false
		for _, pattern := range patterns {
			if strings.Contains(nameLower, pattern) {
				// Skip the main library (already extracted via extractZip)
				if nameLower == strings.ToLower(mainLib) {
					continue
				}
				// For versioned libraries like libllama.0.0.7376.dylib or libllama.so.0.0.0, skip those too
				if strings.HasPrefix(nameLower, "libllama.") || strings.HasPrefix(nameLower, "llama.") {
					continue
				}
				shouldExtract = true
				break
			}
		}

		if !shouldExtract {
			continue
		}

		destPath := filepath.Join(destDir, name)

		// Skip if already exists
		if _, err := os.Stat(destPath); err == nil {
			continue
		}

		// Check file size - skip tiny files that are likely symlink text files
		if f.UncompressedSize64 < 100 {
			utils.LogDebug(fmt.Sprintf("Skipping small file (likely symlink): %s (%d bytes)", name, f.UncompressedSize64))
			continue
		}

		rc, err := f.Open()
		if err != nil {
			utils.LogDebug(fmt.Sprintf("Warning: could not open %s: %v", f.Name, err))
			continue
		}

		destFile, err := os.Create(destPath)
		if err != nil {
			rc.Close()
			utils.LogDebug(fmt.Sprintf("Warning: could not create %s: %v", destPath, err))
			continue
		}

		written, err := io.Copy(destFile, rc)
		rc.Close()
		destFile.Close()

		if err != nil {
			utils.LogDebug(fmt.Sprintf("Warning: could not write %s: %v", destPath, err))
			continue
		}

		// Set executable permission on Unix for libraries
		if runtime.GOOS != "windows" && !strings.HasSuffix(nameLower, ".metal") {
			os.Chmod(destPath, 0755)
		}

		utils.LogDebug(fmt.Sprintf("Extracted dependency: %s (%d bytes)", name, written))
		extractedCount++
	}

	// Create symlinks for versioned libraries on Unix
	if runtime.GOOS != "windows" {
		if err := createDependencySymlinks(destDir); err != nil {
			utils.LogDebug(fmt.Sprintf("Warning: could not create dependency symlinks: %v", err))
		}
	}

	utils.LogDebug(fmt.Sprintf("Extracted %d dependencies", extractedCount))
	return nil
}

// createDependencySymlinks creates symlinks for versioned libraries
// macOS: libggml.0.0.0.dylib -> libggml.0.dylib -> libggml.dylib
// Linux: libggml.so.0.0.0 -> libggml.so.0 -> libggml.so
func createDependencySymlinks(destDir string) error {
	entries, err := os.ReadDir(destDir)
	if err != nil {
		return err
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()

		if runtime.GOOS == "darwin" {
			// macOS: libfoo.MAJOR.MINOR.PATCH.dylib
			if !strings.HasSuffix(name, ".dylib") {
				continue
			}

			// Check if this looks like a versioned library (has version numbers before .dylib)
			// e.g., libggml-base.0.0.0.dylib or libggml.0.0.0.dylib
			parts := strings.Split(name, ".")
			if len(parts) < 5 { // libggml.0.0.0.dylib = 5 parts
				continue
			}

			// Find the base name (everything before the version numbers)
			baseName := ""
			versionStart := -1
			for i, part := range parts {
				if _, err := fmt.Sscanf(part, "%d", new(int)); err == nil {
					versionStart = i
					break
				}
				if baseName != "" {
					baseName += "."
				}
				baseName += part
			}

			if versionStart < 0 || baseName == "" {
				continue
			}

			majorVersion := parts[versionStart]

			// Create libfoo.0.dylib -> libfoo.0.0.0.dylib
			majorSymlink := filepath.Join(destDir, fmt.Sprintf("%s.%s.dylib", baseName, majorVersion))
			if _, err := os.Lstat(majorSymlink); os.IsNotExist(err) {
				if err := os.Symlink(name, majorSymlink); err == nil {
					utils.LogDebug(fmt.Sprintf("Created symlink: %s -> %s", filepath.Base(majorSymlink), name))
				}
			}

			// Create libfoo.dylib -> libfoo.0.dylib
			baseSymlink := filepath.Join(destDir, fmt.Sprintf("%s.dylib", baseName))
			if _, err := os.Lstat(baseSymlink); os.IsNotExist(err) {
				majorName := filepath.Base(majorSymlink)
				if err := os.Symlink(majorName, baseSymlink); err == nil {
					utils.LogDebug(fmt.Sprintf("Created symlink: %s -> %s", filepath.Base(baseSymlink), majorName))
				}
			}
		} else {
			// Linux: libfoo.so.MAJOR.MINOR.PATCH
			if !strings.Contains(name, ".so.") {
				continue
			}

			// Parse libggml.so.0.0.0 or libggml-base.so.0.0.0
			soIndex := strings.Index(name, ".so.")
			if soIndex < 0 {
				continue
			}

			baseName := name[:soIndex]      // libggml or libggml-base
			versionPart := name[soIndex+4:] // 0.0.0
			versionParts := strings.Split(versionPart, ".")
			if len(versionParts) < 1 {
				continue
			}

			majorVersion := versionParts[0]

			// Create libfoo.so.0 -> libfoo.so.0.0.0
			majorSymlink := filepath.Join(destDir, fmt.Sprintf("%s.so.%s", baseName, majorVersion))
			if _, err := os.Lstat(majorSymlink); os.IsNotExist(err) {
				if err := os.Symlink(name, majorSymlink); err == nil {
					utils.LogDebug(fmt.Sprintf("Created symlink: %s -> %s", filepath.Base(majorSymlink), name))
				}
			}

			// Create libfoo.so -> libfoo.so.0
			baseSymlink := filepath.Join(destDir, fmt.Sprintf("%s.so", baseName))
			if _, err := os.Lstat(baseSymlink); os.IsNotExist(err) {
				majorName := filepath.Base(majorSymlink)
				if err := os.Symlink(majorName, baseSymlink); err == nil {
					utils.LogDebug(fmt.Sprintf("Created symlink: %s -> %s", filepath.Base(baseSymlink), majorName))
				}
			}
		}
	}

	return nil
}

// extractZip extracts a specific file from a zip archive, following symlinks if needed
func extractZip(archivePath, srcPath, destPath string) error {
	r, err := zip.OpenReader(archivePath)
	if err != nil {
		return err
	}
	defer r.Close()

	destDir := filepath.Dir(destPath)
	srcName := filepath.Base(srcPath)

	// Build a map of all files in the archive for symlink resolution
	fileMap := make(map[string]*zip.File)
	for _, f := range r.File {
		fileMap[f.Name] = f
	}

	// Find the target file by iterating over r.File (preserves archive order)
	// to ensure deterministic file selection when multiple files match
	var targetFile *zip.File
	var targetPath string
	for _, f := range r.File {
		if strings.HasSuffix(f.Name, srcName) || f.Name == srcPath {
			targetFile = f
			targetPath = f.Name
			break
		}
	}

	if targetFile == nil {
		return fmt.Errorf("file %s not found in archive", srcPath)
	}

	// Follow symlink chain and extract all files in the chain
	return extractFileWithSymlinks(r, fileMap, targetFile, targetPath, destDir, srcName)
}

// extractFileWithSymlinks extracts a file, following and preserving symlink chains
func extractFileWithSymlinks(r *zip.ReadCloser, fileMap map[string]*zip.File, f *zip.File, fPath, destDir, finalName string) error {
	// Validate finalName to prevent path traversal attacks
	if finalName == "" || finalName == "." || finalName == ".." ||
		strings.ContainsAny(finalName, "/\\") || filepath.IsAbs(finalName) {
		return fmt.Errorf("invalid filename: %s", finalName)
	}

	// Check if this is a symlink
	if f.Mode()&os.ModeSymlink != 0 {
		// Read the symlink target
		rc, err := f.Open()
		if err != nil {
			return fmt.Errorf("failed to open symlink %s: %w", f.Name, err)
		}
		targetBytes, err := io.ReadAll(rc)
		rc.Close()
		if err != nil {
			return fmt.Errorf("failed to read symlink target for %s: %w", f.Name, err)
		}
		target := string(targetBytes)
		utils.LogDebug(fmt.Sprintf("Found symlink: %s -> %s", f.Name, target))

		// Resolve the target path relative to the symlink's directory
		symlinkDir := filepath.Dir(fPath)
		resolvedTarget := filepath.Join(symlinkDir, target)
		// Normalize path separators for zip (always forward slashes)
		resolvedTarget = strings.ReplaceAll(resolvedTarget, "\\", "/")
		// Clean the path to handle ../ etc
		resolvedTarget = filepath.Clean(resolvedTarget)
		resolvedTarget = strings.ReplaceAll(resolvedTarget, "\\", "/")

		// Find the target file in the archive
		targetFile, ok := fileMap[resolvedTarget]
		if !ok {
			// Try matching by basename if exact path doesn't work
			targetBase := filepath.Base(target)
			for name, tf := range fileMap {
				if strings.HasSuffix(name, targetBase) && filepath.Dir(name) == symlinkDir {
					targetFile = tf
					resolvedTarget = name
					ok = true
					break
				}
			}
		}

		if !ok {
			return fmt.Errorf("symlink target %s (resolved: %s) not found in archive", target, resolvedTarget)
		}

		// Recursively extract the target (which may also be a symlink)
		targetBaseName := filepath.Base(target)
		if err := extractFileWithSymlinks(r, fileMap, targetFile, resolvedTarget, destDir, targetBaseName); err != nil {
			return err
		}

		// Create the symlink in the destination
		symlinkPath := filepath.Join(destDir, finalName)

		// Validate symlink target to prevent path traversal
		// Resolve where the symlink would point and ensure it stays within destDir
		resolvedSymlinkTarget := filepath.Join(destDir, target)
		resolvedSymlinkTarget = filepath.Clean(resolvedSymlinkTarget)
		if !strings.HasPrefix(resolvedSymlinkTarget, filepath.Clean(destDir)+string(filepath.Separator)) &&
			resolvedSymlinkTarget != filepath.Clean(destDir) {
			return fmt.Errorf("symlink target %s would escape destination directory", target)
		}

		// Remove existing file/symlink if it exists
		os.Remove(symlinkPath)

		if err := os.Symlink(target, symlinkPath); err != nil {
			return fmt.Errorf("failed to create symlink %s -> %s: %w", symlinkPath, target, err)
		}
		utils.LogDebug(fmt.Sprintf("Created symlink: %s -> %s", symlinkPath, target))

		return nil
	}

	// Regular file - extract it
	destPath := filepath.Join(destDir, finalName)
	utils.LogDebug(fmt.Sprintf("Extracting file: %s -> %s", f.Name, destPath))

	rc, err := f.Open()
	if err != nil {
		return fmt.Errorf("failed to open %s: %w", f.Name, err)
	}
	defer rc.Close()

	// Remove existing file if it exists
	os.Remove(destPath)

	destFile, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("failed to create %s: %w", destPath, err)
	}
	defer destFile.Close()

	written, err := io.Copy(destFile, rc)
	if err != nil {
		return fmt.Errorf("failed to write %s: %w", destPath, err)
	}
	utils.LogDebug(fmt.Sprintf("Wrote %d bytes to %s", written, destPath))

	// Set executable permission on Unix
	if runtime.GOOS != "windows" {
		os.Chmod(destPath, 0755)
	}

	return nil
}
