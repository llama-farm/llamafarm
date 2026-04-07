// Package llamabinary downloads and manages llama.cpp binary releases for arbitrary
// target platforms. It is the single source of truth for llama.cpp download URLs,
// extraction, symlink handling, and cache layout.
//
// The package supports cross-platform fetches: a developer on macOS can download a
// Linux ARM64 build for a Raspberry Pi via the llamadrone deployment pipeline. The
// cache is scoped by (os, arch, accelerator, version) so cross-platform fetches do
// not collide with the host's own binary.
package llamabinary

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"

	"github.com/llamafarm/cli/internal/buildinfo"
)

// Version is the pinned llama.cpp release version, overridable at build time via
// -ldflags "-X github.com/llamafarm/cli/internal/llamabinary.Version=bXXXX".
var Version = "b7694"

// Target identifies a platform + accelerator combination for a llama.cpp binary.
type Target struct {
	OS          string // "linux", "darwin", "windows"
	Arch        string // "amd64", "arm64" (canonical Go arch names)
	Accelerator string // "cpu", "cuda", "metal", "vulkan", "rocm"
}

// String returns a compact stable identifier like "linux/arm64/cpu".
func (t Target) String() string {
	return fmt.Sprintf("%s/%s/%s", t.OS, t.Arch, t.Accelerator)
}

// Slug returns a filesystem-safe identifier for cache subdirectory naming.
func (t Target) Slug() string {
	return fmt.Sprintf("%s-%s-%s", t.OS, t.Arch, t.Accelerator)
}

// Spec describes how to download and install a llama.cpp binary for a target.
type Spec struct {
	URL     string // Archive download URL
	SHA256  string // Optional checksum (empty = skip verification)
	LibPath string // Path to the main library inside the archive (relative)
	LibName string // Destination library filename (e.g., "libllama.so")
}

// ValidOSes lists the supported operating systems.
var ValidOSes = []string{"linux", "darwin", "windows"}

// ValidArches lists the supported architectures (canonical Go names).
var ValidArches = []string{"amd64", "arm64"}

// ValidAccelerators lists the supported compute backends.
var ValidAccelerators = []string{"cpu", "cuda", "metal", "vulkan", "rocm"}

// acceptArchAliases maps user-friendly arch names to canonical Go arch names.
var acceptArchAliases = map[string]string{
	"x86_64": "amd64",
	"amd64":  "amd64",
	"arm64":  "arm64",
	"aarch64": "arm64",
}

// CanonicalizeArch normalizes common arch name aliases (x86_64, aarch64) into the
// canonical Go names (amd64, arm64). Returns the canonical name and true, or the
// input and false if unrecognized.
func CanonicalizeArch(arch string) (string, bool) {
	if c, ok := acceptArchAliases[strings.ToLower(arch)]; ok {
		return c, true
	}
	return arch, false
}

// BestAcceleratorFor returns the default accelerator for an OS/arch combination
// when the user does not specify one explicitly.
func BestAcceleratorFor(goos, goarch string) string {
	if goos == "darwin" && goarch == "arm64" {
		return "metal"
	}
	// Default to CPU for everything else. Real CUDA/Vulkan detection requires the
	// host to actually have the hardware; we leave that to the orchestrator path.
	return "cpu"
}

// CurrentHostTarget returns the Target describing the current host machine, using
// the best-known accelerator for the host platform.
func CurrentHostTarget() Target {
	arch := runtime.GOARCH
	return Target{
		OS:          runtime.GOOS,
		Arch:        arch,
		Accelerator: BestAcceleratorFor(runtime.GOOS, arch),
	}
}

// Validate checks that the target has a recognized OS, arch, and accelerator.
func (t Target) Validate() error {
	if !contains(ValidOSes, t.OS) {
		return fmt.Errorf("unsupported OS %q (supported: %s)", t.OS, strings.Join(ValidOSes, ", "))
	}
	if !contains(ValidArches, t.Arch) {
		return fmt.Errorf("unsupported arch %q (supported: %s)", t.Arch, strings.Join(ValidArches, ", "))
	}
	if !contains(ValidAccelerators, t.Accelerator) {
		return fmt.Errorf("unsupported accelerator %q (supported: %s)", t.Accelerator, strings.Join(ValidAccelerators, ", "))
	}
	return nil
}

// LibNameFor returns the platform-specific library filename for a target OS.
func LibNameFor(goos string) string {
	switch goos {
	case "darwin":
		return "libllama.dylib"
	case "windows":
		return "llama.dll"
	default:
		return "libllama.so"
	}
}

// ErrSpecNotAvailable indicates no prebuilt binary exists for the target + version.
var ErrSpecNotAvailable = errors.New("no prebuilt llama.cpp binary available for target")

// testSpecOverride lets tests swap the SpecFor implementation without affecting
// production callers. Tests assign via SetTestSpecForOverride; nil means the
// real implementation runs.
var testSpecOverride func(Target, string) (Spec, error)

// SetTestSpecForOverride installs (or removes, with nil) a SpecFor override for
// testing. It is exported for cross-package tests but its name marks it as
// test-only.
func SetTestSpecForOverride(fn func(Target, string) (Spec, error)) {
	testSpecOverride = fn
}

// SpecFor returns the download Spec for a given target + version. Returns
// ErrSpecNotAvailable if no prebuilt exists for this combination.
func SpecFor(t Target, version string) (Spec, error) {
	if testSpecOverride != nil {
		return testSpecOverride(t, version)
	}
	if version == "" {
		version = Version
	}
	libName := LibNameFor(t.OS)

	urlBase := func(artifact string) string {
		return fmt.Sprintf("https://github.com/ggml-org/llama.cpp/releases/download/%s/%s", version, artifact)
	}

	switch {
	case t.OS == "darwin" && t.Arch == "arm64" && t.Accelerator == "metal":
		return Spec{
			URL:     urlBase(fmt.Sprintf("llama-%s-bin-macos-arm64.tar.gz", version)),
			LibPath: libName,
			LibName: libName,
		}, nil
	case t.OS == "darwin" && t.Arch == "amd64" && t.Accelerator == "cpu":
		return Spec{
			URL:     urlBase(fmt.Sprintf("llama-%s-bin-macos-x64.tar.gz", version)),
			LibPath: libName,
			LibName: libName,
		}, nil
	case t.OS == "linux" && t.Arch == "amd64" && t.Accelerator == "cpu":
		return Spec{
			URL:     urlBase(fmt.Sprintf("llama-%s-bin-ubuntu-x64.tar.gz", version)),
			LibPath: libName,
			LibName: libName,
		}, nil
	case t.OS == "linux" && t.Arch == "amd64" && t.Accelerator == "vulkan":
		return Spec{
			URL:     urlBase(fmt.Sprintf("llama-%s-bin-ubuntu-vulkan-x64.tar.gz", version)),
			LibPath: libName,
			LibName: libName,
		}, nil
	case t.OS == "linux" && t.Arch == "amd64" && (t.Accelerator == "cuda" || t.Accelerator == "rocm"):
		// Upstream no longer ships Linux CUDA or ROCm binaries; fall back to Vulkan which
		// typically gives similar GPU acceleration on supported hardware.
		return Spec{
			URL:     urlBase(fmt.Sprintf("llama-%s-bin-ubuntu-vulkan-x64.tar.gz", version)),
			LibPath: libName,
			LibName: libName,
		}, nil
	case t.OS == "linux" && t.Arch == "arm64" && t.Accelerator == "cpu":
		// LlamaFarm hosts its own Linux ARM64 builds since upstream does not provide them.
		lfVersion := llamafarmReleaseVersion()
		return Spec{
			URL: fmt.Sprintf(
				"https://github.com/llama-farm/llamafarm/releases/download/%s/llama-%s-bin-linux-arm64.tar.gz",
				lfVersion, version,
			),
			LibPath: libName,
			LibName: libName,
		}, nil
	case t.OS == "windows" && t.Arch == "amd64" && t.Accelerator == "cpu":
		return Spec{
			URL:     urlBase(fmt.Sprintf("llama-%s-bin-win-cpu-x64.zip", version)),
			LibPath: libName,
			LibName: libName,
		}, nil
	case t.OS == "windows" && t.Arch == "amd64" && t.Accelerator == "cuda":
		return Spec{
			URL:     urlBase(fmt.Sprintf("llama-%s-bin-win-cuda-12.4-x64.zip", version)),
			LibPath: libName,
			LibName: libName,
		}, nil
	case t.OS == "windows" && t.Arch == "amd64" && t.Accelerator == "vulkan":
		return Spec{
			URL:     urlBase(fmt.Sprintf("llama-%s-bin-win-vulkan-x64.zip", version)),
			LibPath: libName,
			LibName: libName,
		}, nil
	}

	return Spec{}, fmt.Errorf("%w: %s", ErrSpecNotAvailable, t)
}

// llamafarmReleaseVersion returns the LlamaFarm release tag used as the host for
// LlamaFarm-provided Linux ARM64 builds.
func llamafarmReleaseVersion() string {
	if buildinfo.CurrentVersion == "dev" {
		return "v0.0.1"
	}
	return buildinfo.CurrentVersion
}

// CacheRoot returns the top-level cache directory shared by all targets + versions.
// Respects $LLAMAFARM_CACHE_DIR if set, else uses platform-standard locations that
// match the Python llamafarm-llama package.
func CacheRoot() (string, error) {
	if v := os.Getenv("LLAMAFARM_CACHE_DIR"); v != "" {
		return v, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("locate home dir: %w", err)
	}
	switch runtime.GOOS {
	case "darwin":
		return filepath.Join(home, "Library", "Caches", "llamafarm-llama"), nil
	case "windows":
		lad := os.Getenv("LOCALAPPDATA")
		if lad == "" {
			lad = home
		}
		return filepath.Join(lad, "llamafarm-llama", "cache"), nil
	default:
		xdg := os.Getenv("XDG_CACHE_HOME")
		if xdg == "" {
			xdg = filepath.Join(home, ".cache")
		}
		return filepath.Join(xdg, "llamafarm-llama"), nil
	}
}

// CacheDir returns the cache directory for a specific (target, version) scoped so
// cross-platform fetches do not collide. For the current host target the directory
// layout matches the historical `<root>/<version>/` path so existing caches remain
// usable without a re-download.
func CacheDir(t Target, version string) (string, error) {
	if version == "" {
		version = Version
	}
	root, err := CacheRoot()
	if err != nil {
		return "", err
	}
	if t == CurrentHostTarget() {
		return filepath.Join(root, version), nil
	}
	return filepath.Join(root, t.Slug(), version), nil
}

// LibPath returns the expected on-disk path of the main library for a target.
func LibPath(t Target, version string) (string, error) {
	dir, err := CacheDir(t, version)
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, LibNameFor(t.OS)), nil
}

// IsCached reports whether the main library for (target, version) is already
// present on disk.
func IsCached(t Target, version string) bool {
	p, err := LibPath(t, version)
	if err != nil {
		return false
	}
	st, err := os.Stat(p)
	return err == nil && !st.IsDir() && st.Size() > 0
}

// Result describes the outcome of a Download call.
type Result struct {
	LibPath string // Path to the installed main library
	DestDir string // Directory containing the library and its dependencies
	Cached  bool   // True if the binary was already present (no download performed)
}

// Logf is the package's optional debug logger. Consumers may assign a function to
// receive debug output; nil means silent.
var Logf func(format string, args ...any)

func logf(format string, args ...any) {
	if Logf != nil {
		Logf(format, args...)
	}
}

// Download ensures the llama.cpp binary for (target, version) is installed in its
// platform-scoped cache directory. If the binary is already present, Download is a
// no-op and returns Cached=true.
func Download(ctx context.Context, t Target, version string) (Result, error) {
	if err := t.Validate(); err != nil {
		return Result{}, err
	}
	if version == "" {
		version = Version
	}

	destDir, err := CacheDir(t, version)
	if err != nil {
		return Result{}, err
	}
	libPath := filepath.Join(destDir, LibNameFor(t.OS))

	if IsCached(t, version) {
		logf("llama.cpp already cached at %s", libPath)
		return Result{LibPath: libPath, DestDir: destDir, Cached: true}, nil
	}

	spec, err := SpecFor(t, version)
	if err != nil {
		return Result{}, err
	}

	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return Result{}, fmt.Errorf("create dest dir: %w", err)
	}

	logf("downloading %s → %s", spec.URL, destDir)

	tmpFile, err := os.CreateTemp("", "llama-*.archive")
	if err != nil {
		return Result{}, fmt.Errorf("create temp file: %w", err)
	}
	tmpName := tmpFile.Name()
	defer func() {
		tmpFile.Close()
		os.Remove(tmpName)
	}()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, spec.URL, nil)
	if err != nil {
		return Result{}, fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("User-Agent", "lf-cli")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return Result{}, fmt.Errorf("fetch %s: %w", spec.URL, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return Result{}, fmt.Errorf("download %s: HTTP %d", spec.URL, resp.StatusCode)
	}

	if _, err := io.Copy(tmpFile, resp.Body); err != nil {
		return Result{}, fmt.Errorf("write archive: %w", err)
	}

	if spec.SHA256 != "" {
		if _, err := tmpFile.Seek(0, 0); err != nil {
			return Result{}, fmt.Errorf("seek archive: %w", err)
		}
		h := sha256.New()
		if _, err := io.Copy(h, tmpFile); err != nil {
			return Result{}, fmt.Errorf("checksum archive: %w", err)
		}
		got := hex.EncodeToString(h.Sum(nil))
		if got != spec.SHA256 {
			return Result{}, fmt.Errorf("checksum mismatch: expected %s, got %s", spec.SHA256, got)
		}
	}

	// Extract main lib + dependencies based on archive format.
	switch {
	case strings.HasSuffix(spec.URL, ".tar.gz") || strings.HasSuffix(spec.URL, ".tgz"):
		if err := extractTarGz(tmpName, spec.LibPath, libPath); err != nil {
			return Result{}, fmt.Errorf("extract main lib: %w", err)
		}
		if err := extractTarGzDependencies(tmpName, destDir, spec.LibName, t.OS); err != nil {
			logf("warning: extracting tar.gz dependencies: %v", err)
		}
	case strings.HasSuffix(spec.URL, ".zip"):
		if err := extractZip(tmpName, spec.LibPath, libPath); err != nil {
			return Result{}, fmt.Errorf("extract main lib: %w", err)
		}
		if err := extractZipDependencies(tmpName, destDir, spec.LibName, t.OS); err != nil {
			logf("warning: extracting zip dependencies: %v", err)
		}
	default:
		return Result{}, fmt.Errorf("unknown archive format: %s", spec.URL)
	}

	if runtime.GOOS != "windows" && t.OS != "windows" {
		if err := createDependencySymlinks(destDir, t.OS); err != nil {
			logf("warning: creating dep symlinks: %v", err)
		}
	}

	logf("installed llama.cpp to %s", libPath)
	return Result{LibPath: libPath, DestDir: destDir}, nil
}

// Export copies the installed binary and all dependency files for (target, version)
// into destDir as a flat directory suitable for ansible/packer pickup. The cache
// must already be populated (call Download first).
func Export(t Target, version string, destDir string) error {
	if version == "" {
		version = Version
	}
	srcDir, err := CacheDir(t, version)
	if err != nil {
		return err
	}
	if !IsCached(t, version) {
		return fmt.Errorf("binary not cached for %s: run Download first", t)
	}
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return fmt.Errorf("create export dir: %w", err)
	}
	entries, err := os.ReadDir(srcDir)
	if err != nil {
		return fmt.Errorf("read cache dir: %w", err)
	}
	for _, entry := range entries {
		name := entry.Name()
		srcPath := filepath.Join(srcDir, name)
		dstPath := filepath.Join(destDir, name)
		info, err := os.Lstat(srcPath)
		if err != nil {
			return fmt.Errorf("lstat %s: %w", srcPath, err)
		}
		// Preserve symlinks.
		if info.Mode()&os.ModeSymlink != 0 {
			target, err := os.Readlink(srcPath)
			if err != nil {
				return fmt.Errorf("readlink %s: %w", srcPath, err)
			}
			// Remove any existing file at dstPath first.
			os.Remove(dstPath)
			if err := os.Symlink(target, dstPath); err != nil {
				return fmt.Errorf("symlink %s: %w", dstPath, err)
			}
			continue
		}
		if info.IsDir() {
			continue
		}
		if err := copyFile(srcPath, dstPath); err != nil {
			return err
		}
	}
	return nil
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("open %s: %w", src, err)
	}
	defer in.Close()
	// Remove any existing file at dst to avoid permission issues.
	os.Remove(dst)
	out, err := os.Create(dst)
	if err != nil {
		return fmt.Errorf("create %s: %w", dst, err)
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return fmt.Errorf("copy %s -> %s: %w", src, dst, err)
	}
	if err := out.Close(); err != nil {
		return fmt.Errorf("close %s: %w", dst, err)
	}
	// Match original mode on unix (best-effort).
	if st, err := os.Stat(src); err == nil {
		_ = os.Chmod(dst, st.Mode())
	}
	return nil
}

func contains(haystack []string, needle string) bool {
	return slices.Contains(haystack, needle)
}
