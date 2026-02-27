package cmd

import (
	"archive/tar"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/llamafarm/cli/cmd/orchestrator"
	"github.com/llamafarm/cli/cmd/utils"
	"github.com/llamafarm/cli/internal/buildinfo"
	"github.com/spf13/cobra"
)

// BundleManifest describes the contents of a LlamaFarm bundle archive.
type BundleManifest struct {
	Version     string            `json:"version"`
	Platform    string            `json:"platform"`
	Arch        string            `json:"arch"`
	Accelerator string            `json:"accelerator"`
	Components  map[string]string `json:"components"`
	Addons      []string          `json:"addons"`
}

// bundleFlags holds parsed CLI flags for the bundle command.
var bundleFlags struct {
	platform    string
	arch        string
	accelerator string
	addons      string
	version     string
	output      string
	localCLI    string
}

// validPlatforms lists supported OS targets.
var validPlatforms = []string{"linux", "darwin", "windows"}

// validArchitectures lists supported CPU architectures.
var validArchitectures = []string{"x86_64", "arm64"}

// validAccelerators lists supported compute backends.
var validAccelerators = []string{"cuda", "rocm", "vulkan", "cpu", "metal"}

// platformToGoOS maps bundle platform names to Go OS names used in release artifacts.
var platformToGoOS = map[string]string{
	"linux":   "linux",
	"darwin":  "darwin",
	"windows": "windows",
}

// archToGoArch maps bundle arch names to Go arch names used in release artifacts.
var archToGoArch = map[string]string{
	"x86_64": "amd64",
	"arm64":  "arm64",
}

// platformToPyAppOS maps bundle platform names to PyApp binary platform strings.
// PyApp uses "macos" instead of "darwin" for macOS builds.
var platformToPyAppOS = map[string]string{
	"linux":   "linux",
	"darwin":  "macos",
	"windows": "windows",
}

// knownInvalidCombos lists platform/arch combos that don't have release artifacts.
var knownInvalidCombos = map[string]bool{
	"darwin-x86_64":  true, // macOS Intel not supported
	"windows-arm64":  true, // No Windows ARM64 builds
}

var bundleCmd = &cobra.Command{
	Use:   "bundle",
	Short: "Package LlamaFarm for distribution",
	Long: `Package all LlamaFarm components into a single distributable archive.

The bundle includes the CLI binary, PyApp service binaries (server, rag, runtime),
and optionally addon wheel archives and accelerator-specific torch wheels.

The resulting archive can be transferred to a remote machine and installed
with: ./install.sh <archive.tar.gz>

Examples:
  # Bundle for Linux with CUDA support
  lf bundle --platform linux --arch x86_64 --accelerator cuda -o bundle.tar.gz

  # Bundle with addons
  lf bundle --platform linux --arch x86_64 --accelerator cuda --addons stt,tts -o bundle.tar.gz

  # Bundle a specific version
  lf bundle --platform linux --arch arm64 --accelerator cpu --version v0.8.0 -o bundle.tar.gz`,
	RunE: runBundle,
}

// bootstrapCmd triggers PyApp first-run extraction for all installed service binaries.
var bootstrapCmd = &cobra.Command{
	Use:    "bootstrap",
	Short:  "Pre-extract PyApp service binaries",
	Long:   "Triggers first-run extraction for all installed PyApp service binaries so users don't wait on first launch.",
	Hidden: true,
	RunE:   runBootstrap,
}

func init() {
	rootCmd.AddCommand(bundleCmd)
	bundleCmd.AddCommand(bootstrapCmd)

	bundleCmd.Flags().StringVar(&bundleFlags.platform, "platform", "", "Target OS: linux, darwin, windows (required)")
	bundleCmd.Flags().StringVar(&bundleFlags.arch, "arch", "", "Target architecture: x86_64, arm64 (required)")
	bundleCmd.Flags().StringVar(&bundleFlags.accelerator, "accelerator", "", "Compute backend: cuda, rocm, vulkan, cpu, metal (required)")
	bundleCmd.Flags().StringVar(&bundleFlags.addons, "addons", "", "Comma-separated addon names to include (e.g., stt,tts)")
	bundleCmd.Flags().StringVar(&bundleFlags.version, "version", "", "LlamaFarm version to bundle (default: current CLI version)")
	bundleCmd.Flags().StringVarP(&bundleFlags.output, "output", "o", "", "Output file path (required)")
	bundleCmd.Flags().StringVar(&bundleFlags.localCLI, "local-cli", "", "Path to a local CLI binary to include instead of downloading")

	bundleCmd.MarkFlagRequired("platform")
	bundleCmd.MarkFlagRequired("arch")
	bundleCmd.MarkFlagRequired("accelerator")
	bundleCmd.MarkFlagRequired("output")
}

// bootstrapServices are the service names used by ResolveBinaryPath.
var bootstrapServices = []string{"server", "rag", "universal-runtime"}

func runBootstrap(cmd *cobra.Command, args []string) error {
	fmt.Println("Bootstrapping PyApp service binaries...")

	var failed []string
	for _, svc := range bootstrapServices {
		binaryPath, err := orchestrator.ResolveBinaryPath(svc)
		if err != nil {
			fmt.Printf("  Skipping %s (not installed)\n", svc)
			continue
		}

		fmt.Printf("  Extracting %s...\n", svc)
		restore := exec.Command(binaryPath, "self", "restore")
		restore.Stdout = os.Stdout
		restore.Stderr = os.Stderr
		if err := restore.Run(); err != nil {
			fmt.Printf("  Warning: %s bootstrap failed: %v\n", svc, err)
			failed = append(failed, svc)
		} else {
			fmt.Printf("  %s ready\n", svc)
		}
	}

	if len(failed) > 0 {
		fmt.Printf("\nSome services failed to bootstrap: %s\n", strings.Join(failed, ", "))
		fmt.Println("They will extract on first run instead.")
	} else {
		fmt.Println("All service binaries bootstrapped successfully.")
	}

	return nil
}

func runBundle(cmd *cobra.Command, args []string) error {
	// Validate flags
	if err := validateBundleFlags(); err != nil {
		return err
	}

	ver := bundleFlags.version
	if ver == "" {
		ver = buildinfo.CurrentVersion
		if ver == "" || ver == "dev" {
			if bundleFlags.localCLI != "" {
				ver = "v0.0.0-dev"
			} else {
				return fmt.Errorf("cannot determine version; specify --version explicitly")
			}
		}
	}
	// Ensure version has v prefix for GitHub release tags
	if !strings.HasPrefix(ver, "v") {
		ver = "v" + ver
	}

	goOS := platformToGoOS[bundleFlags.platform]
	goArch := archToGoArch[bundleFlags.arch]

	fmt.Printf("Bundling LlamaFarm %s for %s/%s (%s)...\n", ver, bundleFlags.platform, bundleFlags.arch, bundleFlags.accelerator)

	// Create temp directory for downloads
	tmpDir, err := os.MkdirTemp("", "llamafarm-bundle-*")
	if err != nil {
		return fmt.Errorf("failed to create temp directory: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	manifest := BundleManifest{
		Version:     ver,
		Platform:    bundleFlags.platform,
		Arch:        bundleFlags.arch,
		Accelerator: bundleFlags.accelerator,
		Components:  make(map[string]string),
		Addons:      []string{},
	}

	// Include CLI binary (local copy or download)
	cliBinaryName := fmt.Sprintf("lf-%s-%s", goOS, goArch)
	if bundleFlags.platform == "windows" {
		cliBinaryName += ".exe"
	}
	if bundleFlags.localCLI != "" {
		if _, err := os.Stat(bundleFlags.localCLI); err != nil {
			return fmt.Errorf("local CLI binary not found: %w", err)
		}
		fmt.Printf("  Copying local CLI binary (%s)...\n", cliBinaryName)
		if err := utils.CopyFile(bundleFlags.localCLI, filepath.Join(tmpDir, cliBinaryName)); err != nil {
			return fmt.Errorf("failed to copy local CLI binary: %w", err)
		}
	} else {
		fmt.Printf("  Downloading CLI binary (%s)...\n", cliBinaryName)
		if err := downloadReleaseAsset(ver, cliBinaryName, filepath.Join(tmpDir, cliBinaryName)); err != nil {
			return fmt.Errorf("failed to download CLI binary: %w", err)
		}
	}
	manifest.Components["cli"] = cliBinaryName

	// Download PyApp service binaries
	pyappPlatform := fmt.Sprintf("%s-%s", platformToPyAppOS[bundleFlags.platform], bundleFlags.arch)
	for _, component := range []string{"server", "rag", "runtime"} {
		binaryName := fmt.Sprintf("llamafarm-%s-%s", component, pyappPlatform)
		if bundleFlags.platform == "windows" {
			binaryName += ".exe"
		}
		fmt.Printf("  Downloading %s binary (%s)...\n", component, binaryName)
		if err := downloadReleaseAsset(ver, binaryName, filepath.Join(tmpDir, binaryName)); err != nil {
			return fmt.Errorf("failed to download %s binary: %w", component, err)
		}
		manifest.Components[component] = binaryName
	}

	// Download accelerator-specific torch wheels (if not CPU)
	if bundleFlags.accelerator != "cpu" {
		fmt.Printf("  Downloading %s torch wheels...\n", bundleFlags.accelerator)
		torchDir := filepath.Join(tmpDir, "torch")
		if err := os.MkdirAll(torchDir, 0755); err != nil {
			return fmt.Errorf("failed to create torch directory: %w", err)
		}
		if err := downloadTorchWheels(bundleFlags.accelerator, bundleFlags.platform, bundleFlags.arch, torchDir); err != nil {
			return fmt.Errorf("failed to download torch wheels: %w", err)
		}
	}

	// Download addon wheel archives
	if bundleFlags.addons != "" {
		addonNames := strings.Split(bundleFlags.addons, ",")
		addonDir := filepath.Join(tmpDir, "addons")
		registryDir := filepath.Join(tmpDir, "addons-registry")
		if err := os.MkdirAll(addonDir, 0755); err != nil {
			return fmt.Errorf("failed to create addons directory: %w", err)
		}
		if err := os.MkdirAll(registryDir, 0755); err != nil {
			return fmt.Errorf("failed to create addons-registry directory: %w", err)
		}

		addonPlatform := getAddonPlatformString(bundleFlags.platform, bundleFlags.arch)
		for _, addon := range addonNames {
			addon = strings.TrimSpace(addon)
			if addon == "" {
				continue
			}
			fmt.Printf("  Downloading addon wheels: %s...\n", addon)
			wheelArchiveName := fmt.Sprintf("%s-wheels-%s.tar.gz", addon, addonPlatform)
			if err := downloadReleaseAsset(ver, wheelArchiveName, filepath.Join(addonDir, wheelArchiveName)); err != nil {
				return fmt.Errorf("failed to download addon %s wheels: %w", addon, err)
			}
			manifest.Addons = append(manifest.Addons, addon)

			// Copy addon registry YAML
			registryFile := findAddonRegistryFile(addon)
			if registryFile != "" {
				data, err := os.ReadFile(registryFile)
				if err != nil {
					utils.LogDebug(fmt.Sprintf("warning: could not read addon registry file %s: %v", registryFile, err))
				} else {
					if err := os.WriteFile(filepath.Join(registryDir, addon+".yaml"), data, 0644); err != nil {
						return fmt.Errorf("failed to write addon registry file for %s: %w", addon, err)
					}
				}
			}
		}
	}

	// Write manifest.json
	manifestData, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal manifest: %w", err)
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "manifest.json"), manifestData, 0644); err != nil {
		return fmt.Errorf("failed to write manifest: %w", err)
	}

	// Copy install.sh into the bundle
	installScript := findInstallScript()
	if installScript != "" {
		data, err := os.ReadFile(installScript)
		if err != nil {
			return fmt.Errorf("failed to read install.sh: %w", err)
		}
		if err := os.WriteFile(filepath.Join(tmpDir, "install.sh"), data, 0755); err != nil {
			return fmt.Errorf("failed to write install.sh to bundle: %w", err)
		}
	} else {
		utils.LogDebug("warning: install.sh not found, bundle will not include installer")
	}

	// Create the tar.gz archive
	fmt.Printf("  Packaging archive: %s\n", bundleFlags.output)
	if err := createTarGz(bundleFlags.output, tmpDir); err != nil {
		return fmt.Errorf("failed to create archive: %w", err)
	}

	// Print summary
	fi, err := os.Stat(bundleFlags.output)
	if err != nil {
		return fmt.Errorf("failed to stat output archive: %w", err)
	}
	sizeMB := float64(fi.Size()) / (1024 * 1024)
	fmt.Printf("\nBundle created: %s (%.1f MB)\n", bundleFlags.output, sizeMB)
	fmt.Printf("  Version:     %s\n", ver)
	fmt.Printf("  Platform:    %s/%s\n", bundleFlags.platform, bundleFlags.arch)
	fmt.Printf("  Accelerator: %s\n", bundleFlags.accelerator)
	if len(manifest.Addons) > 0 {
		fmt.Printf("  Addons:      %s\n", strings.Join(manifest.Addons, ", "))
	}
	printDeployHint(bundleFlags.platform, filepath.Base(bundleFlags.output))

	return nil
}

// printDeployHint prints platform-appropriate instructions for transferring and installing a bundle.
func printDeployHint(platform, bundleFile string) {
	fmt.Println("\nTo deploy on a remote machine:")
	switch platform {
	case "windows":
		fmt.Printf("  scp %s user@host:C:\\Users\\user\\\n", bundleFile)
		fmt.Printf("  ssh user@host \"tar xzf %s && .\\install.ps1 %s\"\n", bundleFile, bundleFile)
	case "darwin":
		fmt.Printf("  scp %s user@host:~\n", bundleFile)
		fmt.Printf("  ssh user@host 'tar xzf %s && bash install.sh %s'\n", bundleFile, bundleFile)
	default: // linux
		fmt.Printf("  scp %s user@host:~\n", bundleFile)
		fmt.Printf("  ssh user@host 'tar xzf %s && ./install.sh %s'\n", bundleFile, bundleFile)
	}
}

func validateBundleFlags() error {
	if !contains(validPlatforms, bundleFlags.platform) {
		return fmt.Errorf("invalid platform %q; valid options: %s", bundleFlags.platform, strings.Join(validPlatforms, ", "))
	}
	if !contains(validArchitectures, bundleFlags.arch) {
		return fmt.Errorf("invalid arch %q; valid options: %s", bundleFlags.arch, strings.Join(validArchitectures, ", "))
	}
	if !contains(validAccelerators, bundleFlags.accelerator) {
		return fmt.Errorf("invalid accelerator %q; valid options: %s", bundleFlags.accelerator, strings.Join(validAccelerators, ", "))
	}

	combo := bundleFlags.platform + "-" + bundleFlags.arch
	if knownInvalidCombos[combo] {
		return fmt.Errorf("unsupported platform/arch combination: %s/%s", bundleFlags.platform, bundleFlags.arch)
	}

	// Metal is only valid on macOS ARM
	if bundleFlags.accelerator == "metal" && bundleFlags.platform != "darwin" {
		return fmt.Errorf("accelerator 'metal' is only available on macOS (darwin)")
	}

	return nil
}

// downloadReleaseAsset downloads a file from GitHub Releases.
func downloadReleaseAsset(version, assetName, destPath string) error {
	repoOwner := getEnvOrDefault("LF_ADDON_REPO_OWNER", "llama-farm")
	repoName := getEnvOrDefault("LF_ADDON_REPO_NAME", "llamafarm")

	url := fmt.Sprintf("https://github.com/%s/%s/releases/download/%s/%s", repoOwner, repoName, version, assetName)

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Accept", "application/octet-stream")

	// Support authenticated requests for rate limiting
	if token := os.Getenv("GITHUB_TOKEN"); token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("download failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusForbidden {
		return fmt.Errorf("rate limited by GitHub API; set GITHUB_TOKEN environment variable for authenticated access")
	}
	if resp.StatusCode == http.StatusNotFound {
		return fmt.Errorf("release asset not found: %s (version: %s)", assetName, version)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status %d downloading %s", resp.StatusCode, assetName)
	}

	out, err := os.Create(destPath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}

// downloadTorchWheels downloads accelerator-specific torch wheels using pip download.
func downloadTorchWheels(accelerator, platform, arch, destDir string) error {
	// Reuse the PyTorch index URL mappings from hardware_wheels.go
	indexURL := ""
	switch accelerator {
	case "cuda":
		indexURL = "" // Default PyPI has CUDA wheels
	case "rocm":
		indexURL = orchestrator.PyTorchSpec.WheelURLs[orchestrator.HardwareROCm]
	case "vulkan":
		indexURL = "" // Falls back to CPU wheels
	case "metal":
		indexURL = "" // Default PyPI has Metal wheels
	default:
		return fmt.Errorf("unsupported accelerator for torch download: %s", accelerator)
	}

	// Use pip download to fetch platform-specific wheels
	pythonPlatform := getPipPlatformTag(platform, arch)
	args := []string{
		"pip", "download",
		"torch>=2.0.0",
		"--only-binary=:all:",
		"--dest", destDir,
		"--platform", pythonPlatform,
		"--python-version", "3.12",
	}
	if indexURL != "" {
		args = append(args, "--index-url", indexURL)
	}

	cmd := exec.Command("uv", args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// getPipPlatformTag returns the pip platform tag for the given OS/arch.
func getPipPlatformTag(platform, arch string) string {
	switch platform {
	case "linux":
		if arch == "arm64" {
			return "manylinux2014_aarch64"
		}
		return "manylinux2014_x86_64"
	case "darwin":
		return "macosx_14_0_arm64"
	case "windows":
		if arch == "arm64" {
			return "win_arm64"
		}
		return "win_amd64"
	default:
		return "manylinux2014_x86_64"
	}
}

// getAddonPlatformString returns the addon wheel platform string.
func getAddonPlatformString(platform, arch string) string {
	goOS := platformToGoOS[platform]
	switch {
	case goOS == "darwin" && arch == "arm64":
		return "macos-arm64"
	case goOS == "linux" && arch == "x86_64":
		return "linux-x86_64"
	case goOS == "linux" && arch == "arm64":
		return "linux-arm64"
	case goOS == "windows" && arch == "x86_64":
		return "windows-x86_64"
	default:
		return fmt.Sprintf("%s-%s", goOS, arch)
	}
}

// findAddonRegistryFile searches for an addon's registry YAML file.
func findAddonRegistryFile(addonName string) string {
	// Search paths in priority order (same as addon registry lookup)
	searchPaths := []string{}

	if dataDir, err := utils.GetLFDataDir(); err == nil && dataDir != "" {
		searchPaths = append(searchPaths, filepath.Join(dataDir, "src", "addons", "registry"))
	}

	if exe, err := os.Executable(); err == nil {
		searchPaths = append(searchPaths, filepath.Join(filepath.Dir(exe), "..", "addons", "registry"))
	}

	// Development fallback
	if cwd, err := os.Getwd(); err == nil {
		searchPaths = append(searchPaths, filepath.Join(cwd, "..", "addons", "registry"))
		searchPaths = append(searchPaths, filepath.Join(cwd, "addons", "registry"))
	}

	for _, dir := range searchPaths {
		path := filepath.Join(dir, addonName+".yaml")
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	return ""
}

// findInstallScript searches for install.sh.
func findInstallScript() string {
	searchPaths := []string{}

	if exe, err := os.Executable(); err == nil {
		searchPaths = append(searchPaths, filepath.Join(filepath.Dir(exe), "..", "install.sh"))
	}
	if cwd, err := os.Getwd(); err == nil {
		searchPaths = append(searchPaths, filepath.Join(cwd, "install.sh"))
		searchPaths = append(searchPaths, filepath.Join(cwd, "..", "install.sh"))
	}

	for _, path := range searchPaths {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	return ""
}

// createTarGz creates a tar.gz archive from a source directory.
func createTarGz(outputPath, sourceDir string) (retErr error) {
	outFile, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer func() {
		if cerr := outFile.Close(); cerr != nil && retErr == nil {
			retErr = cerr
		}
	}()

	gzWriter := gzip.NewWriter(outFile)
	defer func() {
		if cerr := gzWriter.Close(); cerr != nil && retErr == nil {
			retErr = cerr
		}
	}()

	tw := tar.NewWriter(gzWriter)
	defer func() {
		if cerr := tw.Close(); cerr != nil && retErr == nil {
			retErr = cerr
		}
	}()

	return filepath.Walk(sourceDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Get relative path
		relPath, err := filepath.Rel(sourceDir, path)
		if err != nil {
			return err
		}
		if relPath == "." {
			return nil
		}

		header, err := tar.FileInfoHeader(info, "")
		if err != nil {
			return err
		}
		header.Name = filepath.ToSlash(relPath)

		if err := tw.WriteHeader(header); err != nil {
			return err
		}

		if info.IsDir() {
			return nil
		}

		f, err := os.Open(path)
		if err != nil {
			return err
		}
		defer f.Close()

		_, err = io.Copy(tw, f)
		return err
	})
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func getEnvOrDefault(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}

