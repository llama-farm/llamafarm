package orchestrator

import (
	"fmt"
	"os/exec"
	"strings"

	"github.com/llamafarm/cli/cmd/utils"
)

// HardwarePackageSpec defines a Python package that requires hardware-specific installation
type HardwarePackageSpec struct {
	Name              string                        // Package name (e.g., "torch", "llama-cpp-python")
	Version           string                        // Version constraint (e.g., ">=2.0.0")
	UseIndexURL       bool                          // If true, use --index-url; if false, use --extra-index-url
	WheelURLs         map[HardwareCapability]string // Hardware-specific wheel index URLs
	FallbackToDefault bool                          // If true and URL is empty, use default PyPI
}

// PyTorchSpec defines the hardware-specific installation for PyTorch
var PyTorchSpec = HardwarePackageSpec{
	Name:              "torch",
	Version:           ">=2.0.0",
	UseIndexURL:       true, // PyTorch uses --index-url for hardware-specific builds
	FallbackToDefault: true,
	WheelURLs: map[HardwareCapability]string{
		HardwareCPU:   "https://download.pytorch.org/whl/cpu",
		HardwareCUDA:  "", // Empty = use default PyPI (has CUDA builds)
		HardwareMetal: "", // Empty = use default PyPI (has Metal support)
		HardwareROCm:  "https://download.pytorch.org/whl/rocm6.4",
	},
}

// LlamaCppSpec defines the hardware-specific installation for llama-cpp-python
var LlamaCppSpec = HardwarePackageSpec{
	Name:              "llama-cpp-python",
	Version:           ">=0.3.0",
	UseIndexURL:       false, // llama-cpp-python uses --extra-index-url
	FallbackToDefault: true,
	WheelURLs: map[HardwareCapability]string{
		HardwareCPU:   "https://abetlen.github.io/llama-cpp-python/whl/cpu",
		HardwareCUDA:  "https://abetlen.github.io/llama-cpp-python/whl/cu121",
		HardwareMetal: "https://abetlen.github.io/llama-cpp-python/whl/metal",
		HardwareROCm:  "https://abetlen.github.io/llama-cpp-python/whl/rocm",
	},
}

// GetComponentPackages returns the hardware-dependent packages for a given component
// It looks up the component in ServiceGraph and returns its HardwarePackages field
// Returns an empty slice if the component doesn't exist or has no hardware-specific packages
func GetComponentPackages(componentName string) []HardwarePackageSpec {
	if svc, exists := ServiceGraph[componentName]; exists {
		return svc.HardwarePackages
	}
	return []HardwarePackageSpec{} // Component not found or no hardware packages
}

// InstallHardwarePackages installs hardware-specific Python packages using uv pip install
// This function detects hardware and installs the appropriate wheel versions for each package
func InstallHardwarePackages(pythonEnvMgr *PythonEnvManager, workDir string, packages []HardwarePackageSpec) error {
	// Detect hardware capabilities
	hardware := DetectHardware()
	utils.LogDebug(fmt.Sprintf("Installing hardware-specific packages for %s", hardware))

	uvPath := pythonEnvMgr.uvManager.GetUVPath()

	for _, pkg := range packages {
		// Get the wheel URL for this hardware
		wheelURL, ok := pkg.WheelURLs[hardware]
		if !ok {
			// Hardware not in map - skip or use CPU fallback
			if cpuURL, hasCPU := pkg.WheelURLs[HardwareCPU]; hasCPU && pkg.FallbackToDefault {
				wheelURL = cpuURL
				utils.LogDebug(fmt.Sprintf("Hardware %s not found for %s, using CPU fallback", hardware, pkg.Name))
			} else {
				return fmt.Errorf("no wheel URL found for package %s on hardware %s", pkg.Name, hardware)
			}
		}

		// Build the uv pip install command
		args := []string{"pip", "install"}

		// Add the package with version constraint
		packageSpec := pkg.Name
		if pkg.Version != "" {
			packageSpec = fmt.Sprintf("%s%s", pkg.Name, pkg.Version)
		}

		// Handle index URL parameter
		if wheelURL != "" {
			if pkg.UseIndexURL {
				// Use --index-url (replaces default PyPI)
				args = append(args, "--index-url", wheelURL)
				utils.LogDebug(fmt.Sprintf("Installing %s from index: %s", pkg.Name, wheelURL))
			} else {
				// Use --extra-index-url (supplements default PyPI)
				args = append(args, "--extra-index-url", wheelURL)
				utils.LogDebug(fmt.Sprintf("Installing %s with extra index: %s", pkg.Name, wheelURL))
			}
		} else if pkg.FallbackToDefault {
			// Empty URL = use default PyPI
			utils.LogDebug(fmt.Sprintf("Installing %s from default PyPI (GPU-accelerated)", pkg.Name))
		} else {
			return fmt.Errorf("no wheel URL available for %s on %s and fallback disabled", pkg.Name, hardware)
		}

		args = append(args, packageSpec)

		// Execute the command
		cmd := exec.Command(uvPath, args...)
		cmd.Dir = workDir
		cmd.Env = pythonEnvMgr.GetEnvForProcess()

		utils.LogDebug(fmt.Sprintf("Running: uv %s", strings.Join(args, " ")))

		output, err := cmd.CombinedOutput()
		if err != nil {
			return fmt.Errorf("failed to install %s: %w\nOutput: %s", pkg.Name, err, string(output))
		}

		utils.LogDebug(fmt.Sprintf("Successfully installed %s\nOutput: %s", pkg.Name, string(output)))
	}

	return nil
}
