package orchestrator

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"

	"github.com/llamafarm/cli/cmd/utils"
)

// HardwareCapability represents the compute hardware available on the system
type HardwareCapability int

const (
	HardwareUnknown HardwareCapability = iota
	HardwareCPU                        // CPU-only (fallback)
	HardwareCUDA                       // NVIDIA CUDA
	HardwareMetal                      // Apple Metal (M1/M2/M3)
	HardwareROCm                       // AMD ROCm
)

func (h HardwareCapability) String() string {
	switch h {
	case HardwareCPU:
		return "CPU"
	case HardwareCUDA:
		return "CUDA"
	case HardwareMetal:
		return "Metal"
	case HardwareROCm:
		return "ROCm"
	default:
		return "Unknown"
	}
}

// PyTorchIndexURL returns the appropriate PyTorch index URL for this hardware
func (h HardwareCapability) PyTorchIndexURL() string {
	switch h {
	case HardwareCPU:
		return "https://download.pytorch.org/whl/cpu"
	case HardwareCUDA:
		// Return empty string to use default PyPI (which has CUDA builds)
		// Alternatively, could specify exact CUDA version like cu121
		return ""
	case HardwareMetal:
		// macOS ARM64 builds are on default PyPI
		return ""
	case HardwareROCm:
		// ROCm builds (for AMD GPUs)
		return "https://download.pytorch.org/whl/rocm6.0"
	default:
		// Unknown - fall back to CPU for safety
		return "https://download.pytorch.org/whl/cpu"
	}
}

// DetectHardware detects the compute hardware capabilities of the system
func DetectHardware() HardwareCapability {
	utils.LogDebug("Detecting hardware capabilities...")

	// Check for NVIDIA CUDA (highest priority if available)
	if hasCUDA() {
		utils.LogDebug("Detected NVIDIA CUDA support")
		return HardwareCUDA
	}

	// Check for Apple Metal (macOS ARM64)
	if hasMetal() {
		utils.LogDebug("Detected Apple Metal support (Apple Silicon)")
		return HardwareMetal
	}

	// Check for AMD ROCm
	if hasROCm() {
		utils.LogDebug("Detected AMD ROCm support")
		return HardwareROCm
	}

	// Fallback to CPU
	utils.LogDebug("No GPU acceleration detected, using CPU-only mode")
	return HardwareCPU
}

// hasCUDA checks if NVIDIA CUDA is available
func hasCUDA() bool {
	// Method 1: Check for nvidia-smi command
	if _, err := exec.LookPath("nvidia-smi"); err == nil {
		cmd := exec.Command("nvidia-smi")
		if err := cmd.Run(); err == nil {
			utils.LogDebug("CUDA detected via nvidia-smi")
			return true
		}
	}

	// Method 2: Check for CUDA environment variables
	if cudaPath := os.Getenv("CUDA_PATH"); cudaPath != "" {
		utils.LogDebug("CUDA detected via CUDA_PATH environment variable")
		return true
	}

	if cudaHome := os.Getenv("CUDA_HOME"); cudaHome != "" {
		utils.LogDebug("CUDA detected via CUDA_HOME environment variable")
		return true
	}

	// Method 3: Check for NVIDIA kernel driver (Linux)
	if runtime.GOOS == "linux" {
		if _, err := os.Stat("/proc/driver/nvidia/version"); err == nil {
			utils.LogDebug("CUDA detected via /proc/driver/nvidia/version")
			return true
		}
	}

	// Method 4: Check common CUDA library locations
	cudaPaths := []string{
		"/usr/local/cuda",
		"/usr/lib/cuda",
		"/opt/cuda",
	}
	for _, path := range cudaPaths {
		if _, err := os.Stat(path); err == nil {
			utils.LogDebug(fmt.Sprintf("CUDA detected at %s", path))
			return true
		}
	}

	return false
}

// hasMetal checks if Apple Metal is available (macOS ARM64)
func hasMetal() bool {
	// Metal is available on macOS ARM64 (M1/M2/M3/M4)
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		utils.LogDebug("Metal available on macOS ARM64")
		return true
	}

	// Intel Macs also have Metal, but PyTorch CPU builds are often better
	// We could enable this, but for now stick with ARM64 only
	// if runtime.GOOS == "darwin" {
	// 	return true
	// }

	return false
}

// hasROCm checks if AMD ROCm is available
func hasROCm() bool {
	// Check for rocminfo command
	if _, err := exec.LookPath("rocminfo"); err == nil {
		cmd := exec.Command("rocminfo")
		if err := cmd.Run(); err == nil {
			utils.LogDebug("ROCm detected via rocminfo")
			return true
		}
	}

	// Check for rocm-smi command
	if _, err := exec.LookPath("rocm-smi"); err == nil {
		cmd := exec.Command("rocm-smi")
		if err := cmd.Run(); err == nil {
			utils.LogDebug("ROCm detected via rocm-smi")
			return true
		}
	}

	// Check for ROCm installation paths
	rocmPaths := []string{
		"/opt/rocm",
		"/usr/lib/rocm",
	}
	for _, path := range rocmPaths {
		if _, err := os.Stat(path); err == nil {
			utils.LogDebug(fmt.Sprintf("ROCm detected at %s", path))
			return true
		}
	}

	return false
}

// GetPyTorchIndexURL returns the appropriate PyTorch index URL for the current hardware
// This can be used to set UV_EXTRA_INDEX_URL when installing dependencies
func GetPyTorchIndexURL() string {
	// Allow manual override via environment variable
	if indexURL := os.Getenv("LF_PYTORCH_INDEX"); indexURL != "" {
		utils.LogDebug(fmt.Sprintf("Using manually specified PyTorch index: %s", indexURL))
		return indexURL
	}

	// Also respect UV_EXTRA_INDEX_URL if already set (for CI or manual control)
	if indexURL := os.Getenv("UV_EXTRA_INDEX_URL"); indexURL != "" {
		utils.LogDebug(fmt.Sprintf("UV_EXTRA_INDEX_URL already set: %s", indexURL))
		return indexURL
	}

	// Auto-detect hardware
	hardware := DetectHardware()
	indexURL := hardware.PyTorchIndexURL()

	if indexURL == "" {
		utils.LogDebug(fmt.Sprintf("Using default PyPI for %s (GPU-accelerated)", hardware))
	} else {
		utils.LogDebug(fmt.Sprintf("Using %s builds for detected %s hardware",
			strings.TrimPrefix(strings.TrimPrefix(indexURL, "https://download.pytorch.org/whl/"), "/"),
			hardware))
	}

	return indexURL
}

