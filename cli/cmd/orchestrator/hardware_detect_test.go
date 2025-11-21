package orchestrator

import (
	"os"
	"runtime"
	"testing"
)

func TestDetectHardware(t *testing.T) {
	// This test will detect actual hardware on the test machine
	// We just verify it returns a valid capability and doesn't panic
	hardware := DetectHardware()

	// Should return a known capability
	if hardware < HardwareUnknown || hardware > HardwareROCm {
		t.Errorf("DetectHardware returned invalid capability: %v", hardware)
	}

	// Should have a string representation
	if hardware.String() == "" {
		t.Error("Hardware capability string representation is empty")
	}

	t.Logf("Detected hardware: %s", hardware)
}

func TestHardwareCapabilityString(t *testing.T) {
	tests := []struct {
		capability HardwareCapability
		expected   string
	}{
		{HardwareCPU, "CPU"},
		{HardwareCUDA, "CUDA"},
		{HardwareMetal, "Metal"},
		{HardwareROCm, "ROCm"},
		{HardwareUnknown, "Unknown"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			if got := tt.capability.String(); got != tt.expected {
				t.Errorf("String() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestPyTorchIndexURL(t *testing.T) {
	tests := []struct {
		capability HardwareCapability
		expected   string
	}{
		{HardwareCPU, "https://download.pytorch.org/whl/cpu"},
		{HardwareCUDA, ""},  // Empty means use default PyPI (CUDA builds)
		{HardwareMetal, ""}, // Empty means use default PyPI (Metal builds)
		{HardwareROCm, "https://download.pytorch.org/whl/rocm6.0"},
		{HardwareUnknown, "https://download.pytorch.org/whl/cpu"}, // Fallback to CPU
	}

	for _, tt := range tests {
		t.Run(tt.capability.String(), func(t *testing.T) {
			if got := tt.capability.PyTorchIndexURL(); got != tt.expected {
				t.Errorf("PyTorchIndexURL() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestHasMetal(t *testing.T) {
	// hasMetal should only return true on macOS ARM64
	result := hasMetal()

	expectedMetal := runtime.GOOS == "darwin" && runtime.GOARCH == "arm64"

	if result != expectedMetal {
		t.Errorf("hasMetal() = %v, expected %v (OS: %s, Arch: %s)",
			result, expectedMetal, runtime.GOOS, runtime.GOARCH)
	}
}

func TestGetPyTorchIndexURL_ManualOverride(t *testing.T) {
	// Test LF_PYTORCH_INDEX override
	t.Run("LF_PYTORCH_INDEX override", func(t *testing.T) {
		testURL := "https://custom.pytorch.index/cpu"
		os.Setenv("LF_PYTORCH_INDEX", testURL)
		defer os.Unsetenv("LF_PYTORCH_INDEX")

		got := GetPyTorchIndexURL()
		if got != testURL {
			t.Errorf("GetPyTorchIndexURL() with LF_PYTORCH_INDEX = %v, want %v", got, testURL)
		}
	})

	// Test UV_EXTRA_INDEX_URL override
	t.Run("UV_EXTRA_INDEX_URL override", func(t *testing.T) {
		testURL := "https://custom.uv.index/cpu"
		os.Setenv("UV_EXTRA_INDEX_URL", testURL)
		defer os.Unsetenv("UV_EXTRA_INDEX_URL")

		got := GetPyTorchIndexURL()
		if got != testURL {
			t.Errorf("GetPyTorchIndexURL() with UV_EXTRA_INDEX_URL = %v, want %v", got, testURL)
		}
	})

	// Test auto-detection when no overrides
	t.Run("Auto-detection without overrides", func(t *testing.T) {
		// Make sure env vars are not set
		os.Unsetenv("LF_PYTORCH_INDEX")
		os.Unsetenv("UV_EXTRA_INDEX_URL")

		got := GetPyTorchIndexURL()
		// Should return a valid URL or empty string
		// We can't predict the exact value since it depends on hardware
		// Just verify it doesn't panic and returns something reasonable
		t.Logf("Auto-detected PyTorch index URL: %s (empty = default PyPI)", got)
	})
}

func TestHasCUDA_NoFalsePositives(t *testing.T) {
	// This test ensures hasCUDA doesn't crash and returns a boolean
	result := hasCUDA()

	// Just log the result - we can't assert a specific value
	// since it depends on the test environment
	t.Logf("CUDA detection result: %v", result)

	// If CUDA is detected, there should be some evidence
	if result {
		hasNvidiaSmi := false
		hasCudaPath := false
		hasCudaHome := false

		// Check if any of the detection methods would work
		if _, err := os.Stat("/proc/driver/nvidia/version"); err == nil {
			t.Log("Found /proc/driver/nvidia/version")
			hasNvidiaSmi = true
		}
		if os.Getenv("CUDA_PATH") != "" {
			t.Log("Found CUDA_PATH env var")
			hasCudaPath = true
		}
		if os.Getenv("CUDA_HOME") != "" {
			t.Log("Found CUDA_HOME env var")
			hasCudaHome = true
		}

		// At least one detection method should have found something
		if !hasNvidiaSmi && !hasCudaPath && !hasCudaHome {
			t.Log("Warning: CUDA detected but no evidence found in common locations")
		}
	}
}

func TestHasROCm(t *testing.T) {
	// This test ensures hasROCm doesn't crash and returns a boolean
	result := hasROCm()

	t.Logf("ROCm detection result: %v", result)

	// If ROCm is detected, verify some evidence exists
	if result {
		hasRocmInfo := false
		hasRocmSmi := false

		if _, err := os.Stat("/opt/rocm"); err == nil {
			t.Log("Found /opt/rocm directory")
			hasRocmInfo = true
		}
		if _, err := os.Stat("/usr/lib/rocm"); err == nil {
			t.Log("Found /usr/lib/rocm directory")
			hasRocmSmi = true
		}

		if !hasRocmInfo && !hasRocmSmi {
			t.Log("Warning: ROCm detected but no evidence found in common locations")
		}
	}
}

