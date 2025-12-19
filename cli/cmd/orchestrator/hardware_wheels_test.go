package orchestrator

import (
	"strings"
	"testing"
)

func TestPyTorchSpec(t *testing.T) {
	tests := []struct {
		name     string
		hardware HardwareCapability
		wantURL  string
	}{
		{
			name:     "CPU hardware uses CPU PyTorch index",
			hardware: HardwareCPU,
			wantURL:  "https://download.pytorch.org/whl/cpu",
		},
		{
			name:     "CUDA hardware uses default PyPI (empty URL)",
			hardware: HardwareCUDA,
			wantURL:  "",
		},
		{
			name:     "Metal hardware uses default PyPI (empty URL)",
			hardware: HardwareMetal,
			wantURL:  "",
		},
		{
			name:     "ROCm hardware uses ROCm PyTorch index",
			hardware: HardwareROCm,
			wantURL:  "https://download.pytorch.org/whl/rocm6.4",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotURL, ok := PyTorchSpec.WheelURLs[tt.hardware]
			if !ok {
				t.Errorf("PyTorchSpec.WheelURLs[%s] not found", tt.hardware)
				return
			}
			if gotURL != tt.wantURL {
				t.Errorf("PyTorchSpec.WheelURLs[%s] = %v, want %v", tt.hardware, gotURL, tt.wantURL)
			}
		})
	}

	// Verify package properties
	if PyTorchSpec.Name != "torch" {
		t.Errorf("PyTorchSpec.Name = %v, want torch", PyTorchSpec.Name)
	}
	if !PyTorchSpec.UseIndexURL {
		t.Error("PyTorchSpec.UseIndexURL should be true (uses --index-url)")
	}
	if !PyTorchSpec.FallbackToDefault {
		t.Error("PyTorchSpec.FallbackToDefault should be true")
	}
}

// Note: LlamaFarmLlamaSpec was removed - llama.cpp binaries are now downloaded
// via EnsureLlamaBinary in llama_binary.go, not via pip HardwarePackageSpec.

func TestGetComponentPackages(t *testing.T) {
	t.Run("universal-runtime has hardware packages", func(t *testing.T) {
		packages := GetComponentPackages("universal-runtime")
		if len(packages) == 0 {
			t.Error("universal-runtime should have hardware-specific packages")
		}
		// Only PyTorch is installed via HardwarePackages
		// llama.cpp binaries are downloaded separately via EnsureLlamaBinary
		if len(packages) != 1 {
			t.Errorf("universal-runtime should have 1 package (PyTorch), got %d", len(packages))
		}

		// Verify we have PyTorch
		var hasTorch bool
		for _, pkg := range packages {
			if pkg.Name == "torch" {
				hasTorch = true
			}
		}

		if !hasTorch {
			t.Error("universal-runtime packages missing torch")
		}
	})

	t.Run("unknown component returns empty slice", func(t *testing.T) {
		packages := GetComponentPackages("nonexistent-component")
		if len(packages) != 0 {
			t.Errorf("nonexistent-component should return empty slice, got %d packages", len(packages))
		}
	})

	t.Run("server has no hardware packages (no-op)", func(t *testing.T) {
		packages := GetComponentPackages("server")
		if len(packages) != 0 {
			t.Errorf("server should have no hardware packages, got %d", len(packages))
		}
	})

	t.Run("rag has no hardware packages (no-op)", func(t *testing.T) {
		packages := GetComponentPackages("rag")
		if len(packages) != 0 {
			t.Errorf("rag should have no hardware packages, got %d", len(packages))
		}
	})

	t.Run("config component not in ServiceGraph", func(t *testing.T) {
		// config is not a service, so should return empty
		packages := GetComponentPackages("config")
		if len(packages) != 0 {
			t.Errorf("config should return empty slice, got %d", len(packages))
		}
	})
}

func TestServiceGraphIntegration(t *testing.T) {
	t.Run("ServiceGraph contains universal-runtime", func(t *testing.T) {
		svc, exists := ServiceGraph["universal-runtime"]
		if !exists {
			t.Fatal("universal-runtime not found in ServiceGraph")
		}

		// Only PyTorch is in HardwarePackages; llama.cpp binaries are handled separately
		if len(svc.HardwarePackages) != 1 {
			t.Errorf("universal-runtime.HardwarePackages should have 1 item (PyTorch), got %d", len(svc.HardwarePackages))
		}
	})

	t.Run("ServiceGraph server has no hardware packages", func(t *testing.T) {
		svc, exists := ServiceGraph["server"]
		if !exists {
			t.Fatal("server not found in ServiceGraph")
		}

		if len(svc.HardwarePackages) != 0 {
			t.Errorf("server.HardwarePackages should be empty, got %d items", len(svc.HardwarePackages))
		}
	})

	t.Run("ServiceGraph rag has no hardware packages", func(t *testing.T) {
		svc, exists := ServiceGraph["rag"]
		if !exists {
			t.Fatal("rag not found in ServiceGraph")
		}

		if len(svc.HardwarePackages) != 0 {
			t.Errorf("rag.HardwarePackages should be empty, got %d items", len(svc.HardwarePackages))
		}
	})
}

func TestHardwarePackageSpec_AllHardwareSupported(t *testing.T) {
	// Ensure all hardware types are covered by our specs
	allHardware := []HardwareCapability{
		HardwareCPU,
		HardwareCUDA,
		HardwareMetal,
		HardwareROCm,
	}

	t.Run("PyTorch supports all hardware types", func(t *testing.T) {
		for _, hw := range allHardware {
			if _, ok := PyTorchSpec.WheelURLs[hw]; !ok {
				t.Errorf("PyTorchSpec missing hardware type: %s", hw)
			}
		}
	})

	// Note: llama.cpp binaries are handled via EnsureLlamaBinary, not HardwarePackageSpec
}

func TestHardwarePackageSpec_URLValidation(t *testing.T) {
	// Verify all URLs follow expected patterns
	t.Run("PyTorch URLs", func(t *testing.T) {
		for hw, url := range PyTorchSpec.WheelURLs {
			if url != "" {
				// Non-empty URLs should be from PyTorch download site
				expectedPrefix := "https://download.pytorch.org/whl/"
				if !strings.HasPrefix(url, expectedPrefix) {
					t.Errorf("PyTorchSpec[%s] URL doesn't start with %s: %s", hw, expectedPrefix, url)
				}
			}
		}
	})

	// Note: llama.cpp binaries are handled via EnsureLlamaBinary, not HardwarePackageSpec
}

func TestHardwarePackageSpec_VersionConstraints(t *testing.T) {
	// Verify version constraints are set for PyTorch
	if PyTorchSpec.Version == "" {
		t.Error("PyTorchSpec.Version should not be empty")
	}

	// Verify PyTorch uses >= constraints (allowing newer versions)
	if !strings.HasPrefix(PyTorchSpec.Version, ">=") {
		t.Errorf("PyTorchSpec.Version should start with >=, got: %s", PyTorchSpec.Version)
	}

	// Note: llama.cpp binaries are handled via EnsureLlamaBinary, not HardwarePackageSpec
}
