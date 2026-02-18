package cmd

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNormalizePackageName(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"numpy", "numpy"},
		{"faster-whisper", "faster_whisper"},
		{"PyYAML", "pyyaml"},
		{"Pillow", "pillow"},
		{"typing-extensions", "typing_extensions"},
		{"huggingface-hub", "huggingface_hub"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := normalizePackageName(tt.input)
			if got != tt.want {
				t.Errorf("normalizePackageName(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestExtractAddonPackageNames(t *testing.T) {
	addon := &AddonDefinition{
		Packages: []string{
			"faster-whisper>=1.0.0,<2.0.0",
			"av>=12.0.0,<13.0.0",
			"https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
		},
	}

	result := extractAddonPackageNames(addon)

	expected := map[string]bool{
		"faster_whisper": true,
		"av":             true,
		"en_core_web_sm": true,
	}

	for name := range expected {
		if !result[name] {
			t.Errorf("expected %q in result", name)
		}
	}

	if len(result) != len(expected) {
		t.Errorf("got %d packages, want %d", len(result), len(expected))
	}
}

func TestGetVenvPackageNames_WithMockVenv(t *testing.T) {
	// Create a mock site-packages directory with dist-info dirs
	tempDir := t.TempDir()
	sitePackages := filepath.Join(tempDir, "lib", "python3.12", "site-packages")
	if err := os.MkdirAll(sitePackages, 0755); err != nil {
		t.Fatalf("failed to create site-packages dir: %v", err)
	}

	// Create mock dist-info directories
	mockPackages := []string{
		"numpy-2.4.2.dist-info",
		"torch-2.5.0.dist-info",
		"transformers-4.45.0.dist-info",
		"huggingface_hub-0.26.0.dist-info",
		"typing_extensions-4.12.0.dist-info",
		"PyYAML-6.0.2.dist-info",
	}
	for _, pkg := range mockPackages {
		if err := os.MkdirAll(filepath.Join(sitePackages, pkg), 0755); err != nil {
			t.Fatalf("failed to create %s: %v", pkg, err)
		}
	}

	// Test scanning the mock site-packages
	entries, err := os.ReadDir(sitePackages)
	if err != nil {
		t.Fatalf("failed to read site-packages: %v", err)
	}

	// Manually run the same logic as getVenvPackageNames since we can't
	// point it at a custom directory (it uses GetLFDataDir)
	packages := make(map[string]bool)
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		name := entry.Name()
		if len(name) < 10 { // too short to be .dist-info
			continue
		}
		suffix := ".dist-info"
		if name[len(name)-len(suffix):] != suffix {
			continue
		}
		baseName := name[:len(name)-len(suffix)]
		idx := 0
		for i, c := range baseName {
			if c == '-' {
				idx = i
				break
			}
		}
		if idx == 0 {
			idx = len(baseName)
		}
		pkgName := normalizePackageName(baseName[:idx])
		packages[pkgName] = true
	}

	expectedPackages := []string{"numpy", "torch", "transformers", "huggingface_hub", "typing_extensions", "pyyaml"}
	for _, pkg := range expectedPackages {
		if !packages[pkg] {
			t.Errorf("expected %q to be in venv packages", pkg)
		}
	}
}
