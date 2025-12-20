package orchestrator

import "testing"

func TestShouldBuildFromSourceEnvOverride(t *testing.T) {
	t.Setenv("LLAMAFARM_LLAMA_SOURCE_BUILD", "1")
	if !shouldBuildFromSource() {
		t.Fatal("expected shouldBuildFromSource to be true when override is set")
	}

	t.Setenv("LLAMAFARM_LLAMA_SOURCE_BUILD", "0")
	if shouldBuildFromSource() {
		t.Fatal("expected shouldBuildFromSource to be false when override is disabled")
	}
}

func TestInstallLlamaBinaryUsesSourceBuild(t *testing.T) {
	t.Setenv("LLAMAFARM_LLAMA_SOURCE_BUILD", "1")

	original := buildFromSource
	called := false
	buildFromSource = func(destDir string, hardware HardwareCapability) error {
		called = true
		return nil
	}
	t.Cleanup(func() {
		buildFromSource = original
	})

	if err := InstallLlamaBinary(t.TempDir()); err != nil {
		t.Fatalf("InstallLlamaBinary failed: %v", err)
	}
	if !called {
		t.Fatal("expected buildFromSource to be called")
	}
}
