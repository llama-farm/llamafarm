package llamabinary

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSpecFor_MacOSArm64Metal(t *testing.T) {
	spec, err := SpecFor(Target{OS: "darwin", Arch: "arm64", Accelerator: "metal"}, "b7694")
	if err != nil {
		t.Fatalf("SpecFor: %v", err)
	}
	if !strings.Contains(spec.URL, "bin-macos-arm64.tar.gz") {
		t.Errorf("unexpected URL: %s", spec.URL)
	}
	if spec.LibName != "libllama.dylib" {
		t.Errorf("expected libllama.dylib, got %s", spec.LibName)
	}
}

func TestSpecFor_LinuxAmd64CPU(t *testing.T) {
	spec, err := SpecFor(Target{OS: "linux", Arch: "amd64", Accelerator: "cpu"}, "b7800")
	if err != nil {
		t.Fatalf("SpecFor: %v", err)
	}
	if !strings.Contains(spec.URL, "b7800") {
		t.Errorf("expected version in URL, got %s", spec.URL)
	}
	if !strings.Contains(spec.URL, "bin-ubuntu-x64.tar.gz") {
		t.Errorf("unexpected URL: %s", spec.URL)
	}
	if spec.LibName != "libllama.so" {
		t.Errorf("expected libllama.so, got %s", spec.LibName)
	}
}

func TestSpecFor_LinuxArm64CPUUsesLlamaFarmHost(t *testing.T) {
	spec, err := SpecFor(Target{OS: "linux", Arch: "arm64", Accelerator: "cpu"}, "b7694")
	if err != nil {
		t.Fatalf("SpecFor: %v", err)
	}
	if !strings.Contains(spec.URL, "llama-farm/llamafarm") {
		t.Errorf("expected llama-farm host, got %s", spec.URL)
	}
	if !strings.Contains(spec.URL, "bin-linux-arm64.tar.gz") {
		t.Errorf("unexpected artifact in URL: %s", spec.URL)
	}
}

func TestSpecFor_WindowsAmd64CPU(t *testing.T) {
	spec, err := SpecFor(Target{OS: "windows", Arch: "amd64", Accelerator: "cpu"}, "b7694")
	if err != nil {
		t.Fatalf("SpecFor: %v", err)
	}
	if !strings.HasSuffix(spec.URL, ".zip") {
		t.Errorf("expected .zip URL, got %s", spec.URL)
	}
	if spec.LibName != "llama.dll" {
		t.Errorf("expected llama.dll, got %s", spec.LibName)
	}
}

func TestSpecFor_WindowsAmd64Cuda(t *testing.T) {
	spec, err := SpecFor(Target{OS: "windows", Arch: "amd64", Accelerator: "cuda"}, "b7694")
	if err != nil {
		t.Fatalf("SpecFor: %v", err)
	}
	if !strings.Contains(spec.URL, "cuda") {
		t.Errorf("expected cuda in URL, got %s", spec.URL)
	}
}

func TestSpecFor_InvalidCombo(t *testing.T) {
	_, err := SpecFor(Target{OS: "darwin", Arch: "arm64", Accelerator: "cuda"}, "b7694")
	if err == nil {
		t.Error("expected error for darwin/arm64/cuda")
	}
}

func TestValidate_Valid(t *testing.T) {
	t.Setenv("LLAMAFARM_CACHE_DIR", "") // unrelated cleanliness
	if err := (Target{OS: "linux", Arch: "amd64", Accelerator: "cpu"}).Validate(); err != nil {
		t.Errorf("expected valid, got %v", err)
	}
}

func TestValidate_InvalidOS(t *testing.T) {
	if err := (Target{OS: "bsd", Arch: "amd64", Accelerator: "cpu"}).Validate(); err == nil {
		t.Error("expected error for invalid OS")
	}
}

func TestValidate_InvalidArch(t *testing.T) {
	if err := (Target{OS: "linux", Arch: "riscv64", Accelerator: "cpu"}).Validate(); err == nil {
		t.Error("expected error for invalid arch")
	}
}

func TestValidate_InvalidAccelerator(t *testing.T) {
	if err := (Target{OS: "linux", Arch: "amd64", Accelerator: "fpga"}).Validate(); err == nil {
		t.Error("expected error for invalid accelerator")
	}
}

func TestCanonicalizeArch(t *testing.T) {
	cases := map[string]string{
		"x86_64":  "amd64",
		"amd64":   "amd64",
		"arm64":   "arm64",
		"aarch64": "arm64",
	}
	for input, want := range cases {
		got, ok := CanonicalizeArch(input)
		if !ok {
			t.Errorf("%s should canonicalize", input)
		}
		if got != want {
			t.Errorf("%s → %s, want %s", input, got, want)
		}
	}

	if _, ok := CanonicalizeArch("mips"); ok {
		t.Error("mips should not canonicalize")
	}
}

func TestBestAcceleratorFor(t *testing.T) {
	if BestAcceleratorFor("darwin", "arm64") != "metal" {
		t.Error("expected metal for darwin/arm64")
	}
	if BestAcceleratorFor("linux", "amd64") != "cpu" {
		t.Error("expected cpu default for linux/amd64")
	}
	if BestAcceleratorFor("windows", "amd64") != "cpu" {
		t.Error("expected cpu default for windows/amd64")
	}
}

func TestLibNameFor(t *testing.T) {
	if LibNameFor("darwin") != "libllama.dylib" {
		t.Error()
	}
	if LibNameFor("windows") != "llama.dll" {
		t.Error()
	}
	if LibNameFor("linux") != "libllama.so" {
		t.Error()
	}
}

func TestCacheRoot_HonorsEnvOverride(t *testing.T) {
	t.Setenv("LLAMAFARM_CACHE_DIR", "/custom/cache")
	got, err := CacheRoot()
	if err != nil {
		t.Fatal(err)
	}
	if got != "/custom/cache" {
		t.Errorf("got %q, want /custom/cache", got)
	}
}

func TestCacheDir_CrossPlatformIsScoped(t *testing.T) {
	t.Setenv("LLAMAFARM_CACHE_DIR", "/root")
	host := CurrentHostTarget()
	hostDir, err := CacheDir(host, "b7694")
	if err != nil {
		t.Fatal(err)
	}
	if hostDir != filepath.Join("/root", "b7694") {
		t.Errorf("host dir not at legacy path: %s", hostDir)
	}

	// Pick a target that is guaranteed to differ from the host.
	cross := Target{OS: "linux", Arch: "arm64", Accelerator: "cpu"}
	if cross == host {
		// Change accelerator if host happens to match.
		cross.Accelerator = "vulkan"
	}
	crossDir, err := CacheDir(cross, "b7694")
	if err != nil {
		t.Fatal(err)
	}
	if crossDir == hostDir {
		t.Error("cross target must not share host cache dir")
	}
	if !strings.Contains(crossDir, cross.Slug()) {
		t.Errorf("cross dir should contain slug %s, got %s", cross.Slug(), crossDir)
	}
}

func TestIsCached_FalseWhenMissing(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("LLAMAFARM_CACHE_DIR", tmp)
	if IsCached(Target{OS: "linux", Arch: "arm64", Accelerator: "cpu"}, "bXXXX") {
		t.Error("expected not cached in empty dir")
	}
}

func TestIsCached_TrueWhenFilePresent(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("LLAMAFARM_CACHE_DIR", tmp)
	target := Target{OS: "linux", Arch: "arm64", Accelerator: "cpu"}
	dir, err := CacheDir(target, "b7694")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	libFile := filepath.Join(dir, "libllama.so")
	// Write >0 bytes so IsCached passes.
	if err := os.WriteFile(libFile, []byte("stub"), 0o755); err != nil {
		t.Fatal(err)
	}
	if !IsCached(target, "b7694") {
		t.Error("expected cached after writing file")
	}
}

func TestLibPath_ReturnsExpectedFile(t *testing.T) {
	t.Setenv("LLAMAFARM_CACHE_DIR", "/r")
	target := Target{OS: "darwin", Arch: "arm64", Accelerator: "metal"}
	p, err := LibPath(target, "b7694")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(p, "libllama.dylib") {
		t.Errorf("got %s", p)
	}
}

func TestExport_RequiresCache(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("LLAMAFARM_CACHE_DIR", filepath.Join(tmp, "cache"))
	exportDir := filepath.Join(tmp, "export")

	target := Target{OS: "linux", Arch: "arm64", Accelerator: "cpu"}
	if err := Export(target, "b7694", exportDir); err == nil {
		t.Error("expected error when cache empty")
	}
}

func TestExport_CopiesCachedFilesFlat(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("LLAMAFARM_CACHE_DIR", filepath.Join(tmp, "cache"))

	target := Target{OS: "linux", Arch: "arm64", Accelerator: "cpu"}
	dir, err := CacheDir(target, "b7694")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	files := map[string]string{
		"libllama.so":    "main",
		"libggml.so":     "gg",
		"libggml-cpu.so": "gc",
	}
	for name, content := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o755); err != nil {
			t.Fatal(err)
		}
	}

	exportDir := filepath.Join(tmp, "export")
	if err := Export(target, "b7694", exportDir); err != nil {
		t.Fatalf("Export: %v", err)
	}

	for name, content := range files {
		p := filepath.Join(exportDir, name)
		got, err := os.ReadFile(p)
		if err != nil {
			t.Fatalf("missing %s: %v", name, err)
		}
		if string(got) != content {
			t.Errorf("%s content mismatch", name)
		}
	}
}

func TestExport_PreservesSymlinks(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("LLAMAFARM_CACHE_DIR", filepath.Join(tmp, "cache"))

	target := Target{OS: "linux", Arch: "arm64", Accelerator: "cpu"}
	dir, err := CacheDir(target, "b7694")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "libllama.so"), []byte("stub"), 0o755); err != nil {
		t.Fatal(err)
	}
	// Create a symlink from libllama.so.0 → libllama.so. Windows without
	// developer-mode or admin privileges can't create symlinks, so we skip
	// rather than fail the whole test on that platform.
	if err := os.Symlink("libllama.so", filepath.Join(dir, "libllama.so.0")); err != nil {
		t.Skipf("symlink creation not supported in this environment: %v", err)
	}

	exportDir := filepath.Join(tmp, "export")
	if err := Export(target, "b7694", exportDir); err != nil {
		t.Fatalf("Export: %v", err)
	}
	linkPath := filepath.Join(exportDir, "libllama.so.0")
	info, err := os.Lstat(linkPath)
	if err != nil {
		t.Fatalf("lstat: %v", err)
	}
	if info.Mode()&os.ModeSymlink == 0 {
		t.Error("expected symlink in exported dir")
	}
}
