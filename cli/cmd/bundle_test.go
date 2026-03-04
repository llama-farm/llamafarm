package cmd

import (
	"fmt"
	"testing"
)

func TestPlatformToPyAppOS(t *testing.T) {
	tests := []struct {
		platform string
		want     string
	}{
		{"linux", "linux"},
		{"darwin", "macos"},
		{"windows", "windows"},
	}

	for _, tt := range tests {
		t.Run(tt.platform, func(t *testing.T) {
			got, ok := platformToPyAppOS[tt.platform]
			if !ok {
				t.Fatalf("platform %q not in platformToPyAppOS", tt.platform)
			}
			if got != tt.want {
				t.Errorf("platformToPyAppOS[%q] = %q, want %q", tt.platform, got, tt.want)
			}
		})
	}
}

func TestPyAppBinaryName(t *testing.T) {
	tests := []struct {
		platform  string
		arch      string
		component string
		want      string
	}{
		{"darwin", "arm64", "server", "llamafarm-server-macos-arm64"},
		{"linux", "x86_64", "server", "llamafarm-server-linux-x86_64"},
		{"linux", "arm64", "rag", "llamafarm-rag-linux-arm64"},
		{"windows", "x86_64", "runtime", "llamafarm-runtime-windows-x86_64.exe"},
	}

	for _, tt := range tests {
		name := fmt.Sprintf("%s/%s/%s", tt.platform, tt.arch, tt.component)
		t.Run(name, func(t *testing.T) {
			pyappPlatform := fmt.Sprintf("%s-%s", platformToPyAppOS[tt.platform], tt.arch)
			binaryName := fmt.Sprintf("llamafarm-%s-%s", tt.component, pyappPlatform)
			if tt.platform == "windows" {
				binaryName += ".exe"
			}
			if binaryName != tt.want {
				t.Errorf("got %q, want %q", binaryName, tt.want)
			}
		})
	}
}

func TestGetPipPlatformTag(t *testing.T) {
	tests := []struct {
		platform string
		arch     string
		want     string
	}{
		{"linux", "x86_64", "manylinux2014_x86_64"},
		{"linux", "arm64", "manylinux2014_aarch64"},
		{"darwin", "arm64", "macosx_14_0_arm64"},
		{"windows", "x86_64", "win_amd64"},
		{"windows", "arm64", "win_arm64"},
	}

	for _, tt := range tests {
		name := fmt.Sprintf("%s/%s", tt.platform, tt.arch)
		t.Run(name, func(t *testing.T) {
			got := getPipPlatformTag(tt.platform, tt.arch)
			if got != tt.want {
				t.Errorf("getPipPlatformTag(%q, %q) = %q, want %q", tt.platform, tt.arch, got, tt.want)
			}
		})
	}
}

func TestGetAddonPlatformString(t *testing.T) {
	tests := []struct {
		platform string
		arch     string
		want     string
	}{
		{"darwin", "arm64", "macos-arm64"},
		{"linux", "x86_64", "linux-x86_64"},
		{"linux", "arm64", "linux-arm64"},
		{"windows", "x86_64", "windows-x86_64"},
	}

	for _, tt := range tests {
		name := fmt.Sprintf("%s/%s", tt.platform, tt.arch)
		t.Run(name, func(t *testing.T) {
			got := getAddonPlatformString(tt.platform, tt.arch)
			if got != tt.want {
				t.Errorf("getAddonPlatformString(%q, %q) = %q, want %q", tt.platform, tt.arch, got, tt.want)
			}
		})
	}
}

func TestValidateBundleFlags(t *testing.T) {
	tests := []struct {
		name        string
		platform    string
		arch        string
		accelerator string
		wantErr     bool
	}{
		{
			name:        "valid linux x86_64 cuda",
			platform:    "linux",
			arch:        "x86_64",
			accelerator: "cuda",
		},
		{
			name:        "valid darwin arm64 metal",
			platform:    "darwin",
			arch:        "arm64",
			accelerator: "metal",
		},
		{
			name:        "valid linux arm64 cpu",
			platform:    "linux",
			arch:        "arm64",
			accelerator: "cpu",
		},
		{
			name:        "invalid platform",
			platform:    "freebsd",
			arch:        "x86_64",
			accelerator: "cpu",
			wantErr:     true,
		},
		{
			name:        "invalid arch",
			platform:    "linux",
			arch:        "riscv64",
			accelerator: "cpu",
			wantErr:     true,
		},
		{
			name:        "invalid accelerator",
			platform:    "linux",
			arch:        "x86_64",
			accelerator: "tpu",
			wantErr:     true,
		},
		{
			name:        "darwin x86_64 not supported",
			platform:    "darwin",
			arch:        "x86_64",
			accelerator: "cpu",
			wantErr:     true,
		},
		{
			name:        "windows arm64 not supported",
			platform:    "windows",
			arch:        "arm64",
			accelerator: "cpu",
			wantErr:     true,
		},
		{
			name:        "metal on linux not allowed",
			platform:    "linux",
			arch:        "x86_64",
			accelerator: "metal",
			wantErr:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save and restore bundleFlags
			saved := bundleFlags
			defer func() { bundleFlags = saved }()

			bundleFlags.platform = tt.platform
			bundleFlags.arch = tt.arch
			bundleFlags.accelerator = tt.accelerator

			err := validateBundleFlags()
			if tt.wantErr && err == nil {
				t.Errorf("expected error, got nil")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}
