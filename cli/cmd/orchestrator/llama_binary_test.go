package orchestrator

import (
	"fmt"
	"runtime"
	"testing"
)

func TestGetBinaryInfo_Structure(t *testing.T) {
	// Simple smoke test to ensure GetBinaryInfo returns without error for current platform
	hardware := HardwareCPU
	info, err := GetBinaryInfo(hardware)
	if err != nil {
		t.Fatalf("GetBinaryInfo failed for %s: %v", hardware, err)
	}

	if info.URL == "" {
		t.Error("Expected URL to be populated")
	}
	if info.LibName == "" {
		t.Error("Expected LibName to be populated")
	}
}

func TestLinuxARM64Spec(t *testing.T) {
	// Verify the Linux ARM64 spec is correctly defined
	if LinuxARM64BinarySpec.LibName != "libllama.so" {
		t.Errorf("Expected LibName=libllama.so, got %s", LinuxARM64BinarySpec.LibName)
	}
	if LinuxARM64BinarySpec.LibPath != "libllama.so" {
		t.Errorf("Expected LibPath=libllama.so, got %s", LinuxARM64BinarySpec.LibPath)
	}

	expectedURLPart := fmt.Sprintf("llama-%s-bin-linux-arm64.tar.gz", LlamaCppVersion)
	if !contains(LinuxARM64BinarySpec.URL, expectedURLPart) {
		t.Errorf("Expected URL to contain %s, got %s", expectedURLPart, LinuxARM64BinarySpec.URL)
	}

	// Double check logic mirrors runtime checks (can't easily mock runtime.GOARCH/GOOS)
	if runtime.GOOS == "linux" && runtime.GOARCH == "arm64" {
		info, err := GetBinaryInfo(HardwareCPU)
		if err != nil {
			t.Fatal(err)
		}
		if info != LinuxARM64BinarySpec {
			t.Error("Expected LinuxARM64BinarySpec to be returned on linux/arm64")
		}
	}
}
