package orchestrator

import (
	"context"
	"errors"
	"fmt"
	"runtime"

	"github.com/llamafarm/cli/cmd/utils"
	"github.com/llamafarm/cli/internal/llamabinary"
)

func init() {
	// Forward debug logs from the internal package to the orchestrator's existing
	// utils.LogDebug so the user-visible output is unchanged.
	llamabinary.Logf = func(format string, args ...any) {
		utils.LogDebug(fmt.Sprintf(format, args...))
	}
}

// LlamaCppVersion is the pinned llama.cpp release version.
//
// This variable is kept for backward compatibility with existing callers and
// the llama_binary_test.go test. The authoritative version lives in
// llamabinary.Version, which is the one injected by build-time ldflags.
var LlamaCppVersion = llamabinary.Version

// BinaryInfo retains its historical shape so existing tests compile unchanged.
// New code should use llamabinary.Spec directly.
type BinaryInfo struct {
	URL     string
	SHA256  string
	LibPath string
	LibName string
}

func specToBinaryInfo(s llamabinary.Spec) BinaryInfo {
	return BinaryInfo(s)
}

// LlamaBinarySpec is the legacy per-hardware spec map for Linux/macOS. It is
// recomputed at init time from llamabinary.SpecFor so there is a single source
// of truth for URLs.
var LlamaBinarySpec map[HardwareCapability]BinaryInfo

// WindowsBinarySpec is the legacy per-hardware spec map for Windows targets.
var WindowsBinarySpec map[HardwareCapability]BinaryInfo

// LinuxARM64BinarySpec is the legacy spec for LlamaFarm-hosted Linux ARM64 builds.
var LinuxARM64BinarySpec BinaryInfo

func init() {
	// Build the legacy spec maps from llamabinary.SpecFor so that callers and
	// tests that reference the old symbols see current URLs + version.
	buildLegacySpecs()
}

func buildLegacySpecs() {
	// Linux/macOS specs keyed by HardwareCapability. We target amd64 for the
	// x86_64 Linux/macOS builds that the old map represented.
	darwinArm64Metal, _ := llamabinary.SpecFor(
		llamabinary.Target{OS: "darwin", Arch: "arm64", Accelerator: "metal"},
		llamabinary.Version,
	)
	linuxAmd64Cpu, _ := llamabinary.SpecFor(
		llamabinary.Target{OS: "linux", Arch: "amd64", Accelerator: "cpu"},
		llamabinary.Version,
	)
	linuxAmd64Vulkan, _ := llamabinary.SpecFor(
		llamabinary.Target{OS: "linux", Arch: "amd64", Accelerator: "vulkan"},
		llamabinary.Version,
	)

	LlamaBinarySpec = map[HardwareCapability]BinaryInfo{
		HardwareCPU:   specToBinaryInfo(linuxAmd64Cpu),
		HardwareCUDA:  specToBinaryInfo(linuxAmd64Vulkan), // upstream no longer ships CUDA linux builds
		HardwareMetal: specToBinaryInfo(darwinArm64Metal),
		HardwareROCm:  specToBinaryInfo(linuxAmd64Vulkan),
	}

	winAmd64Cpu, _ := llamabinary.SpecFor(
		llamabinary.Target{OS: "windows", Arch: "amd64", Accelerator: "cpu"},
		llamabinary.Version,
	)
	winAmd64Cuda, _ := llamabinary.SpecFor(
		llamabinary.Target{OS: "windows", Arch: "amd64", Accelerator: "cuda"},
		llamabinary.Version,
	)
	WindowsBinarySpec = map[HardwareCapability]BinaryInfo{
		HardwareCPU:  specToBinaryInfo(winAmd64Cpu),
		HardwareCUDA: specToBinaryInfo(winAmd64Cuda),
	}

	linuxArm64Cpu, _ := llamabinary.SpecFor(
		llamabinary.Target{OS: "linux", Arch: "arm64", Accelerator: "cpu"},
		llamabinary.Version,
	)
	LinuxARM64BinarySpec = specToBinaryInfo(linuxArm64Cpu)
}

// GetLlamaCacheDir returns the cache directory for the current host target. It
// forwards to llamabinary.CacheDir using the current host target so existing
// callers continue to see the historical `<root>/<version>/` path.
func GetLlamaCacheDir() (string, error) {
	root, err := llamabinary.CacheRoot()
	if err != nil {
		return "", err
	}
	return root, nil
}

// GetLlamaLibName returns the platform-specific library filename.
func GetLlamaLibName() string {
	return llamabinary.LibNameFor(runtime.GOOS)
}

// IsLlamaBinaryInstalled reports whether the llama.cpp binary is cached for
// the current host.
func IsLlamaBinaryInstalled() bool {
	return llamabinary.IsCached(llamabinary.CurrentHostTarget(), llamabinary.Version)
}

// EnsureLlamaBinary downloads the llama.cpp binary for the current host if
// missing. Returns the directory containing the installed binary.
func EnsureLlamaBinary() (string, error) {
	target := llamabinary.CurrentHostTarget()
	// Map the detected hardware to the best accelerator for this host. The
	// refactor keeps the orchestrator's HardwareCapability detection path
	// intact; we translate its output into an accelerator name.
	hardware := DetectHardware()
	target.Accelerator = hardwareToAccelerator(hardware, target.OS, target.Arch)

	res, err := llamabinary.Download(context.Background(), target, llamabinary.Version)
	if err != nil {
		// Only fall back to CPU when the failure is specifically "no
		// prebuilt for this accelerator" (ErrSpecNotAvailable). Other
		// failures (network, checksum, extraction, disk full) should
		// propagate — silently downgrading the accelerator for a real
		// I/O problem would mask the root cause and produce a much
		// worse user experience.
		if !errors.Is(err, llamabinary.ErrSpecNotAvailable) {
			return "", err
		}
		utils.LogDebug(fmt.Sprintf(
			"No prebuilt %s binary for %s; falling back to CPU",
			target.Accelerator, target.OS+"/"+target.Arch,
		))
		target.Accelerator = "cpu"
		res, err = llamabinary.Download(context.Background(), target, llamabinary.Version)
		if err != nil {
			return "", err
		}
	}
	return res.DestDir, nil
}

// hardwareToAccelerator maps the orchestrator's HardwareCapability enum to the
// accelerator string used by the llamabinary package.
func hardwareToAccelerator(h HardwareCapability, goos, goarch string) string {
	switch h {
	case HardwareCUDA:
		if goos == "windows" {
			return "cuda"
		}
		// Upstream Linux CUDA is not shipped; Vulkan is the closest substitute.
		return "vulkan"
	case HardwareROCm:
		return "vulkan"
	case HardwareMetal:
		return "metal"
	default:
		return llamabinary.BestAcceleratorFor(goos, goarch)
	}
}

// GetBinaryInfo returns the legacy BinaryInfo for the given hardware, matching
// the original API. It forwards to llamabinary.SpecFor under the hood.
func GetBinaryInfo(hardware HardwareCapability) (BinaryInfo, error) {
	target := llamabinary.CurrentHostTarget()
	target.Accelerator = hardwareToAccelerator(hardware, target.OS, target.Arch)

	spec, err := llamabinary.SpecFor(target, llamabinary.Version)
	if err == nil {
		return specToBinaryInfo(spec), nil
	}
	// Fall back to CPU for the current host/arch, matching old behavior.
	cpuTarget := target
	cpuTarget.Accelerator = "cpu"
	cpuSpec, cpuErr := llamabinary.SpecFor(cpuTarget, llamabinary.Version)
	if cpuErr != nil {
		return BinaryInfo{}, fmt.Errorf("no binary available for hardware %s on %s/%s", hardware, target.OS, target.Arch)
	}
	return specToBinaryInfo(cpuSpec), nil
}

// InstallLlamaBinary downloads the llama.cpp binary for the current host into
// destDir. This is retained for any external code that might still call it.
// New code should use llamabinary.Download directly with an explicit Target.
func InstallLlamaBinary(destDir string) error {
	target := llamabinary.CurrentHostTarget()
	target.Accelerator = hardwareToAccelerator(DetectHardware(), target.OS, target.Arch)

	// The old function installed into an arbitrary destDir. The new path
	// installs into the package-managed cache and then copies out if the
	// caller passed a non-cache destination.
	res, err := llamabinary.Download(context.Background(), target, llamabinary.Version)
	if err != nil {
		return err
	}
	if res.DestDir == destDir {
		return nil
	}
	return llamabinary.Export(target, llamabinary.Version, destDir)
}
