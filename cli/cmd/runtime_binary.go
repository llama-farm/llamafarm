package cmd

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/llamafarm/cli/internal/llamabinary"
	"github.com/spf13/cobra"
)

// runtimeCmd is the parent command for runtime-related operations that are
// distinct from project-level model management.
var runtimeCmd = &cobra.Command{
	Use:   "runtime",
	Short: "Manage LlamaFarm runtime components",
	Long: `Manage LlamaFarm runtime components (llama.cpp binaries, etc.).

Runtime binaries are shared infrastructure that the LlamaFarm server, edge runtime,
and universal runtime all depend on. These commands let you fetch, locate, and
export those binaries for use in deployment pipelines (ansible, packer, Dockerfile
builds).`,
	Run: func(cmd *cobra.Command, args []string) {
		cmd.Help()
	},
}

var runtimeBinaryCmd = &cobra.Command{
	Use:   "binary",
	Short: "Manage llama.cpp runtime binaries",
	Run: func(cmd *cobra.Command, args []string) {
		cmd.Help()
	},
}

// Shared flags for `lf runtime binary pull` and `lf runtime binary path`.
var (
	runtimeBinaryPlatform    string
	runtimeBinaryAccelerator string
	runtimeBinaryVersion     string
	runtimeBinaryExport      string
)

var runtimeBinaryPullCmd = &cobra.Command{
	Use:   "pull",
	Short: "Download the llama.cpp binary for a target platform",
	Long: `Download the pinned llama.cpp shared library and all of its dependency
files for a target platform and accelerator into the local LlamaFarm cache.

Examples:
  # Current host, best accelerator
  lf runtime binary pull

  # Linux ARM64 build for a Raspberry Pi
  lf runtime binary pull --platform linux/arm64 --accelerator cpu

  # Fetch and materialize a flat directory for ansible/packer pickup
  lf runtime binary pull --platform linux/arm64 --export /tmp/lf-bin`,
	RunE:          runRuntimeBinaryPull,
	SilenceUsage:  true,
	SilenceErrors: true,
}

var runtimeBinaryPathCmd = &cobra.Command{
	Use:   "path",
	Short: "Print the cached llama.cpp binary path for a target platform",
	Long: `Print the absolute path to the cached main library for the given target.
Exits non-zero with a remediation message if the binary has not been downloaded
for that target yet.

Examples:
  lf runtime binary path
  lf runtime binary path --platform linux/arm64 --accelerator cpu`,
	RunE:          runRuntimeBinaryPath,
	SilenceUsage:  true,
	SilenceErrors: true,
}

func init() {
	// Shared flag registration: attach the same flag descriptors to both subcommands
	// so the user experience is consistent.
	for _, c := range []*cobra.Command{runtimeBinaryPullCmd, runtimeBinaryPathCmd} {
		c.Flags().StringVar(&runtimeBinaryPlatform, "platform", "", "Target OS/arch (e.g. linux/arm64). Defaults to current host.")
		c.Flags().StringVar(&runtimeBinaryAccelerator, "accelerator", "", "Compute backend (cpu, cuda, metal, vulkan, rocm). Defaults to best-for-platform.")
		c.Flags().StringVar(&runtimeBinaryVersion, "version", "", "llama.cpp version tag. Defaults to the pinned version baked into this CLI.")
	}
	runtimeBinaryPullCmd.Flags().StringVar(&runtimeBinaryExport, "export", "", "After download, copy binary + deps into this flat directory for transport.")

	runtimeBinaryCmd.AddCommand(runtimeBinaryPullCmd)
	runtimeBinaryCmd.AddCommand(runtimeBinaryPathCmd)
	runtimeCmd.AddCommand(runtimeBinaryCmd)
	rootCmd.AddCommand(runtimeCmd)
}

// resolveRuntimeBinaryTarget converts the user's flags into a validated llamabinary.Target.
func resolveRuntimeBinaryTarget() (llamabinary.Target, error) {
	target := llamabinary.CurrentHostTarget()

	if runtimeBinaryPlatform != "" {
		parts := strings.SplitN(runtimeBinaryPlatform, "/", 2)
		if len(parts) != 2 {
			return target, fmt.Errorf("invalid --platform %q: expected format os/arch (e.g. linux/arm64)", runtimeBinaryPlatform)
		}
		target.OS = strings.ToLower(parts[0])
		arch, ok := llamabinary.CanonicalizeArch(parts[1])
		if !ok {
			return target, fmt.Errorf("unrecognized arch %q", parts[1])
		}
		target.Arch = arch
		// Default accelerator for the new platform unless the user also passed --accelerator.
		target.Accelerator = llamabinary.BestAcceleratorFor(target.OS, target.Arch)
	}

	if runtimeBinaryAccelerator != "" {
		target.Accelerator = strings.ToLower(runtimeBinaryAccelerator)
	}

	if err := target.Validate(); err != nil {
		return target, err
	}
	return target, nil
}

func resolveRuntimeBinaryVersion() string {
	if runtimeBinaryVersion != "" {
		return runtimeBinaryVersion
	}
	return llamabinary.Version
}

func runRuntimeBinaryPull(cmd *cobra.Command, args []string) error {
	target, err := resolveRuntimeBinaryTarget()
	if err != nil {
		return err
	}
	version := resolveRuntimeBinaryVersion()

	// Validate that a prebuilt exists before hitting the network.
	if _, err := llamabinary.SpecFor(target, version); err != nil {
		return fmt.Errorf("%v\n\nSupported combinations:\n%s", err, describeSupportedCombos())
	}

	fmt.Fprintf(os.Stderr, "Pulling llama.cpp %s for %s...\n", version, target)
	res, err := llamabinary.Download(context.Background(), target, version)
	if err != nil {
		return fmt.Errorf("download failed: %w", err)
	}
	if res.Cached {
		fmt.Fprintf(os.Stderr, "Already cached: %s\n", res.LibPath)
	} else {
		fmt.Fprintf(os.Stderr, "Installed: %s\n", res.LibPath)
	}

	if runtimeBinaryExport != "" {
		fmt.Fprintf(os.Stderr, "Exporting to %s...\n", runtimeBinaryExport)
		if err := llamabinary.Export(target, version, runtimeBinaryExport); err != nil {
			return fmt.Errorf("export failed: %w", err)
		}
		// Resolve to absolute path for clarity.
		abs := runtimeBinaryExport
		if a, err := filepath.Abs(runtimeBinaryExport); err == nil {
			abs = a
		}
		fmt.Fprintf(os.Stderr, "Exported: %s\n", abs)
	}

	// Always print the main library path on stdout so scripts can pipe into
	// `$(lf runtime binary pull ...)`.
	fmt.Println(res.LibPath)
	return nil
}

func runRuntimeBinaryPath(cmd *cobra.Command, args []string) error {
	target, err := resolveRuntimeBinaryTarget()
	if err != nil {
		return err
	}
	version := resolveRuntimeBinaryVersion()

	if !llamabinary.IsCached(target, version) {
		return fmt.Errorf(
			"llama.cpp binary not found for %s (version %s).\nRun 'lf runtime binary pull --platform %s/%s --accelerator %s' first.",
			target, version, target.OS, target.Arch, target.Accelerator,
		)
	}
	libPath, err := llamabinary.LibPath(target, version)
	if err != nil {
		return err
	}
	fmt.Println(libPath)
	return nil
}

// describeSupportedCombos returns a human-readable list of supported targets.
func describeSupportedCombos() string {
	combos := []llamabinary.Target{
		{OS: "darwin", Arch: "arm64", Accelerator: "metal"},
		{OS: "darwin", Arch: "amd64", Accelerator: "cpu"},
		{OS: "linux", Arch: "amd64", Accelerator: "cpu"},
		{OS: "linux", Arch: "amd64", Accelerator: "vulkan"},
		{OS: "linux", Arch: "arm64", Accelerator: "cpu"},
		{OS: "windows", Arch: "amd64", Accelerator: "cpu"},
		{OS: "windows", Arch: "amd64", Accelerator: "cuda"},
		{OS: "windows", Arch: "amd64", Accelerator: "vulkan"},
	}
	var b strings.Builder
	for _, c := range combos {
		b.WriteString("  ")
		b.WriteString(c.String())
		b.WriteString("\n")
	}
	return strings.TrimRight(b.String(), "\n")
}

