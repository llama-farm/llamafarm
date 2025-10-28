package cmd

import (
	"os"
	"strings"
)

// determineOrchestrationMode determines which orchestration mode to use
// based on environment variables and system capabilities
func determineOrchestrationMode() OrchestrationMode {
	// Check for explicit mode setting
	modeEnv := strings.ToLower(strings.TrimSpace(os.Getenv("LF_ORCHESTRATION_MODE")))

	switch modeEnv {
	case "docker":
		return OrchestrationDocker
	case "native":
		return OrchestrationNative
	case "auto", "":
		// Auto mode - prefer native, fallback to Docker
		return OrchestrationNative
	default:
		// Unknown value, default to native
		return OrchestrationNative
	}
}

// forceDockerMode forces the use of Docker orchestration
// This is useful for backward compatibility or testing
func forceDockerMode() {
	os.Setenv("LF_ORCHESTRATION_MODE", "docker")
}

// forceNativeMode forces the use of native orchestration
func forceNativeMode() {
	os.Setenv("LF_ORCHESTRATION_MODE", "native")
}
