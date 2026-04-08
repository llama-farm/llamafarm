package hfmodel

import (
	"os"
	"strings"
)

// IsOffline reports whether the LLAMAFARM_OFFLINE env var is set to a truthy
// value. Truthy values are: "1", "true", "yes", "on" (case-insensitive),
// matching common/llamafarm_common/offline_mode.py.
//
// When this returns true, all HuggingFace Hub network calls in this package
// MUST be skipped and replaced with an OfflineError that points the user at
// "lf models pull on a host with internet" remediation.
func IsOffline() bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv("LLAMAFARM_OFFLINE")))
	switch v {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}
