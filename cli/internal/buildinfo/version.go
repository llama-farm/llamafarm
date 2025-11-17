package buildinfo

// CurrentVersion will be set by build flags during release builds
// This is in a separate package to avoid import cycles between cmd/version and cmd/orchestrator
var CurrentVersion = "dev"
