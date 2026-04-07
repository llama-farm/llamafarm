package config

// DefaultModelDir is the hardcoded fallback target root for on-device model layouts
// when neither the `--target-root` flag nor `deployment.model_dir` in llamafarm.yaml
// specifies a value.
const DefaultModelDir = "/opt/llamafarm/models"

// ModelDirSource identifies which tier provided a resolved model directory value.
type ModelDirSource string

const (
	// ModelDirSourceFlag indicates the value came from a CLI flag override.
	ModelDirSourceFlag ModelDirSource = "flag"
	// ModelDirSourceConfig indicates the value came from llamafarm.yaml `deployment.model_dir`.
	ModelDirSourceConfig ModelDirSource = "config"
	// ModelDirSourceDefault indicates the value came from the hardcoded DefaultModelDir.
	ModelDirSourceDefault ModelDirSource = "default"
)

// ResolveModelDir returns the effective target root for model layout paths along with
// the source tier that provided it. Precedence: flag > config > default.
//
// The flagValue parameter should be the raw value of a `--target-root` CLI flag (empty
// string when the flag is not set). The config receiver's `deployment.model_dir` field
// is consulted next. The hardcoded fallback `/opt/llamafarm/models` is used last.
func (c *LlamaFarmConfig) ResolveModelDir(flagValue string) (string, ModelDirSource) {
	if flagValue != "" {
		return flagValue, ModelDirSourceFlag
	}
	if c != nil && c.Deployment != nil && c.Deployment.ModelDir != nil && *c.Deployment.ModelDir != "" {
		return *c.Deployment.ModelDir, ModelDirSourceConfig
	}
	return DefaultModelDir, ModelDirSourceDefault
}

// StripDeployment returns a copy of the config with the deployment section removed.
// This is used before pushing config to a remote server, since deployment settings
// are local-only metadata (matching how StripEnvironments works).
func (c *LlamaFarmConfig) StripDeployment() *LlamaFarmConfig {
	copy := *c
	copy.Deployment = nil
	return &copy
}
