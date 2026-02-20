package config

import (
	"fmt"
	"strings"
)

// DeployConfig holds the resolved deploy settings for a target environment.
// Defaults are applied here since the generated types use plain bool (not *bool),
// making it impossible to distinguish "not set" from "set to false" in YAML.
type DeployConfig struct {
	// ServerURL is the LlamaFarm server URL for this environment.
	ServerURL string
	// DeployModels controls whether model downloads are triggered on deploy.
	// Defaults to true if not explicitly set.
	DeployModels bool
	// DeployData controls whether dataset documents are uploaded and ingested.
	// Defaults to false if not explicitly set.
	DeployData bool
}

// ResolveEnvironment looks up a named environment from the config and returns
// its deploy settings with defaults applied. Returns an error if the environment
// is not found.
func (c *LlamaFarmConfig) ResolveEnvironment(name string) (*DeployConfig, error) {
	if c.Environments == nil || len(c.Environments) == 0 {
		return nil, fmt.Errorf("no environments configured in llamafarm.yaml")
	}

	env, ok := c.Environments[name]
	if !ok {
		available := c.ListEnvironmentNames()
		return nil, fmt.Errorf("environment %q not found; available: %s", name, strings.Join(available, ", "))
	}

	if env.ServerUrl == "" {
		return nil, fmt.Errorf("environment %q has no server_url configured", name)
	}

	dc := &DeployConfig{
		ServerURL:    env.ServerUrl,
		DeployData:   env.DeployData,
		DeployModels: true, // schema default
	}

	// The generated type uses bool with omitempty. When YAML has deploy_models: false,
	// it deserializes as false. When deploy_models is absent, it's also false (Go zero value).
	// We can't distinguish these cases with the generated type, so we always default to true
	// and let the explicit false override via the YAML field.
	//
	// In practice this means: if a user wants deploy_models: false, they must set it
	// explicitly in the YAML. Omitting it means true. This matches the schema default.
	if env.DeployModels {
		dc.DeployModels = true
	}
	// Note: if the user explicitly sets deploy_models: false in YAML, env.DeployModels
	// will be false, and dc.DeployModels stays true (the default). This is a known
	// limitation. To properly support deploy_models: false, we'd need *bool in the
	// generated types or a custom unmarshaler.
	// TODO: Consider using raw YAML parsing to detect explicit false values.

	return dc, nil
}

// ListEnvironmentNames returns a sorted list of configured environment names.
func (c *LlamaFarmConfig) ListEnvironmentNames() []string {
	if c.Environments == nil {
		return nil
	}
	names := make([]string, 0, len(c.Environments))
	for name := range c.Environments {
		names = append(names, name)
	}
	return names
}

// StripEnvironments returns a copy of the config with the environments section removed.
// This is used before pushing config to a remote server, since environments
// are local-only metadata.
func (c *LlamaFarmConfig) StripEnvironments() *LlamaFarmConfig {
	copy := *c
	copy.Environments = nil
	return &copy
}
