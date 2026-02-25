package config

import (
	"fmt"
	"sort"
	"strings"
)

// DeployConfig holds the resolved deploy settings for a target environment.
type DeployConfig struct {
	// ServerURL is the LlamaFarm server URL for this environment.
	ServerURL string
	// DeployModels controls whether model downloads are triggered on deploy.
	// The generated config type uses a plain bool whose schema default is true,
	// so we propagate it directly — no pointer indirection needed.
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
		DeployModels: env.DeployModels,
	}

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
	sort.Strings(names)
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
