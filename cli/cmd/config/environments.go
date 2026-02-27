package config

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	toml "github.com/pelletier/go-toml/v2"
	yaml "gopkg.in/yaml.v2"
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

	// The generated type uses bool (not *bool), so we can't distinguish "absent" from
	// "explicitly false". Use the raw config data to check if deploy_models was set.
	if rawConfigData != nil {
		if explicitly, val := envFieldExplicitlySet(rawConfigData, name, "deploy_models"); explicitly {
			dc.DeployModels = val
		}
	} else if env.DeployModels {
		dc.DeployModels = true
	}

	return dc, nil
}

// rawConfigData holds the raw bytes of the loaded config file for secondary parsing.
// This allows us to detect explicitly-set fields that the generated types can't distinguish.
var rawConfigData []byte

// rawConfigFormat holds the format of the raw config data ("yaml", "json", or "toml").
var rawConfigFormat string

// SetRawConfigData stores the raw config bytes and format for use in environment resolution.
func SetRawConfigData(data []byte, format string) {
	rawConfigData = data
	rawConfigFormat = format
}

// envFieldExplicitlySet checks if a boolean field was explicitly set in the raw config
// for a given environment. Returns (true, value) if the field exists, (false, false) otherwise.
func envFieldExplicitlySet(data []byte, envName, field string) (bool, bool) {
	var raw struct {
		Environments map[string]map[string]interface{} `yaml:"environments" json:"environments" toml:"environments"`
	}

	var err error
	switch rawConfigFormat {
	case "json":
		err = json.Unmarshal(data, &raw)
	case "toml":
		err = toml.Unmarshal(data, &raw)
	default: // yaml
		err = yaml.Unmarshal(data, &raw)
	}
	if err != nil {
		return false, false
	}

	env, ok := raw.Environments[envName]
	if !ok {
		return false, false
	}
	val, ok := env[field]
	if !ok {
		return false, false
	}
	if b, ok := val.(bool); ok {
		return true, b
	}
	return false, false
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
