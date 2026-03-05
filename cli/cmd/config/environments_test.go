package config

import (
	"testing"
)

func TestEnvFieldExplicitlySet_YAML(t *testing.T) {
	tests := []struct {
		name         string
		data         string
		envName      string
		field        string
		wantExplicit bool
		wantVal      bool
	}{
		{
			name: "field explicitly false",
			data: `environments:
  staging:
    server_url: https://staging.example.com
    deploy_models: false
`,
			envName:      "staging",
			field:        "deploy_models",
			wantExplicit: true,
			wantVal:      false,
		},
		{
			name: "field explicitly true",
			data: `environments:
  staging:
    server_url: https://staging.example.com
    deploy_models: true
`,
			envName:      "staging",
			field:        "deploy_models",
			wantExplicit: true,
			wantVal:      true,
		},
		{
			name: "field absent",
			data: `environments:
  staging:
    server_url: https://staging.example.com
`,
			envName:      "staging",
			field:        "deploy_models",
			wantExplicit: false,
			wantVal:      false,
		},
		{
			name: "unknown environment",
			data: `environments:
  staging:
    server_url: https://staging.example.com
`,
			envName:      "production",
			field:        "deploy_models",
			wantExplicit: false,
			wantVal:      false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			SetRawConfigData([]byte(tt.data), "yaml")
			defer SetRawConfigData(nil, "")

			gotExplicit, gotVal := envFieldExplicitlySet([]byte(tt.data), tt.envName, tt.field)
			if gotExplicit != tt.wantExplicit || gotVal != tt.wantVal {
				t.Errorf("envFieldExplicitlySet() = (%v, %v), want (%v, %v)",
					gotExplicit, gotVal, tt.wantExplicit, tt.wantVal)
			}
		})
	}
}

func TestEnvFieldExplicitlySet_JSON(t *testing.T) {
	tests := []struct {
		name         string
		data         string
		wantExplicit bool
		wantVal      bool
	}{
		{
			name:         "field explicitly false",
			data:         `{"environments":{"staging":{"server_url":"https://staging.example.com","deploy_models":false}}}`,
			wantExplicit: true,
			wantVal:      false,
		},
		{
			name:         "field absent",
			data:         `{"environments":{"staging":{"server_url":"https://staging.example.com"}}}`,
			wantExplicit: false,
			wantVal:      false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			SetRawConfigData([]byte(tt.data), "json")
			defer SetRawConfigData(nil, "")

			gotExplicit, gotVal := envFieldExplicitlySet([]byte(tt.data), "staging", "deploy_models")
			if gotExplicit != tt.wantExplicit || gotVal != tt.wantVal {
				t.Errorf("envFieldExplicitlySet() = (%v, %v), want (%v, %v)",
					gotExplicit, gotVal, tt.wantExplicit, tt.wantVal)
			}
		})
	}
}

func TestEnvFieldExplicitlySet_TOML(t *testing.T) {
	tests := []struct {
		name         string
		data         string
		wantExplicit bool
		wantVal      bool
	}{
		{
			name: "field explicitly false",
			data: `[environments.staging]
server_url = "https://staging.example.com"
deploy_models = false
`,
			wantExplicit: true,
			wantVal:      false,
		},
		{
			name: "field explicitly true",
			data: `[environments.staging]
server_url = "https://staging.example.com"
deploy_models = true
`,
			wantExplicit: true,
			wantVal:      true,
		},
		{
			name: "field absent",
			data: `[environments.staging]
server_url = "https://staging.example.com"
`,
			wantExplicit: false,
			wantVal:      false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			SetRawConfigData([]byte(tt.data), "toml")
			defer SetRawConfigData(nil, "")

			gotExplicit, gotVal := envFieldExplicitlySet([]byte(tt.data), "staging", "deploy_models")
			if gotExplicit != tt.wantExplicit || gotVal != tt.wantVal {
				t.Errorf("envFieldExplicitlySet() = (%v, %v), want (%v, %v)",
					gotExplicit, gotVal, tt.wantExplicit, tt.wantVal)
			}
		})
	}
}

func TestEnvFieldExplicitlySet_InvalidData(t *testing.T) {
	SetRawConfigData([]byte("not valid yaml: [[["), "yaml")
	defer SetRawConfigData(nil, "")

	gotExplicit, gotVal := envFieldExplicitlySet([]byte("not valid yaml: [[["), "staging", "deploy_models")
	if gotExplicit || gotVal {
		t.Errorf("expected (false, false) for invalid data, got (%v, %v)", gotExplicit, gotVal)
	}
}

func TestResolveEnvironment(t *testing.T) {
	tests := []struct {
		name             string
		config           *LlamaFarmConfig
		rawData          string
		rawFormat        string
		envName          string
		wantErr          bool
		wantDeployModels bool
		wantDeployData   bool
		wantServerURL    string
	}{
		{
			name:    "no environments configured",
			config:  &LlamaFarmConfig{},
			envName: "staging",
			wantErr: true,
		},
		{
			name: "environment not found",
			config: &LlamaFarmConfig{
				Environments: LlamaFarmConfigEnvironments{
					"production": {ServerUrl: "https://prod.example.com"},
				},
			},
			envName: "staging",
			wantErr: true,
		},
		{
			name: "environment missing server_url",
			config: &LlamaFarmConfig{
				Environments: LlamaFarmConfigEnvironments{
					"staging": {ServerUrl: ""},
				},
			},
			envName: "staging",
			wantErr: true,
		},
		{
			name: "deploy_models defaults to true when absent",
			config: &LlamaFarmConfig{
				Environments: LlamaFarmConfigEnvironments{
					"staging": {ServerUrl: "https://staging.example.com"},
				},
			},
			rawData: `environments:
  staging:
    server_url: https://staging.example.com
`,
			rawFormat:        "yaml",
			envName:          "staging",
			wantDeployModels: true,
			wantServerURL:    "https://staging.example.com",
		},
		{
			name: "deploy_models explicitly false in YAML",
			config: &LlamaFarmConfig{
				Environments: LlamaFarmConfigEnvironments{
					"staging": {ServerUrl: "https://staging.example.com"},
				},
			},
			rawData: `environments:
  staging:
    server_url: https://staging.example.com
    deploy_models: false
`,
			rawFormat:        "yaml",
			envName:          "staging",
			wantDeployModels: false,
			wantServerURL:    "https://staging.example.com",
		},
		{
			name: "deploy_models explicitly false in TOML",
			config: &LlamaFarmConfig{
				Environments: LlamaFarmConfigEnvironments{
					"staging": {ServerUrl: "https://staging.example.com"},
				},
			},
			rawData: `[environments.staging]
server_url = "https://staging.example.com"
deploy_models = false
`,
			rawFormat:        "toml",
			envName:          "staging",
			wantDeployModels: false,
			wantServerURL:    "https://staging.example.com",
		},
		{
			name: "deploy_models explicitly false in JSON",
			config: &LlamaFarmConfig{
				Environments: LlamaFarmConfigEnvironments{
					"staging": {ServerUrl: "https://staging.example.com"},
				},
			},
			rawData:          `{"environments":{"staging":{"server_url":"https://staging.example.com","deploy_models":false}}}`,
			rawFormat:        "json",
			envName:          "staging",
			wantDeployModels: false,
			wantServerURL:    "https://staging.example.com",
		},
		{
			name: "deploy_data propagated",
			config: &LlamaFarmConfig{
				Environments: LlamaFarmConfigEnvironments{
					"staging": {ServerUrl: "https://staging.example.com", DeployData: true},
				},
			},
			rawData: `environments:
  staging:
    server_url: https://staging.example.com
    deploy_data: true
`,
			rawFormat:      "yaml",
			envName:        "staging",
			wantDeployData: true,
			// deploy_models absent -> defaults true
			wantDeployModels: true,
			wantServerURL:    "https://staging.example.com",
		},
		{
			name: "no raw config data falls back to default true",
			config: &LlamaFarmConfig{
				Environments: LlamaFarmConfigEnvironments{
					"staging": {ServerUrl: "https://staging.example.com"},
				},
			},
			// rawData intentionally empty — simulates rawConfigData == nil
			envName:          "staging",
			wantDeployModels: true,
			wantServerURL:    "https://staging.example.com",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.rawData != "" {
				SetRawConfigData([]byte(tt.rawData), tt.rawFormat)
			} else {
				SetRawConfigData(nil, "")
			}
			defer SetRawConfigData(nil, "")

			dc, err := tt.config.ResolveEnvironment(tt.envName)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if dc.ServerURL != tt.wantServerURL {
				t.Errorf("ServerURL = %q, want %q", dc.ServerURL, tt.wantServerURL)
			}
			if dc.DeployModels != tt.wantDeployModels {
				t.Errorf("DeployModels = %v, want %v", dc.DeployModels, tt.wantDeployModels)
			}
			if dc.DeployData != tt.wantDeployData {
				t.Errorf("DeployData = %v, want %v", dc.DeployData, tt.wantDeployData)
			}
		})
	}
}

func TestListEnvironmentNames(t *testing.T) {
	config := &LlamaFarmConfig{
		Environments: LlamaFarmConfigEnvironments{
			"production": {ServerUrl: "https://prod.example.com"},
			"staging":    {ServerUrl: "https://staging.example.com"},
			"dev":        {ServerUrl: "http://localhost:14345"},
		},
	}

	names := config.ListEnvironmentNames()
	want := []string{"dev", "production", "staging"}
	if len(names) != len(want) {
		t.Fatalf("got %d names, want %d", len(names), len(want))
	}
	for i, name := range names {
		if name != want[i] {
			t.Errorf("names[%d] = %q, want %q", i, name, want[i])
		}
	}
}

func TestStripEnvironments(t *testing.T) {
	config := &LlamaFarmConfig{
		Name:      "test",
		Namespace: "default",
		Environments: LlamaFarmConfigEnvironments{
			"staging": {ServerUrl: "https://staging.example.com"},
		},
	}

	stripped := config.StripEnvironments()
	if stripped.Environments != nil {
		t.Errorf("expected nil Environments, got %v", stripped.Environments)
	}
	if stripped.Name != "test" {
		t.Errorf("Name should be preserved, got %q", stripped.Name)
	}
	// Original should be unchanged
	if config.Environments == nil {
		t.Error("original config Environments should not be nil")
	}
}
