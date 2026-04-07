package config

import "testing"

func strPtr(s string) *string { return &s }

func TestResolveModelDir_FlagWins(t *testing.T) {
	cfg := &LlamaFarmConfig{
		Deployment: &Deployment{ModelDir: strPtr("/from/config")},
	}
	got, source := cfg.ResolveModelDir("/from/flag")
	if got != "/from/flag" {
		t.Errorf("got %q, want %q", got, "/from/flag")
	}
	if source != ModelDirSourceFlag {
		t.Errorf("got source %q, want %q", source, ModelDirSourceFlag)
	}
}

func TestResolveModelDir_ConfigUsedWhenFlagEmpty(t *testing.T) {
	cfg := &LlamaFarmConfig{
		Deployment: &Deployment{ModelDir: strPtr("/from/config")},
	}
	got, source := cfg.ResolveModelDir("")
	if got != "/from/config" {
		t.Errorf("got %q, want %q", got, "/from/config")
	}
	if source != ModelDirSourceConfig {
		t.Errorf("got source %q, want %q", source, ModelDirSourceConfig)
	}
}

func TestResolveModelDir_DefaultWhenNeitherSet(t *testing.T) {
	cfg := &LlamaFarmConfig{}
	got, source := cfg.ResolveModelDir("")
	if got != DefaultModelDir {
		t.Errorf("got %q, want %q", got, DefaultModelDir)
	}
	if source != ModelDirSourceDefault {
		t.Errorf("got source %q, want %q", source, ModelDirSourceDefault)
	}
}

func TestResolveModelDir_DefaultWhenDeploymentNil(t *testing.T) {
	cfg := &LlamaFarmConfig{Deployment: nil}
	got, source := cfg.ResolveModelDir("")
	if got != DefaultModelDir {
		t.Errorf("got %q, want %q", got, DefaultModelDir)
	}
	if source != ModelDirSourceDefault {
		t.Errorf("got source %q, want %q", source, ModelDirSourceDefault)
	}
}

func TestResolveModelDir_DefaultWhenModelDirNil(t *testing.T) {
	cfg := &LlamaFarmConfig{Deployment: &Deployment{ModelDir: nil}}
	got, source := cfg.ResolveModelDir("")
	if got != DefaultModelDir {
		t.Errorf("got %q, want %q", got, DefaultModelDir)
	}
	if source != ModelDirSourceDefault {
		t.Errorf("got source %q, want %q", source, ModelDirSourceDefault)
	}
}

func TestResolveModelDir_DefaultWhenModelDirEmptyString(t *testing.T) {
	cfg := &LlamaFarmConfig{Deployment: &Deployment{ModelDir: strPtr("")}}
	got, source := cfg.ResolveModelDir("")
	if got != DefaultModelDir {
		t.Errorf("got %q, want %q", got, DefaultModelDir)
	}
	if source != ModelDirSourceDefault {
		t.Errorf("got source %q, want %q", source, ModelDirSourceDefault)
	}
}

func TestResolveModelDir_NilReceiver(t *testing.T) {
	var cfg *LlamaFarmConfig
	got, source := cfg.ResolveModelDir("")
	if got != DefaultModelDir {
		t.Errorf("got %q, want %q", got, DefaultModelDir)
	}
	if source != ModelDirSourceDefault {
		t.Errorf("got source %q, want %q", source, ModelDirSourceDefault)
	}
}

func TestStripDeployment(t *testing.T) {
	cfg := &LlamaFarmConfig{
		Deployment: &Deployment{ModelDir: strPtr("/opt/lf")},
	}
	stripped := cfg.StripDeployment()
	if stripped.Deployment != nil {
		t.Errorf("expected Deployment to be nil after StripDeployment, got %+v", stripped.Deployment)
	}
	// Original should remain unchanged.
	if cfg.Deployment == nil {
		t.Error("StripDeployment should not mutate the receiver")
	}
}

// TestStripEnvironmentsAndDeployment verifies both local-only sections are
// removed when chained (the pattern used by `lf deploy` before pushing to a
// remote server).
func TestStripEnvironmentsAndDeployment(t *testing.T) {
	cfg := &LlamaFarmConfig{
		Name:      "test",
		Namespace: "default",
		Environments: LlamaFarmConfigEnvironments{
			"staging": {ServerUrl: "https://staging.example.com"},
		},
		Deployment: &Deployment{ModelDir: strPtr("/opt/lf")},
	}
	stripped := cfg.StripEnvironments().StripDeployment()
	if stripped.Environments != nil {
		t.Errorf("expected Environments nil, got %v", stripped.Environments)
	}
	if stripped.Deployment != nil {
		t.Errorf("expected Deployment nil, got %v", stripped.Deployment)
	}
	// Originals must remain intact.
	if cfg.Environments == nil {
		t.Error("original Environments was mutated")
	}
	if cfg.Deployment == nil {
		t.Error("original Deployment was mutated")
	}
}
