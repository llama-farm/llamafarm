package config

import (
    "os"
    "path/filepath"
    "testing"
)

func writeTempConfig(t *testing.T, content string) string {
    t.Helper()
    dir := t.TempDir()
    path := filepath.Join(dir, "llamafarm.yaml")
    if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
        t.Fatalf("failed to write temp config: %v", err)
    }
    return path
}

func TestGetProjectInfo(t *testing.T) {
    cfg := &LlamaFarmConfig{Name: "acme/shop"}
    pi, err := cfg.GetProjectInfo()
    if err != nil {
        t.Fatalf("unexpected err: %v", err)
    }
    if pi.Namespace != "acme" || pi.Project != "shop" {
        t.Fatalf("unexpected project info: %+v", pi)
    }
}

func TestGetServerConfig_Strict(t *testing.T) {
    // No config file, expect error if namespace/project missing
    _, err := GetServerConfig("", "", "", "")
    if err == nil {
        t.Fatalf("expected error when namespace/project missing")
    }

    // With config file containing name
    path := writeTempConfig(t, "name: acme/shop\nversion: v1\n")
    sc, err := GetServerConfig(path, "", "", "")
    if err != nil {
        t.Fatalf("unexpected err: %v", err)
    }
    if sc.URL != "http://localhost:8000" || sc.Namespace != "acme" || sc.Project != "shop" {
        t.Fatalf("unexpected server config: %+v", sc)
    }
}

func TestGetServerConfig_Lenient(t *testing.T) {
    // No config file, but lenient should still return defaults
    sc, err := GetServerConfigLenient("", "", "", "")
    if err != nil {
        t.Fatalf("unexpected err: %v", err)
    }
    if sc.URL != "http://localhost:8000" || sc.Namespace != "" || sc.Project != "" {
        t.Fatalf("unexpected server config: %+v", sc)
    }

    // With config file and overrides
    path := writeTempConfig(t, "name: acme/shop\nversion: v1\n")
    sc, err = GetServerConfigLenient(path, "http://x", "ns", "proj")
    if err != nil {
        t.Fatalf("unexpected err: %v", err)
    }
    if sc.URL != "http://x" || sc.Namespace != "ns" || sc.Project != "proj" {
        t.Fatalf("unexpected server config: %+v", sc)
    }

    // Config file missing 'name' field, should fallback to empty namespace/project
    pathMissingName := writeTempConfig(t, "version: v1\n")
    sc, err = GetServerConfigLenient(pathMissingName, "", "", "")
    if err != nil {
        t.Fatalf("unexpected err: %v", err)
    }
    if sc.Namespace != "" || sc.Project != "" {
        t.Fatalf("expected empty namespace/project for missing name, got: %+v", sc)
    }
}
