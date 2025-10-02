package cmd

import (
	"bytes"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type fakeHTTPClient struct {
	resp      *http.Response
	err       error
	callCount int
}

func (f *fakeHTTPClient) Do(req *http.Request) (*http.Response, error) {
	f.callCount++
	return f.resp, f.err
}

func TestMaybeCheckForUpgrade_NewVersionAvailable(t *testing.T) {
	t.Setenv(upgradeStateEnvVar, filepath.Join(t.TempDir(), "state.json"))

	originalHTTPClient := httpClient
	originalTimeNow := timeNow
	originalVersion := Version
	defer func() {
		httpClient = originalHTTPClient
		timeNow = originalTimeNow
		Version = originalVersion
	}()

	Version = "1.0.0"
	fixedTime := time.Date(2025, time.January, 1, 10, 0, 0, 0, time.UTC)
	timeNow = func() time.Time { return fixedTime }

	body := `{"tag_name":"v1.1.0","html_url":"https://example.com/release","published_at":"2025-01-01T00:00:00Z","draft":false,"prerelease":false,"body":"Release notes"}`
	httpClient = &fakeHTTPClient{
		resp: &http.Response{
			StatusCode: http.StatusOK,
			Status:     "200 OK",
			Body:       io.NopCloser(strings.NewReader(body)),
			Header:     make(http.Header),
		},
	}

	info, err := maybeCheckForUpgrade(true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if info == nil {
		t.Fatalf("expected upgrade info, got nil")
	}
	if !info.UpdateAvailable {
		t.Fatalf("expected update available")
	}
	if info.LatestVersion != "v1.1.0" {
		t.Fatalf("expected latest version v1.1.0, got %s", info.LatestVersion)
	}

	statePath, err := getUpgradeStatePath()
	if err != nil {
		t.Fatalf("unexpected state path error: %v", err)
	}
	data, err := os.ReadFile(statePath)
	if err != nil {
		t.Fatalf("expected state file to exist: %v", err)
	}
	if len(bytes.TrimSpace(data)) == 0 {
		t.Fatalf("state file should not be empty")
	}
}

func TestMaybeCheckForUpgrade_SkipsWithinInterval(t *testing.T) {
	tmpDir := t.TempDir()
	statePath := filepath.Join(tmpDir, "state.json")
	t.Setenv(upgradeStateEnvVar, statePath)

	originalHTTPClient := httpClient
	originalTimeNow := timeNow
	defer func() {
		httpClient = originalHTTPClient
		timeNow = originalTimeNow
	}()

	// Any HTTP call should fail this test if made.
	httpClient = &fakeHTTPClient{
		err: errors.New("http call not expected"),
	}

	baseTime := time.Date(2025, time.January, 2, 9, 0, 0, 0, time.UTC)
	timeNow = func() time.Time { return baseTime.Add(3 * time.Hour) }

	state := upgradeState{
		LastChecked:   baseTime,
		LatestVersion: "v1.0.0",
	}
	if err := persistUpgradeState(statePath, state); err != nil {
		t.Fatalf("failed to write state: %v", err)
	}

	info, err := maybeCheckForUpgrade(false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if info != nil {
		t.Fatalf("expected no upgrade info when within interval")
	}

	fakeClient := httpClient.(*fakeHTTPClient)
	if fakeClient.callCount != 0 {
		t.Fatalf("expected no HTTP calls, got %d", fakeClient.callCount)
	}
}

func TestMaybeCheckForUpgrade_ForceErrorPropagates(t *testing.T) {
	t.Setenv(upgradeStateEnvVar, filepath.Join(t.TempDir(), "state.json"))

	originalHTTPClient := httpClient
	defer func() {
		httpClient = originalHTTPClient
	}()

	httpClient = &fakeHTTPClient{
		err: errors.New("network down"),
	}

	if _, err := maybeCheckForUpgrade(true); err == nil {
		t.Fatalf("expected error when HTTP call fails during forced upgrade check")
	}
}

func TestMaybeCheckForUpgrade_IgnoresDraftAndPrerelease(t *testing.T) {
	t.Setenv(upgradeStateEnvVar, filepath.Join(t.TempDir(), "state.json"))

	originalHTTPClient := httpClient
	originalVersion := Version
	defer func() {
		httpClient = originalHTTPClient
		Version = originalVersion
	}()

	Version = "v1.2.2"

	// Simulate a GitHub API response with a draft release
	draftRelease := `{
		"tag_name": "v1.2.3",
		"draft": true,
		"prerelease": false,
		"html_url": "https://github.com/example/repo/releases/tag/v1.2.3",
		"published_at": "2025-01-01T00:00:00Z",
		"body": "Draft release"
	}`
	prerelease := `{
		"tag_name": "v1.2.4-beta",
		"draft": false,
		"prerelease": true,
		"html_url": "https://github.com/example/repo/releases/tag/v1.2.4-beta",
		"published_at": "2025-01-01T00:00:00Z",
		"body": "Beta release"
	}`

	for _, body := range []string{draftRelease, prerelease} {
		httpClient = &fakeHTTPClient{
			resp: &http.Response{
				StatusCode: 200,
				Status:     "200 OK",
				Body:       io.NopCloser(strings.NewReader(body)),
				Header:     make(http.Header),
			},
		}
		// Use a version lower than the release to trigger upgrade logic
		_, err := maybeCheckForUpgrade(true)
		if err == nil {
			t.Fatalf("expected error for draft/prerelease, got nil")
		}
		if !strings.Contains(err.Error(), "draft or prerelease") {
			t.Fatalf("expected error about draft/prerelease, got: %v", err)
		}
	}
}
