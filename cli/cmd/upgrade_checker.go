package cmd

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	semver "github.com/Masterminds/semver/v3"
)

const (
	upgradeStateEnvVar     = "LF_UPGRADE_STATE_PATH"
	upgradeStateFileName   = "upgrade_state.json"
	upgradeCheckInterval   = 6 * time.Hour
	githubLatestReleaseURL = "https://api.github.com/repos/llama-farm/llamafarm/releases/latest"
)

var (
	// timeNow is overridden in tests.
	timeNow = time.Now
)

type upgradeState struct {
	LastChecked   time.Time `json:"last_checked"`
	LatestVersion string    `json:"latest_version"`
}

type releaseInfo struct {
	TagName     string
	HTMLURL     string
	PublishedAt time.Time
	Body        string
}

// UpgradeInfo contains details about the currently running version and the latest release.
type UpgradeInfo struct {
	CurrentVersion           string
	CurrentVersionNormalized string
	LatestVersion            string
	LatestVersionNormalized  string
	ReleaseURL               string
	PublishedAt              time.Time
	ReleaseNotes             string
	UpdateAvailable          bool
	CurrentVersionIsSemver   bool
}

func maybeCheckForUpgrade(force bool) (*UpgradeInfo, error) {
	now := timeNow()

	state, statePath, stateErr := readUpgradeState()
	if stateErr != nil {
		logDebug(fmt.Sprintf("upgrade state read failed: %v", stateErr))
	}

	if !force && !shouldCheckForUpgrade(now, state) {
		return nil, nil
	}

	release, err := fetchLatestRelease()
	if err != nil {
		if force {
			// Return the error to the caller for explicit commands.
			return nil, err
		}
		logDebug(fmt.Sprintf("upgrade check failed: %v", err))
		return nil, nil
	}

	if err := persistUpgradeState(statePath, upgradeState{LastChecked: now, LatestVersion: release.TagName}); err != nil {
		logDebug(fmt.Sprintf("upgrade state write failed: %v", err))
	}

	info := buildUpgradeInfo(release)
	return info, nil
}

func buildUpgradeInfo(release *releaseInfo) *UpgradeInfo {
	currentRaw := strings.TrimSpace(Version)
	latestRaw := strings.TrimSpace(release.TagName)

	currentNormalized, currentSemver := normalizeForSemver(currentRaw)
	latestNormalized, latestSemver := normalizeForSemver(latestRaw)

	updateAvailable := false
	if currentSemver != nil && latestSemver != nil {
		updateAvailable = latestSemver.GreaterThan(currentSemver)
	}

	// Development builds (non-semver) should never auto-notify, but we still surface info to the command.
	return &UpgradeInfo{
		CurrentVersion:           displayVersion(currentRaw),
		CurrentVersionNormalized: currentNormalized,
		LatestVersion:            displayVersion(latestRaw),
		LatestVersionNormalized:  latestNormalized,
		ReleaseURL:               release.HTMLURL,
		PublishedAt:              release.PublishedAt,
		ReleaseNotes:             strings.TrimSpace(release.Body),
		UpdateAvailable:          updateAvailable,
		CurrentVersionIsSemver:   currentSemver != nil,
	}
}

func displayVersion(v string) string {
	if v == "" {
		return "unknown"
	}
	return v
}

func normalizeForSemver(raw string) (string, *semver.Version) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return trimmed, nil
	}
	normalized := strings.TrimPrefix(strings.TrimPrefix(trimmed, "v"), "V")
	parsed, err := semver.NewVersion(normalized)
	if err != nil {
		return normalized, nil
	}
	return normalized, parsed
}

func shouldCheckForUpgrade(now time.Time, state upgradeState) bool {
	if state.LastChecked.IsZero() {
		return true
	}
	return now.Sub(state.LastChecked) >= upgradeCheckInterval
}

func fetchLatestRelease() (*releaseInfo, error) {
	req, err := http.NewRequest(http.MethodGet, githubLatestReleaseURL, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Accept", "application/vnd.github+json")
	ua := fmt.Sprintf("LlamaFarmCLI/%s", strings.TrimSpace(Version))
	if ua == "LlamaFarmCLI/" {
		ua = "LlamaFarmCLI"
	}
	req.Header.Set("User-Agent", ua)

	if token := strings.TrimSpace(os.Getenv("LF_GITHUB_TOKEN")); token != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))
	}

	resp, err := getHTTPClient().Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		return nil, fmt.Errorf(
			"github releases responded with %d %s: %s",
			resp.StatusCode,
			http.StatusText(resp.StatusCode),
			strings.TrimSpace(string(body)),
		)
	}

	var payload struct {
		TagName     string `json:"tag_name"`
		HTMLURL     string `json:"html_url"`
		PublishedAt string `json:"published_at"`
		Draft       bool   `json:"draft"`
		Prerelease  bool   `json:"prerelease"`
		Body        string `json:"body"`
	}

	decoder := json.NewDecoder(resp.Body)
	if err := decoder.Decode(&payload); err != nil {
		return nil, fmt.Errorf("failed to decode GitHub release payload: %w", err)
	}

	// Validate required fields
	if payload.TagName == "" {
		return nil, errors.New("missing or invalid tag_name in release info")
	}
	if payload.HTMLURL == "" {
		return nil, errors.New("missing or invalid html_url in release info")
	}

	if payload.Draft || payload.Prerelease {
		return nil, errors.New("latest release is marked as draft or prerelease")
	}

	publishedAt, err := time.Parse(time.RFC3339, payload.PublishedAt)
	if err != nil {
		// Treat a parse failure as non-fatal; keep zero time.
		publishedAt = time.Time{}
	}

	return &releaseInfo{
		TagName:     payload.TagName,
		HTMLURL:     payload.HTMLURL,
		PublishedAt: publishedAt,
		Body:        payload.Body,
	}, nil
}

func readUpgradeState() (upgradeState, string, error) {
	path, err := getUpgradeStatePath()
	if err != nil {
		return upgradeState{}, "", err
	}

	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return upgradeState{}, path, nil
	}
	if err != nil {
		return upgradeState{}, path, err
	}

	var state upgradeState
	if err := json.Unmarshal(data, &state); err != nil {
		return upgradeState{}, path, err
	}

	return state, path, nil
}

func persistUpgradeState(path string, state upgradeState) error {
	if path == "" {
		return nil
	}

	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}

	if err := os.WriteFile(path, data, 0o644); err != nil {
		return err
	}

	return nil
}

func getUpgradeStatePath() (string, error) {
	if override := strings.TrimSpace(os.Getenv(upgradeStateEnvVar)); override != "" {
		if err := os.MkdirAll(filepath.Dir(override), 0o755); err != nil {
			return "", err
		}
		return override, nil
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}

	dir := filepath.Join(home, ".llamafarm", "cli")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}

	return filepath.Join(dir, upgradeStateFileName), nil
}
