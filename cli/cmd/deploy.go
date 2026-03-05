package cmd

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/llamafarm/cli/cmd/config"
	"github.com/llamafarm/cli/cmd/utils"
	"github.com/spf13/cobra"
)

var deployFlags struct {
	withData   bool
	skipModels bool
	dryRun     bool
}

var deployCmd = &cobra.Command{
	Use:   "deploy [environment]",
	Short: "Deploy project to a LlamaFarm server",
	Long: `Deploy the current project config to a LlamaFarm server and trigger model downloads.

When called with no arguments, deploys to the local server (localhost:14345).
When called with an environment name, uses the server_url from the environments
section of llamafarm.yaml.

Examples:
  # Deploy to local server
  lf deploy

  # Deploy to a named environment
  lf deploy staging

  # Deploy to an ad-hoc server
  lf deploy --server-url http://10.0.1.50:14345

  # Deploy with dataset upload
  lf deploy production --with-data

  # Dry run to see what would happen
  lf deploy staging --dry-run`,
	Args: cobra.MaximumNArgs(1),
	RunE: runDeploy,
}

func init() {
	rootCmd.AddCommand(deployCmd)

	deployCmd.Flags().BoolVar(&deployFlags.withData, "with-data", false, "Upload and ingest dataset documents")
	deployCmd.Flags().BoolVar(&deployFlags.skipModels, "skip-models", false, "Skip model download step")
	deployCmd.Flags().BoolVar(&deployFlags.dryRun, "dry-run", false, "Show what would happen without executing")
}

func runDeploy(cmd *cobra.Command, args []string) error {
	startTime := time.Now()

	// Load project config
	cwd := utils.GetEffectiveCWD()
	cfg, err := config.LoadConfig(cwd)
	if err != nil {
		return fmt.Errorf("failed to load project config: %w", err)
	}

	projectInfo, err := cfg.GetProjectInfo()
	if err != nil {
		return err
	}

	// Resolve target server URL and deploy settings
	targetURL, deployModels, deployData, envName, err := resolveDeployTarget(cfg, args)
	if err != nil {
		return err
	}

	// Apply flag overrides
	if deployFlags.withData {
		deployData = true
	}
	if deployFlags.skipModels {
		deployModels = false
	}

	// Display deploy plan
	if envName != "" {
		fmt.Printf("Deploying %s/%s to %s (%s)\n", projectInfo.Namespace, projectInfo.Project, envName, targetURL)
	} else {
		fmt.Printf("Deploying %s/%s to %s\n", projectInfo.Namespace, projectInfo.Project, targetURL)
	}

	if deployFlags.dryRun {
		return printDryRun(cfg, targetURL, projectInfo, deployModels, deployData)
	}

	// Step 1: Health check
	fmt.Print("Checking server health... ")
	if err := healthCheck(targetURL); err != nil {
		fmt.Println("FAILED")
		return fmt.Errorf("server at %s is not reachable: %w", targetURL, err)
	}
	fmt.Println("OK")

	// Step 2: Upsert project config
	fmt.Print("Pushing project config... ")
	if err := upsertProjectConfig(targetURL, projectInfo.Namespace, projectInfo.Project, cfg); err != nil {
		fmt.Println("FAILED")
		return fmt.Errorf("failed to push config: %w", err)
	}
	fmt.Println("OK")

	// Step 3: Model downloads
	modelResults := []modelDeployResult{}
	if deployModels && cfg.Runtime.Models != nil && len(cfg.Runtime.Models) > 0 {
		results, err := deployModelsToServer(targetURL, cfg.Runtime.Models)
		if err != nil {
			return fmt.Errorf("model deployment failed: %w", err)
		}
		modelResults = results
	}

	// Step 4: Dataset upload (Phase 3 placeholder)
	if deployData {
		fmt.Println("\nDataset upload not yet implemented (Phase 3)")
	}

	// Step 5: Summary
	printDeploySummary(projectInfo, targetURL, envName, modelResults, startTime)

	return nil
}

// resolveDeployTarget determines the server URL and deploy settings from environment name,
// --server-url flag, or defaults.
func resolveDeployTarget(cfg *config.LlamaFarmConfig, args []string) (targetURL string, deployModels bool, deployData bool, envName string, err error) {
	// Defaults
	deployModels = true
	deployData = false

	if len(args) > 0 {
		// Named environment
		envName = args[0]
		dc, resolveErr := cfg.ResolveEnvironment(envName)
		if resolveErr != nil {
			return "", false, false, "", resolveErr
		}
		targetURL = dc.ServerURL
		deployModels = dc.DeployModels
		deployData = dc.DeployData
	} else {
		// Use --server-url flag (which defaults to localhost:14345)
		targetURL = serverURL
	}

	return targetURL, deployModels, deployData, envName, nil
}

// healthCheck verifies the target server is reachable.
func healthCheck(targetURL string) error {
	url := fmt.Sprintf("%s/health", strings.TrimSuffix(targetURL, "/"))

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return err
	}

	resp, err := utils.GetHTTPClient().Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check returned status %d", resp.StatusCode)
	}
	return nil
}

// upsertProjectConfig pushes the config to the remote server, creating the project if needed.
func upsertProjectConfig(targetURL, namespace, project string, cfg *config.LlamaFarmConfig) error {
	baseURL := strings.TrimSuffix(targetURL, "/")

	// Strip environments before sending
	stripped := cfg.StripEnvironments()

	configJSON, err := json.Marshal(map[string]interface{}{
		"config": stripped,
	})
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	// Try PUT first (update existing)
	putURL := fmt.Sprintf("%s/v1/projects/%s/%s", baseURL, namespace, project)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "PUT", putURL, bytes.NewReader(configJSON))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := utils.GetHTTPClient().Do(req)
	if err != nil {
		return err
	}
	resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		return nil
	}

	if resp.StatusCode == http.StatusNotFound {
		// Project doesn't exist, create it first
		utils.LogDebug("Project not found on remote, creating...")
		if err := createProject(baseURL, namespace, project); err != nil {
			return fmt.Errorf("failed to create project: %w", err)
		}

		// Now retry the PUT
		ctx2, cancel2 := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel2()

		req2, err := http.NewRequestWithContext(ctx2, "PUT", putURL, bytes.NewReader(configJSON))
		if err != nil {
			return err
		}
		req2.Header.Set("Content-Type", "application/json")

		resp2, err := utils.GetHTTPClient().Do(req2)
		if err != nil {
			return err
		}
		resp2.Body.Close()

		if resp2.StatusCode != http.StatusOK {
			return fmt.Errorf("failed to update config after project creation (status %d)", resp2.StatusCode)
		}
		return nil
	}

	return fmt.Errorf("unexpected status %d from server", resp.StatusCode)
}

// createProject creates a new project on the remote server.
func createProject(baseURL, namespace, project string) error {
	url := fmt.Sprintf("%s/v1/projects/%s", baseURL, namespace)

	body, err := json.Marshal(map[string]string{
		"name": project,
	})
	if err != nil {
		return err
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := utils.GetHTTPClient().Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		bodyBytes, _ := io.ReadAll(resp.Body)
		utils.LogDebug(fmt.Sprintf("create project response body: %s", string(bodyBytes)))
		return fmt.Errorf("failed to create project (server returned status %d)", resp.StatusCode)
	}

	return nil
}

// modelDeployResult tracks the outcome of a single model deployment.
type modelDeployResult struct {
	Name   string
	Model  string
	Status string // "downloaded", "cached", "failed"
	Error  error
}

// deployModelsToServer triggers parallel model downloads on the remote server.
func deployModelsToServer(targetURL string, models []config.LlamaFarmConfigRuntimeModelsElem) ([]modelDeployResult, error) {
	baseURL := strings.TrimSuffix(targetURL, "/")

	// Filter to universal provider models (other providers don't need download)
	var universalModels []config.LlamaFarmConfigRuntimeModelsElem
	for _, m := range models {
		if m.Provider == config.LlamaFarmConfigRuntimeModelsElemProviderUniversal {
			universalModels = append(universalModels, m)
		}
	}

	if len(universalModels) == 0 {
		fmt.Println("\nNo universal models to download")
		return nil, nil
	}

	fmt.Printf("\nDownloading %d model(s)...\n", len(universalModels))

	// Pre-check disk space for each model
	var diskWarnings []string
	for _, m := range universalModels {
		if err := validateModelDiskSpace(baseURL, m.Model); err != nil {
			diskWarnings = append(diskWarnings, fmt.Sprintf("  %s: %v", m.Model, err))
		}
	}
	if len(diskWarnings) > 0 {
		fmt.Println("  Disk space warnings:")
		for _, w := range diskWarnings {
			fmt.Println(w)
		}
		fmt.Print("  Continue anyway? [y/N] ")
		var answer string
		fmt.Scanln(&answer)
		if answer != "y" && answer != "Y" {
			return nil, fmt.Errorf("aborted by user due to disk space warnings")
		}
	}

	// Download models in parallel
	var wg sync.WaitGroup
	results := make([]modelDeployResult, len(universalModels))

	for i, m := range universalModels {
		wg.Add(1)
		go func(idx int, model config.LlamaFarmConfigRuntimeModelsElem) {
			defer wg.Done()
			result := modelDeployResult{
				Name:  model.Name,
				Model: model.Model,
			}
			status, err := pullModelFromRemote(baseURL, model.Model, model.Name)
			if err != nil {
				result.Status = "failed"
				result.Error = err
			} else {
				result.Status = status
			}
			results[idx] = result
		}(i, m)
	}

	wg.Wait()

	// Check if any models failed
	var failedNames []string
	for _, r := range results {
		if r.Status == "failed" {
			failedNames = append(failedNames, r.Name)
		}
	}
	if len(failedNames) == len(results) {
		return results, fmt.Errorf("all model downloads failed")
	}

	return results, nil
}

// progressMu serializes terminal progress output across concurrent model downloads.
var progressMu sync.Mutex

// pullModelFromRemote triggers a model download on the remote server and streams progress.
// Returns the status ("downloaded" or "cached") and any error.
func pullModelFromRemote(baseURL, modelID, displayName string) (string, error) {
	url := fmt.Sprintf("%s/v1/models/download", baseURL)

	requestBody, err := json.Marshal(map[string]string{
		"provider":   "universal",
		"model_name": modelID,
	})
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Minute)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(requestBody))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	resp, err := utils.GetHTTPClientWithTimeout(0).Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to connect to server: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("server returned status %d", resp.StatusCode)
	}

	reader := bufio.NewReader(resp.Body)
	prefix := fmt.Sprintf("  [%s]", displayName)
	status := "downloaded"

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return "", fmt.Errorf("error reading response: %w", err)
		}

		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		var event downloadEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		if event.Keepalive {
			continue
		}

		switch event.Event {
		case "init":
			if event.TotalSize > 0 {
				progressMu.Lock()
				fmt.Printf("%s %s (%s)\n", prefix, event.ModelID, utils.FormatBytes(event.TotalSize))
				progressMu.Unlock()
			}
		case "progress":
			if event.Total > 1024*1024 {
				rateStr := ""
				if event.BytesPerSec > 0 {
					rateStr = fmt.Sprintf(" @ %s", utils.FormatTransferRate(event.BytesPerSec))
				}
				progressMu.Lock()
				fmt.Printf("\r%s %.1f%%%s", prefix, event.Percent, rateStr)
				os.Stdout.Sync()
				progressMu.Unlock()
			}
		case "cached":
			progressMu.Lock()
			fmt.Printf("%s cached\n", prefix)
			progressMu.Unlock()
			status = "cached"
		case "end":
			progressMu.Lock()
			fmt.Printf("\r%s 100%%\n", prefix)
			progressMu.Unlock()
		case "done":
			return status, nil
		case "error":
			return "", fmt.Errorf("%s", event.Message)
		}
	}

	return "", fmt.Errorf("download incomplete: connection closed before completion")
}

// validateModelDiskSpace checks if the server has enough disk space for a model.
func validateModelDiskSpace(baseURL, modelID string) error {
	url := fmt.Sprintf("%s/v1/models/validate-download", baseURL)

	body, err := json.Marshal(map[string]string{
		"provider":   "universal",
		"model_name": modelID,
	})
	if err != nil {
		return err
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := utils.GetHTTPClient().Do(req)
	if err != nil {
		return fmt.Errorf("disk space check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errResp struct {
			Detail string `json:"detail"`
		}
		json.NewDecoder(resp.Body).Decode(&errResp)
		if errResp.Detail != "" {
			return fmt.Errorf("%s", errResp.Detail)
		}
		return fmt.Errorf("disk space validation returned status %d", resp.StatusCode)
	}

	return nil
}

// printDryRun displays what would happen during deploy without executing.
func printDryRun(cfg *config.LlamaFarmConfig, targetURL string, projectInfo *config.ProjectInfo, deployModels, deployData bool) error {
	fmt.Println("\n[DRY RUN] Actions that would be taken:")
	fmt.Printf("  1. Health check: GET %s/health\n", targetURL)
	fmt.Printf("  2. Push config: PUT /v1/projects/%s/%s (or create if not found)\n", projectInfo.Namespace, projectInfo.Project)

	if deployModels && cfg.Runtime.Models != nil {
		fmt.Printf("  3. Download models:\n")
		for _, m := range cfg.Runtime.Models {
			if m.Provider == config.LlamaFarmConfigRuntimeModelsElemProviderUniversal {
				fmt.Printf("     - %s (%s)\n", m.Name, m.Model)
			}
		}
	} else {
		fmt.Println("  3. Model downloads: SKIPPED")
	}

	if deployData {
		fmt.Println("  4. Dataset upload: would upload and ingest datasets")
	} else {
		fmt.Println("  4. Dataset upload: SKIPPED")
	}

	return nil
}

// printDeploySummary displays the deploy completion summary.
func printDeploySummary(projectInfo *config.ProjectInfo, targetURL, envName string, results []modelDeployResult, startTime time.Time) {
	elapsed := time.Since(startTime)

	fmt.Println("\n--- Deploy Summary ---")
	fmt.Printf("Project:  %s/%s\n", projectInfo.Namespace, projectInfo.Project)
	if envName != "" {
		fmt.Printf("Target:   %s (%s)\n", envName, targetURL)
	} else {
		fmt.Printf("Target:   %s\n", targetURL)
	}
	fmt.Println("Config:   pushed")

	if len(results) > 0 {
		downloaded := 0
		cached := 0
		failed := 0
		for _, r := range results {
			switch r.Status {
			case "downloaded":
				downloaded++
			case "cached":
				cached++
			case "failed":
				failed++
			}
		}
		fmt.Printf("Models:   %d downloaded, %d cached, %d failed\n", downloaded, cached, failed)

		for _, r := range results {
			switch r.Status {
			case "downloaded":
				fmt.Printf("  + %s (%s)\n", r.Name, r.Model)
			case "cached":
				fmt.Printf("  = %s (%s) cached\n", r.Name, r.Model)
			case "failed":
				fmt.Printf("  ! %s (%s) FAILED: %v\n", r.Name, r.Model, r.Error)
			}
		}
	}

	fmt.Printf("Time:     %s\n", elapsed.Round(time.Second))
}
