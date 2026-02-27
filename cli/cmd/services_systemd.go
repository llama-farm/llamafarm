package cmd

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"text/template"

	"github.com/llamafarm/cli/cmd/orchestrator"
	"github.com/llamafarm/cli/cmd/utils"
	"github.com/spf13/cobra"
)

// serviceNames defines the install order (server first, then deps).
var serviceNames = []string{"server", "universal-runtime", "rag"}

// systemd unit file template
var unitTemplate = template.Must(template.New("unit").Parse(`[Unit]
Description=LlamaFarm {{ .Description }}
{{ .After }}
{{- if .Requires }}
{{ .Requires }}
{{- end }}

[Service]
Type=simple
ExecStart={{ .ExecStart }}
WorkingDirectory={{ .WorkingDirectory }}
{{- if .UserGroup }}
{{ .UserGroup }}
{{- end }}
Environment="LF_DEPLOY_MODE=binary"
Environment="LF_DATA_DIR={{ .DataDir }}"
Environment="HF_HUB_DISABLE_PROGRESS_BARS=1"
{{- if .ExtraEnv }}
{{ .ExtraEnv }}
{{- end }}
KillSignal=SIGINT
TimeoutStopSec=10
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy={{ .WantedBy }}
`))

// systemdConfig holds resolved paths and settings for systemd operations.
type systemdConfig struct {
	systemMode  bool
	unitDir     string
	systemctl   string
	systemctlAs []string // extra args like "--user"
	dataDir     string
	user        string
	group       string
}

// unitParams holds the values passed to the unit template.
type unitParams struct {
	Description      string
	After            string
	Requires         string
	ExecStart        string
	WorkingDirectory string
	UserGroup        string
	DataDir          string
	ExtraEnv         string
	WantedBy         string
}

var servicesInstallCmd = &cobra.Command{
	Use:   "install",
	Short: "Install LlamaFarm services as systemd units",
	Long: `Generate systemd unit files, enable, and start all LlamaFarm services.

By default, installs user-level units (no root required). Use --system
to install system-wide units that run under a dedicated 'llamafarm' user.

Requires Linux with systemctl available. Service binaries must already
be installed (see 'lf deploy').

Examples:
  lf services install            # User-level units (~/.config/systemd/user/)
  sudo lf services install --system  # System-wide units (/etc/systemd/system/)`,
	Run: runServicesInstall,
}

var servicesUninstallCmd = &cobra.Command{
	Use:   "uninstall",
	Short: "Remove LlamaFarm systemd units",
	Long: `Stop, disable, and remove all LlamaFarm systemd unit files.

Examples:
  lf services uninstall            # Remove user-level units
  sudo lf services uninstall --system  # Remove system-wide units`,
	Run: runServicesUninstall,
}

func init() {
	servicesCmd.AddCommand(servicesInstallCmd)
	servicesCmd.AddCommand(servicesUninstallCmd)

	servicesInstallCmd.Flags().Bool("system", false, "Install system-wide units (requires root)")
	servicesUninstallCmd.Flags().Bool("system", false, "Remove system-wide units (requires root)")
}

// ---------------------------------------------------------------------------
// Install
// ---------------------------------------------------------------------------

func runServicesInstall(cmd *cobra.Command, args []string) {
	systemMode, _ := cmd.Flags().GetBool("system")

	if err := checkSystemdAvailable(); err != nil {
		utils.OutputError("%v", err)
		os.Exit(1)
	}

	cfg, err := newSystemdConfig(systemMode)
	if err != nil {
		utils.OutputError("%v", err)
		os.Exit(1)
	}

	// Verify all binaries exist
	for _, name := range serviceNames {
		if _, err := orchestrator.ResolveBinaryPath(name); err != nil {
			utils.OutputError("Binary for %s not found: %v", name, err)
			fmt.Fprintln(os.Stderr, "Install binaries first (see 'lf deploy').")
			os.Exit(1)
		}
	}

	// Check that unit files don't already exist
	for _, name := range serviceNames {
		unitPath := filepath.Join(cfg.unitDir, unitFileName(name))
		if _, err := os.Stat(unitPath); err == nil {
			utils.OutputError("Unit file already exists: %s", unitPath)
			fmt.Fprintln(os.Stderr, "Run 'lf services uninstall' first to remove existing units.")
			os.Exit(1)
		}
	}

	// System mode: create llamafarm user if needed and chown data dir
	if cfg.systemMode {
		if err := ensureSystemUser(cfg); err != nil {
			utils.OutputError("Failed to create system user: %v", err)
			os.Exit(1)
		}
		if err := chownDir(cfg.dataDir, cfg.user, cfg.group); err != nil {
			utils.OutputError("Failed to set ownership on data dir: %v", err)
			os.Exit(1)
		}
	}

	// Ensure unit directory exists (user mode may need to create it)
	if err := os.MkdirAll(cfg.unitDir, 0755); err != nil {
		utils.OutputError("Failed to create unit directory %s: %v", cfg.unitDir, err)
		os.Exit(1)
	}

	// Generate and write unit files
	for _, name := range serviceNames {
		params, err := buildUnitParams(name, cfg)
		if err != nil {
			utils.OutputError("Failed to build unit for %s: %v", name, err)
			os.Exit(1)
		}
		content, err := generateUnitFile(params)
		if err != nil {
			utils.OutputError("Failed to render unit for %s: %v", name, err)
			os.Exit(1)
		}
		unitPath := filepath.Join(cfg.unitDir, unitFileName(name))
		if err := os.WriteFile(unitPath, []byte(content), 0644); err != nil {
			utils.OutputError("Failed to write %s: %v", unitPath, err)
			os.Exit(1)
		}
		utils.OutputInfo("Wrote %s", unitPath)
	}

	// Reload, enable, start
	if err := runSystemctl(cfg, "daemon-reload"); err != nil {
		utils.OutputError("daemon-reload failed: %v", err)
		os.Exit(1)
	}

	unitNames := make([]string, len(serviceNames))
	for i, name := range serviceNames {
		unitNames[i] = unitFileName(name)
	}

	if err := runSystemctl(cfg, append([]string{"enable"}, unitNames...)...); err != nil {
		utils.OutputError("enable failed: %v", err)
		os.Exit(1)
	}

	// Start in dependency order
	for _, name := range serviceNames {
		if err := runSystemctl(cfg, "start", unitFileName(name)); err != nil {
			utils.OutputError("Failed to start %s: %v", name, err)
			os.Exit(1)
		}
	}

	utils.OutputSuccess("All services installed and started.")
	fmt.Println()
	printStatusHints(cfg)
}

// ---------------------------------------------------------------------------
// Uninstall
// ---------------------------------------------------------------------------

func runServicesUninstall(cmd *cobra.Command, args []string) {
	systemMode, _ := cmd.Flags().GetBool("system")

	if err := checkSystemdAvailable(); err != nil {
		utils.OutputError("%v", err)
		os.Exit(1)
	}

	cfg, err := newSystemdConfig(systemMode)
	if err != nil {
		utils.OutputError("%v", err)
		os.Exit(1)
	}

	// Check if any unit files exist
	found := false
	for _, name := range serviceNames {
		unitPath := filepath.Join(cfg.unitDir, unitFileName(name))
		if _, err := os.Stat(unitPath); err == nil {
			found = true
			break
		}
	}
	if !found {
		utils.OutputInfo("No LlamaFarm systemd units found. Nothing to uninstall.")
		return
	}

	// Stop in reverse dependency order (rag, universal-runtime, server)
	for i := len(serviceNames) - 1; i >= 0; i-- {
		unit := unitFileName(serviceNames[i])
		// Ignore errors on stop/disable — unit may already be inactive
		_ = runSystemctl(cfg, "stop", unit)
	}

	unitNames := make([]string, len(serviceNames))
	for i, name := range serviceNames {
		unitNames[i] = unitFileName(name)
	}
	_ = runSystemctl(cfg, append([]string{"disable"}, unitNames...)...)

	// Remove unit files
	for _, name := range serviceNames {
		unitPath := filepath.Join(cfg.unitDir, unitFileName(name))
		if err := os.Remove(unitPath); err != nil && !os.IsNotExist(err) {
			utils.OutputWarning("Failed to remove %s: %v", unitPath, err)
		} else if err == nil {
			utils.OutputInfo("Removed %s", unitPath)
		}
	}

	if err := runSystemctl(cfg, "daemon-reload"); err != nil {
		utils.OutputWarning("daemon-reload failed: %v", err)
	}

	utils.OutputSuccess("LlamaFarm systemd units removed.")
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func checkSystemdAvailable() error {
	if runtime.GOOS != "linux" {
		return fmt.Errorf("systemd service installation is only supported on Linux")
	}
	if _, err := exec.LookPath("systemctl"); err != nil {
		return fmt.Errorf("systemctl not found in PATH; is systemd installed?")
	}
	return nil
}

func newSystemdConfig(systemMode bool) (*systemdConfig, error) {
	dataDir, err := utils.GetLFDataDir()
	if err != nil {
		return nil, fmt.Errorf("could not determine data directory: %w", err)
	}

	cfg := &systemdConfig{
		systemMode: systemMode,
		dataDir:    dataDir,
	}

	if systemMode {
		// Require root
		if os.Geteuid() != 0 {
			return nil, fmt.Errorf("system-wide install requires root; run with sudo")
		}
		cfg.unitDir = "/etc/systemd/system"
		cfg.systemctl = "systemctl"
		cfg.user = "llamafarm"
		cfg.group = "llamafarm"
	} else {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("could not determine home directory: %w", err)
		}
		cfg.unitDir = filepath.Join(homeDir, ".config", "systemd", "user")
		cfg.systemctl = "systemctl"
		cfg.systemctlAs = []string{"--user"}

		currentUser, err := user.Current()
		if err != nil {
			return nil, fmt.Errorf("could not determine current user: %w", err)
		}
		cfg.user = currentUser.Username
		cfg.group = currentUser.Username
	}

	return cfg, nil
}

func buildUnitParams(serviceName string, cfg *systemdConfig) (*unitParams, error) {
	svcDef, ok := orchestrator.ServiceGraph[serviceName]
	if !ok {
		return nil, fmt.Errorf("unknown service: %s", serviceName)
	}

	binaryPath, err := orchestrator.ResolveBinaryPath(serviceName)
	if err != nil {
		return nil, err
	}

	// Build After= and Requires= from dependencies
	var afterParts []string
	afterParts = append(afterParts, "network-online.target")
	for _, dep := range svcDef.Dependencies {
		afterParts = append(afterParts, unitFileName(dep))
	}
	afterLine := "Wants=network-online.target\nAfter=" + strings.Join(afterParts, " ")

	requiresLine := ""
	if len(svcDef.Dependencies) > 0 {
		depUnits := make([]string, len(svcDef.Dependencies))
		for i, dep := range svcDef.Dependencies {
			depUnits[i] = unitFileName(dep)
		}
		requiresLine = "Requires=" + strings.Join(depUnits, " ")
	}

	// Build extra environment lines from ServiceGraph env, skipping
	// LOG_FILE (journal handles it) and empty-value "inherit" vars.
	// Sort keys for deterministic unit file output.
	var envKeys []string
	for key := range svcDef.Env {
		envKeys = append(envKeys, key)
	}
	sort.Strings(envKeys)

	var envLines []string
	for _, key := range envKeys {
		val := svcDef.Env[key]
		if key == "LOG_FILE" {
			continue
		}
		if val == "" {
			continue
		}
		// Skip vars that reference env-var placeholders (inherit vars)
		if strings.HasPrefix(val, "${") {
			continue
		}
		envLines = append(envLines, fmt.Sprintf("Environment=%q", key+"="+val))
	}

	// Description map
	descMap := map[string]string{
		"server":            "Server",
		"universal-runtime": "Universal Runtime",
		"rag":               "RAG Worker",
	}
	description := descMap[serviceName]
	if description == "" {
		description = serviceName
	}

	// User/Group line (system mode only)
	userGroup := ""
	if cfg.systemMode {
		userGroup = fmt.Sprintf("User=%s\nGroup=%s", cfg.user, cfg.group)
	}

	wantedBy := "default.target"
	if cfg.systemMode {
		wantedBy = "multi-user.target"
	}

	return &unitParams{
		Description:      description,
		After:            afterLine,
		Requires:         requiresLine,
		ExecStart:        binaryPath,
		WorkingDirectory: cfg.dataDir,
		UserGroup:        userGroup,
		DataDir:          cfg.dataDir,
		ExtraEnv:         strings.Join(envLines, "\n"),
		WantedBy:         wantedBy,
	}, nil
}

func generateUnitFile(params *unitParams) (string, error) {
	var buf bytes.Buffer
	if err := unitTemplate.Execute(&buf, params); err != nil {
		return "", err
	}
	return buf.String(), nil
}

func unitFileName(serviceName string) string {
	return "llamafarm-" + strings.ReplaceAll(serviceName, "universal-", "") + ".service"
}

func runSystemctl(cfg *systemdConfig, args ...string) error {
	fullArgs := make([]string, 0, len(cfg.systemctlAs)+len(args))
	fullArgs = append(fullArgs, cfg.systemctlAs...)
	fullArgs = append(fullArgs, args...)
	cmd := exec.Command(cfg.systemctl, fullArgs...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func ensureSystemUser(cfg *systemdConfig) error {
	// Check if user already exists
	if _, err := user.Lookup(cfg.user); err == nil {
		return nil // user exists
	}

	utils.OutputInfo("Creating system user %s...", cfg.user)
	cmd := exec.Command("useradd",
		"--system",
		"--user-group",
		"--home-dir", cfg.dataDir,
		"--shell", "/usr/sbin/nologin",
		cfg.user,
	)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func chownDir(dir, userName, group string) error {
	cmd := exec.Command("chown", "-R", userName+":"+group, dir)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func printStatusHints(cfg *systemdConfig) {
	userFlag := ""
	if !cfg.systemMode {
		userFlag = " --user"
	}

	fmt.Println("Check status:")
	for _, name := range serviceNames {
		fmt.Printf("  systemctl%s status %s\n", userFlag, unitFileName(name))
	}
	fmt.Println()
	fmt.Println("View logs:")
	for _, name := range serviceNames {
		unit := unitFileName(name)
		if cfg.systemMode {
			fmt.Printf("  journalctl -u %s -f\n", unit)
		} else {
			fmt.Printf("  journalctl --user-unit %s -f\n", unit)
		}
	}
}
