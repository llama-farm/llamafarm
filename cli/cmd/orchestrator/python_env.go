package orchestrator

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/llamafarm/cli/cmd/utils"
)

const (
	// Python version to install
	pythonVersion = "3.12"
)

// PythonEnvManager handles Python environment setup via UV
type PythonEnvManager struct {
	uvManager *UVManager
	homeDir   string
	pythonDir string
}

// NewPythonEnvManager creates a new Python environment manager
func NewPythonEnvManager(uvManager *UVManager) (*PythonEnvManager, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("failed to get user home directory: %w", err)
	}

	pythonDir := filepath.Join(homeDir, ".llamafarm", "python")

	return &PythonEnvManager{
		uvManager: uvManager,
		homeDir:   homeDir,
		pythonDir: pythonDir,
	}, nil
}

// EnsurePython ensures Python is installed via UV and returns the python executable path
func (m *PythonEnvManager) EnsurePython() (string, error) {
	// First ensure UV is available
	uvPath, err := m.uvManager.EnsureUV()
	if err != nil {
		return "", fmt.Errorf("UV not available: %w", err)
	}

	// Ensure Python directory exists
	if err := os.MkdirAll(m.pythonDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create Python directory: %w", err)
	}

	// Check if Python is already installed
	pythonPath, err := m.findPython(uvPath)
	if err == nil && pythonPath != "" {
		utils.LogDebug("Python already available via UV: " + pythonPath)
		return pythonPath, nil
	}

	// Install Python via UV
	utils.LogDebug(fmt.Sprintf("Installing Python %s via UV (this may take a few minutes)...", pythonVersion))

	cmd := exec.Command(uvPath, "python", "install", pythonVersion)
	cmd.Dir = m.pythonDir // Run from Python directory context for proper isolation
	cmd.Env = m.getEnv()
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed to install Python: %w\nOutput: %s", err, string(output))
	}

	utils.LogDebug(fmt.Sprintf("UV Python install output: %s", string(output)))

	// Find the installed Python
	pythonPath, err = m.findPython(uvPath)
	if err != nil {
		return "", fmt.Errorf("Python installation succeeded but cannot find Python: %w", err)
	}

	utils.LogDebug(fmt.Sprintf("Python %s installed successfully", pythonVersion))
	return pythonPath, nil
}

// findPython finds the Python executable managed by UV
func (m *PythonEnvManager) findPython(uvPath string) (string, error) {
	cmd := exec.Command(uvPath, "--managed-python", "python", "find", pythonVersion)
	cmd.Dir = m.pythonDir // Run from Python directory context
	cmd.Env = m.getEnv()
	output, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("Python not found: %w", err)
	}

	pythonPath := strings.TrimSpace(string(output))
	if pythonPath == "" {
		return "", fmt.Errorf("Python path is empty")
	}

	// Verify the Python executable exists
	if _, err := os.Stat(pythonPath); err != nil {
		return "", fmt.Errorf("Python executable not found at %s: %w", pythonPath, err)
	}

	return pythonPath, nil
}

// GetPythonPath returns the path to the Python executable
func (m *PythonEnvManager) GetPythonPath() (string, error) {
	uvPath := m.uvManager.GetUVPath()
	return m.findPython(uvPath)
}

// ValidatePythonInstallation validates that Python is properly installed and functional
func (m *PythonEnvManager) ValidatePythonInstallation() error {
	pythonPath, err := m.GetPythonPath()
	if err != nil {
		return fmt.Errorf("Python not found: %w", err)
	}

	// Try to run Python with --version
	cmd := exec.Command(pythonPath, "--version")
	cmd.Dir = m.pythonDir // Run from Python directory context
	cmd.Env = m.getEnv()
	output, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("Python is not functional: %w", err)
	}

	version := strings.TrimSpace(string(output))
	if !strings.Contains(version, pythonVersion) {
		return fmt.Errorf("unexpected Python version: %s (expected %s)", version, pythonVersion)
	}

	utils.LogDebug(fmt.Sprintf("Python validation successful: %s", version))

	return nil
}

// getEnv returns the environment variables for UV commands
// This ensures UV uses our managed directories
func (m *PythonEnvManager) getEnv() []string {
	// Start with the current environment
	env := os.Environ()

	// Filter out Python-related environment variables that could interfere
	// with UV's managed Python environment
	filteredEnv := make([]string, 0, len(env))
	pythonEnvVars := map[string]bool{
		"VIRTUAL_ENV":            true,
		"PYTHONHOME":             true,
		"PYTHONPATH":             true,
		"PYTHONSTARTUP":          true,
		"PYTHONEXECUTABLE":       true,
		"PYTHONUSERBASE":         true,
		"CONDA_DEFAULT_ENV":      true,
		"CONDA_PREFIX":           true,
		"CONDA_PYTHON_EXE":       true,
		"PYENV_VERSION":          true,
		"PYENV_VIRTUAL_ENV":      true,
		"PIPENV_ACTIVE":          true,
		"POETRY_ACTIVE":          true,
		"PDM_PYTHON":             true,
	}

	for _, e := range env {
		parts := strings.SplitN(e, "=", 2)
		if len(parts) == 0 {
			continue
		}
		varName := parts[0]

		// Case-insensitive comparison to prevent bypass attacks (e.g., Virtual_Env, virtual_env)
		// This is critical on Windows where env vars are case-insensitive
		if _, isPythonEnv := pythonEnvVars[strings.ToUpper(varName)]; !isPythonEnv {
			filteredEnv = append(filteredEnv, e)
		}
	}

	env = filteredEnv

	// Add UV-specific environment variables if needed
	llamafarmDir := filepath.Join(m.homeDir, ".llamafarm")

	// Set UV_CACHE_DIR to keep all UV data in our directory
	env = append(env, fmt.Sprintf("UV_CACHE_DIR=%s", filepath.Join(llamafarmDir, "uv-cache")))

	// Set UV_PYTHON_INSTALL_DIR for Python installations
	env = append(env, fmt.Sprintf("UV_PYTHON_INSTALL_DIR=%s", m.pythonDir))

	// Ensure UV bin directory is first in PATH
	uvBinDir := m.uvManager.binDir
	currentPath := os.Getenv("PATH")
	newPath := uvBinDir + string(os.PathListSeparator) + currentPath

	// Update or add PATH in environment
	pathUpdated := false
	for i, e := range env {
		if strings.HasPrefix(e, "PATH=") {
			env[i] = "PATH=" + newPath
			pathUpdated = true
			break
		}
	}
	if !pathUpdated {
		env = append(env, "PATH="+newPath)
	}

	return env
}

// GetEnvForProcess returns environment variables that should be used when running processes
func (m *PythonEnvManager) GetEnvForProcess() []string {
	env := m.getEnv()
	lfDataDir, err := utils.GetLFDataDir()
	if err == nil {
		env = append(env, fmt.Sprintf("LF_DATA_DIR=%s", lfDataDir))
	}
	return env
}
