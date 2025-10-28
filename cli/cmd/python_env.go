package cmd

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
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
		if debug {
			logDebug("Python already available via UV: " + pythonPath)
		}
		return pythonPath, nil
	}

	// Install Python via UV
	OutputProgress("Installing Python %s via UV (this may take a few minutes)...\n", pythonVersion)

	cmd := exec.Command(uvPath, "python", "install", pythonVersion)
	cmd.Dir = m.pythonDir // Run from Python directory context for proper isolation
	cmd.Env = m.getEnv()
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed to install Python: %w\nOutput: %s", err, string(output))
	}

	if debug {
		logDebug(fmt.Sprintf("UV Python install output: %s", string(output)))
	}

	// Find the installed Python
	pythonPath, err = m.findPython(uvPath)
	if err != nil {
		return "", fmt.Errorf("Python installation succeeded but cannot find Python: %w", err)
	}

	OutputProgress("Python %s installed successfully\n", pythonVersion)
	return pythonPath, nil
}

// findPython finds the Python executable managed by UV
func (m *PythonEnvManager) findPython(uvPath string) (string, error) {
	cmd := exec.Command(uvPath, "python", "find", pythonVersion)
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

	if debug {
		logDebug(fmt.Sprintf("Python validation successful: %s", version))
	}

	return nil
}

// RunWithUV runs a command using UV's Python environment
// This is the preferred way to run Python commands as it handles all environment setup
func (m *PythonEnvManager) RunWithUV(workDir string, args ...string) *exec.Cmd {
	uvPath := m.uvManager.GetUVPath()

	// Build the command: uv run python <args>
	fullArgs := append([]string{"run", "python"}, args...)

	cmd := exec.Command(uvPath, fullArgs...)
	cmd.Dir = workDir
	cmd.Env = m.getEnv()

	return cmd
}

// getEnv returns the environment variables for UV commands
// This ensures UV uses our managed directories
func (m *PythonEnvManager) getEnv() []string {
	env := os.Environ()

	// Add UV-specific environment variables if needed
	llamafarmDir := filepath.Join(m.homeDir, ".llamafarm")

	// Set UV_CACHE_DIR to keep all UV data in our directory
	env = append(env, fmt.Sprintf("UV_CACHE_DIR=%s", filepath.Join(llamafarmDir, "uv-cache")))

	// Set UV_PYTHON_INSTALL_DIR for Python installations
	env = append(env, fmt.Sprintf("UV_PYTHON_INSTALL_DIR=%s", m.pythonDir))

	return env
}

// GetEnvForProcess returns environment variables that should be used when running processes
func (m *PythonEnvManager) GetEnvForProcess() []string {
	return m.getEnv()
}
