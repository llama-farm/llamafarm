package orchestrator

import (
	"os"
	"os/exec"
	"testing"
	"time"
)

// TestIsProcessAlive_CurrentProcess tests isProcessAlive with the current process PID.
// The current process is guaranteed to be alive.
func TestIsProcessAlive_CurrentProcess(t *testing.T) {
	pid := os.Getpid()
	if !isProcessAlive(pid) {
		t.Errorf("current process (PID %d) should be alive", pid)
	}
}

// TestIsProcessAlive_DeadProcess tests isProcessAlive with a process that has exited.
func TestIsProcessAlive_DeadProcess(t *testing.T) {
	// Start a short-lived process
	cmd := exec.Command("true") // Unix: exits immediately with 0
	if err := cmd.Start(); err != nil {
		// Try "cmd /c exit 0" on Windows
		cmd = exec.Command("cmd", "/c", "exit", "0")
		if err := cmd.Start(); err != nil {
			t.Skipf("could not start test process: %v", err)
		}
	}

	pid := cmd.Process.Pid

	// Wait for it to complete
	if err := cmd.Wait(); err != nil {
		t.Logf("process exited with: %v", err)
	}

	// Give the OS a moment to clean up
	time.Sleep(100 * time.Millisecond)

	// The process should now be dead
	if isProcessAlive(pid) {
		t.Errorf("dead process (PID %d) should not be reported as alive (possible PID reuse, but unlikely)", pid)
	}
}

// TestIsProcessAlive_ParentProcess tests isProcessAlive with parent process PID.
func TestIsProcessAlive_ParentProcess(t *testing.T) {
	ppid := os.Getppid()
	// Parent process should be alive (it's running this test)
	if !isProcessAlive(ppid) {
		t.Errorf("parent process (PID %d) should be alive", ppid)
	}
}

// TestIsProcessAlive_LongRunningProcess tests isProcessAlive with a process we control.
func TestIsProcessAlive_LongRunningProcess(t *testing.T) {
	// Start a process that sleeps
	cmd := exec.Command("sleep", "10")
	if err := cmd.Start(); err != nil {
		// Try Windows equivalent
		cmd = exec.Command("cmd", "/c", "timeout", "/t", "10")
		if err := cmd.Start(); err != nil {
			t.Skipf("could not start sleep process: %v", err)
		}
	}

	pid := cmd.Process.Pid
	defer func() {
		cmd.Process.Kill()
		cmd.Wait()
	}()

	// Process should be alive
	if !isProcessAlive(pid) {
		t.Errorf("running process (PID %d) should be alive", pid)
	}

	// Kill it
	if err := cmd.Process.Kill(); err != nil {
		t.Fatalf("failed to kill process: %v", err)
	}
	cmd.Wait()

	// Give OS time to clean up
	time.Sleep(100 * time.Millisecond)

	// Process should now be dead
	if isProcessAlive(pid) {
		t.Errorf("killed process (PID %d) should not be reported as alive (possible PID reuse, but unlikely)", pid)
	}
}
