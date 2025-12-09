package orchestrator

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

// newTestProcessManager creates a ProcessManager with a temporary directory for testing.
// This isolates tests from the real system and ensures clean state.
func newTestProcessManager(t *testing.T) *ProcessManager {
	t.Helper()
	tmpDir := t.TempDir()

	logsDir := filepath.Join(tmpDir, "logs")
	if err := os.MkdirAll(logsDir, 0755); err != nil {
		t.Fatalf("failed to create logs directory: %v", err)
	}

	pidsDir := filepath.Join(tmpDir, "pids")
	if err := os.MkdirAll(pidsDir, 0755); err != nil {
		t.Fatalf("failed to create pids directory: %v", err)
	}

	return &ProcessManager{
		logsDir:   logsDir,
		pidsDir:   pidsDir,
		processes: make(map[string]*ProcessInfo),
	}
}

// writePIDFile writes a PID to a service's PID file for testing.
func writePIDFile(t *testing.T, pidsDir, serviceName string, pid int) {
	t.Helper()
	pidFile := filepath.Join(pidsDir, fmt.Sprintf("%s.pid", serviceName))
	if err := os.WriteFile(pidFile, []byte(fmt.Sprintf("%d", pid)), 0644); err != nil {
		t.Fatalf("failed to write PID file: %v", err)
	}
}

// findDeadPID returns a PID that is guaranteed not to be running.
// It scans from 99999 down to find an unused PID.
func findDeadPID() int {
	for pid := 99999; pid > 1000; pid-- {
		if !isProcessAlive(pid) {
			return pid
		}
	}
	// Fallback - this PID is very unlikely to exist
	return 99999
}

// TestReadPIDFile_ValidPID tests reading a valid PID from a PID file.
func TestReadPIDFile_ValidPID(t *testing.T) {
	pm := newTestProcessManager(t)

	// Write a valid PID file
	expectedPID := 12345
	writePIDFile(t, pm.pidsDir, "test-service", expectedPID)

	// Read it back
	pid, found := pm.ReadPIDFile("test-service")
	if !found {
		t.Fatal("expected to find PID file, but didn't")
	}
	if pid != expectedPID {
		t.Errorf("expected PID %d, got %d", expectedPID, pid)
	}
}

// TestReadPIDFile_NonExistent tests reading a PID file that doesn't exist.
func TestReadPIDFile_NonExistent(t *testing.T) {
	pm := newTestProcessManager(t)

	pid, found := pm.ReadPIDFile("nonexistent-service")
	if found {
		t.Errorf("expected not to find PID file, but found PID %d", pid)
	}
}

// TestReadPIDFile_InvalidContent tests reading a PID file with invalid content.
func TestReadPIDFile_InvalidContent(t *testing.T) {
	pm := newTestProcessManager(t)

	// Write invalid content to PID file
	pidFile := filepath.Join(pm.pidsDir, "invalid-service.pid")
	if err := os.WriteFile(pidFile, []byte("not-a-number"), 0644); err != nil {
		t.Fatalf("failed to write PID file: %v", err)
	}

	pid, found := pm.ReadPIDFile("invalid-service")
	if found {
		t.Errorf("expected not to find valid PID, but found %d", pid)
	}
}

// TestReadPIDFile_EmptyFile tests reading an empty PID file.
func TestReadPIDFile_EmptyFile(t *testing.T) {
	pm := newTestProcessManager(t)

	// Write empty PID file
	pidFile := filepath.Join(pm.pidsDir, "empty-service.pid")
	if err := os.WriteFile(pidFile, []byte(""), 0644); err != nil {
		t.Fatalf("failed to write PID file: %v", err)
	}

	pid, found := pm.ReadPIDFile("empty-service")
	if found {
		t.Errorf("expected not to find valid PID in empty file, but found %d", pid)
	}
}

// TestRemovePIDFile tests removing a PID file.
func TestRemovePIDFile(t *testing.T) {
	pm := newTestProcessManager(t)

	// Write a PID file
	writePIDFile(t, pm.pidsDir, "remove-test", 12345)

	// Verify it exists
	pidFile := filepath.Join(pm.pidsDir, "remove-test.pid")
	if _, err := os.Stat(pidFile); os.IsNotExist(err) {
		t.Fatal("PID file should exist before removal")
	}

	// Remove it
	pm.removePIDFile("remove-test")

	// Verify it's gone
	if _, err := os.Stat(pidFile); !os.IsNotExist(err) {
		t.Error("PID file should not exist after removal")
	}
}

// TestRemovePIDFile_NonExistent tests removing a PID file that doesn't exist.
func TestRemovePIDFile_NonExistent(t *testing.T) {
	pm := newTestProcessManager(t)

	// This should not panic or error
	pm.removePIDFile("nonexistent-service")
}

// TestGetPIDFilePath tests the PID file path generation.
func TestGetPIDFilePath(t *testing.T) {
	pm := newTestProcessManager(t)

	path := pm.getPIDFilePath("my-service")
	expected := filepath.Join(pm.pidsDir, "my-service.pid")
	if path != expected {
		t.Errorf("expected path %s, got %s", expected, path)
	}
}

// TestGetLockFilePath tests the lock file path generation.
func TestGetLockFilePath(t *testing.T) {
	pm := newTestProcessManager(t)

	path := pm.getLockFilePath("my-service")
	expected := filepath.Join(pm.pidsDir, "my-service.lock")
	if path != expected {
		t.Errorf("expected path %s, got %s", expected, path)
	}
}

// TestAcquireServiceLock_Basic tests basic lock acquisition and release.
func TestAcquireServiceLock_Basic(t *testing.T) {
	pm := newTestProcessManager(t)

	lockFile, err := pm.acquireServiceLock("test-service")
	if err != nil {
		t.Fatalf("failed to acquire lock: %v", err)
	}

	// Verify lock file was created
	lockPath := pm.getLockFilePath("test-service")
	if _, err := os.Stat(lockPath); os.IsNotExist(err) {
		t.Error("lock file should exist after acquiring lock")
	}

	// Release the lock
	pm.releaseServiceLock(lockFile)
}

// TestAcquireServiceLock_NilRelease tests that releasing a nil lock doesn't panic.
func TestAcquireServiceLock_NilRelease(t *testing.T) {
	pm := newTestProcessManager(t)

	// This should not panic
	pm.releaseServiceLock(nil)
}

// TestAcquireServiceLock_Concurrent tests that concurrent lock attempts work correctly.
// One goroutine should acquire the lock, others should wait and eventually succeed.
func TestAcquireServiceLock_Concurrent(t *testing.T) {
	pm := newTestProcessManager(t)
	serviceName := "concurrent-test"

	const numGoroutines = 5
	var wg sync.WaitGroup
	successCount := 0
	var mu sync.Mutex

	// Track the order of lock acquisitions
	acquisitionOrder := make([]int, 0, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			lockFile, err := pm.acquireServiceLock(serviceName)
			if err != nil {
				t.Errorf("goroutine %d failed to acquire lock: %v", id, err)
				return
			}

			mu.Lock()
			successCount++
			acquisitionOrder = append(acquisitionOrder, id)
			mu.Unlock()

			// Hold the lock briefly to simulate work
			time.Sleep(50 * time.Millisecond)

			pm.releaseServiceLock(lockFile)
		}(i)
	}

	wg.Wait()

	if successCount != numGoroutines {
		t.Errorf("expected all %d goroutines to acquire lock, but only %d succeeded", numGoroutines, successCount)
	}

	// All goroutines should have acquired the lock sequentially
	if len(acquisitionOrder) != numGoroutines {
		t.Errorf("expected %d acquisitions, got %d", numGoroutines, len(acquisitionOrder))
	}
}

// TestAcquireServiceLock_ExclusiveTwoManagers tests that two ProcessManagers
// cannot hold the same lock simultaneously.
func TestAcquireServiceLock_ExclusiveTwoManagers(t *testing.T) {
	// Use a shared temp directory to simulate two CLI instances
	tmpDir := t.TempDir()

	pidsDir := filepath.Join(tmpDir, "pids")
	logsDir := filepath.Join(tmpDir, "logs")
	if err := os.MkdirAll(pidsDir, 0755); err != nil {
		t.Fatalf("failed to create pids directory: %v", err)
	}
	if err := os.MkdirAll(logsDir, 0755); err != nil {
		t.Fatalf("failed to create logs directory: %v", err)
	}

	pm1 := &ProcessManager{
		logsDir:   logsDir,
		pidsDir:   pidsDir,
		processes: make(map[string]*ProcessInfo),
	}

	pm2 := &ProcessManager{
		logsDir:   logsDir,
		pidsDir:   pidsDir,
		processes: make(map[string]*ProcessInfo),
	}

	serviceName := "shared-service"

	// pm1 acquires lock
	lockFile1, err := pm1.acquireServiceLock(serviceName)
	if err != nil {
		t.Fatalf("pm1 failed to acquire lock: %v", err)
	}

	// pm2 tries to acquire lock with a short timeout
	// We modify the test to use a channel to detect if lock was acquired
	lockAcquired := make(chan bool, 1)
	go func() {
		lockFile2, err := pm2.acquireServiceLock(serviceName)
		if err == nil {
			pm2.releaseServiceLock(lockFile2)
			lockAcquired <- true
		} else {
			lockAcquired <- false
		}
	}()

	// Wait a bit - pm2 should be blocked
	select {
	case acquired := <-lockAcquired:
		if acquired {
			// This is expected - pm2 acquired after pm1 released, but pm1 hasn't released yet
			// so this means the lock wasn't exclusive
			t.Error("pm2 should not have been able to acquire lock while pm1 holds it immediately")
		}
	case <-time.After(200 * time.Millisecond):
		// Expected - pm2 is blocked waiting for lock
	}

	// Release pm1's lock
	pm1.releaseServiceLock(lockFile1)

	// Now pm2 should be able to acquire
	select {
	case acquired := <-lockAcquired:
		if !acquired {
			t.Error("pm2 should have acquired lock after pm1 released it")
		}
	case <-time.After(ServiceLockTimeout + time.Second):
		t.Error("pm2 timed out waiting for lock")
	}
}

// TestWaitForPIDFile_AlreadyExists tests waitForPIDFile when PID file already exists.
func TestWaitForPIDFile_AlreadyExists(t *testing.T) {
	pm := newTestProcessManager(t)

	// Write PID file with current process PID (guaranteed to be alive)
	currentPID := os.Getpid()
	writePIDFile(t, pm.pidsDir, "existing-service", currentPID)

	// Should return immediately
	err := pm.waitForPIDFile("existing-service", 1*time.Second)
	if err != nil {
		t.Errorf("expected no error for existing PID file, got: %v", err)
	}
}

// TestWaitForPIDFile_CreatedDuringWait tests waitForPIDFile when PID file is created while waiting.
func TestWaitForPIDFile_CreatedDuringWait(t *testing.T) {
	pm := newTestProcessManager(t)
	serviceName := "delayed-service"

	// Use a channel to signal when the goroutine completes
	done := make(chan struct{})

	// Start a goroutine that writes the PID file after a delay
	go func() {
		defer close(done)
		time.Sleep(300 * time.Millisecond)
		// Write PID file directly instead of using writePIDFile helper
		// to avoid calling t.Fatalf from a goroutine (violates Go testing conventions)
		pidFile := filepath.Join(pm.pidsDir, fmt.Sprintf("%s.pid", serviceName))
		if err := os.WriteFile(pidFile, []byte(fmt.Sprintf("%d", os.Getpid())), 0644); err != nil {
			// Log error but don't call t.Fatalf from goroutine
			t.Logf("failed to write PID file in goroutine: %v", err)
		}
	}()

	start := time.Now()
	err := pm.waitForPIDFile(serviceName, 2*time.Second)
	elapsed := time.Since(start)

	// Wait for goroutine to complete before test exits
	<-done

	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}

	// Should have waited some time
	if elapsed < 200*time.Millisecond {
		t.Errorf("expected to wait at least 200ms, but only waited %v", elapsed)
	}
}

// TestWaitForPIDFile_Timeout tests waitForPIDFile timeout behavior.
func TestWaitForPIDFile_Timeout(t *testing.T) {
	pm := newTestProcessManager(t)

	start := time.Now()
	err := pm.waitForPIDFile("nonexistent-service", 500*time.Millisecond)
	elapsed := time.Since(start)

	if err == nil {
		t.Error("expected timeout error, got nil")
	}

	// Should have waited approximately the timeout duration
	if elapsed < 400*time.Millisecond {
		t.Errorf("expected to wait at least 400ms, but only waited %v", elapsed)
	}
}

// TestWaitForPIDFile_DeadPID tests waitForPIDFile when PID file exists but process is dead.
func TestWaitForPIDFile_DeadPID(t *testing.T) {
	pm := newTestProcessManager(t)

	// Write PID file with a dead PID
	deadPID := findDeadPID()
	writePIDFile(t, pm.pidsDir, "dead-service", deadPID)

	// Should timeout because isProcessAlive returns false for the dead PID
	err := pm.waitForPIDFile("dead-service", 500*time.Millisecond)
	if err == nil {
		t.Error("expected timeout error for dead PID, got nil")
	}
}

// TestStalePIDFileCleanup tests that stale PID files are cleaned up during lock acquisition flow.
// This simulates the scenario in StartProcess where a stale PID file from a crashed process is found.
func TestStalePIDFileCleanup(t *testing.T) {
	pm := newTestProcessManager(t)
	serviceName := "stale-service"

	// Write a PID file with a dead PID (simulating crashed process)
	deadPID := findDeadPID()
	writePIDFile(t, pm.pidsDir, serviceName, deadPID)

	// Verify the file exists
	pidFile := pm.getPIDFilePath(serviceName)
	if _, err := os.Stat(pidFile); os.IsNotExist(err) {
		t.Fatal("PID file should exist before cleanup")
	}

	// Simulate the flow in StartProcess: check PID file and clean up if stale
	if pid, found := pm.ReadPIDFile(serviceName); found {
		if !isProcessAlive(pid) {
			pm.removePIDFile(serviceName)
		}
	}

	// Verify the stale PID file was cleaned up
	if _, err := os.Stat(pidFile); !os.IsNotExist(err) {
		t.Error("stale PID file should have been removed")
	}
}

// TestLivePIDFileNotRemoved tests that a PID file for a running process is not removed.
func TestLivePIDFileNotRemoved(t *testing.T) {
	pm := newTestProcessManager(t)
	serviceName := "live-service"

	// Write a PID file with current process PID (guaranteed alive)
	currentPID := os.Getpid()
	writePIDFile(t, pm.pidsDir, serviceName, currentPID)

	// Simulate the check in StartProcess
	if pid, found := pm.ReadPIDFile(serviceName); found {
		if !isProcessAlive(pid) {
			pm.removePIDFile(serviceName)
		}
	}

	// Verify the PID file still exists (process is alive)
	pidFile := pm.getPIDFilePath(serviceName)
	if _, err := os.Stat(pidFile); os.IsNotExist(err) {
		t.Error("PID file for live process should not have been removed")
	}
}

// TestFindProcess_InMemory tests findProcess when process is in memory.
func TestFindProcess_InMemory(t *testing.T) {
	pm := newTestProcessManager(t)
	serviceName := "memory-service"

	// Add a process to memory
	pm.processes[serviceName] = &ProcessInfo{
		Name:   serviceName,
		Status: "running",
		PID:    12345,
	}

	proc, found := pm.findProcess(serviceName)
	if !found {
		t.Fatal("expected to find process in memory")
	}
	if proc.Name != serviceName {
		t.Errorf("expected name %s, got %s", serviceName, proc.Name)
	}
}

// TestFindProcess_ViaPIDFile tests findProcess when process is found via PID file.
func TestFindProcess_ViaPIDFile(t *testing.T) {
	pm := newTestProcessManager(t)
	serviceName := "pidfile-service"

	// Write PID file with current process PID
	currentPID := os.Getpid()
	writePIDFile(t, pm.pidsDir, serviceName, currentPID)

	proc, found := pm.findProcess(serviceName)
	if !found {
		t.Fatal("expected to find process via PID file")
	}
	if proc.PID != currentPID {
		t.Errorf("expected PID %d, got %d", currentPID, proc.PID)
	}
	if proc.Status != "running" {
		t.Errorf("expected status 'running', got %s", proc.Status)
	}
}

// TestFindProcess_DeadPIDFile tests findProcess when PID file points to dead process.
func TestFindProcess_DeadPIDFile(t *testing.T) {
	pm := newTestProcessManager(t)
	serviceName := "dead-pidfile-service"

	// Write PID file with dead PID
	deadPID := findDeadPID()
	writePIDFile(t, pm.pidsDir, serviceName, deadPID)

	proc, found := pm.findProcess(serviceName)
	if found {
		t.Errorf("should not find process with dead PID, but found: %+v", proc)
	}
}

// TestFindProcess_NotFound tests findProcess when process doesn't exist.
func TestFindProcess_NotFound(t *testing.T) {
	pm := newTestProcessManager(t)

	proc, found := pm.findProcess("nonexistent-service")
	if found {
		t.Errorf("should not find nonexistent process, but found: %+v", proc)
	}
}
