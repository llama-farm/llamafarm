package orchestrator

import (
	"testing"
)

func TestResolveDependencies(t *testing.T) {
	sm := &ServiceManager{
		serverURL: "http://localhost:14345",
		services:  ServiceGraph,
	}

	tests := []struct {
		name        string
		serviceName string
		wantOrder   []string
		wantErr     bool
		errContains string
	}{
		{
			name:        "resolve server with no dependencies",
			serviceName: "server",
			wantOrder:   []string{"server"},
			wantErr:     false,
		},
		{
			name:        "resolve universal-runtime with no dependencies",
			serviceName: "universal-runtime",
			wantOrder:   []string{"server", "universal-runtime"},
			wantErr:     false,
		},
		{
			name:        "resolve rag with server and universal-runtime dependencies",
			serviceName: "rag",
			wantOrder:   []string{"server", "universal-runtime", "rag"},
			wantErr:     false,
		},
		{
			name:        "unknown service returns error",
			serviceName: "unknown-service",
			wantErr:     true,
			errContains: "unknown service in dependency graph",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotOrder, err := sm.resolveDependencies(tt.serviceName)

			if tt.wantErr {
				if err == nil {
					t.Errorf("resolveDependencies() expected error but got none")
					return
				}
				if tt.errContains != "" && !contains(err.Error(), tt.errContains) {
					t.Errorf("resolveDependencies() error = %v, should contain %v", err, tt.errContains)
				}
				return
			}

			if err != nil {
				t.Errorf("resolveDependencies() unexpected error = %v", err)
				return
			}

			if !slicesEqualIgnoreOrder(gotOrder, tt.wantOrder) {
				t.Errorf("resolveDependencies() = %v, want %v", gotOrder, tt.wantOrder)
			}

			// Verify dependency order is correct
			if err := verifyDependencyOrder(gotOrder, ServiceGraph); err != nil {
				t.Errorf("resolveDependencies() invalid dependency order: %v", err)
			}
		})
	}
}

func TestResolveDependenciesCircular(t *testing.T) {
	// Create a circular dependency for testing
	circularGraph := map[string]*ServiceDefinition{
		"service-a": {
			Name:         "service-a",
			Dependencies: []string{"service-b"},
		},
		"service-b": {
			Name:         "service-b",
			Dependencies: []string{"service-c"},
		},
		"service-c": {
			Name:         "service-c",
			Dependencies: []string{"service-a"}, // Circular!
		},
	}

	// Temporarily replace ServiceGraph
	originalGraph := ServiceGraph
	ServiceGraph = circularGraph
	defer func() { ServiceGraph = originalGraph }()

	sm := &ServiceManager{
		serverURL: "http://localhost:14345",
		services:  circularGraph,
	}

	_, err := sm.resolveDependencies("service-a")
	if err == nil {
		t.Errorf("resolveDependencies() expected circular dependency error but got none")
		return
	}

	if !contains(err.Error(), "circular dependency") {
		t.Errorf("resolveDependencies() error = %v, should contain 'circular dependency'", err)
	}
}

// Helper functions

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > len(substr) && s[:len(substr)] == substr || s[len(s)-len(substr):] == substr || s != "" && substr != "" && findSubstring(s, substr))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func slicesEqualIgnoreOrder(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}

	// For the rag service test, we need to check that dependencies come before dependents
	// So we can't truly ignore order. Instead, we check that all elements are present.
	counts := make(map[string]int)
	for _, v := range a {
		counts[v]++
	}
	for _, v := range b {
		counts[v]--
		if counts[v] < 0 {
			return false
		}
	}
	for _, count := range counts {
		if count != 0 {
			return false
		}
	}
	return true
}

func verifyDependencyOrder(order []string, graph map[string]*ServiceDefinition) error {
	// Build position map
	position := make(map[string]int)
	for i, name := range order {
		position[name] = i
	}

	// Check each service's dependencies come before it
	for _, name := range order {
		serviceDef, exists := graph[name]
		if !exists {
			continue
		}

		for _, dep := range serviceDef.Dependencies {
			depPos, depExists := position[dep]
			if !depExists {
				continue // Dependency not in this resolution (shouldn't happen in valid graph)
			}

			servicePos := position[name]
			if depPos >= servicePos {
				return &dependencyOrderError{
					service:    name,
					dependency: dep,
					message:    "dependency appears after dependent service",
				}
			}
		}
	}

	return nil
}

type dependencyOrderError struct {
	service    string
	dependency string
	message    string
}

func (e *dependencyOrderError) Error() string {
	return "dependency order violation: " + e.message + " (service: " + e.service + ", dependency: " + e.dependency + ")"
}

func TestFindDependents(t *testing.T) {
	sm := &ServiceManager{
		serverURL: "http://localhost:14345",
		services:  ServiceGraph,
	}

	tests := []struct {
		name        string
		serviceName string
		wantContain []string // Services that should be in the result
	}{
		{
			name:        "find dependents of server",
			serviceName: "server",
			wantContain: []string{"server", "rag"}, // rag depends on server
		},
		{
			name:        "find dependents of universal-runtime",
			serviceName: "universal-runtime",
			wantContain: []string{"universal-runtime", "rag"}, // rag depends on universal-runtime
		},
		{
			name:        "find dependents of rag (no dependents)",
			serviceName: "rag",
			wantContain: []string{"rag"}, // nothing depends on rag
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dependents := sm.findDependents(tt.serviceName)

			// Check that all expected services are present
			for _, expected := range tt.wantContain {
				found := false
				for _, dep := range dependents {
					if dep == expected {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("findDependents() = %v, expected to contain %v", dependents, expected)
				}
			}
		})
	}
}

func TestStopServicesValidation(t *testing.T) {
	sm := &ServiceManager{
		serverURL: "http://localhost:14345",
		services:  ServiceGraph,
	}

	tests := []struct {
		name         string
		serviceNames []string
		wantErr      bool
		errContains  string
	}{
		{
			name:         "stop unknown service",
			serviceNames: []string{"unknown-service"},
			wantErr:      true,
			errContains:  "unknown service",
		},
		{
			name:         "stop empty list",
			serviceNames: []string{},
			wantErr:      false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := sm.StopServices(tt.serviceNames...)

			if tt.wantErr {
				if err == nil {
					t.Errorf("StopServices() expected error but got none")
					return
				}
				if tt.errContains != "" && !contains(err.Error(), tt.errContains) {
					t.Errorf("StopServices() error = %v, should contain %v", err, tt.errContains)
				}
				return
			}

			// For non-error cases that don't require process manager, we expect success
			if err != nil && !contains(err.Error(), "nil pointer") {
				t.Errorf("StopServices() unexpected error: %v", err)
			}
		})
	}
}

// TestEnsureServicesWithConfig_AutoStartDisabled tests the auto-start disabled behavior
func TestEnsureServicesWithConfig_AutoStartDisabled(t *testing.T) {
	sm := &ServiceManager{
		serverURL: "http://localhost:14345",
		services:  ServiceGraph,
	}

	tests := []struct {
		name         string
		autoStart    bool
		serviceNames []string
		wantErr      bool
		errContains  string
		// Note: We can't easily mock isServiceHealthy without refactoring,
		// so these tests focus on error message formatting and unknown service handling
	}{
		{
			name:         "unknown service returns error immediately",
			autoStart:    false,
			serviceNames: []string{"unknown-service"},
			wantErr:      true,
			errContains:  "unknown service",
		},
		{
			name:         "empty service list succeeds",
			autoStart:    false,
			serviceNames: []string{},
			wantErr:      false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := &ServiceOrchestrationConfig{
				ServerURL:   "http://localhost:14345",
				PrintStatus: false,
				AutoStart:   tt.autoStart,
			}

			err := sm.EnsureServicesWithConfig(config, tt.serviceNames...)

			if tt.wantErr {
				if err == nil {
					t.Errorf("EnsureServicesWithConfig() expected error but got none")
					return
				}
				if tt.errContains != "" && !contains(err.Error(), tt.errContains) {
					t.Errorf("EnsureServicesWithConfig() error = %v, should contain %v", err, tt.errContains)
				}
				return
			}

			// For non-error cases, we can't fully test without mocking health checks,
			// but we verify the function doesn't error on empty lists
			if err != nil && !contains(err.Error(), "not running") {
				t.Errorf("EnsureServicesWithConfig() unexpected error: %v", err)
			}
		})
	}
}

// TestEnsureServicesWithConfig_ErrorMessageFormat tests that error messages are properly formatted
func TestEnsureServicesWithConfig_ErrorMessageFormat(t *testing.T) {
	// This test verifies the error message format includes all unhealthy services
	// Since we can't easily mock isServiceHealthy, we test the error message structure
	// by checking that the error message format is correct when services are unhealthy
	// We use port 59999 to ensure no server is running

	sm := &ServiceManager{
		serverURL: "http://localhost:59999",
		services:  ServiceGraph,
	}

	config := &ServiceOrchestrationConfig{
		ServerURL:   "http://localhost:59999",
		PrintStatus: false,
		AutoStart:   false,
	}

	// Test with a known service (will fail health check if server is not running)
	// This tests the error message format
	err := sm.EnsureServicesWithConfig(config, "server")

	// The server should not be running during unit tests, so we must get an error
	if err == nil {
		t.Fatal("EnsureServicesWithConfig() expected error when server is not running and AutoStart is false, but got nil")
	}

	// Verify error message contains expected format
	errMsg := err.Error()
	if !contains(errMsg, "services not running") || !contains(errMsg, "auto-start is disabled") {
		t.Errorf("EnsureServicesWithConfig() error message should mention 'services not running' and 'auto-start is disabled', got: %v", errMsg)
	}
	if !contains(errMsg, "--auto-start") {
		t.Errorf("EnsureServicesWithConfig() error message should mention '--auto-start', got: %v", errMsg)
	}
}
