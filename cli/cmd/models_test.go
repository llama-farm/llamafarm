package cmd

import "testing"

func TestFormatModelRuntimeSummary(t *testing.T) {
	tests := []struct {
		name     string
		model    ModelInfo
		expected string
	}{
		{
			name:     "empty when no runtime status",
			model:    ModelInfo{},
			expected: "",
		},
		{
			name: "status only",
			model: ModelInfo{
				RuntimeStatus: "idle",
			},
			expected: "idle",
		},
		{
			name: "status with metrics",
			model: ModelInfo{
				RuntimeStatus:    "running",
				MemoryUsageHuman: "4.2 GB",
				GPUAllocation:    "3.6 GB",
				UptimeHuman:      "12m",
			},
			expected: "running | Memory: 4.2 GB | GPU: 3.6 GB | Uptime: 12m",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := formatModelRuntimeSummary(tt.model)
			if got != tt.expected {
				t.Fatalf("formatModelRuntimeSummary() = %q, want %q", got, tt.expected)
			}
		})
	}
}

func TestFormatModelRuntimeMessage(t *testing.T) {
	tests := []struct {
		name     string
		model    ModelInfo
		expected string
	}{
		{
			name: "prints error-like runtime message",
			model: ModelInfo{
				RuntimeStatus:  "missing",
				RuntimeMessage: "Configured model is not installed in Ollama",
			},
			expected: "Configured model is not installed in Ollama",
		},
		{
			name: "suppresses redundant loaded message",
			model: ModelInfo{
				RuntimeStatus:  "loaded",
				RuntimeMessage: "Model is loaded in Universal Runtime",
			},
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := formatModelRuntimeMessage(tt.model)
			if got != tt.expected {
				t.Fatalf("formatModelRuntimeMessage() = %q, want %q", got, tt.expected)
			}
		})
	}
}
