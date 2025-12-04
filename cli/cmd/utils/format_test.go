package utils

import (
	"testing"
)

func TestFormatBytes(t *testing.T) {
	tests := []struct {
		name     string
		bytes    int64
		expected string
	}{
		{
			name:     "bytes",
			bytes:    500,
			expected: "500 B",
		},
		{
			name:     "kilobytes",
			bytes:    1024,
			expected: "1.0 KB",
		},
		{
			name:     "megabytes",
			bytes:    1024 * 1024,
			expected: "1.0 MB",
		},
		{
			name:     "gigabytes",
			bytes:    1024 * 1024 * 1024,
			expected: "1.0 GB",
		},
		{
			name:     "terabytes",
			bytes:    1024 * 1024 * 1024 * 1024,
			expected: "1.0 TB",
		},
		{
			name:     "petabytes",
			bytes:    1024 * 1024 * 1024 * 1024 * 1024,
			expected: "1.0 PB",
		},
		{
			name:     "exabytes (should cap at PB and not panic)",
			bytes:    1024 * 1024 * 1024 * 1024 * 1024 * 1024,
			expected: "1024.0 PB",
		},
		{
			name:     "fractional MB",
			bytes:    1536 * 1024,
			expected: "1.5 MB",
		},
		{
			name:     "zero bytes",
			bytes:    0,
			expected: "0 B",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := FormatBytes(tt.bytes)
			if result != tt.expected {
				t.Errorf("FormatBytes(%d) = %s, want %s", tt.bytes, result, tt.expected)
			}
		})
	}
}

func TestFormatDuration(t *testing.T) {
	tests := []struct {
		name     string
		seconds  float64
		expected string
	}{
		{
			name:     "negative duration",
			seconds:  -1,
			expected: "unknown",
		},
		{
			name:     "seconds only",
			seconds:  45,
			expected: "45s",
		},
		{
			name:     "minutes with seconds",
			seconds:  150,
			expected: "2m 30s",
		},
		{
			name:     "minutes only",
			seconds:  120,
			expected: "2m",
		},
		{
			name:     "hours with minutes",
			seconds:  5400,
			expected: "1h 30m",
		},
		{
			name:     "hours only",
			seconds:  7200,
			expected: "2h",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := FormatDuration(tt.seconds)
			if result != tt.expected {
				t.Errorf("FormatDuration(%f) = %s, want %s", tt.seconds, result, tt.expected)
			}
		})
	}
}

func TestFormatTransferRate(t *testing.T) {
	tests := []struct {
		name        string
		bytesPerSec int64
		expected    string
	}{
		{
			name:        "kilobytes per second",
			bytesPerSec: 1024,
			expected:    "1.0 KB/s",
		},
		{
			name:        "megabytes per second",
			bytesPerSec: 1024 * 1024,
			expected:    "1.0 MB/s",
		},
		{
			name:        "fractional rate",
			bytesPerSec: 1536 * 1024,
			expected:    "1.5 MB/s",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := FormatTransferRate(tt.bytesPerSec)
			if result != tt.expected {
				t.Errorf("FormatTransferRate(%d) = %s, want %s", tt.bytesPerSec, result, tt.expected)
			}
		})
	}
}
