package utils

import "fmt"

// FormatBytes converts bytes to a human-readable string with appropriate units.
// Uses binary units (1024-based): KB, MB, GB, TB, PB.
func FormatBytes(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}

	units := []string{"KB", "MB", "GB", "TB", "PB"}

	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit && exp < len(units)-1; n /= unit {
		div *= unit
		exp++
	}

	return fmt.Sprintf("%.1f %s", float64(bytes)/float64(div), units[exp])
}

// FormatDuration formats seconds into a human-readable duration string.
// Examples: "5s", "2m 30s", "1h 15m", "2h 30m"
func FormatDuration(seconds float64) string {
	if seconds < 0 {
		return "unknown"
	}

	totalSecs := int(seconds)

	if totalSecs < 60 {
		return fmt.Sprintf("%ds", totalSecs)
	}

	minutes := totalSecs / 60
	secs := totalSecs % 60

	if minutes < 60 {
		if secs == 0 {
			return fmt.Sprintf("%dm", minutes)
		}
		return fmt.Sprintf("%dm %ds", minutes, secs)
	}

	hours := minutes / 60
	mins := minutes % 60

	if mins == 0 {
		return fmt.Sprintf("%dh", hours)
	}
	return fmt.Sprintf("%dh %dm", hours, mins)
}

// FormatTransferRate formats bytes per second into a human-readable rate string.
// Examples: "1.5 MB/s", "500 KB/s"
func FormatTransferRate(bytesPerSec int64) string {
	return FormatBytes(bytesPerSec) + "/s"
}
