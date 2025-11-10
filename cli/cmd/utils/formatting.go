package utils

import "strings"

func IconForStatus(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	switch s {
	case "healthy":
		return "✅"
	case "degraded":
		return "⚠️ "
	case "unhealthy":
		return "❌"
	default:
		return "❓"
	}
}
