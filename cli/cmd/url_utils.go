package cmd

import "strings"

// buildServerURL safely joins base and path, ensuring single slash separation
func buildServerURL(base string, path string) string {
    b := strings.TrimRight(base, "/")
    if path == "" {
        return b
    }
    if strings.HasPrefix(path, "/") {
        return b + path
    }
    return b + "/" + path
}
