package cmd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"
)

// HTTPClient interface for testing
type HTTPClient interface {
	Do(req *http.Request) (*http.Response, error)
}

// DefaultHTTPClient is the default HTTP client
type DefaultHTTPClient struct{ Timeout time.Duration }

// Do implements the HTTPClient interface
func (c *DefaultHTTPClient) Do(req *http.Request) (*http.Response, error) {
	client := &http.Client{Timeout: 60 * time.Second}
	return client.Do(req)
}

var httpClient HTTPClient = &DefaultHTTPClient{}

// logBodyContent safely reads and logs a body, restoring it for later use.
// Returns the restored body (or nil if input was nil).
// Truncates very large bodies to avoid flooding logs.
func logBodyContent(body io.ReadCloser, label string) io.ReadCloser {
	if body == nil {
		logDebug(fmt.Sprintf("  -> %s: <nil>", label))
		return nil
	}

	bodyBytes, err := io.ReadAll(body)
	body.Close()

	if err != nil {
		logDebug(fmt.Sprintf("  -> %s: <error reading: %v>", label, err))
		return io.NopCloser(bytes.NewReader([]byte{}))
	}

	if len(bodyBytes) == 0 {
		logDebug(fmt.Sprintf("  -> %s: <empty>", label))
		return io.NopCloser(bytes.NewReader(bodyBytes))
	}

	// Truncate very large bodies for readability
	const maxLogSize = 1024
	bodyStr := string(bodyBytes)
	if len(bodyStr) > maxLogSize {
		bodyStr = bodyStr[:maxLogSize] + "... (truncated)"
	}

	logDebug(fmt.Sprintf("  -> %s: %s", label, bodyStr))
	return io.NopCloser(bytes.NewReader(bodyBytes))
}

// VerboseHTTPClient wraps another HTTPClient and logs request/response basics and headers.
type VerboseHTTPClient struct{ Inner HTTPClient }

func (v *VerboseHTTPClient) Do(req *http.Request) (*http.Response, error) {
	inner := v.Inner
	if inner == nil {
		inner = &DefaultHTTPClient{}
	}
	logDebug(fmt.Sprintf("HTTP %s %s", req.Method, req.URL.String()))
	logHeaders("request", req.Header)

	// Log and restore request body
	req.Body = logBodyContent(req.Body, "request body")

	resp, err := inner.Do(req)
	if err != nil {
		logDebug(fmt.Sprintf("  -> error: %v", err))
		return nil, err
	}
	logDebug(fmt.Sprintf("  -> %d %s", resp.StatusCode, http.StatusText(resp.StatusCode)))
	logHeaders("response", resp.Header)

	// Log and restore response body
	resp.Body = logBodyContent(resp.Body, "response body")

	return resp, nil
}

func getHTTPClient() HTTPClient {
	return &VerboseHTTPClient{Inner: httpClient}
}

func getHTTPClientWithTimeout(timeout time.Duration) HTTPClient {
	return &VerboseHTTPClient{Inner: &DefaultHTTPClient{Timeout: timeout}}
}

func logHeaders(kind string, hdr http.Header) {
	if len(hdr) == 0 {
		return
	}
	// List of sensitive headers to redact, lower-case for comparison
	sensitiveHeaders := map[string]struct{}{
		"authorization": {},
		"cookie":        {},
		"x-session-id":  {},
		"set-cookie":    {},
		// Add others as needed
	}
	keys := make([]string, 0, len(hdr))
	for k := range hdr {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		vals := hdr.Values(k)
		_, isSensitive := sensitiveHeaders[strings.ToLower(k)]
		for _, v := range vals {
			if isSensitive {
				logDebug(fmt.Sprintf("  %s header: %s: [REDACTED]", kind, k))
			} else {
				logDebug(fmt.Sprintf("  %s header: %s: %s", kind, k, v))
			}
		}
	}
}

// prettyServerError extracts a readable message from a server error response body.
// It parses common JSON shapes like {"detail":...}, {"message":...}, {"error":...}.
func prettyServerError(resp *http.Response, body []byte) string {
	// Try to parse JSON error envelopes
	var env struct {
		Detail    any    `json:"detail"`
		Message   string `json:"message"`
		Error     string `json:"error"`
		RequestID string `json:"request_id"`
	}
	if json.Unmarshal(body, &env) == nil {
		switch v := env.Detail.(type) {
		case string:
			if v != "" {
				return v
			}
		case map[string]any:
			if m, ok := v["message"].(string); ok && m != "" {
				return m
			}
			if m, ok := v["detail"].(string); ok && m != "" {
				return m
			}
		case []any:
			if len(v) > 0 {
				if m, ok := v[0].(map[string]any); ok {
					if s, ok := m["message"].(string); ok && s != "" {
						return s
					}
					if s, ok := m["detail"].(string); ok && s != "" {
						return s
					}
				}
			}
		}
		if env.Message != "" {
			return env.Message
		}
		if env.Error != "" {
			return env.Error
		}
	}
	s := strings.TrimSpace(string(body))
	if s == "" {
		return http.StatusText(resp.StatusCode)
	}
	// Append request id if present to aid debugging
	if env.RequestID != "" {
		return s + " (request_id=" + env.RequestID + ")"
	}
	return s
}
