package cmd

import (
    "net/http"
    "testing"
)

func TestAddLocalhostCWDHeader_Localhost(t *testing.T) {
    req, _ := http.NewRequest(http.MethodGet, "http://localhost:8000/info", nil)
    if err := addLocalhostCWDHeader(req); err != nil {
        t.Fatalf("unexpected error: %v", err)
    }
    if req.Header.Get("X-LF-Client-CWD") == "" {
        t.Fatalf("expected X-LF-Client-CWD to be set for localhost URL")
    }
}

func TestAddLocalhostCWDHeader_Remote(t *testing.T) {
    req, _ := http.NewRequest(http.MethodGet, "https://example.com", nil)
    if err := addLocalhostCWDHeader(req); err != nil {
        t.Fatalf("unexpected error: %v", err)
    }
    if req.Header.Get("X-LF-Client-CWD") != "" {
        t.Fatalf("did not expect X-LF-Client-CWD to be set for remote URL")
    }
}

func TestPrettyServerError_BestEffort(t *testing.T) {
    resp := &http.Response{StatusCode: 500, Header: make(http.Header)}
    body := []byte(`{"detail":"boom"}`)
    got := prettyServerError(resp, body)
    if got != "boom" {
        t.Fatalf("prettyServerError expected 'boom', got %q", got)
    }
}
