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

func TestPrettyServerError_AlternateShapes(t *testing.T) {
    resp := &http.Response{StatusCode: 400, Header: make(http.Header)}

    // message at root
    if got := prettyServerError(resp, []byte(`{"message":"oops"}`)); got != "oops" {
        t.Fatalf("want 'oops', got %q", got)
    }

    // error at root
    if got := prettyServerError(resp, []byte(`{"error":"bad"}`)); got != "bad" {
        t.Fatalf("want 'bad', got %q", got)
    }

    // nested array detail
    if got := prettyServerError(resp, []byte(`{"detail":[{"message":"deep"}]}`)); got != "deep" {
        t.Fatalf("want 'deep', got %q", got)
    }
}
