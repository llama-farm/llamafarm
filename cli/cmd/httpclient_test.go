package cmd

import (
	"net/http"
	"testing"
)

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
