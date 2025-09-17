package cmd

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// mock streaming SSE server
func sseHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Session-ID", "test-session")
	// initial role delta
	io.WriteString(w, `data: {"choices":[{"delta":{"role":"assistant"}}]}`+"\n\n")
	// two content chunks
	io.WriteString(w, `data: {"choices":[{"delta":{"content":"Hello"}}]}`+"\n\n")
	io.WriteString(w, `data: {"choices":[{"delta":{"content":" world"}}]}`+"\n\n")
	// done
	io.WriteString(w, `data: [DONE]`+"\n\n")
}

func TestSendChatRequestStream_SSE(t *testing.T) {
	// Spin up test server
	ts := httptest.NewServer(http.HandlerFunc(sseHandler))
	defer ts.Close()

	// Point globals to test server
	serverURL = ts.URL
	sessionID = ""
	namespace = "llamafarm"
	projectID = "project-seed"

	// Prepare messages
	msgs := []ChatMessage{{Role: "user", Content: "hi"}}
	got, err := sendChatRequest(msgs, nil)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if got != "Hello world" {
		t.Fatalf("unexpected assembled text: %q", got)
	}
	if sessionID != "test-session" {
		t.Fatalf("expected sessionID to be 'test-session', got %q", sessionID)
	}
}
