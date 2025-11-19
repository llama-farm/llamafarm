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

	// Create ChatManager with test server
	cfg := &ChatConfig{
		ServerURL:        ts.URL,
		Namespace:        "llamafarm",
		ProjectID:        "project-seed",
		SessionMode:      SessionModeProject,
		SessionNamespace: "llamafarm",
		SessionProject:   "project-seed",
		RAGEnabled:       false,
	}

	mgr, err := NewChatManager(cfg)
	if err != nil {
		t.Fatalf("failed to create manager: %v", err)
	}

	// Prepare messages and send
	msgs := []Message{{Role: "user", Content: "hi"}}
	got, err := mgr.SendMessages(msgs)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if got != "Hello world" {
		t.Fatalf("unexpected assembled text: %q", got)
	}

	// Verify session ID was set from response header
	gotSessionID := mgr.GetSessionID()
	if gotSessionID != "test-session" {
		t.Fatalf("expected sessionID to be 'test-session', got %q", gotSessionID)
	}
}
