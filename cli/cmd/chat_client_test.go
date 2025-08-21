package cmd

import "testing"

func TestBuildChatAPIURL(t *testing.T) {
    ctx := &ChatSessionContext{ServerURL: "http://localhost:8000"}
    // Inference path when no ns/project
    got := buildChatAPIURL(ctx)
    want := "http://localhost:8000/v1/inference/chat"
    if got != want {
        t.Fatalf("expected %q, got %q", want, got)
    }

    // Project-scoped path when ns/project provided
    ctx.Namespace = "org"
    ctx.ProjectID = "proj"
    got = buildChatAPIURL(ctx)
    want = "http://localhost:8000/v1/projects/org/proj/chat/completions"
    if got != want {
        t.Fatalf("expected %q, got %q", want, got)
    }
}

func TestNewDefaultContextFromGlobals(t *testing.T) {
    serverURL = "http://localhost:8000"
    namespace = "ns"
    projectID = "proj"
    sessionID = "sess"
    temperature = 0.5
    maxTokens = 123
    streaming = true

    ctx := newDefaultContextFromGlobals()
    if ctx.ServerURL != serverURL || ctx.Namespace != namespace || ctx.ProjectID != projectID || ctx.SessionID != sessionID || ctx.Temperature != temperature || ctx.MaxTokens != maxTokens || ctx.Streaming != streaming {
        t.Fatalf("context not initialized from globals correctly")
    }
    if ctx.HTTPClient == nil {
        t.Fatalf("expected HTTPClient set")
    }
}
