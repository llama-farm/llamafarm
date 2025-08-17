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
