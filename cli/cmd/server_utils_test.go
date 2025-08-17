package cmd

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

func TestIsLocalhost(t *testing.T) {
	cases := []struct {
		in  string
		yes bool
	}{
		{"http://localhost:8000", true},
		{"http://127.0.0.1:8000", true},
		{"http://::1:8000", true},
		{"http://example.com", false},
	}
	for _, c := range cases {
		if isLocalhost(c.in) != c.yes {
			t.Fatalf("isLocalhost(%q) mismatch; expected %v", c.in, c.yes)
		}
	}
}

func TestResolvePort(t *testing.T) {
	cases := []struct {
		url  string
		def  int
		want int
	}{
		{"http://localhost:8000", 8000, 8000},
		{"http://localhost", 8000, 80},
		{"https://localhost", 8000, 443},
		{"invalid://", 8000, 8000},
		{"http://localhost:1234", 8000, 1234},
	}
	for _, c := range cases {
		if got := resolvePort(c.url, c.def); got != c.want {
			t.Fatalf("resolvePort(%q,%d) = %d; want %d", c.url, c.def, got, c.want)
		}
	}
}

func TestPingURL_SuccessAndFailure(t *testing.T) {
	// success
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/tags" {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`[]`))
	}))
	defer ts.Close()

	if err := pingURL(ts.URL); err != nil {
		t.Fatalf("expected nil error from pingURL on 200, got %v", err)
	}

	// failure (not found)
	ts2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	defer ts2.Close()
	if err := pingURL(ts2.URL); err == nil {
		t.Fatalf("expected error from pingURL on 404, got nil")
	}
}

func TestCheckServerHealth_SuccessAndFailure(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/info" {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer ts.Close()

	if err := checkServerHealth(ts.URL); err != nil {
		t.Fatalf("expected nil from checkServerHealth on 200, got %v", err)
	}

	ts2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
		_, _ = w.Write([]byte(`error`))
	}))
	defer ts2.Close()
	if err := checkServerHealth(ts2.URL); err == nil {
		t.Fatalf("expected error from checkServerHealth on 500, got nil")
	}
}

func TestEnsureServerAvailable_LocalhostUp(t *testing.T) {
	// httptest server binds to localhost/127.0.0.1 which is considered localhost
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/info" {
			w.WriteHeader(200)
			_, _ = w.Write([]byte(`{"ok":true}`))
			return
		}
		http.NotFound(w, r)
	}))
	defer ts.Close()

	if err := ensureServerAvailable(ts.URL); err != nil {
		t.Fatalf("expected ensureServerAvailable to succeed for running localhost server, got %v", err)
	}
}

// dummy HTTP client used to test VerboseHTTPClient behavior
type dummyClient struct {
	resp   *http.Response
	err    error
	gotReq *http.Request
}

func (d *dummyClient) Do(req *http.Request) (*http.Response, error) {
	d.gotReq = req
	return d.resp, d.err
}

func TestVerboseHTTPClient_Logs(t *testing.T) {
	// Prepare dummy response with headers
	resp := &http.Response{
		StatusCode: 200,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader("ok")),
	}
	resp.Header.Set("X-Test", "val")

	d := &dummyClient{resp: resp}
	// swap global client and enable debug
	prev := httpClient
	httpClient = d
	prevDebug := debug
	debug = true
	defer func() { httpClient = prev; debug = prevDebug }()

	// capture stderr
	r, w, _ := os.Pipe()
	old := os.Stderr
	os.Stderr = w
	defer func() { w.Close(); os.Stderr = old }()

	req, _ := http.NewRequest(http.MethodGet, "http://localhost:8000/foo", nil)
	client := getHTTPClient()
	gotResp, err := client.Do(req)
	// flush writer
	w.Close()

	// read stderr
	var b strings.Builder
	_, _ = io.Copy(&b, r)
	out := b.String()

	if err != nil {
		t.Fatalf("unexpected error from VerboseHTTPClient: %v", err)
	}
	if gotResp.StatusCode != 200 {
		t.Fatalf("unexpected response from VerboseHTTPClient: %v", gotResp.StatusCode)
	}
	if !strings.Contains(out, "HTTP GET") || !strings.Contains(out, "-> 200 OK") || !strings.Contains(out, "response header: X-Test: val") {
		t.Fatalf("expected verbose logs to contain request/response lines and header; got: %q", out)
	}
}
