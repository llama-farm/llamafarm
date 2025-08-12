package cmd

import (
    "fmt"
    "net/http"
    "os"
    "sort"
    "strings"
    "time"
)

// HTTPClient interface for testing
type HTTPClient interface {
    Do(req *http.Request) (*http.Response, error)
}

// DefaultHTTPClient is the default HTTP client
type DefaultHTTPClient struct{}

// Do implements the HTTPClient interface
func (c *DefaultHTTPClient) Do(req *http.Request) (*http.Response, error) {
    _ = addLocalhostCWDHeader(req)
    client := &http.Client{Timeout: 30 * time.Second}
    return client.Do(req)
}

var httpClient HTTPClient = &DefaultHTTPClient{}

// VerboseHTTPClient wraps another HTTPClient and logs request/response basics and headers.
type VerboseHTTPClient struct{ Inner HTTPClient }

func (v *VerboseHTTPClient) Do(req *http.Request) (*http.Response, error) {
    inner := v.Inner
    if inner == nil {
        inner = &DefaultHTTPClient{}
    }
    _ = addLocalhostCWDHeader(req)
    fmt.Fprintf(os.Stderr, "HTTP %s %s\n", req.Method, req.URL.String())
    logHeaders("request", req.Header)
    resp, err := inner.Do(req)
    if err != nil {
        fmt.Fprintf(os.Stderr, "  -> error: %v\n", err)
        return nil, err
    }
    fmt.Fprintf(os.Stderr, "  -> %d %s\n", resp.StatusCode, http.StatusText(resp.StatusCode))
    logHeaders("response", resp.Header)
    return resp, nil
}

func getHTTPClient() HTTPClient {
    if verbose {
        return &VerboseHTTPClient{Inner: httpClient}
    }
    return httpClient
}

// addLocalhostCWDHeader attaches the current working directory as X-LF-Client-CWD
// if the request is targeting a localhost URL.
func addLocalhostCWDHeader(req *http.Request) error {
    host := strings.ToLower(req.URL.Hostname())
    if host == "localhost" || host == "127.0.0.1" || host == "::1" {
        if wd, err := os.Getwd(); err == nil {
            req.Header.Set("X-LF-Client-CWD", wd)
        } else {
            return err
        }
    }
    return nil
}

func logHeaders(kind string, hdr http.Header) {
    if len(hdr) == 0 {
        return
    }
    keys := make([]string, 0, len(hdr))
    for k := range hdr {
        keys = append(keys, k)
    }
    sort.Strings(keys)
    for _, k := range keys {
        vals := hdr.Values(k)
        for _, v := range vals {
            fmt.Fprintf(os.Stderr, "  %s header: %s: %s\n", kind, k, v)
        }
    }
}
