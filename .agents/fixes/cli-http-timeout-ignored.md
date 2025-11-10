# CLI HTTP Timeout Not Respected

## Problem
Dataset processing was timing out after 60 seconds with error:
```
context deadline exceeded (Client.Timeout exceeded while awaiting headers)
```

Even though the code explicitly requested no timeout via `GetHTTPClientWithTimeout(0)`, the request still failed after 60 seconds.

## Root Cause
The `DefaultHTTPClient.Do()` method was **hardcoded** to always use a 60-second timeout, completely ignoring the `Timeout` field:

```go
func (c *DefaultHTTPClient) Do(req *http.Request) (*http.Response, error) {
    client := &http.Client{Timeout: 60 * time.Second}  // ⚠️ IGNORED c.Timeout!
    return client.Do(req)
}
```

So when `datasets.go` called:
```go
resp, err := utils.GetHTTPClientWithTimeout(0).Do(req)  // Intended: no timeout
```

It actually got a 60-second timeout instead!

## Solution
Fixed `DefaultHTTPClient.Do()` to respect the configured timeout:

```go
func (c *DefaultHTTPClient) Do(req *http.Request) (*http.Response, error) {
    // Use the configured timeout (0 means no timeout in Go's http.Client)
    client := &http.Client{Timeout: c.Timeout}
    return client.Do(req)
}
```

Also set a sensible default timeout for normal API calls:
```go
var httpClient HTTPClient = &DefaultHTTPClient{Timeout: 60 * time.Second}
```

## Behavior After Fix

| Call | Timeout | Use Case |
|------|---------|----------|
| `GetHTTPClient()` | 60 seconds | Normal API calls |
| `GetHTTPClientWithTimeout(0)` | None (infinite) | Dataset processing, file uploads |
| `GetHTTPClientWithTimeout(5*time.Minute)` | 5 minutes | Custom timeout |

## Testing
After this fix:
1. Dataset processing can run indefinitely without timing out
2. File uploads can handle large files without timeouts
3. Normal API calls still have sensible 60-second timeouts

## Related Files
- `cli/cmd/utils/httpclient.go` - Fixed timeout handling
- `cli/cmd/datasets.go` - Uses `GetHTTPClientWithTimeout(0)` for processing

