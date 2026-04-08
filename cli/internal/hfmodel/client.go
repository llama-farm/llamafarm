package hfmodel

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// DefaultEndpoint is the canonical HuggingFace Hub URL.
const DefaultEndpoint = "https://huggingface.co"

// Client is an HTTP client for the HuggingFace Hub. It is intentionally
// minimal: only the endpoints `lf models pull` actually needs are wrapped.
//
// All methods honor LLAMAFARM_OFFLINE — if set, they return an *OfflineError
// before any network call.
type Client struct {
	httpClient *http.Client
	endpoint   string
	userAgent  string
	token      string // empty means anonymous
	// tokenExplicit signals that WithToken was passed (including
	// WithToken("") for forced anonymous). When false, NewClient runs
	// DiscoverToken to pick up the user's stored credentials.
	tokenExplicit bool
}

// ClientOption configures a Client.
type ClientOption func(*Client)

// WithHTTPClient overrides the default HTTP client (useful for tests).
func WithHTTPClient(c *http.Client) ClientOption { return func(o *Client) { o.httpClient = c } }

// WithEndpoint overrides the HuggingFace Hub endpoint (useful for tests
// against an httptest.Server).
func WithEndpoint(url string) ClientOption {
	return func(o *Client) { o.endpoint = strings.TrimRight(url, "/") }
}

// WithUserAgent sets the User-Agent header sent on every request.
func WithUserAgent(ua string) ClientOption { return func(o *Client) { o.userAgent = ua } }

// WithToken explicitly sets a bearer token, overriding token discovery.
// Pass an empty string to force anonymous requests (suppresses the default
// DiscoverToken lookup that would otherwise pick up HF_TOKEN, etc.).
func WithToken(token string) ClientOption {
	return func(o *Client) {
		o.token = token
		o.tokenExplicit = true
	}
}

// NewClient constructs a Client. By default it uses a 30s HTTP timeout for
// metadata requests (downloads use a separate, no-timeout client because
// they may take hours), the official endpoint, and discovers a token via
// DiscoverToken.
func NewClient(opts ...ClientOption) (*Client, error) {
	c := &Client{
		httpClient: &http.Client{Timeout: 30 * time.Second},
		endpoint:   DefaultEndpoint,
		userAgent:  "llamafarm-cli/hfmodel",
	}
	for _, opt := range opts {
		opt(c)
	}
	// Discover a token only when the caller did not pass WithToken
	// explicitly. WithToken("") is a valid way to force anonymous requests
	// even when HF_TOKEN is set in the environment — important for tests
	// and for callers that need to suppress credential leaks.
	if !c.tokenExplicit {
		if tok, err := DiscoverToken(); err == nil {
			c.token = tok
		}
	}
	return c, nil
}

// TreeEntry describes one file or directory in a HuggingFace repo tree.
// Mirrors the subset of the /api/models/<id>/tree response we need.
type TreeEntry struct {
	Type string `json:"type"` // "file" or "directory"
	Path string `json:"path"`
	Size int64  `json:"size"`
	OID  string `json:"oid"` // git blob oid
}

// FileMetadata describes a single file's downloadable metadata.
type FileMetadata struct {
	Filename     string
	URL          string // resolved CDN URL after redirects
	Size         int64
	ETag         string // unquoted etag (HF uses this as the blob filename)
	CommitHash   string // git commit hash that contains this file
	LinkedSize   int64  // X-Linked-Size if present (LFS files)
	LinkedETag   string // X-Linked-ETag if present (LFS files)
}

// ListRepoTree fetches the recursive file tree for a repo at a given
// revision. `revision` may be a branch name (e.g. "main") or a commit hash.
func (c *Client) ListRepoTree(ctx context.Context, repoID, revision string) ([]TreeEntry, error) {
	if err := ValidateModelID(repoID); err != nil {
		return nil, err
	}
	if IsOffline() {
		return nil, &OfflineError{ModelID: repoID, Op: "list_repo_tree"}
	}
	if revision == "" {
		revision = "main"
	}
	endpoint := fmt.Sprintf("%s/api/models/%s/tree/%s?recursive=true",
		c.endpoint, escapeRepoID(repoID), url.PathEscape(revision))

	var out []TreeEntry
	cursor := endpoint
	for cursor != "" {
		batch, next, err := c.fetchTreePage(ctx, cursor, repoID)
		if err != nil {
			return nil, err
		}
		out = append(out, batch...)
		cursor = next
	}
	return out, nil
}

// fetchTreePage fetches one page of tree results and returns the entries plus
// the URL of the next page (empty when there are no more pages). HF paginates
// large repos via a Link header.
func (c *Client) fetchTreePage(ctx context.Context, requestURL, repoID string) ([]TreeEntry, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL, nil)
	if err != nil {
		return nil, "", err
	}
	c.applyHeaders(req)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, "", classifyTransportError(err)
	}
	defer resp.Body.Close()

	if err := classifyHTTPStatus(resp, repoID); err != nil {
		return nil, "", err
	}
	var entries []TreeEntry
	if err := json.NewDecoder(resp.Body).Decode(&entries); err != nil {
		return nil, "", fmt.Errorf("decode tree response: %w", err)
	}
	return entries, parseNextLink(resp.Header.Get("Link")), nil
}

// parseNextLink extracts the rel="next" URL from an RFC 5988 Link header.
// Returns empty when no next link is present.
func parseNextLink(header string) string {
	for _, part := range strings.Split(header, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		// Format: <https://...>; rel="next"
		semi := strings.Index(part, ";")
		if semi < 0 {
			continue
		}
		urlPart := strings.TrimSpace(part[:semi])
		params := strings.TrimSpace(part[semi+1:])
		if !strings.HasPrefix(urlPart, "<") || !strings.HasSuffix(urlPart, ">") {
			continue
		}
		if strings.Contains(params, `rel="next"`) {
			return urlPart[1 : len(urlPart)-1]
		}
	}
	return ""
}

// ListGGUFFiles is a thin convenience wrapper around ListRepoTree that filters
// for `*.gguf` files.
func (c *Client) ListGGUFFiles(ctx context.Context, repoID string) ([]string, error) {
	tree, err := c.ListRepoTree(ctx, repoID, "main")
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range tree {
		if e.Type == "file" && strings.HasSuffix(strings.ToLower(e.Path), ".gguf") {
			out = append(out, e.Path)
		}
	}
	return out, nil
}

// modelInfoResponse is the subset of /api/models/<id> we care about.
type modelInfoResponse struct {
	SHA string `json:"sha"`
}

// GetModelCommitHash returns the commit sha for a model's main revision via
// /api/models/<id>. Used to resolve a single canonical commit hash up-front
// for a download, since per-file HEAD responses for LFS files don't reliably
// echo X-Repo-Commit through CDN redirects.
func (c *Client) GetModelCommitHash(ctx context.Context, repoID string) (string, error) {
	if err := ValidateModelID(repoID); err != nil {
		return "", err
	}
	if IsOffline() {
		return "", &OfflineError{ModelID: repoID, Op: "get_model_commit_hash"}
	}
	infoURL := fmt.Sprintf("%s/api/models/%s", c.endpoint, escapeRepoID(repoID))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, infoURL, nil)
	if err != nil {
		return "", err
	}
	c.applyHeaders(req)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", classifyTransportError(err)
	}
	defer resp.Body.Close()
	if err := classifyHTTPStatus(resp, repoID); err != nil {
		return "", err
	}
	var info modelInfoResponse
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return "", fmt.Errorf("decode model info: %w", err)
	}
	return info.SHA, nil
}

// GetFileMetadata fetches metadata for a single file via a HEAD request that
// follows redirects. Captures ETag, size, commit hash, and the resolved CDN
// URL for the subsequent body fetch.
func (c *Client) GetFileMetadata(ctx context.Context, repoID, revision, filename string) (*FileMetadata, error) {
	if err := ValidateModelID(repoID); err != nil {
		return nil, err
	}
	if IsOffline() {
		return nil, &OfflineError{ModelID: repoID, Op: "get_file_metadata"}
	}
	if revision == "" {
		revision = "main"
	}
	resolveURL := fmt.Sprintf("%s/%s/resolve/%s/%s",
		c.endpoint, escapeRepoID(repoID), url.PathEscape(revision), escapeFilename(filename))

	req, err := http.NewRequestWithContext(ctx, http.MethodHead, resolveURL, nil)
	if err != nil {
		return nil, err
	}
	c.applyHeaders(req)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, classifyTransportError(err)
	}
	defer resp.Body.Close()

	if err := classifyHTTPStatus(resp, repoID); err != nil {
		return nil, err
	}

	md := &FileMetadata{
		Filename:   filename,
		URL:        resp.Request.URL.String(),
		ETag:       unquoteETag(resp.Header.Get("ETag")),
		CommitHash: resp.Header.Get("X-Repo-Commit"),
		LinkedETag: unquoteETag(resp.Header.Get("X-Linked-ETag")),
	}
	if cl := resp.Header.Get("Content-Length"); cl != "" {
		if n, err := strconv.ParseInt(cl, 10, 64); err == nil {
			md.Size = n
		}
	}
	if ls := resp.Header.Get("X-Linked-Size"); ls != "" {
		if n, err := strconv.ParseInt(ls, 10, 64); err == nil {
			md.LinkedSize = n
			// LFS files: X-Linked-Size is the real size; Content-Length on
			// the redirect is the size of the redirect chain payload.
			md.Size = n
		}
	}
	// LFS files: prefer the linked etag, which is the actual blob's content
	// hash, over the redirect etag.
	if md.LinkedETag != "" {
		md.ETag = md.LinkedETag
	}
	if md.ETag == "" {
		// Without an etag we cannot uniquely identify the blob. Fall back
		// to the commit hash so the file is at least addressable.
		md.ETag = md.CommitHash
	}
	return md, nil
}

// escapeRepoID URL-escapes each segment of "org/name" individually but
// preserves the literal slash separator. HF Hub rejects requests where the
// slash is percent-encoded.
func escapeRepoID(repoID string) string {
	parts := strings.SplitN(repoID, "/", 2)
	for i, p := range parts {
		parts[i] = url.PathEscape(p)
	}
	return strings.Join(parts, "/")
}

// escapeFilename URL-escapes each path segment of a filename individually
// but preserves slashes between segments. Files inside subdirectories
// (e.g. `voices/af.bin`) need slash preservation for the resolve endpoint
// to find them.
func escapeFilename(filename string) string {
	parts := strings.Split(filename, "/")
	for i, p := range parts {
		parts[i] = url.PathEscape(p)
	}
	return strings.Join(parts, "/")
}

// unquoteETag strips surrounding quotes and the optional `W/` weak prefix
// from an HTTP ETag header. Used to derive a filesystem-safe blob name from
// the server's ETag (HF cache stores blobs as `blobs/<unquoted-etag>`).
func unquoteETag(etag string) string {
	etag = strings.TrimPrefix(etag, "W/")
	etag = strings.Trim(etag, `"`)
	return etag
}

// quoteETagForHeader formats a stored (unquoted) etag back into the wire
// format HTTP headers expect (RFC 7232/7233: `"<value>"`). Used for
// `If-Range`, `If-Match`, `If-None-Match`, etc.
//
// We deliberately store etags unquoted everywhere else in this package
// because they double as filesystem blob names. This helper is the
// single place that re-adds the quotes when sending the value back to
// the server. If the etag already starts with a quote (defensive — should
// not happen given unquoteETag is the only writer) it is returned unchanged.
func quoteETagForHeader(etag string) string {
	if etag == "" {
		return ""
	}
	if strings.HasPrefix(etag, `"`) || strings.HasPrefix(etag, `W/"`) {
		return etag
	}
	return `"` + etag + `"`
}

// applyHeaders sets the User-Agent and (when present) Authorization headers.
func (c *Client) applyHeaders(req *http.Request) {
	if c.userAgent != "" {
		req.Header.Set("User-Agent", c.userAgent)
	}
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
}

// classifyHTTPStatus turns a non-2xx response into a structured error.
// 2xx responses return nil. Reads (and discards) the body so the connection
// can be reused.
func classifyHTTPStatus(resp *http.Response, repoID string) error {
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	bodyStr := strings.ToLower(string(body))
	switch resp.StatusCode {
	case http.StatusUnauthorized:
		return ErrUnauthorized
	case http.StatusForbidden:
		// Distinguish gated from generic 403 via response body / X-Error-Code header.
		errCode := resp.Header.Get("X-Error-Code")
		if errCode == "GatedRepo" || strings.Contains(bodyStr, "gated") || strings.Contains(bodyStr, "access to this repo") {
			return &GatedError{ModelID: repoID}
		}
		return ErrForbidden
	case http.StatusNotFound:
		return &NotFoundError{ModelID: repoID}
	case http.StatusGone, http.StatusUnavailableForLegalReasons:
		return ErrRemoved
	default:
		return fmt.Errorf("huggingface hub returned status %d for %q", resp.StatusCode, repoID)
	}
}

// classifyTransportError wraps a low-level transport error (DNS, connection
// refused, timeout, ...) in a NetworkError with actionable guidance.
func classifyTransportError(err error) error {
	if err == nil {
		return nil
	}
	// Already classified — pass through.
	var oe *OfflineError
	if errors.As(err, &oe) {
		return err
	}
	var dnsErr *net.DNSError
	if errors.As(err, &dnsErr) {
		return &NetworkError{Cause: err}
	}
	var netErr net.Error
	if errors.As(err, &netErr) {
		return &NetworkError{Cause: err}
	}
	// Wrap context errors as transport so callers see the same shape.
	if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
		return &NetworkError{Cause: err}
	}
	return &NetworkError{Cause: err}
}
