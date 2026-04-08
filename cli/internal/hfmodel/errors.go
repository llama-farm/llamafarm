package hfmodel

import (
	"errors"
	"fmt"
)

// Sentinel errors for matching with errors.Is. Each has an Error() that
// includes user-actionable remediation guidance — generic 4xx/5xx messages
// without remediation are not acceptable in this package.

// ErrUnauthorized signals an HTTP 401 from the HuggingFace Hub: the request
// requires authentication. Remediation: run `huggingface-cli login` or set
// $HF_TOKEN.
var ErrUnauthorized = errors.New("huggingface hub returned 401 unauthorized: run `huggingface-cli login` or set HF_TOKEN")

// ErrForbidden signals an HTTP 403 from the HuggingFace Hub for a non-gated
// reason. Most often this means the token is valid but lacks permission for
// the resource.
var ErrForbidden = errors.New("huggingface hub returned 403 forbidden: your token does not have access to this resource")

// ErrGated signals an HTTP 403 specifically because the model is gated.
// The error message includes a URL the user can visit to accept terms.
type GatedError struct {
	ModelID string
}

func (e *GatedError) Error() string {
	return fmt.Sprintf("model %q is gated: visit https://huggingface.co/%s to accept the terms", e.ModelID, e.ModelID)
}

func (e *GatedError) Is(target error) bool {
	_, ok := target.(*GatedError)
	return ok || errors.Is(target, errGatedSentinel)
}

// errGatedSentinel allows `errors.Is(err, ErrGated)` matching.
var errGatedSentinel = errors.New("gated")

// ErrGated is the sentinel value for `errors.Is`. Use *GatedError as the
// concrete type when you need to extract the model id.
var ErrGated = errGatedSentinel

// NotFoundError signals an HTTP 404 from the HuggingFace Hub. Includes the
// model id so the message can suggest spelling correction.
type NotFoundError struct {
	ModelID string
}

func (e *NotFoundError) Error() string {
	return fmt.Sprintf("model %q not found on huggingface hub: check the spelling and that the repo is public (or that your token has access)", e.ModelID)
}

func (e *NotFoundError) Is(target error) bool {
	_, ok := target.(*NotFoundError)
	return ok || errors.Is(target, errNotFoundSentinel)
}

var errNotFoundSentinel = errors.New("not found")

// ErrNotFound is the sentinel value for `errors.Is`.
var ErrNotFound = errNotFoundSentinel

// ErrRemoved signals an HTTP 410/451 — the model was removed or is
// unavailable for legal reasons in your region.
var ErrRemoved = errors.New("model has been removed or is unavailable in your region")

// NetworkError wraps a low-level transport error (DNS failure, connection
// refused, timeout) with a message that points at LLAMAFARM_OFFLINE.
type NetworkError struct {
	Cause error
}

func (e *NetworkError) Error() string {
	return fmt.Sprintf("cannot reach huggingface.co: %v — check your connection or set LLAMAFARM_OFFLINE=1 to use only the local cache", e.Cause)
}

func (e *NetworkError) Unwrap() error { return e.Cause }

// OfflineError signals that a network call was refused because
// LLAMAFARM_OFFLINE is set. The message names the model and points at the
// remediation `lf models pull` would emit.
type OfflineError struct {
	ModelID string
	Op      string // operation that was refused, e.g. "list_repo_tree", "download_file"
}

func (e *OfflineError) Error() string {
	if e.ModelID != "" {
		return fmt.Sprintf("refused %s for %q in offline mode (LLAMAFARM_OFFLINE=1): run `lf models pull %s` on a host with internet, then sync the files",
			e.Op, e.ModelID, e.ModelID)
	}
	return fmt.Sprintf("refused %s in offline mode (LLAMAFARM_OFFLINE=1)", e.Op)
}

func (e *OfflineError) Is(target error) bool {
	_, ok := target.(*OfflineError)
	return ok || errors.Is(target, errOfflineSentinel)
}

var errOfflineSentinel = errors.New("offline")

// ErrOffline is the sentinel value for `errors.Is`.
var ErrOffline = errOfflineSentinel
