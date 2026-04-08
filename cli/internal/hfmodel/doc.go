// Package hfmodel is the write-side companion to cli/internal/hfcache. It
// owns the HuggingFace Hub cache write path bit-perfectly so that Python
// loaders (transformers.from_pretrained, huggingface_hub.try_to_load_from_cache,
// llama_cpp.Llama) can find files written by Go without any code changes on
// the Python side.
//
// This package exists so that `lf models pull` can fetch models from the
// HuggingFace Hub without booting the LlamaFarm server or requiring a Python
// interpreter. It mirrors the relevant subset of huggingface_hub's download
// machinery and the GGUF quantization-selection logic from
// common/llamafarm_common/model_utils.py.
//
// Layered structure:
//
//   - gguf_select.go    Pure-logic GGUF quantization selection. Mirrors
//                       common/llamafarm_common/model_utils.py.
//   - token.go          HF token discovery (env vars + token files).
//   - offline.go        LLAMAFARM_OFFLINE detection.
//   - errors.go         Structured error types with remediation messages.
//   - client.go         HF Hub HTTP API client (tree, metadata, file fetch).
//   - downloader.go     Streaming downloader, blob writes, snapshot symlinks.
//   - lock_unix.go      File locks compatible with huggingface_hub.filelock.
//   - lock_windows.go   Windows equivalent via LockFileEx.
//
// The "drift contract" with common/llamafarm_common/model_utils.py is
// maintained by table-driven tests using fixtures copied from
// common/tests/test_model_utils*.py. Any divergence is a bug.
package hfmodel
