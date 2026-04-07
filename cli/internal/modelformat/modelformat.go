// Package modelformat detects the format of model files by file extension and
// magic bytes. It mirrors the subset of logic used by runtimes/edge/utils/model_format.py
// that the CLI needs for `lf models path`.
package modelformat

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Kind identifies a model file format.
type Kind string

const (
	// KindUnknown is returned when the file cannot be classified.
	KindUnknown Kind = "unknown"
	// KindGGUF is a .gguf file — llama.cpp compatible quantized weights.
	KindGGUF Kind = "gguf"
	// KindMmproj is a GGUF multimodal projector file (carries an mmproj hint in its name).
	KindMmproj Kind = "mmproj"
	// KindUltralytics is an Ultralytics .pt/.pth checkpoint (YOLO family).
	KindUltralytics Kind = "ultralytics"
	// KindTransformers is a HuggingFace Transformers model directory
	// (config.json + weights). Represented here only for rejection in V1.
	KindTransformers Kind = "transformers"
	// KindUnsupported marks kinds that the CLI recognizes but does not yet
	// emit a transport plan for. V1 only supports GGUF.
	KindUnsupported Kind = "unsupported"
)

// ErrFileNotFound is returned when Detect is called on a missing path.
var ErrFileNotFound = errors.New("file not found")

// ggufMagic is the 4-byte prefix of every GGUF v2/v3 file.
var ggufMagic = []byte{'G', 'G', 'U', 'F'}

// Detect inspects a path and returns its model Kind. The path may be a file or
// a directory; directories are classified if they contain transformers-style
// files. Unknown files return KindUnknown, not an error.
func Detect(path string) (Kind, error) {
	info, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return KindUnknown, fmt.Errorf("%w: %s", ErrFileNotFound, path)
		}
		return KindUnknown, err
	}
	if info.IsDir() {
		return detectDir(path)
	}
	return detectFile(path)
}

// detectFile classifies a single file by extension + magic bytes.
func detectFile(path string) (Kind, error) {
	nameLower := strings.ToLower(filepath.Base(path))

	// GGUF: check extension and magic bytes. The magic-byte check guards
	// against misleading filenames.
	if strings.HasSuffix(nameLower, ".gguf") {
		ok, err := hasGGUFMagic(path)
		if err != nil {
			return KindUnknown, err
		}
		if !ok {
			return KindUnknown, nil
		}
		if IsMmprojName(nameLower) {
			return KindMmproj, nil
		}
		return KindGGUF, nil
	}

	// Ultralytics: .pt / .pth extension (we don't attempt to verify torch
	// pickle contents; the runtime will sanity-check at load time).
	if strings.HasSuffix(nameLower, ".pt") || strings.HasSuffix(nameLower, ".pth") {
		return KindUltralytics, nil
	}

	return KindUnknown, nil
}

// detectDir classifies a directory as transformers if it contains both a
// config.json and a weights file (*.safetensors or *.bin).
func detectDir(path string) (Kind, error) {
	hasConfig := fileExists(filepath.Join(path, "config.json"))
	if !hasConfig {
		return KindUnknown, nil
	}
	entries, err := os.ReadDir(path)
	if err != nil {
		return KindUnknown, err
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		n := strings.ToLower(entry.Name())
		if strings.HasSuffix(n, ".safetensors") || strings.HasSuffix(n, ".bin") {
			// V1 does not yet emit plans for transformers models. Return
			// KindUnsupported to let callers fail loudly with a clear message.
			return KindUnsupported, nil
		}
	}
	return KindUnknown, nil
}

// hasGGUFMagic returns true if the first four bytes of the file are "GGUF".
func hasGGUFMagic(path string) (bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer f.Close()
	buf := make([]byte, 4)
	n, err := f.Read(buf)
	if err != nil {
		return false, nil // Short reads are not an error for classification.
	}
	if n < 4 {
		return false, nil
	}
	return string(buf) == string(ggufMagic), nil
}

// IsMmprojName reports whether a GGUF filename represents a multimodal
// projector companion file rather than the main model weights.
// Mirrors the heuristic in _is_mmproj_file() in common/llamafarm_common/model_utils.py.
func IsMmprojName(name string) bool {
	name = strings.ToLower(name)
	if !strings.HasSuffix(name, ".gguf") {
		return false
	}
	if !strings.Contains(name, "mmproj") && !strings.Contains(name, "multimodal") {
		return false
	}
	// Exclude main weights files that happen to contain "multimodal" in the name.
	quants := []string{"q2_", "q3_", "q4_", "q5_", "q6_", "q8_"}
	for _, q := range quants {
		if strings.Contains(name, q) {
			return false
		}
	}
	return true
}

// ParseQuantization extracts a GGUF quantization label from a filename such as
// "qwen3-1.7b.Q4_K_M.gguf". Returns "" if none can be detected. Normalizes
// FP16 → F16 and FP32 → F32.
func ParseQuantization(name string) string {
	name = strings.ToLower(name)
	// Most common quantization families, ordered from most specific to least.
	// We check after splitting on common delimiters.
	candidates := []string{
		"q2_k", "q3_k_s", "q3_k_m", "q3_k_l",
		"q4_0", "q4_1", "q4_k_s", "q4_k_m", "q4_k",
		"q5_0", "q5_1", "q5_k_s", "q5_k_m", "q5_k",
		"q6_k", "q8_0",
		"iq2_k", "iq3_k", "iq4_xs", "iq3_xs",
		"f16", "f32", "fp16", "fp32",
	}
	// Longest-first matching avoids Q4 shadowing Q4_K_M.
	type sized struct {
		label string
	}
	// Naive: try each candidate as a token surrounded by delimiters.
	name = strings.ReplaceAll(name, "-", ".")
	name = strings.ReplaceAll(name, "_", "_") // no-op; kept for intent
	for _, c := range candidates {
		// Match patterns like .q4_k_m. or .q4_k_m at end before .gguf
		needleA := "." + c + "."
		needleB := "." + c
		if strings.Contains(name, needleA) || strings.HasSuffix(name, needleB) {
			label := strings.ToUpper(c)
			if label == "FP16" {
				return "F16"
			}
			if label == "FP32" {
				return "F32"
			}
			return label
		}
	}
	return ""
}

// ParseMmprojPrecision extracts the precision tag from an mmproj filename,
// e.g. "mmproj-qwen-f16.gguf" → "f16". Returns "" if unknown.
func ParseMmprojPrecision(name string) string {
	name = strings.ToLower(name)
	for _, p := range []string{"f16", "bf16", "fp16", "f32", "fp32"} {
		for _, sep := range []string{"-", "_", "."} {
			if strings.Contains(name, sep+p+".") || strings.HasSuffix(name, sep+p+".gguf") {
				// Normalize.
				switch p {
				case "fp16":
					return "f16"
				case "fp32":
					return "f32"
				}
				return p
			}
		}
	}
	return ""
}

func fileExists(p string) bool {
	info, err := os.Stat(p)
	return err == nil && !info.IsDir()
}
