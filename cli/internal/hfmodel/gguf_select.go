package hfmodel

// GGUF selection logic ported from
// common/llamafarm_common/model_utils.py. This file is the "drift contract"
// with the Python implementation: any change here MUST be mirrored in Python
// (or vice versa), and the table-driven tests in gguf_select_test.go MUST
// stay in sync with common/tests/test_model_utils*.py.
//
// Functions in this file are pure: no I/O, no network, no logging side
// effects. They take filename strings and return filename strings.

import (
	"regexp"
	"strings"
)

// QuantPreferenceOrder mirrors GGUF_QUANTIZATION_PREFERENCE_ORDER in
// model_utils.py. The order encodes a balance of size vs quality, with Q4_K_M
// as the default for most users.
var QuantPreferenceOrder = []string{
	"Q4_K_M", // Best default: good balance of size and quality
	"Q4_K",   // Generic Q4_K
	"Q5_K_M", // Slightly higher quality, larger size
	"Q5_K",   // Generic Q5_K
	"Q8_0",   // High quality, larger size
	"Q6_K",   // Between Q5 and Q8
	"Q4_K_S", // Smaller Q4 variant
	"Q5_K_S", // Smaller Q5 variant
	"Q3_K_M", // Smaller, lower quality
	"Q2_K",   // Very small, lower quality
	"F16",    // Full precision, very large
}

// ParseModelWithQuantization splits a model name like "org/repo:Q4_K_M" into
// the base model id and the (uppercased) quantization. Returns the input and
// "" when no suffix is present. Mirrors parse_model_with_quantization.
func ParseModelWithQuantization(name string) (modelID string, quant string) {
	idx := strings.LastIndex(name, ":")
	if idx == -1 {
		return name, ""
	}
	q := name[idx+1:]
	return name[:idx], strings.ToUpper(q)
}

// Quantization regex patterns ported from parse_quantization_from_filename.
// Order matters — more specific patterns first so they win over generic ones.
//
// The "I?" prefix on Q-patterns covers imatrix quantization variants
// (IQ2_K, IQ3_K, IQ4_XS, ...). FP16/FP32 are normalized to F16/F32 below.
var quantPatterns = []*regexp.Regexp{
	// Patterns with separators (most common)
	regexp.MustCompile(`(?i)[._-](I?Q[2-8]_K_[SML])`),    // Q3_K_S, Q4_K_M, IQ2_K_S, etc.
	regexp.MustCompile(`(?i)[._-](I?Q[2-8]_[01])`),       // Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
	regexp.MustCompile(`(?i)[._-](I?Q[2-8]_K)(?:[^_.]|$)`), // Q2_K, Q3_K, Q6_K (not followed by _ or .)
	regexp.MustCompile(`(?i)[._-](I?Q[2-8]_XS)`),         // IQ4_XS, IQ3_XS, etc.
	regexp.MustCompile(`(?i)[._-](F16|F32|FP16|FP32)`),

	// Patterns without separators (less common but possible)
	regexp.MustCompile(`(?i)(I?Q[2-8]_K_[SML])`),
	regexp.MustCompile(`(?i)(I?Q[2-8]_[01])`),
	regexp.MustCompile(`(?i)(I?Q[2-8]_K)(?:[^_.]|$)`),
	regexp.MustCompile(`(?i)(I?Q[2-8]_XS)`),
	regexp.MustCompile(`(?i)(F16|F32|FP16|FP32)`),
}

// quantValidator matches the canonical normalized form. Mirrors the final
// `re.match` in parse_quantization_from_filename.
var quantValidator = regexp.MustCompile(`^(I?Q[2-8](?:_K(?:_[SML])?|_[01]|_K|_XS)|F(?:16|32))$`)

// ParseQuantizationFromFilename extracts a quantization label like "Q4_K_M",
// "IQ4_XS", or "F16" from a GGUF filename. Returns "" when no quantization is
// recognizable. Mirrors parse_quantization_from_filename.
func ParseQuantizationFromFilename(filename string) string {
	for _, pat := range quantPatterns {
		m := pat.FindStringSubmatch(filename)
		if len(m) < 2 {
			continue
		}
		quant := strings.ToUpper(m[1])
		// Normalize FP16/FP32 to F16/F32, matching the Python implementation.
		switch quant {
		case "FP16":
			quant = "F16"
		case "FP32":
			quant = "F32"
		}
		if quantValidator.MatchString(quant) {
			return quant
		}
	}
	return ""
}

// splitGGUFRegex matches multi-part GGUF shards like
// "model-00001-of-00002.gguf" or "qwen-q4_k_m-00001-of-00002.gguf". Mirrors
// is_split_gguf_file.
var splitGGUFRegex = regexp.MustCompile(`(?i)-\d{5}-of-\d{5}[.\-]`)

// IsSplitGGUFFile reports whether a filename is part of a multi-file GGUF
// shard set. Mirrors is_split_gguf_file.
func IsSplitGGUFFile(filename string) bool {
	return splitGGUFRegex.MatchString(filename)
}

// SelectGGUFFile picks the best GGUF file from a list based on a quantization
// preference. Selection rules, in order:
//
//  1. If only one file, return it.
//  2. Filter out split-shard files when a non-split version exists.
//  3. If preferred is non-empty, return the file matching it (case-insensitive).
//     Falls back to a split-file with the same quant if no non-split matches.
//  4. Walk QuantPreferenceOrder and return the first matching file.
//  5. Fall back to the first file in the working set.
//
// Returns "" only when the input list is empty. Mirrors select_gguf_file.
func SelectGGUFFile(files []string, preferred string) string {
	if len(files) == 0 {
		return ""
	}
	if len(files) == 1 {
		return files[0]
	}

	var nonSplit, splits []string
	for _, f := range files {
		if IsSplitGGUFFile(f) {
			splits = append(splits, f)
		} else {
			nonSplit = append(nonSplit, f)
		}
	}
	working := nonSplit
	if len(working) == 0 {
		working = splits
	}

	if preferred != "" {
		want := strings.ToUpper(preferred)
		for _, f := range working {
			if strings.EqualFold(ParseQuantizationFromFilename(f), want) {
				return f
			}
		}
		// Fall back to split files when the preferred quant isn't in the
		// non-split set. Mirrors the Python branch.
		if len(nonSplit) > 0 && len(splits) > 0 {
			for _, f := range splits {
				if strings.EqualFold(ParseQuantizationFromFilename(f), want) {
					return f
				}
			}
		}
	}

	// Default preference order.
	for _, pref := range QuantPreferenceOrder {
		for _, f := range working {
			if strings.EqualFold(ParseQuantizationFromFilename(f), pref) {
				return f
			}
		}
	}

	if len(working) > 0 {
		return working[0]
	}
	return files[0]
}

// modelIDPattern matches a HuggingFace repo id: "org/name" or just "name".
// Allows alphanumerics, hyphens, underscores, and periods. Mirrors
// _validate_model_id.
var modelIDPattern = regexp.MustCompile(`^[a-zA-Z0-9_.\-]+(/[a-zA-Z0-9_.\-]+)?$`)

// ValidateModelID returns nil iff the model id is safe for use in filesystem
// paths and HF API URLs. Rejects path-traversal attempts and unrecognized
// characters. Mirrors _validate_model_id.
func ValidateModelID(id string) error {
	if strings.Contains(id, "..") || strings.HasPrefix(id, "/") || strings.HasPrefix(id, `\`) {
		return &InvalidModelIDError{ID: id, Reason: "path traversal not allowed"}
	}
	if !modelIDPattern.MatchString(id) {
		return &InvalidModelIDError{ID: id, Reason: "invalid format"}
	}
	return nil
}

// InvalidModelIDError signals a malformed or unsafe HuggingFace model id.
type InvalidModelIDError struct {
	ID     string
	Reason string
}

func (e *InvalidModelIDError) Error() string {
	return "invalid model id " + e.ID + ": " + e.Reason
}
