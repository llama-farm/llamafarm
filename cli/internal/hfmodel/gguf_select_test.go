package hfmodel

// Tests in this file are mirrored from common/tests/test_model_utils.py.
// Any new test added here should be added to the Python suite as well, and
// vice versa. The two implementations form a "drift contract" that catches
// regressions on either side.

import (
	"errors"
	"testing"
)

func TestParseQuantizationFromFilename(t *testing.T) {
	tests := []struct {
		name     string
		filename string
		want     string
	}{
		{"q4_k_m", "qwen3-1.7b.Q4_K_M.gguf", "Q4_K_M"},
		{"q8_0", "model.Q8_0.gguf", "Q8_0"},
		{"f16", "llama-3.2-3b.F16.gguf", "F16"},
		{"q5_k_s", "model.Q5_K_S.gguf", "Q5_K_S"},
		{"case_insensitive", "model.q4_k_m.gguf", "Q4_K_M"},
		{"no_quantization", "model.gguf", ""},
		{"complex_filename", "unsloth_qwen3-1.7b-instruct.Q4_K_M.gguf", "Q4_K_M"},
		{"fp16_normalized", "model.FP16.gguf", "F16"},
		{"fp32_normalized", "model.FP32.gguf", "F32"},
		{"imatrix_iq4_xs", "model.IQ4_XS.gguf", "IQ4_XS"},
		// Q2_K/Q6_K/Q3_K return None in Python (verified):
		// the negative lookahead `(?![_.])` rejects the trailing `.gguf`
		// extension. This is a Python quirk we must mirror to keep the
		// drift contract.
		{"q6_k_returns_empty", "model.Q6_K.gguf", ""},
		{"q2_k_returns_empty", "model.Q2_K.gguf", ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ParseQuantizationFromFilename(tc.filename)
			if got != tc.want {
				t.Errorf("ParseQuantizationFromFilename(%q) = %q, want %q", tc.filename, got, tc.want)
			}
		})
	}
}

func TestParseModelWithQuantization(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantID    string
		wantQuant string
	}{
		{"with_quant", "unsloth/Qwen3-4B-GGUF:Q4_K_M", "unsloth/Qwen3-4B-GGUF", "Q4_K_M"},
		{"lowercase_normalized", "unsloth/Qwen3-4B-GGUF:q8_0", "unsloth/Qwen3-4B-GGUF", "Q8_0"},
		{"no_quant", "unsloth/Qwen3-4B-GGUF", "unsloth/Qwen3-4B-GGUF", ""},
		{"multiple_colons_last_wins", "org:user/model:Q4_K_M", "org:user/model", "Q4_K_M"},
		{"empty_quant", "unsloth/Qwen3-4B-GGUF:", "unsloth/Qwen3-4B-GGUF", ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gotID, gotQuant := ParseModelWithQuantization(tc.input)
			if gotID != tc.wantID || gotQuant != tc.wantQuant {
				t.Errorf("ParseModelWithQuantization(%q) = (%q, %q), want (%q, %q)",
					tc.input, gotID, gotQuant, tc.wantID, tc.wantQuant)
			}
		})
	}
}

func TestIsSplitGGUFFile(t *testing.T) {
	tests := []struct {
		filename string
		want     bool
	}{
		{"model-00001-of-00002.gguf", true},
		{"model-00001-of-00002.Q4_K_M.gguf", true},
		{"qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf", true},
		{"model.Q4_K_M.gguf", false},
		{"model-v2.Q4_K_M.gguf", false},
		{"model.gguf", false},
	}
	for _, tc := range tests {
		t.Run(tc.filename, func(t *testing.T) {
			if got := IsSplitGGUFFile(tc.filename); got != tc.want {
				t.Errorf("IsSplitGGUFFile(%q) = %v, want %v", tc.filename, got, tc.want)
			}
		})
	}
}

func TestSelectGGUFFile(t *testing.T) {
	tests := []struct {
		name      string
		files     []string
		preferred string
		want      string
	}{
		{
			name:  "single_file",
			files: []string{"model.Q8_0.gguf"},
			want:  "model.Q8_0.gguf",
		},
		{
			name: "default_q4_k_m",
			files: []string{
				"model.Q2_K.gguf",
				"model.Q4_K_M.gguf",
				"model.Q8_0.gguf",
				"model.F16.gguf",
			},
			want: "model.Q4_K_M.gguf",
		},
		{
			name:      "preferred_quantization",
			files:     []string{"model.Q4_K_M.gguf", "model.Q8_0.gguf", "model.F16.gguf"},
			preferred: "Q8_0",
			want:      "model.Q8_0.gguf",
		},
		{
			name:      "preferred_case_insensitive",
			files:     []string{"model.Q4_K_M.gguf", "model.Q8_0.gguf"},
			preferred: "q8_0",
			want:      "model.Q8_0.gguf",
		},
		{
			name:      "fallback_when_preferred_not_found",
			files:     []string{"model.Q4_K_M.gguf", "model.Q8_0.gguf"},
			preferred: "F16",
			want:      "model.Q4_K_M.gguf",
		},
		{
			name:  "priority_q5_k_m_when_no_q4",
			files: []string{"model.Q8_0.gguf", "model.Q5_K_M.gguf", "model.F16.gguf"},
			want:  "model.Q5_K_M.gguf",
		},
		{
			name:  "priority_q8_0_when_no_q4_q5",
			files: []string{"model.Q8_0.gguf", "model.F16.gguf", "model.Q2_K.gguf"},
			want:  "model.Q8_0.gguf",
		},
		{
			name:  "first_when_no_quantization",
			files: []string{"model_a.gguf", "model_b.gguf"},
			want:  "model_a.gguf",
		},
		{
			name:  "empty_list",
			files: []string{},
			want:  "",
		},
		{
			name: "prefers_non_split",
			files: []string{
				"model-00001-of-00002.Q4_K_M.gguf",
				"model-00002-of-00002.Q4_K_M.gguf",
				"model.Q4_K_M.gguf",
				"model.Q8_0.gguf",
			},
			want: "model.Q4_K_M.gguf",
		},
		{
			name: "uses_split_for_unique_quant",
			files: []string{
				"model-00001-of-00002.F16.gguf",
				"model-00002-of-00002.F16.gguf",
				"model.Q4_K_M.gguf",
			},
			preferred: "F16",
			want:      "model-00001-of-00002.F16.gguf",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := SelectGGUFFile(tc.files, tc.preferred)
			if got != tc.want {
				t.Errorf("SelectGGUFFile(%v, %q) = %q, want %q", tc.files, tc.preferred, got, tc.want)
			}
		})
	}
}

func TestQuantPreferenceOrderDefined(t *testing.T) {
	if QuantPreferenceOrder[0] != "Q4_K_M" {
		t.Errorf("first preference should be Q4_K_M, got %q", QuantPreferenceOrder[0])
	}
	found := map[string]bool{}
	for _, q := range QuantPreferenceOrder {
		found[q] = true
	}
	for _, expected := range []string{"Q8_0", "F16"} {
		if !found[expected] {
			t.Errorf("preference order missing %q", expected)
		}
	}
}

func TestValidateModelID(t *testing.T) {
	tests := []struct {
		id      string
		wantErr bool
	}{
		{"unsloth/Qwen3-1.7B-GGUF", false},
		{"meta-llama/Llama-2-7b-hf", false},
		{"some-model", false},
		{"org_with_underscores/model.with.dots", false},
		{"../etc/passwd", true},
		{"/absolute/path", true},
		{"\\windows\\path", true},
		{"has spaces/model", true},
		{"has@special/chars", true},
	}
	for _, tc := range tests {
		t.Run(tc.id, func(t *testing.T) {
			err := ValidateModelID(tc.id)
			if (err != nil) != tc.wantErr {
				t.Errorf("ValidateModelID(%q) err=%v, wantErr=%v", tc.id, err, tc.wantErr)
			}
			if tc.wantErr && err != nil {
				var modelErr *InvalidModelIDError
				if !errors.As(err, &modelErr) {
					t.Errorf("err should be *InvalidModelIDError, got %T", err)
				}
			}
		})
	}
}
