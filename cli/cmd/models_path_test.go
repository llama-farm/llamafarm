package cmd

import (
	"testing"

	"github.com/llamafarm/cli/cmd/config"
)

func TestParseModelSpec(t *testing.T) {
	cases := map[string]struct {
		repo, quant string
	}{
		"unsloth/Qwen3-1.7B-GGUF":          {"unsloth/Qwen3-1.7B-GGUF", ""},
		"unsloth/Qwen3-1.7B-GGUF:Q4_K_M":   {"unsloth/Qwen3-1.7B-GGUF", "Q4_K_M"},
		"unsloth/Qwen3-1.7B-GGUF:q8_0":     {"unsloth/Qwen3-1.7B-GGUF", "Q8_0"},
	}
	for input, want := range cases {
		r, q := parseModelSpec(input)
		if r != want.repo || q != want.quant {
			t.Errorf("parseModelSpec(%q) = (%q,%q), want (%q,%q)", input, r, q, want.repo, want.quant)
		}
	}
}

func TestCanonicalWeightsName(t *testing.T) {
	if got := canonicalWeightsName("Q4_K_M"); got != "model.Q4_K_M.gguf" {
		t.Errorf("got %q", got)
	}
	if got := canonicalWeightsName(""); got != "model.gguf" {
		t.Errorf("got %q", got)
	}
}

func TestSelectModelsByAlias_All(t *testing.T) {
	all := []config.LlamaFarmConfigRuntimeModelsElem{
		{Name: "a", Model: "org/a-gguf"},
		{Name: "b", Model: "org/b-gguf"},
	}
	got, err := selectModelsByAlias(all, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Errorf("got %d, want 2", len(got))
	}
}

func TestSelectModelsByAlias_Filter(t *testing.T) {
	all := []config.LlamaFarmConfigRuntimeModelsElem{
		{Name: "a", Model: "org/a-gguf"},
		{Name: "b", Model: "org/b-gguf"},
		{Name: "c", Model: "org/c-gguf"},
	}
	got, err := selectModelsByAlias(all, []string{"b", "a"})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d, want 2", len(got))
	}
	if got[0].Name != "b" || got[1].Name != "a" {
		t.Errorf("expected order preserved: %v", got)
	}
}

func TestSelectModelsByAlias_UnknownNameErrors(t *testing.T) {
	all := []config.LlamaFarmConfigRuntimeModelsElem{
		{Name: "a", Model: "org/a-gguf"},
	}
	_, err := selectModelsByAlias(all, []string{"nope"})
	if err == nil {
		t.Error("expected error for unknown alias")
	}
}

func TestLooksLikeGGUFModel(t *testing.T) {
	cases := []struct {
		name string
		m    config.LlamaFarmConfigRuntimeModelsElem
		want bool
	}{
		{
			name: "explicit quant suffix",
			m:    config.LlamaFarmConfigRuntimeModelsElem{Model: "unsloth/Qwen3:Q4_K_M"},
			want: true,
		},
		{
			name: "GGUF in repo name",
			m:    config.LlamaFarmConfigRuntimeModelsElem{Model: "unsloth/Qwen3-1.7B-GGUF"},
			want: true,
		},
		{
			name: "non-gguf transformers model",
			m:    config.LlamaFarmConfigRuntimeModelsElem{Model: "bert-base-uncased"},
			want: false,
		},
	}
	for _, c := range cases {
		got := looksLikeGGUFModel(c.m)
		if got != c.want {
			t.Errorf("%s: got %v, want %v", c.name, got, c.want)
		}
	}
}
