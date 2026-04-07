package modelformat

import (
	"os"
	"path/filepath"
	"testing"
)

func writeFile(t *testing.T, dir, name string, content []byte) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, content, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestDetect_GGUF(t *testing.T) {
	tmp := t.TempDir()
	p := writeFile(t, tmp, "model.Q4_K_M.gguf", append([]byte{'G', 'G', 'U', 'F'}, 0x00, 0x01, 0x02))
	kind, err := Detect(p)
	if err != nil {
		t.Fatal(err)
	}
	if kind != KindGGUF {
		t.Errorf("got %s, want %s", kind, KindGGUF)
	}
}

func TestDetect_GGUFWithoutMagic(t *testing.T) {
	tmp := t.TempDir()
	p := writeFile(t, tmp, "fake.gguf", []byte("NOTGGUF"))
	kind, err := Detect(p)
	if err != nil {
		t.Fatal(err)
	}
	if kind != KindUnknown {
		t.Errorf("got %s, want %s", kind, KindUnknown)
	}
}

func TestDetect_Mmproj(t *testing.T) {
	tmp := t.TempDir()
	p := writeFile(t, tmp, "mmproj-qwen-f16.gguf", []byte("GGUF\x00\x01\x02"))
	kind, err := Detect(p)
	if err != nil {
		t.Fatal(err)
	}
	if kind != KindMmproj {
		t.Errorf("got %s, want %s", kind, KindMmproj)
	}
}

func TestDetect_UltralyticsPt(t *testing.T) {
	tmp := t.TempDir()
	p := writeFile(t, tmp, "yolo11n.pt", []byte{0x80, 0x04})
	kind, err := Detect(p)
	if err != nil {
		t.Fatal(err)
	}
	if kind != KindUltralytics {
		t.Errorf("got %s, want %s", kind, KindUltralytics)
	}
}

func TestDetect_UltralyticsPth(t *testing.T) {
	tmp := t.TempDir()
	p := writeFile(t, tmp, "yolo11n.pth", []byte{0x80, 0x04})
	kind, err := Detect(p)
	if err != nil {
		t.Fatal(err)
	}
	if kind != KindUltralytics {
		t.Errorf("got %s, want %s", kind, KindUltralytics)
	}
}

func TestDetect_TransformersDirectory(t *testing.T) {
	tmp := t.TempDir()
	writeFile(t, tmp, "config.json", []byte(`{"model_type": "llama"}`))
	writeFile(t, tmp, "model.safetensors", []byte("stub"))
	kind, err := Detect(tmp)
	if err != nil {
		t.Fatal(err)
	}
	if kind != KindUnsupported {
		t.Errorf("got %s, want %s (V1 does not yet emit plans for transformers)", kind, KindUnsupported)
	}
}

func TestDetect_TransformersDirectoryWithBinWeights(t *testing.T) {
	tmp := t.TempDir()
	writeFile(t, tmp, "config.json", []byte(`{}`))
	writeFile(t, tmp, "pytorch_model.bin", []byte("stub"))
	kind, err := Detect(tmp)
	if err != nil {
		t.Fatal(err)
	}
	if kind != KindUnsupported {
		t.Errorf("got %s, want %s", kind, KindUnsupported)
	}
}

func TestDetect_UnknownFile(t *testing.T) {
	tmp := t.TempDir()
	p := writeFile(t, tmp, "random.bin", []byte("random bytes"))
	kind, err := Detect(p)
	if err != nil {
		t.Fatal(err)
	}
	if kind != KindUnknown {
		t.Errorf("got %s, want %s", kind, KindUnknown)
	}
}

func TestDetect_EmptyFile(t *testing.T) {
	tmp := t.TempDir()
	p := writeFile(t, tmp, "empty.gguf", nil)
	kind, err := Detect(p)
	if err != nil {
		t.Fatal(err)
	}
	if kind != KindUnknown {
		t.Errorf("got %s, want %s (empty .gguf has no magic)", kind, KindUnknown)
	}
}

func TestDetect_MissingFile(t *testing.T) {
	_, err := Detect("/nonexistent/path/does/not/exist")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

func TestIsMmprojName(t *testing.T) {
	cases := []struct {
		name string
		want bool
	}{
		{"mmproj-qwen-f16.gguf", true},
		{"mmproj.f16.gguf", true},
		{"multimodal-f16.gguf", true},
		{"qwen.Q4_K_M.gguf", false},
		{"model-multimodal.Q4_K_M.gguf", false}, // quant present → treat as weights
		{"readme.txt", false},
	}
	for _, c := range cases {
		got := IsMmprojName(c.name)
		if got != c.want {
			t.Errorf("IsMmprojName(%q) = %v, want %v", c.name, got, c.want)
		}
	}
}

func TestParseQuantization(t *testing.T) {
	cases := map[string]string{
		"qwen3-1.7b.Q4_K_M.gguf":    "Q4_K_M",
		"model.Q8_0.gguf":            "Q8_0",
		"model.F16.gguf":             "F16",
		"model-f16.gguf":             "F16",
		"model.fp32.gguf":            "F32",
		"model-q3_k_l.gguf":          "Q3_K_L",
		"smollm-135m-q4_k_m.gguf":    "Q4_K_M",
		"noquant.gguf":               "",
	}
	for input, want := range cases {
		got := ParseQuantization(input)
		if got != want {
			t.Errorf("ParseQuantization(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestParseMmprojPrecision(t *testing.T) {
	cases := map[string]string{
		"mmproj-qwen-f16.gguf":    "f16",
		"mmproj.f16.gguf":         "f16",
		"mmproj_f32.gguf":         "f32",
		"mmproj-qwen-bf16.gguf":   "bf16",
		"mmproj-qwen-fp16.gguf":   "f16",
		"mmproj-qwen.gguf":        "",
	}
	for input, want := range cases {
		got := ParseMmprojPrecision(input)
		if got != want {
			t.Errorf("ParseMmprojPrecision(%q) = %q, want %q", input, got, want)
		}
	}
}
