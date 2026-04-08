package cmd

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/llamafarm/cli/cmd/config"
	"github.com/llamafarm/cli/cmd/utils"
	"github.com/llamafarm/cli/internal/hfcache"
	"github.com/llamafarm/cli/internal/modelformat"
	"github.com/spf13/cobra"
)

// aliasPattern matches the subset of characters we accept in a model alias
// when it will be used as a filesystem subdirectory name under the target
// root. The pattern mirrors the Python `validate_alias` helper so tooling on
// both sides of the pipeline rejects the same inputs.
var aliasPattern = regexp.MustCompile(`^[a-zA-Z0-9._][a-zA-Z0-9._\-]*$`)

// validateAlias returns an error if the alias cannot be safely joined into
// the target path. The checks mirror the Python validator and satisfy static
// analysis tools that flag unsanitized identifiers used in path expressions.
func validateAlias(alias string) error {
	if alias == "" {
		return fmt.Errorf("invalid alias: must be a non-empty string")
	}
	// Explicit ".." rejection — the pattern recognized by CodeQL's
	// go/zipslip family as a traversal sanitizer.
	if strings.Contains(alias, "..") {
		return fmt.Errorf("invalid alias %q: path traversal not allowed", alias)
	}
	if strings.ContainsAny(alias, `/\`) {
		return fmt.Errorf("invalid alias %q: path separator not allowed", alias)
	}
	if filepath.IsAbs(alias) {
		return fmt.Errorf("invalid alias %q: absolute path not allowed", alias)
	}
	if !aliasPattern.MatchString(alias) {
		return fmt.Errorf("invalid alias %q: must match %s", alias, aliasPattern.String())
	}
	return nil
}

// Output shape for `lf models path --format json`. Matches the schema called
// out in the spec so that downstream tooling can rely on stable field names.
type modelsPathOutput struct {
	TargetRoot      string              `json:"target_root"`
	ManifestTarget  string              `json:"manifest_target"`
	Models          []modelsPathOutModel `json:"models"`
}

type modelsPathOutModel struct {
	Name  string              `json:"name"`
	Kind  string              `json:"kind"`
	Quant string              `json:"quant,omitempty"`
	Files []modelsPathOutFile `json:"files"`
}

type modelsPathOutFile struct {
	Role   string `json:"role"`
	Source string `json:"source"`
	Target string `json:"target"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256,omitempty"`
}

// Role values used throughout the command.
const (
	roleWeights   = "weights"
	roleMmproj    = "mmproj"
	roleTokenizer = "tokenizer"
	roleAll       = "all"
)

var (
	modelsPathFormat     string
	modelsPathTargetRoot string
	modelsPathRole       string
	modelsPathSourceOnly bool
	modelsPathEnsure     bool
)

var modelsPathCmd = &cobra.Command{
	Use:   "path [alias...]",
	Short: "Emit a source→target transport plan for cached model files",
	Long: `Query the local HuggingFace cache for each model in llamafarm.yaml and emit
a transport plan describing where the files live now and where they should go
on a deployment target. Useful for ansible/packer/Dockerfile workflows that
need to push models to a device without copying the entire HF cache.

The command does not download, copy, or modify any files. Run 'lf models pull'
first (or pass --ensure) to populate the cache.

Output shapes:
  json  Structured plan with size + sha256 per file
  tsv   Tab-separated (name, role, source, target) — no sha256

Examples:
  # Emit a plan for every model in the project, as JSON
  lf models path --format json

  # Just the source path of a single model's weights — ideal for scripting
  lf models path qwen3-1.7b --role weights --source-only

  # Pull + emit in one shot on a fresh build host
  lf models path --format json --ensure`,
	RunE:          runModelsPath,
	SilenceUsage:  true,
	SilenceErrors: true,
}

func init() {
	modelsPathCmd.Flags().StringVar(&modelsPathFormat, "format", "tsv", "Output format: json or tsv")
	modelsPathCmd.Flags().StringVar(&modelsPathTargetRoot, "target-root", "", "Override the target root used for computed 'target' paths")
	modelsPathCmd.Flags().StringVar(&modelsPathRole, "role", roleAll, "Filter files by role: all, weights, mmproj, tokenizer")
	modelsPathCmd.Flags().BoolVar(&modelsPathSourceOnly, "source-only", false, "Print only source paths, one per line (shell-friendly)")
	modelsPathCmd.Flags().BoolVar(&modelsPathEnsure, "ensure", false, "Run 'lf models pull' for any missing models before emitting the plan")
	modelsCmd.AddCommand(modelsPathCmd)
}

func runModelsPath(cmd *cobra.Command, args []string) error {
	// Validate --format early.
	switch modelsPathFormat {
	case "json", "tsv":
	default:
		return fmt.Errorf("invalid --format %q (expected json or tsv)", modelsPathFormat)
	}
	// Validate --role.
	switch modelsPathRole {
	case roleAll, roleWeights, roleMmproj, roleTokenizer:
	default:
		return fmt.Errorf("invalid --role %q (expected one of: all, weights, mmproj, tokenizer)", modelsPathRole)
	}

	cwd := utils.GetEffectiveCWD()
	cfg, err := config.LoadConfig(cwd)
	if err != nil {
		return fmt.Errorf("load llamafarm config: %w", err)
	}
	if len(cfg.Runtime.Models) == 0 {
		return fmt.Errorf("no runtime.models[] configured in llamafarm config")
	}

	// Filter models by positional args.
	selectedModels, err := selectModelsByAlias(cfg.Runtime.Models, args)
	if err != nil {
		return err
	}

	// Resolve target-root tier.
	targetRoot, _ := cfg.ResolveModelDir(modelsPathTargetRoot)

	// --ensure: pull any missing models before proceeding.
	if modelsPathEnsure {
		if err := ensureModelsCached(selectedModels); err != nil {
			return err
		}
	}

	// Build the plan by walking each selected model.
	plan := modelsPathOutput{
		TargetRoot:     targetRoot,
		ManifestTarget: filepath.Join(targetRoot, "manifest.json"),
		Models:         make([]modelsPathOutModel, 0, len(selectedModels)),
	}

	includeSHA := modelsPathFormat == "json"

	for _, m := range selectedModels {
		modelEntry, err := buildModelEntry(m, targetRoot, modelsPathRole, includeSHA)
		if err != nil {
			return err
		}
		plan.Models = append(plan.Models, modelEntry)
	}

	// Emit.
	switch {
	case modelsPathSourceOnly:
		return emitSourceOnly(plan)
	case modelsPathFormat == "json":
		return emitJSON(plan)
	default:
		return emitTSV(plan)
	}
}

// selectModelsByAlias filters the project's model list to those matching the
// user-supplied alias args. An empty args list selects all models.
func selectModelsByAlias(all []config.LlamaFarmConfigRuntimeModelsElem, args []string) ([]config.LlamaFarmConfigRuntimeModelsElem, error) {
	if len(args) == 0 {
		out := make([]config.LlamaFarmConfigRuntimeModelsElem, len(all))
		copy(out, all)
		return out, nil
	}
	byName := make(map[string]config.LlamaFarmConfigRuntimeModelsElem, len(all))
	for _, m := range all {
		byName[m.Name] = m
	}
	out := make([]config.LlamaFarmConfigRuntimeModelsElem, 0, len(args))
	for _, want := range args {
		m, ok := byName[want]
		if !ok {
			names := make([]string, 0, len(all))
			for _, v := range all {
				names = append(names, v.Name)
			}
			sort.Strings(names)
			return nil, fmt.Errorf(
				"model %q not found in llamafarm config; available: %s",
				want, strings.Join(names, ", "),
			)
		}
		out = append(out, m)
	}
	return out, nil
}

// parseModelSpec splits a "repo:quant" model identifier into its parts. The
// quantization suffix is optional.
func parseModelSpec(spec string) (repoID, quant string) {
	if idx := strings.LastIndex(spec, ":"); idx != -1 {
		return spec[:idx], strings.ToUpper(spec[idx+1:])
	}
	return spec, ""
}

// buildModelEntry looks up a single model in the HF cache and builds its
// transport-plan entry. Returns an error if the model is not cached OR if the
// model is a non-GGUF kind (V1 scope).
func buildModelEntry(m config.LlamaFarmConfigRuntimeModelsElem, targetRoot, roleFilter string, includeSHA bool) (modelsPathOutModel, error) {
	// Validate the alias before it gets joined into target paths. The name
	// comes from runtime.models[].name in llamafarm.yaml — which is not
	// direct HTTP input, but we still reject path traversal attempts
	// (e.g. `../etc`) so the emitted `target` paths cannot escape
	// targetRoot. This also catches mistyped names with embedded slashes.
	if err := validateAlias(m.Name); err != nil {
		return modelsPathOutModel{}, err
	}

	repoID, quant := parseModelSpec(m.Model)

	entry := modelsPathOutModel{
		Name: m.Name,
		Kind: string(modelformat.KindGGUF),
		Files: []modelsPathOutFile{},
	}

	// For V1, only attempt GGUF via the HF cache helper. Non-GGUF model
	// formats (transformers, ultralytics) get a clear error from this
	// command rather than a misleading empty plan.
	if !looksLikeGGUFModel(m) {
		return entry, fmt.Errorf(
			"model %q is not a GGUF model; 'lf models path' only supports GGUF models in V1",
			m.Name,
		)
	}

	// Look up main weights. Only ErrNotCached maps to the "run lf models pull"
	// remediation message — any other error (malformed repo id, filesystem
	// I/O failure, etc.) is a genuine problem that should surface with its
	// original context rather than be masked behind the cache-miss message.
	weights, err := hfcache.LocateGGUF(repoID, quant)
	if err != nil {
		if errors.Is(err, hfcache.ErrNotCached) {
			return entry, fmt.Errorf(
				"%s not found in cache. Run 'lf models pull %s' first.",
				m.Name, m.Model,
			)
		}
		return entry, fmt.Errorf("locate %s: %w", m.Name, err)
	}
	detectedQuant := modelformat.ParseQuantization(weights.Filename)
	entry.Quant = detectedQuant

	// Look up optional mmproj.
	var mmproj hfcache.SnapshotFile
	mmprojOK := false
	if roleFilter == roleAll || roleFilter == roleMmproj {
		mm, err := hfcache.LocateMmproj(repoID)
		if err != nil && !errors.Is(err, hfcache.ErrNotCached) {
			return entry, err
		}
		if mm.Filename != "" {
			mmproj = mm
			mmprojOK = true
		}
	}

	// Apply role filter.
	includeWeights := roleFilter == roleAll || roleFilter == roleWeights
	includeMmproj := (roleFilter == roleAll || roleFilter == roleMmproj) && mmprojOK

	if includeWeights {
		targetName := canonicalWeightsName(detectedQuant)
		f := modelsPathOutFile{
			Role:   roleWeights,
			Source: weights.SnapshotPath,
			Target: filepath.Join(targetRoot, m.Name, targetName),
			Size:   weights.Size,
		}
		if includeSHA {
			sha, err := hfcache.SHA256(weights)
			if err != nil {
				return entry, fmt.Errorf("sha256 for %s: %w", weights.Filename, err)
			}
			f.SHA256 = sha
		}
		entry.Files = append(entry.Files, f)
	}

	if includeMmproj {
		precision := modelformat.ParseMmprojPrecision(mmproj.Filename)
		if precision == "" {
			precision = "f16"
		}
		targetName := fmt.Sprintf("mmproj.%s.gguf", precision)
		f := modelsPathOutFile{
			Role:   roleMmproj,
			Source: mmproj.SnapshotPath,
			Target: filepath.Join(targetRoot, m.Name, targetName),
			Size:   mmproj.Size,
		}
		if includeSHA {
			sha, err := hfcache.SHA256(mmproj)
			if err != nil {
				return entry, fmt.Errorf("sha256 for %s: %w", mmproj.Filename, err)
			}
			f.SHA256 = sha
		}
		entry.Files = append(entry.Files, f)
	}

	return entry, nil
}

// looksLikeGGUFModel heuristically decides whether a config model entry refers
// to a GGUF model. V1 supports only GGUF via this command; future work will
// extend to transformers/ultralytics.
func looksLikeGGUFModel(m config.LlamaFarmConfigRuntimeModelsElem) bool {
	// GGUF models typically use provider "llama.cpp", "llama-cpp", "runtime",
	// or similar — but the canonical signal is the model ID itself. The
	// strongest signal is ":quant" suffixes or model IDs containing "GGUF".
	repoID, quant := parseModelSpec(m.Model)
	if quant != "" {
		return true
	}
	if strings.Contains(strings.ToLower(repoID), "gguf") {
		return true
	}
	// Fall back to provider inspection. The provider enum string values for
	// llama.cpp-style runtimes vary; this is intentionally permissive.
	provider := strings.ToLower(string(m.Provider))
	if strings.Contains(provider, "llama") || provider == "runtime" || provider == "llamafarm" {
		return true
	}
	return false
}

// canonicalWeightsName returns the on-device filename for a GGUF weights file.
// Format: model.<QUANT>.gguf, or model.gguf if quant is unknown.
func canonicalWeightsName(quant string) string {
	if quant == "" {
		return "model.gguf"
	}
	return fmt.Sprintf("model.%s.gguf", quant)
}

// ensureModelsCached walks the selected models and invokes `lf models pull`
// for any that are not present in the HF cache.
func ensureModelsCached(selected []config.LlamaFarmConfigRuntimeModelsElem) error {
	missing := make([]config.LlamaFarmConfigRuntimeModelsElem, 0)
	for _, m := range selected {
		if !looksLikeGGUFModel(m) {
			continue
		}
		repoID, quant := parseModelSpec(m.Model)
		// Only treat ErrNotCached as "needs pull". Other errors (invalid
		// repo id, filesystem I/O) should surface so the user sees a
		// meaningful diagnostic instead of a misleading pull attempt.
		_, err := hfcache.LocateGGUF(repoID, quant)
		if err == nil {
			continue
		}
		if !errors.Is(err, hfcache.ErrNotCached) {
			return fmt.Errorf("locate %s: %w", m.Name, err)
		}
		missing = append(missing, m)
	}
	if len(missing) == 0 {
		return nil
	}
	fmt.Fprintf(os.Stderr, "Pulling %d missing model(s)...\n", len(missing))
	for _, m := range missing {
		fmt.Fprintf(os.Stderr, "  → %s (%s)\n", m.Name, m.Model)
		if err := pullModelNative(context.Background(), m.Model); err != nil {
			return fmt.Errorf("pull %s: %w", m.Name, err)
		}
	}
	return nil
}

func emitJSON(plan modelsPathOutput) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(plan)
}

func emitTSV(plan modelsPathOutput) error {
	for _, m := range plan.Models {
		for _, f := range m.Files {
			fmt.Printf("%s\t%s\t%s\t%s\n", m.Name, f.Role, f.Source, f.Target)
		}
	}
	return nil
}

func emitSourceOnly(plan modelsPathOutput) error {
	for _, m := range plan.Models {
		for _, f := range m.Files {
			fmt.Println(f.Source)
		}
	}
	return nil
}
