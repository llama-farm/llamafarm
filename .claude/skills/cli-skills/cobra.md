# Cobra Command Patterns

Best practices for building Cobra CLI commands in LlamaFarm.

## Checklist

### 1. Use RunE for Error Handling

**Description**: Prefer `RunE` over `Run` to properly propagate errors up the command chain.

**Search Pattern**:
```bash
grep -rn "Run: func" cli/cmd/*.go | grep -v "RunE"
```

**Pass Criteria**: Commands use `RunE` and return errors instead of calling `os.Exit(1)` directly.

**Fail Criteria**: Commands use `Run` and call `os.Exit(1)` for error handling.

**Severity**: Medium

**Note**: The root command (`root.go`) uses `Run` instead of `RunE` as an exception, since the root command itself performs no work and defers to subcommands.

**Recommendation**:
```go
// Good
var myCmd = &cobra.Command{
    Use: "mycommand",
    RunE: func(cmd *cobra.Command, args []string) error {
        if err := doWork(); err != nil {
            return fmt.Errorf("failed to do work: %w", err)
        }
        return nil
    },
}

// Avoid
var myCmd = &cobra.Command{
    Use: "mycommand",
    Run: func(cmd *cobra.Command, args []string) {
        if err := doWork(); err != nil {
            fmt.Fprintf(os.Stderr, "Error: %v\n", err)
            os.Exit(1)  // Bypasses Cobra's error handling
        }
    },
}
```

---

### 2. Use Args Validators

**Description**: Use Cobra's built-in argument validators instead of manual validation.

**Search Pattern**:
```bash
grep -rn "Args:" cli/cmd/*.go
```

**Pass Criteria**: Commands use appropriate `Args` validators like `cobra.ExactArgs`, `cobra.MaximumNArgs`, `cobra.MinimumNArgs`.

**Fail Criteria**: Manual argument count validation inside `Run`/`RunE` functions.

**Severity**: Low

**Recommendation**:
```go
// Good
var myCmd = &cobra.Command{
    Use:  "mycommand <required-arg>",
    Args: cobra.ExactArgs(1),
}

// For custom validation
var myCmd = &cobra.Command{
    Use: "chat [namespace/project] \"input\"",
    Args: func(cmd *cobra.Command, args []string) error {
        if len(args) > 0 && strings.Contains(args[0], "/") {
            if strings.Count(args[0], "/") != 1 {
                return fmt.Errorf("project must be in format 'namespace/project'")
            }
        }
        return nil
    },
}
```

---

### 3. Register Flags in init()

**Description**: Register all flags in `init()` functions for predictable initialization order.

**Search Pattern**:
```bash
grep -rn "func init()" cli/cmd/*.go -A10 | grep "Flags()"
```

**Pass Criteria**: All flag registration happens in `init()` functions.

**Fail Criteria**: Flags registered inside command Run functions or at arbitrary points.

**Severity**: Medium

**Recommendation**:
```go
var myFlag string

var myCmd = &cobra.Command{
    Use: "mycommand",
    RunE: runMyCommand,
}

func init() {
    rootCmd.AddCommand(myCmd)
    myCmd.Flags().StringVar(&myFlag, "myflag", "default", "Description of flag")
    myCmd.Flags().BoolVar(&verbose, "verbose", false, "Enable verbose output")
}
```

---

### 4. Use Persistent Flags for Shared Options

**Description**: Use `PersistentFlags()` on parent commands for options shared by subcommands.

**Search Pattern**:
```bash
grep -rn "PersistentFlags()" cli/cmd/*.go
```

**Pass Criteria**: Global options like `--debug`, `--server-url` are persistent flags on root command.

**Fail Criteria**: Same flag defined on multiple commands instead of parent.

**Severity**: Low

**Recommendation**:
```go
// In root.go
func init() {
    rootCmd.PersistentFlags().BoolVarP(&debug, "debug", "d", false, "Enable debug output")
    rootCmd.PersistentFlags().StringVar(&serverURL, "server-url", "http://localhost:8000", "Server URL")
}

// Subcommands automatically inherit these flags
```

---

### 5. Provide Comprehensive Help Text

**Description**: Commands should have clear `Short`, `Long`, and usage examples.

**Search Pattern**:
```bash
grep -rn "Long:" cli/cmd/*.go -A5
```

**Pass Criteria**: Commands have `Short` description, `Long` description with examples, and clear `Use` pattern.

**Fail Criteria**: Missing or unhelpful descriptions.

**Severity**: Low

**Recommendation**:
```go
var myCmd = &cobra.Command{
    Use:   "mycommand [flags] <required-arg>",
    Short: "Brief one-line description",
    Long: `Extended description explaining the command's purpose.

Examples:
  # Basic usage
  lf mycommand value

  # With flags
  lf mycommand --flag=option value

  # Common use case
  lf mycommand --verbose my-value`,
}
```

---

### 6. Use Subcommand Hierarchy

**Description**: Group related commands under parent commands for better organization.

**Search Pattern**:
```bash
grep -rn "AddCommand" cli/cmd/*.go
```

**Pass Criteria**: Related commands grouped under logical parent commands (e.g., `services start`, `services stop`).

**Fail Criteria**: Flat command structure with many top-level commands.

**Severity**: Low

**Recommendation**:
```go
// Parent command (no Run function needed)
var servicesCmd = &cobra.Command{
    Use:   "services",
    Short: "Manage LlamaFarm services",
}

// Subcommands
var servicesStartCmd = &cobra.Command{
    Use:   "start [service-name]",
    Short: "Start LlamaFarm services",
    RunE:  runServicesStart,
}

func init() {
    rootCmd.AddCommand(servicesCmd)
    servicesCmd.AddCommand(servicesStartCmd)
    servicesCmd.AddCommand(servicesStopCmd)
    servicesCmd.AddCommand(servicesStatusCmd)
}
```

---

### 7. Handle PersistentPreRunE for Setup

**Description**: Use `PersistentPreRunE` for common setup that applies to all subcommands.

**Search Pattern**:
```bash
grep -rn "PersistentPreRunE" cli/cmd/*.go
```

**Pass Criteria**: Common initialization (like debug logging setup) in `PersistentPreRunE` on root.

**Fail Criteria**: Same setup code duplicated in multiple command `RunE` functions.

**Severity**: Medium

**Recommendation**:
```go
var rootCmd = &cobra.Command{
    Use:   "lf",
    Short: "LlamaFarm CLI",
    PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
        if debug {
            utils.InitDebugLogger("", true)
        }
        return nil
    },
}
```

---

### 8. Support JSON Output for Machine Readability

**Description**: Commands that output structured data should support `--json` flag.

**Search Pattern**:
```bash
grep -rn '"json"' cli/cmd/*.go
```

**Pass Criteria**: Status and list commands support `--json` for scripting and automation.

**Fail Criteria**: Only human-readable output, making scripting difficult.

**Severity**: Medium

**Recommendation**:
```go
func init() {
    myCmd.Flags().Bool("json", false, "Output in JSON format")
}

func runMyCommand(cmd *cobra.Command, args []string) error {
    jsonOutput, _ := cmd.Flags().GetBool("json")

    result := getResult()

    if jsonOutput {
        encoder := json.NewEncoder(os.Stdout)
        encoder.SetIndent("", "  ")
        return encoder.Encode(result)
    }

    // Human-readable output
    fmt.Printf("Result: %s\n", result.Name)
    return nil
}
```

---

### 9. Validate Flag Combinations

**Description**: Validate mutually exclusive or dependent flag combinations.

**Search Pattern**:
```bash
grep -rn "MarkFlagsMutuallyExclusive\|MarkFlagsRequiredTogether" cli/cmd/*.go
```

**Pass Criteria**: Conflicting flags are properly validated.

**Fail Criteria**: Invalid flag combinations lead to undefined behavior.

**Severity**: Medium

**Recommendation**:
```go
func init() {
    myCmd.Flags().StringVar(&inputFile, "file", "", "Input from file")
    myCmd.Flags().StringVar(&inputText, "text", "", "Input as text")

    // Cobra 1.5+ built-in validation
    myCmd.MarkFlagsMutuallyExclusive("file", "text")

    // Or manual validation in RunE
}

func runMyCommand(cmd *cobra.Command, args []string) error {
    if inputFile != "" && inputText != "" {
        return fmt.Errorf("specify either --file or --text, not both")
    }
    // ...
}
```

---

### 10. Use Context for Cancellation

**Description**: Pass context through commands for proper cancellation support.

**Search Pattern**:
```bash
grep -rn "cmd.Context()" cli/cmd/*.go
```

**Pass Criteria**: Long-running commands use `cmd.Context()` for cancellation.

**Fail Criteria**: No cancellation support, leading to stuck processes on Ctrl+C.

**Severity**: High

**Recommendation**:
```go
func runMyCommand(cmd *cobra.Command, args []string) error {
    ctx := cmd.Context()

    // Pass context to long-running operations
    result, err := longRunningOperation(ctx)
    if err != nil {
        if errors.Is(err, context.Canceled) {
            return fmt.Errorf("operation cancelled")
        }
        return err
    }

    return nil
}
```
