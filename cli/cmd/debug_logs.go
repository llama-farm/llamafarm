// cli/cmd/debug_logs.go
package cmd

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/spf13/cobra"
)

var (
	logsFollow      bool
	logsTail        int
	logsSince       string
	logsServices    []string
	logsComposeFile string
	logsNoPrefix    bool
	logsOut         string
)

func NewCmdDebugLogs() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "logs",
		Short: "Show recent or live logs from LlamaFarm dev containers",
		Long: `Show recent or live logs from the LlamaFarm Docker stack started by "lf start".
By default shows the last 200 lines from all services. Use --follow for live tail.`,
		RunE: runDebugLogs,
		Example: `
  # Show last 200 lines from all services
  lf debug logs

  # Live tail everything
  lf debug logs -f

  # Show the last 500 lines from server + rag services since 30 minutes ago
  lf debug logs -n 500 --since 30m -s server -s rag

  # Write logs to a file while also printing to console
  lf debug logs -f -o .llamafarm/logs/dev-$(date +%Y%m%d-%H%M%S).log

  # Use a custom compose file
  lf debug logs --compose-file ./deployment/docker_compose/docker-compose.yml`,
	}

	cmd.Flags().BoolVarP(&logsFollow, "follow", "f", false, "Follow log output (live tail)")
	cmd.Flags().IntVarP(&logsTail, "tail", "n", 200, "Number of lines to show from the end of logs (per container)")
	cmd.Flags().StringVar(&logsSince, "since", "", "Show logs since timestamp or relative time (e.g. 2025-10-09T12:00:00Z, 30m, 1h)")
	cmd.Flags().StringSliceVarP(&logsServices, "service", "s", nil, "Limit to one or more services (repeat flag)")
	cmd.Flags().StringVar(&logsComposeFile, "compose-file", "", "Path to docker compose file (default: repo's compose file)")
	cmd.Flags().BoolVar(&logsNoPrefix, "no-prefix", false, "Don't print service name prefixes")
	cmd.Flags().StringVarP(&logsOut, "out", "o", "", "Write combined logs to a file (also prints to stdout)")

	return cmd
}


func init() {
	debugCmd.AddCommand(NewCmdDebugLogs)
}

func runDebugLogs(cmd *cobra.Command, args []string) error {
	if _, err := exec.LookPath("docker"); err != nil {
		return fmt.Errorf("docker not found in PATH: %w", err)
	}

	ctx := cmd.Context()
	if ctx == nil {
		var cancel context.CancelFunc
		ctx, cancel = context.WithCancel(context.Background())
		defer cancel()
	}

	composeFile, usingCompose := resolveComposeFile(logsComposeFile)
	if usingCompose {
		return runComposeLogs(ctx, composeFile)
	}

	// Fallback: gather docker ps and use docker logs per container
	return runDockerLogsFallback(ctx)
}

func resolveComposeFile(flagPath string) (string, bool) {
	// Priority: flag > common repo path > env > none
	if flagPath != "" {
		if fileExists(flagPath) {
			return flagPath, true
		}
	}

	// Try repo path (works when run from repo root)
	defaultRepoPath := filepath.Join(".", "deployment", "docker_compose", "docker-compose.yml")
	if fileExists(defaultRepoPath) {
		return defaultRepoPath, true
	}

	// Env override (for advanced setups)
	if p := os.Getenv("LF_COMPOSE_FILE"); p != "" && fileExists(p) {
		return p, true
	}

	return "", false
}

func fileExists(p string) bool {
	if p == "" {
		return false
	}
	info, err := os.Stat(p)
	return err == nil && !info.IsDir()
}

func runComposeLogs(ctx context.Context, composeFile string) error {
	args := []string{"compose", "-f", composeFile, "logs"}

	// flags (compose supports --since, --tail, --follow, --no-log-prefix)
	if logsFollow {
		args = append(args, "--follow")
	}
	if logsNoPrefix {
		args = append(args, "--no-log-prefix")
	}
	if logsSince != "" {
		args = append(args, "--since", logsSince)
	}
	if logsTail >= 0 {
		args = append(args, "--tail", fmt.Sprintf("%d", logsTail))
	}
	// services (optional)
	args = append(args, logsServices...)

	c := exec.CommandContext(ctx, "docker", args...)
	stdout, err := c.StdoutPipe()
	if err != nil {
		return err
	}
	stderr, err := c.StderrPipe()
	if err != nil {
		return err
	}

	var writer io.Writer = os.Stdout
	var f *os.File
	if logsOut != "" {
		if err := os.MkdirAll(filepath.Dir(logsOut), 0o755); err != nil {
			return fmt.Errorf("creating log output dir: %w", err)
		}
		f, err = os.Create(logsOut)
		if err != nil {
			return fmt.Errorf("creating log output file: %w", err)
		}
		defer f.Close()
		writer = io.MultiWriter(os.Stdout, f)
	}

	if err := c.Start(); err != nil {
		return fmt.Errorf("docker compose logs: %w", err)
	}

	wg := &sync.WaitGroup{}
	wg.Add(2)
	go pump(stdout, writer, wg)
	go pump(stderr, writer, wg)

	err = c.Wait()
	wg.Wait()

	// Compose returns exit code 0 when logs stream ends or process is killed (Ctrl-C).
	// If follow is false, a non-zero here should be surfaced.
	if !logsFollow && err != nil {
		return err
	}
	return nil
}

func pump(r io.Reader, w io.Writer, wg *sync.WaitGroup) {
	defer wg.Done()
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		_, _ = fmt.Fprintln(w, sc.Text())
	}
}

func runDockerLogsFallback(ctx context.Context) error {
	// List containers whose name hints they belong to LlamaFarm
	// Conservative match: names containing "llamafarm" OR service name filters if provided.
	psArgs := []string{"ps", "--format", "{{.ID}} {{.Names}}"}
	out, err := exec.CommandContext(ctx, "docker", psArgs...).Output()
	if err != nil {
		return fmt.Errorf("docker ps failed: %w", err)
	}
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	var targets []string // container names

	include := func(name string) bool {
		if len(logsServices) == 0 {
			return strings.Contains(strings.ToLower(name), "llamafarm")
		}
		low := strings.ToLower(name)
		for _, s := range logsServices {
			if strings.Contains(low, strings.ToLower(s)) {
				return true
			}
		}
		return false
	}

	for _, ln := range lines {
		if ln == "" {
			continue
		}
		parts := strings.SplitN(ln, " ", 2)
		if len(parts) != 2 {
			continue
		}
		name := strings.TrimSpace(parts[1])
		if include(name) {
			targets = append(targets, name)
		}
	}
	if len(targets) == 0 {
		return errors.New("no matching running containers found (try --compose-file or check that lf start is running)")
	}

	// Stream each container's logs concurrently, prefixing container name unless --no-prefix
	var writers []io.Writer
	writers = append(writers, os.Stdout)
	var f *os.File
	if logsOut != "" {
		if err := os.MkdirAll(filepath.Dir(logsOut), 0o755); err != nil {
			return fmt.Errorf("creating log output dir: %w", err)
		}
		f, err = os.Create(logsOut)
		if err != nil {
			return fmt.Errorf("creating log output file: %w", err)
		}
		defer f.Close()
		writers = append(writers, f)
	}
	writer := io.MultiWriter(writers...)

	wg := &sync.WaitGroup{}
	errCh := make(chan error, len(targets))

	for _, name := range targets {
		name := name
		wg.Add(1)
		go func() {
			defer wg.Done()
			args := []string{"logs"}
			if logsFollow {
				args = append(args, "--follow")
			}
			if logsSince != "" {
				args = append(args, "--since", logsSince)
			}
			if logsTail >= 0 {
				args = append(args, "--tail", fmt.Sprintf("%d", logsTail))
			}
			args = append(args, name)

			c := exec.CommandContext(ctx, "docker", args...)

			stdout, err := c.StdoutPipe()
			if err != nil {
				errCh <- fmt.Errorf("docker logs %s: failed to get stdout pipe: %w", name, err)
				return
			}
			stderr, err := c.StderrPipe()
			if err != nil {
				errCh <- fmt.Errorf("docker logs %s: failed to get stderr pipe: %w", name, err)
				return
			}

			if err := c.Start(); err != nil {
				errCh <- fmt.Errorf("docker logs %s: %w", name, err)
				return
			}

			p := func(r io.Reader) {
				sc := bufio.NewScanner(r)
				for sc.Scan() {
					line := sc.Text()
					if logsNoPrefix {
						fmt.Fprintln(writer, line)
					} else {
						now := time.Now().Format(time.RFC3339)
						fmt.Fprintf(writer, "[%s] [%s] %s\n", now, name, line)
					}
				}
			}
			done := make(chan struct{})
			go func() { p(stdout); close(done) }()
			p(stderr)
			<-done

			if err := c.Wait(); err != nil && !logsFollow {
				errCh <- fmt.Errorf("docker logs %s finished with error: %w", name, err)
				return
			}
		}()
	}

	wg.Wait()
	close(errCh)
	var combined error
	for e := range errCh {
		if combined == nil {
			combined = e
		} else {
			combined = fmt.Errorf("%v; %w", combined, e)
		}
	}
	return combined
}
