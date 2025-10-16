// cli/cmd/debug_logs.go
package cmd

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/client"
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
	// ✅ must call the function, not pass the symbol
	debugCmd.AddCommand(NewCmdDebugLogs())
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

	// Fallback: use Docker SDK directly
	return runDockerLogsFallback(ctx)
}

func resolveComposeFile(flagPath string) (string, bool) {
	if flagPath != "" && fileExists(flagPath) {
		return flagPath, true
	}

	defaultRepoPath := filepath.Join(".", "deployment", "docker_compose", "docker-compose.yml")
	if fileExists(defaultRepoPath) {
		return defaultRepoPath, true
	}

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

	writer, cleanup, err := setupOutputWriter(logsOut)
	if err != nil {
		return err
	}
	defer cleanup()

	if err := c.Start(); err != nil {
		return fmt.Errorf("docker compose logs: %w", err)
	}

	wg := &sync.WaitGroup{}
	wg.Add(2)
	go pump(stdout, writer, wg)
	go pump(stderr, writer, wg)

	err = c.Wait()
	wg.Wait()

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
	cli, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
	if err != nil {
		return fmt.Errorf("failed to create Docker client: %w", err)
	}
	defer cli.Close()

	containers, err := cli.ContainerList(ctx, container.ListOptions{All: false})
	if err != nil {
		return fmt.Errorf("failed to list containers: %w", err)
	}

	var targets []string
	for _, c := range containers {
		include := false
		for _, name := range c.Names {
			name = strings.TrimPrefix(name, "/")
			if len(logsServices) == 0 && strings.Contains(name, "llamafarm") {
				include = true
				break
			}
			for _, s := range logsServices {
				if strings.Contains(strings.ToLower(name), strings.ToLower(s)) {
					include = true
					break
				}
			}
		}
		if include {
			targets = append(targets, c.ID)
		}
	}

	if len(targets) == 0 {
		return fmt.Errorf("no matching containers found (try --compose-file or run lf start)")
	}

	writer, cleanup, err := setupOutputWriter(logsOut)
	if err != nil {
		return err
	}
	defer cleanup()

	wg := &sync.WaitGroup{}
	errCh := make(chan error, len(targets))

	for _, id := range targets {
		id := id
		wg.Add(1)
		go func() {
			defer wg.Done()

			reader, err := cli.ContainerLogs(ctx, id, container.LogsOptions{
				ShowStdout: true,
				ShowStderr: true,
				Follow:     logsFollow,
				Tail:       fmt.Sprintf("%d", logsTail),
				Since:      logsSince,
				Timestamps: false,
			})
			if err != nil {
				errCh <- fmt.Errorf("failed to stream logs for container %s: %w", id[:12], err)
				return
			}
			defer reader.Close()

			buf := make([]byte, 4096)
			for {
				n, readErr := reader.Read(buf)
				if n > 0 {
					line := string(buf[:n])
					fmt.Fprint(writer, formatLogLine(id[:12], line))
				}
				if readErr == io.EOF {
					break
				}
				if readErr != nil {
					errCh <- fmt.Errorf("log read error (%s): %w", id[:12], readErr)
					return
				}
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

func setupOutputWriter(path string) (io.Writer, func(), error) {
	if path == "" {
		return os.Stdout, func() {}, nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, func() {}, fmt.Errorf("creating output dir: %w", err)
	}
	f, err := os.Create(path)
	if err != nil {
		return nil, func() {}, fmt.Errorf("creating output file: %w", err)
	}
	cleanup := func() { _ = f.Close() }
	return io.MultiWriter(os.Stdout, f), cleanup, nil
}

func formatLogLine(service, line string) string {
	if logsNoPrefix {
		return line
	}
	return fmt.Sprintf("[%s] %s", service, line)
}
