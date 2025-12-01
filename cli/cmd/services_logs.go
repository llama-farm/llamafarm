package cmd

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/llamafarm/cli/cmd/utils"
	"github.com/spf13/cobra"
)

// servicesLogsCmd displays logs for LlamaFarm services
var servicesLogsCmd = &cobra.Command{
	Use:   "logs",
	Short: "View logs for LlamaFarm services",
	Long: `Display logs from LlamaFarm services.

Logs are stored in ~/.llamafarm/logs/<service>.log and can be viewed in real-time
or as a static snapshot. When no service is specified, logs from all services are
displayed with service name prefixes.

Available services:
  - server: The main FastAPI server
  - rag: The RAG/Celery worker
  - universal-runtime: The universal runtime server

Examples:
  lf services logs                           # Show logs from all services
  lf services logs --tail 50                 # Show last 50 lines from all services
  lf services logs --follow                  # Follow all service logs in real-time
  lf services logs --service server          # Show all server logs
  lf services logs -s rag --tail 50          # Show last 50 lines of RAG logs
  lf services logs -s server --follow        # Follow server logs in real-time
  lf services logs -s rag -f -n 100          # Follow RAG logs, starting with last 100 lines`,
	Run: runServicesLogs,
}

func init() {
	servicesCmd.AddCommand(servicesLogsCmd)

	// Optional flag: which service to show logs for (if empty, show all)
	servicesLogsCmd.Flags().StringP("service", "s", "", "Service to view logs for (server, rag, universal-runtime). If omitted, shows all services.")

	// Optional flags
	servicesLogsCmd.Flags().BoolP("follow", "f", false, "Follow log output (like tail -f)")
	servicesLogsCmd.Flags().IntP("tail", "n", 100, "Number of lines to show from the end (Default: 100, 0 = show all)")
}

// runServicesLogs is the main entry point for the services logs command
func runServicesLogs(cmd *cobra.Command, args []string) {
	serviceName, _ := cmd.Flags().GetString("service")
	follow, _ := cmd.Flags().GetBool("follow")
	tailLines, _ := cmd.Flags().GetInt("tail")

	// Determine which services to show logs for
	validServices := []string{"server", "rag", "universal-runtime"}
	var servicesToShow []string

	if serviceName == "" {
		// Show all services
		servicesToShow = validServices
	} else {
		// Validate single service name
		if !isValidService(serviceName, validServices) {
			utils.OutputError("Invalid service: %s\n", serviceName)
			fmt.Fprintf(os.Stderr, "Valid services are: %s\n", strings.Join(validServices, ", "))
			os.Exit(1)
		}
		servicesToShow = []string{serviceName}
	}

	// Check which log files exist and build log file map
	logFiles := make(map[string]string)
	for _, svc := range servicesToShow {
		logFile, err := getServiceLogFile(svc)
		if err != nil {
			utils.OutputError("Failed to determine log file location for %s: %v\n", svc, err)
			os.Exit(1)
		}

		if _, err := os.Stat(logFile); os.IsNotExist(err) {
			if len(servicesToShow) == 1 {
				// Only error out if user specifically requested this service
				utils.OutputError("Log file not found: %s\n", logFile)
				fmt.Fprintf(os.Stderr, "\nThe %s service may not have been started yet.\n", svc)
				fmt.Fprintf(os.Stderr, "Run 'lf services start %s' to start the service.\n", svc)
				os.Exit(1)
			}
			// Skip missing logs when showing all services
			continue
		}

		logFiles[svc] = logFile
	}

	if len(logFiles) == 0 {
		utils.OutputError("No log files found for any service.\n")
		fmt.Fprintf(os.Stderr, "\nServices may not have been started yet.\n")
		fmt.Fprintf(os.Stderr, "Run 'lf services start' to start services.\n")
		os.Exit(1)
	}

	// Display logs based on flags
	if len(logFiles) == 1 {
		// Single service - simple display without prefixes
		for _, logFile := range logFiles {
			if follow {
				if err := followLogs(logFile, tailLines, ""); err != nil {
					utils.OutputError("Failed to follow logs: %v\n", err)
					os.Exit(1)
				}
			} else {
				if err := displayLogs(logFile, tailLines, ""); err != nil {
					utils.OutputError("Failed to display logs: %v\n", err)
					os.Exit(1)
				}
			}
		}
	} else {
		// Multiple services - interleaved display with prefixes
		if follow {
			if err := followMultipleLogs(logFiles, tailLines); err != nil {
				utils.OutputError("Failed to follow logs: %v\n", err)
				os.Exit(1)
			}
		} else {
			if err := displayMultipleLogs(logFiles, tailLines); err != nil {
				utils.OutputError("Failed to display logs: %v\n", err)
				os.Exit(1)
			}
		}
	}
}

// isValidService checks if the service name is valid
func isValidService(serviceName string, validServices []string) bool {
	for _, valid := range validServices {
		if serviceName == valid {
			return true
		}
	}
	return false
}

// getServiceLogFile returns the path to a service's log file
func getServiceLogFile(serviceName string) (string, error) {
	dataDir, err := utils.GetLFDataDir()
	if err != nil {
		return "", fmt.Errorf("failed to get llamafarm data directory: %w", err)
	}

	logsDir := filepath.Join(dataDir, "logs")
	logFile := filepath.Join(logsDir, fmt.Sprintf("%s.log", serviceName))

	return logFile, nil
}

// displayLogs displays logs from a file (optionally showing only the tail)
func displayLogs(logFile string, tailLines int, prefix string) error {
	file, err := os.Open(logFile)
	if err != nil {
		return fmt.Errorf("failed to open log file: %w", err)
	}
	defer file.Close()

	// If tail is requested, read all lines and show only the last N
	if tailLines > 0 {
		return displayTailLines(file, tailLines, prefix)
	}

	// Otherwise, show all lines
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		if prefix != "" {
			fmt.Print(prefix)
		}
		fmt.Println(scanner.Text())
	}

	return scanner.Err()
}

// displayTailLines displays only the last N lines from a file
func displayTailLines(file *os.File, n int, prefix string) error {
	// Read all lines into a buffer
	var lines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return err
	}

	// Calculate start index
	start := 0
	if len(lines) > n {
		start = len(lines) - n
	}

	// Print the last N lines
	for i := start; i < len(lines); i++ {
		if prefix != "" {
			fmt.Print(prefix)
		}
		fmt.Println(lines[i])
	}

	return nil
}

// followLogs follows a log file in real-time (like tail -f)
func followLogs(logFile string, initialTailLines int, prefix string) error {
	file, err := os.Open(logFile)
	if err != nil {
		return fmt.Errorf("failed to open log file: %w", err)
	}
	defer file.Close()

	// If tail is specified, first show the last N lines
	if initialTailLines > 0 {
		if err := displayTailLines(file, initialTailLines, prefix); err != nil {
			return fmt.Errorf("failed to display initial tail: %w", err)
		}
	} else {
		// Otherwise, seek to the end of the file
		if _, err := file.Seek(0, io.SeekEnd); err != nil {
			return fmt.Errorf("failed to seek to end of file: %w", err)
		}
	}

	// Set up signal handling for graceful exit
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	// Create a reader that will follow the file
	reader := bufio.NewReader(file)

	// Channel to signal when we should stop
	done := make(chan struct{})

	// Goroutine to handle signals
	go func() {
		<-sigChan
		close(done)
	}()

	// Follow the file
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-done:
			return nil
		case <-ticker.C:
			// Try to read lines
			for {
				line, err := reader.ReadString('\n')
				if err != nil {
					if err == io.EOF {
						// No more data available right now
						break
					}
					return fmt.Errorf("failed to read from log file: %w", err)
				}
				// Print the line with prefix (already includes newline)
				if prefix != "" {
					fmt.Print(prefix)
				}
				fmt.Print(line)
			}
		}
	}
}

// logLine represents a line from a log file with metadata
type logLine struct {
	timestamp time.Time
	service   string
	content   string
}

// displayMultipleLogs displays logs from multiple files, interleaved by timestamp
func displayMultipleLogs(logFiles map[string]string, tailLines int) error {
	var allLines []logLine

	// Read lines from all files
	for service, logFile := range logFiles {
		file, err := os.Open(logFile)
		if err != nil {
			return fmt.Errorf("failed to open log file for %s: %w", service, err)
		}

		scanner := bufio.NewScanner(file)
		for scanner.Scan() {
			line := scanner.Text()
			timestamp := extractTimestamp(line)
			allLines = append(allLines, logLine{
				timestamp: timestamp,
				service:   service,
				content:   line,
			})
		}
		file.Close()

		if err := scanner.Err(); err != nil {
			return fmt.Errorf("failed to read log file for %s: %w", service, err)
		}
	}

	// Sort by timestamp
	sort.Slice(allLines, func(i, j int) bool {
		return allLines[i].timestamp.Before(allLines[j].timestamp)
	})

	// Apply tail limit if specified
	if tailLines > 0 && len(allLines) > tailLines {
		allLines = allLines[len(allLines)-tailLines:]
	}

	// Print all lines with service prefix
	for _, line := range allLines {
		prefix := getServicePrefix(line.service)
		fmt.Printf("%s%s\n", prefix, line.content)
	}

	return nil
}

// followMultipleLogs follows multiple log files in real-time
func followMultipleLogs(logFiles map[string]string, initialTailLines int) error {
	// First, show initial tail if requested
	if initialTailLines > 0 {
		if err := displayMultipleLogs(logFiles, initialTailLines); err != nil {
			return err
		}
	}

	// Set up signal handling for graceful exit
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	done := make(chan struct{})

	// Goroutine to handle signals
	go func() {
		<-sigChan
		close(done)
	}()

	// Open all files and seek to end
	type fileInfo struct {
		file   *os.File
		reader *bufio.Reader
	}
	files := make(map[string]*fileInfo)

	for service, logFile := range logFiles {
		file, err := os.Open(logFile)
		if err != nil {
			// Close any already opened files before returning
			for _, fi := range files {
				fi.file.Close()
			}
			return fmt.Errorf("failed to open log file for %s: %w", service, err)
		}

		// Seek to end if we didn't show initial tail
		if initialTailLines == 0 {
			if _, err := file.Seek(0, io.SeekEnd); err != nil {
				file.Close()
				// Close any already opened files before returning
				for _, fi := range files {
					fi.file.Close()
				}
				return fmt.Errorf("failed to seek to end for %s: %w", service, err)
			}
		}

		files[service] = &fileInfo{
			file:   file,
			reader: bufio.NewReader(file),
		}
	}

	// Defer closing all files
	defer func() {
		for _, fi := range files {
			fi.file.Close()
		}
	}()

	// Use a mutex to serialize output from different goroutines
	var outputMu sync.Mutex

	// Start a goroutine for each service
	var wg sync.WaitGroup
	for service, fi := range files {
		wg.Add(1)
		go func(svc string, info *fileInfo) {
			defer wg.Done()
			prefix := getServicePrefix(svc)
			ticker := time.NewTicker(100 * time.Millisecond)
			defer ticker.Stop()

			for {
				select {
				case <-done:
					return
				case <-ticker.C:
					for {
						line, err := info.reader.ReadString('\n')
						if err != nil {
							if err == io.EOF {
								break
							}
							return
						}
						outputMu.Lock()
						fmt.Printf("%s%s", prefix, line)
						outputMu.Unlock()
					}
				}
			}
		}(service, fi)
	}

	wg.Wait()
	return nil
}

// getServicePrefix returns a plain text prefix for a service
func getServicePrefix(service string) string {
	// Pad service name to 17 chars for alignment
	paddedService := fmt.Sprintf("%-17s", service)
	return fmt.Sprintf("[%s] ", paddedService)
}

// extractTimestamp extracts timestamp from a log line
// Expected format: [2025-11-17 11:49:02] ...
func extractTimestamp(line string) time.Time {
	if len(line) < 21 || line[0] != '[' {
		return time.Time{} // Return zero time if no timestamp
	}

	timestampStr := line[1:20] // Extract "2025-11-17 11:49:02"
	t, err := time.Parse("2006-01-02 15:04:05", timestampStr)
	if err != nil {
		return time.Time{}
	}

	return t
}
