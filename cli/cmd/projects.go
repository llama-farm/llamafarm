package cmd

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/llamafarm/cli/cmd/config"
	"github.com/llamafarm/cli/cmd/orchestrator"
	"github.com/llamafarm/cli/cmd/utils"
	"github.com/spf13/cobra"
)

// Chat CLI state variables
var (
	namespace   string
	projectID   string
	sessionID   string
	temperature float64
	maxTokens   int
	streaming   bool
)

// projectsCmd represents the projects command
var projectsCmd = &cobra.Command{
	Use:   "projects",
	Short: "Manage LlamaFarm projects and interact with them",
	Long: `Manage LlamaFarm projects and interact with them through various interfaces.

Available commands:
  chat - Start an interactive chat session with a project`,
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("LlamaFarm Projects Management")
		cmd.Help()
	},
}

// projectsListCmd lists projects for a namespace from the server
var projectsListCmd = &cobra.Command{
	Use:     "list",
	Aliases: []string{"ls"},
	Short:   "List local LlamaFarm projects",
	Long:    "List local LlamaFarm projects discovered in ~/.llamafarm/projects, marking the current project when inside one.",
	Run: func(cmd *cobra.Command, args []string) {
		projectsRoot, err := utils.GetProjectsRoot()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error resolving projects directory: %v\n", err)
			os.Exit(1)
		}
		projectsRoot = filepath.Clean(projectsRoot)

		projects, warnings, err := discoverProjects(projectsRoot)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error listing projects: %v\n", err)
			os.Exit(1)
		}

		sortProjectsByModTime(projects)

		if len(projects) == 0 {
			fmt.Printf("No projects found in %s\n", projectsRoot)
			return
		}

		printProjectsTable(projects)

		for _, w := range warnings {
			fmt.Fprintf(os.Stderr, "Warning: %s\n", w)
		}
	},
}

// projectsDeleteCmd represents the projects delete command
var projectsDeleteCmd = &cobra.Command{
	Use:     "delete [project-id]",
	Aliases: []string{"rm", "remove", "del"},
	Short:   "Delete a project and all its associated resources",
	Long: `Delete a project and all its associated resources from the LlamaFarm server.

This operation is irreversible and will delete all project data.`,
	Args: cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		projectToDelete := args[0]

		// Reuse existing config resolution pattern
		serverCfg, err := config.GetServerConfig(utils.GetEffectiveCWD(), serverURL, namespace, projectToDelete)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		serverURL = serverCfg.URL
		ns := strings.TrimSpace(serverCfg.Namespace)
		if ns == "" {
			fmt.Fprintln(os.Stderr, "Error: namespace is required. Provide --namespace or set it in llamafarm.yaml")
			os.Exit(1)
		}

		// Ensure server is running
		factory := GetServiceConfigFactory()
		config := factory.ServerOnly(serverURL)
		orchestrator.EnsureServicesOrExitWithConfig(config, "server")

		// Handle confirmation (follow rag_manage.go pattern)
		force, _ := cmd.Flags().GetBool("force")
		if !force {
			fmt.Printf("⚠️  WARNING: This will permanently delete project '%s/%s' and all associated data\n", ns, projectToDelete)
			fmt.Print("Are you sure? Type 'yes' to confirm: ")

			var response string
			fmt.Scanln(&response)
			if response != "yes" {
				fmt.Println("Operation cancelled")
				return
			}
		}

		// Execute delete (follow datasets.go delete pattern)
		url := buildServerURL(serverCfg.URL, fmt.Sprintf("/v1/projects/%s/%s", ns, projectToDelete))
		req, err := http.NewRequest("DELETE", url, nil)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error creating request: %v\n", err)
			os.Exit(1)
		}

		resp, err := utils.GetHTTPClient().Do(req)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error sending request: %v\n", err)
			os.Exit(1)
		}
		defer resp.Body.Close()

		body, readErr := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			if readErr != nil {
				fmt.Fprintf(os.Stderr, "Failed to delete project '%s/%s' (%d), and body read failed: %v\n",
					ns, projectToDelete, resp.StatusCode, readErr)
				os.Exit(1)
			}
			fmt.Fprintf(os.Stderr, "Failed to delete project '%s/%s' (%d): %s\n",
				ns, projectToDelete, resp.StatusCode, utils.PrettyServerError(resp, body))
			os.Exit(1)
		}

		fmt.Printf("✅ Successfully deleted project '%s/%s'\n", ns, projectToDelete)
	},
}

type projectRow struct {
	Namespace string
	Name      string
	ModTime   time.Time
	Path      string
	IsCurrent bool
}

func findConfigFileInDir(dir string) string {
	for _, name := range config.SupportedLlamaFarmConfigFiles {
		candidate := filepath.Join(dir, name)
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			return candidate
		}
	}
	return ""
}

func discoverProjects(projectsRoot string) ([]projectRow, []string, error) {
	currentInfo, currentPath := findCurrentProject(utils.GetEffectiveCWD())
	var rows []projectRow
	var warnings []string

	info, err := os.Stat(projectsRoot)
	if err != nil {
		if os.IsNotExist(err) {
			return rows, warnings, nil
		}
		return nil, warnings, fmt.Errorf("failed to read projects directory %s: %w", projectsRoot, err)
	}
	if !info.IsDir() {
		return rows, warnings, nil
	}

	namespaceEntries, err := os.ReadDir(projectsRoot)
	if err != nil {
		return nil, warnings, fmt.Errorf("failed to read projects directory %s: %w", projectsRoot, err)
	}

	for _, nsEntry := range namespaceEntries {
		if !nsEntry.IsDir() {
			continue
		}
		nsPath := filepath.Join(projectsRoot, nsEntry.Name())
		projectEntries, err := os.ReadDir(nsPath)
		if err != nil {
			warnings = append(warnings, fmt.Sprintf("failed to read namespace %s: %v", nsEntry.Name(), err))
			continue
		}

		for _, projEntry := range projectEntries {
			if !projEntry.IsDir() {
				continue
			}
			projPath := filepath.Join(nsPath, projEntry.Name())
			cfgPath := findConfigFileInDir(projPath)
			if cfgPath == "" {
				warnings = append(warnings, fmt.Sprintf("skipping %s/%s: no llamafarm config found", nsEntry.Name(), projEntry.Name()))
				continue
			}

			cfg, err := config.LoadConfigFile(cfgPath)
			if err != nil {
				warnings = append(warnings, fmt.Sprintf("skipping %s/%s: %v", nsEntry.Name(), projEntry.Name(), err))
				continue
			}

			projectInfo, err := cfg.GetProjectInfo()
			if err != nil {
				warnings = append(warnings, fmt.Sprintf("skipping %s/%s: %v", nsEntry.Name(), projEntry.Name(), err))
				continue
			}

			stat, err := os.Stat(cfgPath)
			if err != nil {
				warnings = append(warnings, fmt.Sprintf("skipping %s/%s: %v", nsEntry.Name(), projEntry.Name(), err))
				continue
			}

			absProjPath, err := filepath.Abs(projPath)
			if err != nil {
				absProjPath = projPath
			}

			isCurrent := currentInfo != nil &&
				projectsEqual(projectInfo, currentInfo)
			if isCurrent && currentPath != "" {
				absProjPath = currentPath
			}

			rows = append(rows, projectRow{
				Namespace: projectInfo.Namespace,
				Name:      projectInfo.Project,
				ModTime:   stat.ModTime(),
				Path:      absProjPath,
				IsCurrent: isCurrent,
			})
		}
	}

	return rows, warnings, nil
}

func findCurrentProject(startDir string) (*config.ProjectInfo, string) {
	dir := filepath.Clean(startDir)
	for {
		cfgPath, err := config.FindConfigFile(dir)
		if err == nil && cfgPath != "" {
			cfg, loadErr := config.LoadConfigFile(cfgPath)
			if loadErr == nil {
				if info, infoErr := cfg.GetProjectInfo(); infoErr == nil {
					return info, dir
				}
			}
		}

		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return nil, ""
}

func projectsEqual(a *config.ProjectInfo, b *config.ProjectInfo) bool {
	if a == nil || b == nil {
		return false
	}
	return strings.TrimSpace(a.Namespace) == strings.TrimSpace(b.Namespace) &&
		strings.TrimSpace(a.Project) == strings.TrimSpace(b.Project)
}

func sortProjectsByModTime(projects []projectRow) {
	sort.SliceStable(projects, func(i, j int) bool {
		return projects[i].ModTime.After(projects[j].ModTime)
	})
}

func printProjectsTable(projects []projectRow) {
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "PROJECT NAME\tNAMESPACE\tLAST MODIFIED\tPATH")
	fmt.Fprintln(w, "------------\t---------\t-------------\t----")
	for _, p := range projects {
		marker := " "
		if p.IsCurrent {
			marker = "*"
		}
		name := fmt.Sprintf("%s %s", marker, p.Name)
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\n", name, p.Namespace, p.ModTime.Local().Format("2006-01-02 15:04"), p.Path)
	}
	w.Flush()
}

func init() {
	// Server routing flags (align with datasets)
	projectsCmd.PersistentFlags().StringVar(&serverURL, "server-url", "", "LlamaFarm server URL (default: http://localhost:8000)")
	projectsCmd.PersistentFlags().StringVar(&namespace, "namespace", "", "Project namespace (default: from llamafarm.yaml)")
	projectsCmd.PersistentFlags().StringVar(&projectID, "project", "", "Project ID (default: from llamafarm.yaml)")

	// Add delete subcommand with force flag
	projectsDeleteCmd.Flags().BoolP("force", "f", false, "Skip confirmation prompt")

	// Register commands
	projectsCmd.AddCommand(projectsListCmd)
	projectsCmd.AddCommand(projectsDeleteCmd)

	// Add the projects command to root
	rootCmd.AddCommand(projectsCmd)
}
