package cmd

import (
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

func TestIsGlobPattern(t *testing.T) {
	tests := []struct {
		path     string
		expected bool
		desc     string
	}{
		{"*.txt", true, "asterisk wildcard"},
		{"file?.log", true, "question mark wildcard"},
		{"[abc].txt", true, "bracket range"},
		{"file[0-9].txt", true, "bracket numeric range"},
		{"normal_file.txt", false, "normal file path"},
		{"/path/to/file.txt", false, "absolute path without globs"},
		{"*.{txt,log}", true, "asterisk with braces"},
		{"test/[a-z]*.txt", true, "mixed glob patterns"},
		{"file\\*.txt", true, "escaped asterisk (still detected as glob)"},
		{"", false, "empty string"},
		{"just_a_directory/", false, "directory path without globs"},
		{"**/*.txt", true, "double asterisk pattern"},
		{"docs/**/files", true, "double asterisk in middle"},
	}

	for _, tt := range tests {
		t.Run(tt.desc, func(t *testing.T) {
			result := isGlobPattern(tt.path)
			if result != tt.expected {
				t.Errorf("isGlobPattern(%q) = %t, want %t", tt.path, result, tt.expected)
			}
		})
	}
}

func TestRecursiveGlob(t *testing.T) {
	// Create a temporary test directory structure
	testDir := t.TempDir()

	// Create directory structure:
	// testdir/
	//   ├── file1.txt
	//   ├── file2.md
	//   ├── subdir1/
	//   │   ├── sub1.txt
	//   │   └── sub1.pdf
	//   └── subdir2/
	//       ├── sub2.txt
	//       └── nested/
	//           └── deep.txt

	dirs := []string{
		filepath.Join(testDir, "subdir1"),
		filepath.Join(testDir, "subdir2"),
		filepath.Join(testDir, "subdir2", "nested"),
	}

	files := []string{
		filepath.Join(testDir, "file1.txt"),
		filepath.Join(testDir, "file2.md"),
		filepath.Join(testDir, "subdir1", "sub1.txt"),
		filepath.Join(testDir, "subdir1", "sub1.pdf"),
		filepath.Join(testDir, "subdir2", "sub2.txt"),
		filepath.Join(testDir, "subdir2", "nested", "deep.txt"),
	}

	// Create directories
	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatalf("Failed to create directory %s: %v", dir, err)
		}
	}

	// Create files
	for _, file := range files {
		if err := os.WriteFile(file, []byte("test content"), 0644); err != nil {
			t.Fatalf("Failed to create file %s: %v", file, err)
		}
	}

	tests := []struct {
		name        string
		pattern     string
		expectedCount int
		shouldContain []string // Files that should be in results (relative to testDir)
		shouldError   bool
	}{
		{
			name:        "non-recursive glob",
			pattern:     filepath.Join(testDir, "*.txt"),
			expectedCount: 1,
			shouldContain: []string{"file1.txt"},
		},
		{
			name:        "recursive all files",
			pattern:     filepath.Join(testDir, "**", "*"),
			expectedCount: 6,
			shouldContain: []string{"file1.txt", "file2.md", "subdir1/sub1.txt", "subdir1/sub1.pdf", "subdir2/sub2.txt", "subdir2/nested/deep.txt"},
		},
		{
			name:        "recursive txt files only",
			pattern:     filepath.Join(testDir, "**", "*.txt"),
			expectedCount: 4,
			shouldContain: []string{"file1.txt", "subdir1/sub1.txt", "subdir2/sub2.txt", "subdir2/nested/deep.txt"},
		},
		{
			name:        "recursive pdf files only",
			pattern:     filepath.Join(testDir, "**", "*.pdf"),
			expectedCount: 1,
			shouldContain: []string{"subdir1/sub1.pdf"},
		},
		{
			name:        "specific subdirectory",
			pattern:     filepath.Join(testDir, "subdir1", "*"),
			expectedCount: 2,
			shouldContain: []string{"subdir1/sub1.txt", "subdir1/sub1.pdf"},
		},
		{
			name:        "two-level glob for txt files",
			pattern:     filepath.Join(testDir, "*", "*.txt"),
			expectedCount: 2,
			shouldContain: []string{"subdir1/sub1.txt", "subdir2/sub2.txt"},
		},
		{
			name:        "non-existent path",
			pattern:     filepath.Join(testDir, "nonexistent", "**", "*"),
			expectedCount: 0,
			shouldContain: []string{},
		},
		{
			name:        "invalid pattern with multiple **",
			pattern:     filepath.Join(testDir, "**", "middle", "**", "*"),
			shouldError: true,
		},
		{
			name:        "fallback to standard glob",
			pattern:     filepath.Join(testDir, "file?.txt"),
			expectedCount: 1,
			shouldContain: []string{"file1.txt"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := recursiveGlob(tt.pattern)

			if tt.shouldError {
				if err == nil {
					t.Errorf("recursiveGlob(%q) expected error but got none", tt.pattern)
				}
				return
			}

			if err != nil {
				t.Errorf("recursiveGlob(%q) unexpected error: %v", tt.pattern, err)
				return
			}

			if len(results) != tt.expectedCount {
				t.Errorf("recursiveGlob(%q) returned %d files, want %d", tt.pattern, len(results), tt.expectedCount)
				t.Logf("Results: %v", results)
			}

			// Check that expected files are present
			for _, expectedFile := range tt.shouldContain {
				expectedPath := filepath.Join(testDir, expectedFile)
				found := false
				for _, result := range results {
					if result == expectedPath {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("recursiveGlob(%q) missing expected file: %s", tt.pattern, expectedPath)
					t.Logf("Results: %v", results)
				}
			}
		})
	}
}

func TestExpandPathsToFiles(t *testing.T) {
	// Create a temporary test directory structure
	testDir := t.TempDir()

	// Create directory structure similar to above
	dirs := []string{
		filepath.Join(testDir, "docs"),
		filepath.Join(testDir, "pdfs"),
		filepath.Join(testDir, "docs", "nested"),
	}

	files := []string{
		filepath.Join(testDir, "readme.txt"),
		filepath.Join(testDir, "config.yaml"),
		filepath.Join(testDir, "docs", "guide.md"),
		filepath.Join(testDir, "docs", "api.txt"),
		filepath.Join(testDir, "docs", "nested", "deep.md"),
		filepath.Join(testDir, "pdfs", "manual.pdf"),
		filepath.Join(testDir, "pdfs", "spec.pdf"),
	}

	// Create directories
	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatalf("Failed to create directory %s: %v", dir, err)
		}
	}

	// Create files
	for _, file := range files {
		if err := os.WriteFile(file, []byte("test content"), 0644); err != nil {
			t.Fatalf("Failed to create file %s: %v", file, err)
		}
	}

	tests := []struct {
		name          string
		paths         []string
		expectedCount int
		shouldContain []string // Files that should be in results (basenames for simplicity)
		shouldError   bool
	}{
		{
			name:          "single file",
			paths:         []string{filepath.Join(testDir, "readme.txt")},
			expectedCount: 1,
			shouldContain: []string{"readme.txt"},
		},
		{
			name:          "multiple files",
			paths:         []string{filepath.Join(testDir, "readme.txt"), filepath.Join(testDir, "config.yaml")},
			expectedCount: 2,
			shouldContain: []string{"readme.txt", "config.yaml"},
		},
		{
			name:          "directory non-recursive",
			paths:         []string{filepath.Join(testDir, "docs")},
			expectedCount: 2, // guide.md and api.txt only (not nested/deep.md)
			shouldContain: []string{"guide.md", "api.txt"},
		},
		{
			name:          "glob pattern",
			paths:         []string{filepath.Join(testDir, "*.txt")},
			expectedCount: 1,
			shouldContain: []string{"readme.txt"},
		},
		{
			name:          "recursive glob pattern",
			paths:         []string{filepath.Join(testDir, "**", "*.md")},
			expectedCount: 2,
			shouldContain: []string{"guide.md", "deep.md"},
		},
		{
			name:          "mixed inputs",
			paths:         []string{
				filepath.Join(testDir, "readme.txt"),
				filepath.Join(testDir, "pdfs"),
				filepath.Join(testDir, "docs", "*.md"),
			},
			expectedCount: 4, // readme.txt + 2 pdfs + guide.md
			shouldContain: []string{"readme.txt", "manual.pdf", "spec.pdf", "guide.md"},
		},
		{
			name:          "non-existent file",
			paths:         []string{filepath.Join(testDir, "nonexistent.txt")},
			expectedCount: 0, // Should warn but not error
			shouldContain: []string{},
		},
		{
			name:          "duplicate files handled",
			paths:         []string{
				filepath.Join(testDir, "readme.txt"),
				filepath.Join(testDir, "readme.txt"), // duplicate
			},
			expectedCount: 1, // Should deduplicate
			shouldContain: []string{"readme.txt"},
		},
		{
			name:        "invalid glob pattern",
			paths:       []string{filepath.Join(testDir, "**", "middle", "**", "*")},
			shouldError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := expandPathsToFiles(tt.paths)

			if tt.shouldError {
				if err == nil {
					t.Errorf("expandPathsToFiles(%v) expected error but got none", tt.paths)
				}
				return
			}

			if err != nil {
				t.Errorf("expandPathsToFiles(%v) unexpected error: %v", tt.paths, err)
				return
			}

			if len(results) != tt.expectedCount {
				t.Errorf("expandPathsToFiles(%v) returned %d files, want %d", tt.paths, len(results), tt.expectedCount)
				t.Logf("Results: %v", results)
			}

			// Check that expected files are present (by basename)
			resultBasenames := make([]string, len(results))
			for i, result := range results {
				resultBasenames[i] = filepath.Base(result)
			}

			for _, expectedFile := range tt.shouldContain {
				found := false
				for _, basename := range resultBasenames {
					if basename == expectedFile {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("expandPathsToFiles(%v) missing expected file: %s", tt.paths, expectedFile)
					t.Logf("Result basenames: %v", resultBasenames)
				}
			}
		})
	}
}

func TestExpandPathsToFiles_ErrorCases(t *testing.T) {
	tests := []struct {
		name    string
		paths   []string
		wantErr bool
		errMsg  string
	}{
		{
			name:    "invalid permission on directory",
			paths:   []string{filepath.Join(os.TempDir(), "some-random-nonexistent-folder-xyz12345")},
			wantErr: false, // Should not error, just skip inaccessible or nonexistent directories
		},
		{
			name:    "invalid glob pattern with multiple **",
			paths:   []string{"some/**/path/**/pattern"},
			wantErr: true,
			errMsg:  "invalid pattern: only one ** supported per pattern",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := expandPathsToFiles(tt.paths)

			if tt.wantErr {
				if err == nil {
					t.Errorf("expandPathsToFiles(%v) expected error but got none", tt.paths)
					return
				}
				if tt.errMsg != "" && !strings.Contains(err.Error(), tt.errMsg) {
					t.Errorf("expandPathsToFiles(%v) error = %q, want error containing %q", tt.paths, err.Error(), tt.errMsg)
				}
			} else {
				if err != nil {
					t.Errorf("expandPathsToFiles(%v) unexpected error: %v", tt.paths, err)
				}
			}
		})
	}
}

func TestRecursiveGlob_EdgeCases(t *testing.T) {
	// Create a temporary test directory
	testDir := t.TempDir()

	// Create a simple file
	testFile := filepath.Join(testDir, "test.txt")
	if err := os.WriteFile(testFile, []byte("content"), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	tests := []struct {
		name     string
		pattern  string
		expected []string
		wantErr  bool
	}{
		{
			name:     "pattern without **",
			pattern:  filepath.Join(testDir, "*.txt"),
			expected: []string{testFile},
		},
		{
			name:     "pattern with ** but empty remaining",
			pattern:  filepath.Join(testDir, "**"),
			expected: []string{testFile}, // Should find the file
		},
		{
			name:     "pattern with ** and *",
			pattern:  filepath.Join(testDir, "**", "*"),
			expected: []string{testFile},
		},
		{
			name:    "pattern with multiple **",
			pattern: filepath.Join(testDir, "**", "middle", "**"),
			wantErr: true,
		},
		{
			name:     "empty base path",
			pattern:  "**/*.txt",
			expected: nil, // Don't check exact results, just ensure no error
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := recursiveGlob(tt.pattern)

			if tt.wantErr {
				if err == nil {
					t.Errorf("recursiveGlob(%q) expected error but got none", tt.pattern)
				}
				return
			}

			if err != nil {
				t.Errorf("recursiveGlob(%q) unexpected error: %v", tt.pattern, err)
				return
			}

			// Skip exact comparison for nil expected (just check no error)
			if tt.expected != nil {
				// Sort both slices for comparison
				sort.Strings(results)
				sort.Strings(tt.expected)

				if !reflect.DeepEqual(results, tt.expected) {
					t.Errorf("recursiveGlob(%q) = %v, want %v", tt.pattern, results, tt.expected)
				}
			}
		})
	}
}

func TestRecursiveGlob_SymlinksAndPermissions(t *testing.T) {
	// Create a temporary test directory
	testDir := t.TempDir()

	// Create a file and directory
	testFile := filepath.Join(testDir, "file.txt")
	testSubdir := filepath.Join(testDir, "subdir")
	testSubFile := filepath.Join(testSubdir, "sub.txt")

	if err := os.WriteFile(testFile, []byte("content"), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	if err := os.MkdirAll(testSubdir, 0755); err != nil {
		t.Fatalf("Failed to create subdirectory: %v", err)
	}

	if err := os.WriteFile(testSubFile, []byte("sub content"), 0644); err != nil {
		t.Fatalf("Failed to create sub file: %v", err)
	}

	// Test that the function handles inaccessible directories gracefully
	t.Run("handles inaccessible directories", func(t *testing.T) {
		// Make subdirectory inaccessible (if we can)
		if err := os.Chmod(testSubdir, 0000); err != nil {
			t.Skip("Cannot change directory permissions on this system")
		}

		// Restore permissions after test
		defer os.Chmod(testSubdir, 0755)

		pattern := filepath.Join(testDir, "**", "*.txt")
		results, err := recursiveGlob(pattern)

		// Should not error, but might not find the sub file
		if err != nil {
			t.Errorf("recursiveGlob(%q) unexpected error: %v", pattern, err)
		}

		// Should at least find the main file
		found := false
		for _, result := range results {
			if result == testFile {
				found = true
				break
			}
		}

		if !found {
			t.Errorf("recursiveGlob(%q) should find accessible file %s", pattern, testFile)
		}
	})
}

func TestDatasetsRecursiveIntegration(t *testing.T) {
	// Integration test demonstrating the complete workflow
	testDir := t.TempDir()

	// Create a complex directory structure
	dirs := []string{
		filepath.Join(testDir, "docs"),
		filepath.Join(testDir, "docs", "api"),
		filepath.Join(testDir, "docs", "guides"),
		filepath.Join(testDir, "src"),
		filepath.Join(testDir, "src", "components"),
		filepath.Join(testDir, "tests"),
	}

	files := []string{
		filepath.Join(testDir, "README.md"),
		filepath.Join(testDir, "package.json"),
		filepath.Join(testDir, "docs", "overview.md"),
		filepath.Join(testDir, "docs", "api", "endpoints.md"),
		filepath.Join(testDir, "docs", "api", "auth.txt"),
		filepath.Join(testDir, "docs", "guides", "quickstart.md"),
		filepath.Join(testDir, "src", "main.js"),
		filepath.Join(testDir, "src", "components", "Button.tsx"),
		filepath.Join(testDir, "src", "components", "Modal.tsx"),
		filepath.Join(testDir, "tests", "main.test.js"),
		filepath.Join(testDir, "tests", "integration.spec.ts"),
	}

	// Create directories
	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatalf("Failed to create directory %s: %v", dir, err)
		}
	}

	// Create files
	for _, file := range files {
		if err := os.WriteFile(file, []byte("test content"), 0644); err != nil {
			t.Fatalf("Failed to create file %s: %v", file, err)
		}
	}

	testCases := []struct {
		name          string
		paths         []string
		expectedCount int
		description   string
	}{
		{
			name:          "mixed patterns like CLI usage",
			paths:         []string{
				filepath.Join(testDir, "README.md"),           // Single file
				filepath.Join(testDir, "docs"),                // Directory (non-recursive)
				filepath.Join(testDir, "src", "**", "*.tsx"),  // Recursive pattern for specific extension
				filepath.Join(testDir, "tests", "*"),          // Single-level glob
			},
			expectedCount: 6, // README.md + 1 docs file (overview.md, non-recursive) + 2 tsx files + 2 test files
			description:   "Demonstrates real-world usage mixing files, directories, and glob patterns",
		},
		{
			name:          "all markdown files recursively",
			paths:         []string{filepath.Join(testDir, "**", "*.md")},
			expectedCount: 4, // README.md + overview.md + endpoints.md + quickstart.md
			description:   "Find all markdown files in entire project tree",
		},
		{
			name:          "source files only",
			paths:         []string{filepath.Join(testDir, "src", "**", "*")},
			expectedCount: 3, // main.js + Button.tsx + Modal.tsx
			description:   "Recursively find all source files",
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			results, err := expandPathsToFiles(tt.paths)
			if err != nil {
				t.Errorf("expandPathsToFiles(%v) unexpected error: %v", tt.paths, err)
				return
			}

			if len(results) != tt.expectedCount {
				t.Errorf("expandPathsToFiles(%v) returned %d files, want %d", tt.paths, len(results), tt.expectedCount)
				t.Logf("Description: %s", tt.description)
				t.Logf("Results: %v", results)
			}
		})
	}
}
