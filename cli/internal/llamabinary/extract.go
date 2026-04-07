package llamabinary

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// extractZip extracts a single file (matched by base name or exact path) from a
// zip archive, following any symlink chain and recreating it in the destination.
func extractZip(archivePath, srcPath, destPath string) error {
	r, err := zip.OpenReader(archivePath)
	if err != nil {
		return err
	}
	defer r.Close()

	destDir := filepath.Dir(destPath)
	srcName := filepath.Base(srcPath)

	fileMap := make(map[string]*zip.File)
	for _, f := range r.File {
		fileMap[f.Name] = f
	}

	var targetFile *zip.File
	var targetPath string
	for _, f := range r.File {
		if strings.HasSuffix(f.Name, srcName) || f.Name == srcPath {
			targetFile = f
			targetPath = f.Name
			break
		}
	}
	if targetFile == nil {
		return fmt.Errorf("file %s not found in archive", srcPath)
	}
	return extractZipFileWithSymlinks(fileMap, targetFile, targetPath, destDir, srcName)
}

// extractZipFileWithSymlinks recursively follows and preserves symlink chains.
func extractZipFileWithSymlinks(fileMap map[string]*zip.File, f *zip.File, fPath, destDir, finalName string) error {
	if !safeBaseName(finalName) {
		return fmt.Errorf("invalid filename: %s", finalName)
	}

	if f.Mode()&os.ModeSymlink != 0 {
		rc, err := f.Open()
		if err != nil {
			return fmt.Errorf("open symlink %s: %w", f.Name, err)
		}
		targetBytes, err := io.ReadAll(rc)
		rc.Close()
		if err != nil {
			return fmt.Errorf("read symlink %s: %w", f.Name, err)
		}
		target := string(targetBytes)

		symlinkDir := filepath.Dir(fPath)
		resolved := filepath.Join(symlinkDir, target)
		resolved = strings.ReplaceAll(filepath.Clean(resolved), "\\", "/")

		next, ok := fileMap[resolved]
		if !ok {
			// Fallback: match basename within symlink's directory.
			base := filepath.Base(target)
			for name, tf := range fileMap {
				if strings.HasSuffix(name, base) && filepath.Dir(name) == symlinkDir {
					next = tf
					resolved = name
					ok = true
					break
				}
			}
		}
		if !ok {
			return fmt.Errorf("symlink target %s not found", target)
		}

		targetBase := filepath.Base(target)
		if err := extractZipFileWithSymlinks(fileMap, next, resolved, destDir, targetBase); err != nil {
			return err
		}

		// Validate: symlink target must stay within destDir.
		resolvedSym := filepath.Clean(filepath.Join(destDir, target))
		if !strings.HasPrefix(resolvedSym, filepath.Clean(destDir)+string(filepath.Separator)) &&
			resolvedSym != filepath.Clean(destDir) {
			return fmt.Errorf("symlink target %s would escape %s", target, destDir)
		}

		symPath := filepath.Join(destDir, finalName)
		os.Remove(symPath)
		if err := os.Symlink(target, symPath); err != nil {
			return fmt.Errorf("create symlink %s: %w", symPath, err)
		}
		return nil
	}

	// Regular file.
	destPath := filepath.Join(destDir, finalName)
	rc, err := f.Open()
	if err != nil {
		return fmt.Errorf("open %s: %w", f.Name, err)
	}
	defer rc.Close()
	os.Remove(destPath)
	out, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("create %s: %w", destPath, err)
	}
	if _, err := io.Copy(out, rc); err != nil {
		out.Close()
		return fmt.Errorf("write %s: %w", destPath, err)
	}
	if err := out.Close(); err != nil {
		return err
	}
	_ = os.Chmod(destPath, 0o755)
	return nil
}

// extractTarGz extracts a single named file (following symlinks) from a tar.gz.
func extractTarGz(archivePath, srcPath, destPath string) error {
	srcName := filepath.Base(srcPath)

	entries := make(map[string]*tar.Header)
	if err := readTarGzEntries(archivePath, entries); err != nil {
		return err
	}

	var targetEntry *tar.Header
	var targetName string
	for name, h := range entries {
		if strings.HasSuffix(name, srcName) {
			targetEntry = h
			targetName = name
			break
		}
	}
	if targetEntry == nil {
		return fmt.Errorf("file %s not found in archive", srcPath)
	}

	// Follow the symlink chain.
	resolvedName := targetName
	for targetEntry.Typeflag == tar.TypeSymlink {
		symlinkDir := filepath.Dir(resolvedName)
		target := filepath.Join(symlinkDir, targetEntry.Linkname)
		target = strings.ReplaceAll(filepath.Clean(target), "\\", "/")

		next, ok := entries[target]
		if !ok {
			base := filepath.Base(targetEntry.Linkname)
			for name, h := range entries {
				if filepath.Dir(name) == symlinkDir && filepath.Base(name) == base {
					next = h
					target = name
					ok = true
					break
				}
			}
		}
		if !ok {
			return fmt.Errorf("symlink target %s not found", target)
		}
		targetEntry = next
		resolvedName = target
	}

	return extractTarGzFile(archivePath, resolvedName, destPath)
}

func readTarGzEntries(archivePath string, entries map[string]*tar.Header) error {
	f, err := os.Open(archivePath)
	if err != nil {
		return err
	}
	defer f.Close()
	gzr, err := gzip.NewReader(f)
	if err != nil {
		return fmt.Errorf("gzip: %w", err)
	}
	defer gzr.Close()
	tr := tar.NewReader(gzr)
	for {
		h, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("tar: %w", err)
		}
		name := strings.ReplaceAll(filepath.Clean(h.Name), "\\", "/")
		cp := *h
		entries[name] = &cp
	}
	return nil
}

func extractTarGzFile(archivePath, fileName, destPath string) error {
	f, err := os.Open(archivePath)
	if err != nil {
		return err
	}
	defer f.Close()
	gzr, err := gzip.NewReader(f)
	if err != nil {
		return fmt.Errorf("gzip: %w", err)
	}
	defer gzr.Close()
	tr := tar.NewReader(gzr)
	destDir := filepath.Dir(destPath)

	for {
		h, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("tar: %w", err)
		}
		name := strings.ReplaceAll(filepath.Clean(h.Name), "\\", "/")
		if name != fileName {
			continue
		}
		base := filepath.Base(h.Name)
		if !safeBaseName(base) {
			continue
		}
		target := filepath.Join(destDir, filepath.Base(destPath))
		out, err := os.Create(target)
		if err != nil {
			return fmt.Errorf("create %s: %w", target, err)
		}
		if _, err := io.Copy(out, tr); err != nil {
			out.Close()
			return fmt.Errorf("write %s: %w", target, err)
		}
		out.Close()
		_ = os.Chmod(target, 0o755)
		return nil
	}
	return fmt.Errorf("file %s not found in archive", fileName)
}

// extractTarGzDependencies extracts all sibling dependency libraries (ggml, metal,
// cuda, etc.) into destDir for the given target OS. The main library is identified
// by mainLib and excluded from the dependency copy.
func extractTarGzDependencies(archivePath, destDir, mainLib, targetOS string) error {
	f, err := os.Open(archivePath)
	if err != nil {
		return err
	}
	defer f.Close()
	gzr, err := gzip.NewReader(f)
	if err != nil {
		return fmt.Errorf("gzip: %w", err)
	}
	defer gzr.Close()
	tr := tar.NewReader(gzr)

	patterns := depPatternsFor(targetOS)
	mainLower := strings.ToLower(mainLib)

	for {
		h, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("tar: %w", err)
		}
		if h.Typeflag == tar.TypeDir {
			continue
		}
		name := filepath.Base(h.Name)
		if !safeBaseName(name) {
			continue
		}
		nameLower := strings.ToLower(name)
		if !matchesDepPattern(nameLower, patterns) {
			continue
		}
		if nameLower == mainLower {
			continue
		}
		if strings.HasPrefix(nameLower, "libllama.") || strings.HasPrefix(nameLower, "llama.") {
			continue
		}
		if h.Size < 100 {
			continue
		}
		destPath := filepath.Join(destDir, name)
		if _, err := os.Stat(destPath); err == nil {
			continue
		}
		out, err := os.Create(destPath)
		if err != nil {
			continue
		}
		if _, err := io.Copy(out, tr); err != nil {
			out.Close()
			continue
		}
		out.Close()
		if targetOS != "windows" && !strings.HasSuffix(nameLower, ".metal") {
			_ = os.Chmod(destPath, 0o755)
		}
	}
	return nil
}

// extractZipDependencies is the zip-archive equivalent of extractTarGzDependencies.
func extractZipDependencies(archivePath, destDir, mainLib, targetOS string) error {
	r, err := zip.OpenReader(archivePath)
	if err != nil {
		return err
	}
	defer r.Close()

	patterns := depPatternsFor(targetOS)
	mainLower := strings.ToLower(mainLib)

	for _, f := range r.File {
		if f.FileInfo().IsDir() {
			continue
		}
		if f.Mode()&os.ModeSymlink != 0 {
			continue
		}
		name := filepath.Base(f.Name)
		if !safeBaseName(name) {
			continue
		}
		nameLower := strings.ToLower(name)
		if !matchesDepPattern(nameLower, patterns) {
			continue
		}
		if nameLower == mainLower {
			continue
		}
		if strings.HasPrefix(nameLower, "libllama.") || strings.HasPrefix(nameLower, "llama.") {
			continue
		}
		if f.UncompressedSize64 < 100 {
			continue
		}
		destPath := filepath.Join(destDir, name)
		if _, err := os.Stat(destPath); err == nil {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			continue
		}
		out, err := os.Create(destPath)
		if err != nil {
			rc.Close()
			continue
		}
		if _, err := io.Copy(out, rc); err != nil {
			out.Close()
			rc.Close()
			continue
		}
		out.Close()
		rc.Close()
		if targetOS != "windows" && !strings.HasSuffix(nameLower, ".metal") {
			_ = os.Chmod(destPath, 0o755)
		}
	}
	return nil
}

func depPatternsFor(targetOS string) []string {
	switch targetOS {
	case "windows":
		return []string{".dll"}
	case "darwin":
		return []string{".dylib", ".metal"}
	default:
		return []string{".so.", ".so"}
	}
}

func matchesDepPattern(nameLower string, patterns []string) bool {
	for _, p := range patterns {
		if strings.Contains(nameLower, p) {
			return true
		}
	}
	return false
}

// createDependencySymlinks recreates major/unversioned symlinks for versioned libs
// installed into destDir. On Linux it creates libfoo.so → libfoo.so.N → libfoo.so.N.M.K,
// and the analogous macOS dylib chain.
func createDependencySymlinks(destDir, targetOS string) error {
	entries, err := os.ReadDir(destDir)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()

		if targetOS == "darwin" {
			if !strings.HasSuffix(name, ".dylib") {
				continue
			}
			parts := strings.Split(name, ".")
			if len(parts) < 5 {
				continue
			}
			baseName := ""
			versionStart := -1
			for i, part := range parts {
				if _, err := fmt.Sscanf(part, "%d", new(int)); err == nil {
					versionStart = i
					break
				}
				if baseName != "" {
					baseName += "."
				}
				baseName += part
			}
			if versionStart < 0 || baseName == "" {
				continue
			}
			major := parts[versionStart]
			majorSym := filepath.Join(destDir, fmt.Sprintf("%s.%s.dylib", baseName, major))
			if _, err := os.Lstat(majorSym); os.IsNotExist(err) {
				_ = os.Symlink(name, majorSym)
			}
			baseSym := filepath.Join(destDir, fmt.Sprintf("%s.dylib", baseName))
			if _, err := os.Lstat(baseSym); os.IsNotExist(err) {
				_ = os.Symlink(filepath.Base(majorSym), baseSym)
			}
		} else {
			if !strings.Contains(name, ".so.") {
				continue
			}
			soIdx := strings.Index(name, ".so.")
			if soIdx < 0 {
				continue
			}
			baseName := name[:soIdx]
			versionPart := name[soIdx+4:]
			vParts := strings.Split(versionPart, ".")
			if len(vParts) < 1 {
				continue
			}
			major := vParts[0]
			majorSym := filepath.Join(destDir, fmt.Sprintf("%s.so.%s", baseName, major))
			if _, err := os.Lstat(majorSym); os.IsNotExist(err) {
				_ = os.Symlink(name, majorSym)
			}
			baseSym := filepath.Join(destDir, fmt.Sprintf("%s.so", baseName))
			if _, err := os.Lstat(baseSym); os.IsNotExist(err) {
				_ = os.Symlink(filepath.Base(majorSym), baseSym)
			}
		}
	}
	return nil
}

// safeBaseName rejects names that would escape a destination directory.
func safeBaseName(name string) bool {
	if name == "" || name == "." || name == ".." {
		return false
	}
	if strings.ContainsAny(name, "/\\") {
		return false
	}
	if filepath.IsAbs(name) {
		return false
	}
	return true
}
