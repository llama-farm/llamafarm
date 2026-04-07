package llamabinary

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// sortStrings is a tiny helper to keep call sites terse.
func sortStrings(s []string) {
	sort.Strings(s)
}

// maxSymlinkDepth caps how many hops we will follow through a symlink chain
// before giving up. Matches Linux's MAXSYMLINKS. Without this limit a
// malicious archive with cyclic symlinks (A → B → A) or a very long chain
// could trigger unbounded recursion and stack overflow.
const maxSymlinkDepth = 40

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

	// Deterministic selection: prefer an exact path / base match; otherwise
	// pick the lexicographically smallest suffix match. This ensures the
	// same archive always yields the same extracted file regardless of zip
	// iteration order.
	var targetFile *zip.File
	var targetPath string
	var suffixCandidates []string
	for _, f := range r.File {
		if f.Name == srcPath || filepath.Base(f.Name) == srcName {
			targetFile = f
			targetPath = f.Name
			break
		}
		if strings.HasSuffix(f.Name, srcName) {
			suffixCandidates = append(suffixCandidates, f.Name)
		}
	}
	if targetFile == nil {
		if len(suffixCandidates) == 0 {
			return fmt.Errorf("file %s not found in archive", srcPath)
		}
		sortStrings(suffixCandidates)
		targetPath = suffixCandidates[0]
		targetFile = fileMap[targetPath]
	}
	return extractZipFileWithSymlinks(fileMap, targetFile, targetPath, destDir, srcName, 0)
}

// extractZipFileWithSymlinks recursively follows and preserves symlink chains.
// The depth parameter is bounded by maxSymlinkDepth to prevent unbounded
// recursion when an archive contains cyclic or pathologically long symlink
// chains.
func extractZipFileWithSymlinks(fileMap map[string]*zip.File, f *zip.File, fPath, destDir, finalName string, depth int) error {
	if depth >= maxSymlinkDepth {
		return fmt.Errorf("symlink chain exceeded max depth (%d): possible cycle at %s", maxSymlinkDepth, fPath)
	}
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
		if err := extractZipFileWithSymlinks(fileMap, next, resolved, destDir, targetBase, depth+1); err != nil {
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

	// Pick the target entry deterministically. Map iteration order in Go is
	// randomized, so when multiple entries have the same basename suffix the
	// previous "first match" approach could return different entries across
	// runs. Instead, collect all matches, prefer exact path / base matches,
	// and sort to make ties deterministic.
	var targetEntry *tar.Header
	var targetName string
	candidates := make([]string, 0)
	for name := range entries {
		if name == srcName || filepath.Base(name) == srcName || strings.HasSuffix(name, "/"+srcName) {
			candidates = append(candidates, name)
		}
	}
	if len(candidates) == 0 {
		return fmt.Errorf("file %s not found in archive", srcPath)
	}
	// Deterministic ordering + exact-path preference.
	for _, name := range candidates {
		if name == srcName || filepath.Base(name) == srcName {
			targetEntry = entries[name]
			targetName = name
			break
		}
	}
	if targetEntry == nil {
		// Sort remaining candidates to break ties deterministically.
		sortedCandidates := append([]string(nil), candidates...)
		sortStrings(sortedCandidates)
		targetName = sortedCandidates[0]
		targetEntry = entries[targetName]
	}

	// Follow the symlink chain, capped at maxSymlinkDepth to prevent
	// infinite loops on cyclic or pathologically deep symlink chains.
	resolvedName := targetName
	for depth := 0; targetEntry.Typeflag == tar.TypeSymlink; depth++ {
		if depth >= maxSymlinkDepth {
			return fmt.Errorf("symlink chain exceeded max depth (%d): possible cycle at %s", maxSymlinkDepth, resolvedName)
		}
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
		// The destination filename comes from the caller-supplied destPath,
		// not the archive entry, but we still validate it against destDir to
		// ensure the write cannot escape. This uses the same layered Zip
		// Slip sanitization (explicit ".." rejection + absolute-path prefix
		// check) that CodeQL recognizes for go/zipslip.
		target, ok := safeDestPath(destDir, filepath.Base(destPath))
		if !ok {
			return fmt.Errorf("refused to extract %s: path traversal", h.Name)
		}
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
		// Apply the full safeDestPath Zip Slip sanitization (explicit ".."
		// rejection + absolute-path containment) before any filesystem op.
		destPath, ok := safeDestPath(destDir, h.Name)
		if !ok {
			continue
		}
		name := filepath.Base(destPath)
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
		// Apply the full safeDestPath Zip Slip sanitization (explicit ".."
		// rejection + absolute-path containment) before any filesystem op.
		destPath, ok := safeDestPath(destDir, f.Name)
		if !ok {
			continue
		}
		name := filepath.Base(destPath)
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

// safeDestPath validates that an archive entry's name can be safely joined with
// destDir without escaping it (Zip Slip protection). It returns the validated
// absolute destination path on success.
//
// The function applies four layered checks:
//  1. Explicit rejection of entries whose path contains ".." anywhere. This is
//     the CWE-22 sanitization pattern recognized by CodeQL's go/zipslip rule.
//  2. Reduction to a pure base filename (no directory components), rejecting
//     names that would traverse via safeBaseName.
//  3. filepath.Join with destDir, followed by filepath.Abs normalization.
//  4. A strings.HasPrefix containment check ensuring the resolved absolute
//     path lives under destDir.
//
// Returns ("", false) if any check fails; the caller should skip the entry.
func safeDestPath(destDir, entryName string) (string, bool) {
	// CodeQL-recognized sanitization: reject any ".." component explicitly
	// before using the entry name in a filesystem operation.
	if strings.Contains(entryName, "..") {
		return "", false
	}
	base := filepath.Base(entryName)
	if !safeBaseName(base) {
		return "", false
	}
	absDestDir, err := filepath.Abs(destDir)
	if err != nil {
		return "", false
	}
	joined := filepath.Join(absDestDir, base)
	absJoined, err := filepath.Abs(joined)
	if err != nil {
		return "", false
	}
	// Ensure the resolved path stays within destDir. The trailing separator
	// check prevents "/tmp/destdir-evil" from matching "/tmp/destdir".
	sep := string(filepath.Separator)
	if absJoined != absDestDir && !strings.HasPrefix(absJoined, absDestDir+sep) {
		return "", false
	}
	return absJoined, true
}
