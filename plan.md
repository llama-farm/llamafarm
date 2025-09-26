# Directory Upload Feature Plan

## Objective
Add directory upload capability to the CLI while fixing the existing single-file upload issue. The CLI must handle the directory traversal locally since the server (especially in Docker) cannot access local directories.

## Current Issue Analysis
The `lf datasets ingest` command is failing with 404 errors. Investigation shows:
- Direct API calls to `/v1/projects/{namespace}/{project}/datasets/{dataset}/data` work correctly
- The CLI's upload function appears correct but something in the execution path is broken

## Implementation Plan

### Phase 1: Debug and Fix Current Upload Issue
1. **Identify the exact failure point**
   - Check where the ingest command is defined
   - Trace the path from command to `uploadFileToDataset()`
   - Verify the server URL construction
   - Check if the dataset name is being passed correctly

2. **Fix the single-file upload**
   - Ensure proper error handling and logging
   - Verify multipart form construction
   - Test with single file upload

### Phase 2: Add Directory Upload Features
1. **Enhance CLI argument parsing**
   - Support for directory paths: `/path/to/dir/`
   - Support for glob patterns: `/path/*.pdf`, `/path/**/*.txt`
   - Add `--recursive` flag for directory traversal
   - Handle both relative and absolute paths

2. **Implement file discovery logic**
   ```
   Input patterns to support:
   - `/path/to/dir` - All files in directory (non-recursive by default)
   - `/path/to/dir/` - Same as above (trailing slash optional)
   - `/path/to/dir/*` - All files in directory (explicit)
   - `/path/to/dir/*.pdf` - Files matching pattern in directory
   - `/path/to/dir/**/*` - All files recursively
   - `/path/to/dir/**/*.pdf` - PDF files recursively
   ```

3. **File processing strategy**
   - Collect all matching files first
   - Display count and total size to user
   - Upload files individually to existing API endpoint
   - Show progress (e.g., "Uploading 3/25 files...")
   - Handle failures gracefully (continue with remaining files)
   - Provide summary at the end

### Phase 3: Implementation Details

#### CLI Changes (cli/cmd/datasets.go)
1. **Modify ingest command**
   - Accept multiple path arguments
   - Add `--recursive` flag
   - Add `--pattern` flag for filtering (e.g., `--pattern "*.pdf"`)

2. **Add file discovery function**
   ```go
   func expandPaths(paths []string, recursive bool, pattern string) ([]string, error) {
       var files []string
       for _, path := range paths {
           // Handle glob patterns
           // Handle directories
           // Apply recursive flag
           // Filter by pattern if provided
       }
       return files, nil
   }
   ```

3. **Batch upload with progress**
   ```go
   func uploadFiles(server, namespace, project, dataset string, files []string) error {
       fmt.Printf("Found %d files to upload\n", len(files))
       for i, file := range files {
           fmt.Printf("Uploading %d/%d: %s\n", i+1, len(files), filepath.Base(file))
           if err := uploadFileToDataset(server, namespace, project, dataset, file); err != nil {
               fmt.Printf("  ❌ Failed: %v\n", err)
               // Continue with next file
           } else {
               fmt.Printf("  ✅ Success\n")
           }
       }
       return nil
   }
   ```

#### Server Changes (Minimal)
- No server changes needed! The existing `/v1/projects/{namespace}/{project}/datasets/{dataset}/data` endpoint handles individual files
- The CLI will handle all directory traversal and send files one by one

### Phase 4: Testing
1. **Single file upload** (fix existing issue first)
   ```bash
   lf datasets ingest my-dataset /path/to/file.txt
   ```

2. **Directory upload** (non-recursive)
   ```bash
   lf datasets ingest my-dataset /path/to/dir/
   ```

3. **Recursive directory upload**
   ```bash
   lf datasets ingest my-dataset /path/to/dir/ --recursive
   ```

4. **Glob patterns**
   ```bash
   lf datasets ingest my-dataset "/path/to/dir/*.pdf"
   lf datasets ingest my-dataset "/path/to/dir/**/*.txt" 
   ```

5. **Multiple paths**
   ```bash
   lf datasets ingest my-dataset /path1/ /path2/*.pdf /path3/file.txt
   ```

### Phase 5: Error Handling
- **File not found**: Skip and log
- **Permission denied**: Skip and log
- **Upload failure**: Log and continue with next file
- **Network errors**: Implement retry logic (3 attempts)
- **Large files**: Show progress for individual file uploads

### Success Criteria
1. ✅ Single file upload works
2. ✅ Directory upload works (non-recursive)
3. ✅ Recursive directory upload works
4. ✅ Glob patterns work
5. ✅ Multiple path arguments work
6. ✅ Progress indication is clear
7. ✅ Errors don't stop the entire batch
8. ✅ Summary shows success/failure counts

### Implementation Order
1. First: Debug and fix the current single-file upload issue
2. Second: Add basic directory support (non-recursive)
3. Third: Add recursive flag
4. Fourth: Add glob pattern support
5. Fifth: Add progress and error handling improvements

## Minimal Change Approach
- **CLI only**: All changes confined to `cli/cmd/datasets.go`
- **No server changes**: Reuse existing upload endpoint
- **No API changes**: Same multipart form upload
- **Backward compatible**: Existing single-file usage unchanged