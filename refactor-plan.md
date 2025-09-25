# Refactoring Plan: Client-Side File Processing for Docker Compatibility

## Current Problem

The current implementation has the server-side processing paths and expanding directories, which **won't work in Docker** because:
1. Docker container doesn't have access to the user's local filesystem
2. Paths sent to the server are meaningless in a containerized environment
3. The server cannot read files from the client's machine

## What Was Implemented (Current State)

### Server-Side (`server/api/routers/datasets/datasets.py`)
- Smart endpoint that accepts:
  - Direct file uploads (✅ Works with Docker)
  - File paths for server-side processing (❌ Won't work with Docker)
  - Directory paths for server-side expansion (❌ Won't work with Docker)
  - Glob patterns for server-side matching (❌ Won't work with Docker)

### CLI-Side (`cli/cmd/datasets.go`)
- Determines input type (files/paths/mixed)
- For actual files: uploads them (✅ Works)
- For paths/patterns: sends them to server for processing (❌ Won't work)

## What Needs to Change

### Core Architecture Change
**ALL file processing, path expansion, and glob matching must happen CLIENT-SIDE**

The server should ONLY receive:
1. Actual file content (via multipart upload)
2. Metadata about the files

## New Implementation Plan

### 1. Server-Side Simplification

The server endpoint should be simplified to ONLY handle file uploads:

```python
@router.post("/{dataset}/ingest", response_model=BatchIngestResponse)
async def ingest_files(
    namespace: str,
    project: str,
    dataset: str,
    files: List[UploadFile] = File(...),  # Always expect files
    # Optional metadata
    batch_id: Optional[str] = Form(None),
    total_batches: Optional[int] = Form(None),
    current_batch: Optional[int] = Form(None)
):
    """
    Ingest uploaded files into a dataset.
    ALL path expansion and file reading happens client-side.
    """
```

### 2. CLI-Side Enhancement

The CLI needs to:

#### A. Path Expansion (Client-Side)
```go
func expandPaths(paths []string, recursive bool, pattern string) ([]string, error) {
    var allFiles []string
    
    for _, path := range paths {
        // Handle glob patterns
        if containsGlobPattern(path) {
            matches, _ := filepath.Glob(path)
            allFiles = append(allFiles, filterFiles(matches)...)
        }
        // Handle directories
        else if info, _ := os.Stat(path); info.IsDir() {
            files := walkDirectory(path, recursive, pattern)
            allFiles = append(allFiles, files...)
        }
        // Handle regular files
        else if isFile(path) {
            allFiles = append(allFiles, path)
        }
    }
    
    return allFiles, nil
}
```

#### B. Batch Processing with Streaming
```go
func ingestFilesInBatches(dataset string, files []string, batchSize int) error {
    totalBatches := (len(files) + batchSize - 1) / batchSize
    
    for i := 0; i < len(files); i += batchSize {
        end := min(i+batchSize, len(files))
        batch := files[i:end]
        currentBatch := (i / batchSize) + 1
        
        err := uploadBatch(dataset, batch, totalBatches, currentBatch)
        if err != nil {
            return err
        }
        
        // Progress indication
        fmt.Printf("Uploaded batch %d/%d\n", currentBatch, totalBatches)
    }
    
    return nil
}
```

#### C. Streaming for Large Files
```go
func uploadBatch(dataset string, filePaths []string, totalBatches, currentBatch int) error {
    // Create multipart writer
    body := &bytes.Buffer{}
    writer := multipart.NewWriter(body)
    
    // Add metadata
    writer.WriteField("total_batches", strconv.Itoa(totalBatches))
    writer.WriteField("current_batch", strconv.Itoa(currentBatch))
    
    // Add each file
    for _, filePath := range filePaths {
        file, err := os.Open(filePath)
        if err != nil {
            continue // Log and skip
        }
        defer file.Close()
        
        part, _ := writer.CreateFormFile("files", filepath.Base(filePath))
        io.Copy(part, file)
    }
    
    writer.Close()
    
    // Send request
    req, _ := http.NewRequest("POST", url, body)
    req.Header.Set("Content-Type", writer.FormDataContentType())
    
    // For very large uploads, we might want to use chunked transfer encoding
    // req.TransferEncoding = []string{"chunked"}
    
    return sendRequest(req)
}
```

### 3. Progressive Upload Strategy

For better UX and handling large datasets:

1. **Immediate feedback**: Start uploading as soon as first batch is ready
2. **Progress tracking**: Show upload progress for each batch
3. **Failure recovery**: Track which files failed and offer retry
4. **Memory efficiency**: Stream files instead of loading all into memory

### 4. Optimizations

#### A. Parallel Uploads (Optional)
```go
func uploadBatchesInParallel(batches [][]string, workers int) {
    sem := make(chan struct{}, workers)
    var wg sync.WaitGroup
    
    for i, batch := range batches {
        wg.Add(1)
        sem <- struct{}{}
        
        go func(batchNum int, files []string) {
            defer wg.Done()
            defer func() { <-sem }()
            
            uploadBatch(dataset, files, len(batches), batchNum)
        }(i+1, batch)
    }
    
    wg.Wait()
}
```

#### B. Compression (Optional)
For many small files, we could compress before sending:
```go
func compressAndUpload(files []string) {
    // Create tar.gz of files
    // Upload as single compressed file
    // Server decompresses and processes
}
```

## Migration Path

### Phase 1: Update Server (Backward Compatible)
1. Keep existing endpoint but deprecate path-based inputs
2. Add new simplified endpoint for file-only uploads
3. Server only processes uploaded files, never local paths

### Phase 2: Update CLI
1. Move ALL path expansion to client-side
2. Implement batching and streaming
3. Add progress indicators
4. Test with large datasets

### Phase 3: Remove Deprecated Code
1. Remove server-side path expansion
2. Remove path-based inputs from API
3. Clean up unused code

## Benefits of This Approach

1. **Docker Compatible**: Works regardless of where server is running
2. **More Secure**: Server never accesses local filesystem
3. **Better Progress Tracking**: Client knows exactly what's being uploaded
4. **More Efficient**: Can optimize batching based on file sizes
5. **Better Error Recovery**: Client can retry failed files
6. **Platform Agnostic**: Same code works for local, Docker, cloud deployments

## Implementation Priority

1. **Critical**: Refactor CLI to expand paths client-side
2. **Critical**: Update server to only accept file uploads
3. **Important**: Implement batching for large datasets
4. **Nice-to-have**: Add compression for many small files
5. **Nice-to-have**: Parallel uploads for faster processing

## Testing Strategy

1. Test with local server
2. Test with Dockerized server
3. Test with large datasets (1000+ files)
4. Test with mixed file sizes
5. Test failure recovery
6. Test progress reporting

## Estimated Changes

### Files to Modify
- `cli/cmd/datasets.go` - Major refactor (80% rewrite)
- `server/api/routers/datasets/datasets.py` - Simplification (50% reduction)
- Tests - Update to match new architecture

### New Functions Needed (CLI)
- `expandPathsLocally()` - Handle all glob/directory expansion
- `batchFiles()` - Group files into uploadable batches  
- `streamUpload()` - Stream large files efficiently
- `trackProgress()` - Show upload progress to user

### Functions to Remove (Server)
- `detect_and_expand_path()` - No longer needed
- Path-based processing in `smart_ingest()`
- Server-side glob handling

## Next Steps

1. Get approval on this approach
2. Create feature branch for refactor
3. Implement server changes first (backward compatible)
4. Implement CLI changes
5. Test thoroughly with Docker
6. Update documentation