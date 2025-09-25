# Simplified Refactoring Plan: Client-Side Path Expansion for Docker Compatibility

## Problem
Server running in Docker cannot access client's filesystem. Current implementation sends paths to server, which won't work.

## Solution
Move ALL path/glob expansion to CLI. Server only receives uploaded file contents.

## Changes Required

### 1. CLI Changes (`cli/cmd/datasets.go`)

#### A. Client-Side Path Expansion
```go
// Expand all paths/patterns locally BEFORE sending
func expandPathsLocally(paths []string, recursive bool, pattern string) ([]string, error) {
    var allFiles []string
    
    for _, path := range paths {
        // Check for glob patterns (*, ?, [)
        if strings.ContainsAny(path, "*?[") {
            matches, _ := filepath.Glob(path)
            for _, match := range matches {
                if info, _ := os.Stat(match); !info.IsDir() {
                    allFiles = append(allFiles, match)
                }
            }
        } else if info, err := os.Stat(path); err == nil {
            if info.IsDir() {
                // Walk directory
                if recursive {
                    filepath.Walk(path, func(p string, info os.FileInfo, err error) error {
                        if !info.IsDir() {
                            if pattern == "" || matched, _ := filepath.Match(pattern, filepath.Base(p)) {
                                allFiles = append(allFiles, p)
                            }
                        }
                        return nil
                    })
                } else {
                    // Just immediate files
                    files, _ := ioutil.ReadDir(path)
                    for _, f := range files {
                        if !f.IsDir() {
                            if pattern == "" || matched, _ := filepath.Match(pattern, f.Name()) {
                                allFiles = append(allFiles, filepath.Join(path, f.Name()))
                            }
                        }
                    }
                }
            } else {
                // Regular file
                allFiles = append(allFiles, path)
            }
        }
    }
    
    return allFiles, nil
}
```

#### B. Upload Files in Batches
```go
func uploadExpandedFiles(url string, files []string, batchSize int) {
    total := len(files)
    fmt.Printf("Uploading %d files...\n", total)
    
    for i := 0; i < len(files); i += batchSize {
        end := i + batchSize
        if end > len(files) {
            end = len(files)
        }
        
        batch := files[i:end]
        fmt.Printf("Uploading batch %d-%d of %d\n", i+1, end, total)
        
        // Upload this batch
        uploadBatchOfFiles(url, batch)
    }
}

func uploadBatchOfFiles(url string, filePaths []string) error {
    var buf bytes.Buffer
    writer := multipart.NewWriter(&buf)
    
    // Add each file to the request
    for _, path := range filePaths {
        file, err := os.Open(path)
        if err != nil {
            fmt.Printf("  ⚠️ Skipping %s: %v\n", path, err)
            continue
        }
        
        part, _ := writer.CreateFormFile("files", filepath.Base(path))
        io.Copy(part, file)
        file.Close()
    }
    
    writer.Close()
    
    // Send request
    req, _ := http.NewRequest("POST", url, &buf)
    req.Header.Set("Content-Type", writer.FormDataContentType())
    
    resp, _ := getHTTPClient().Do(req)
    // Handle response...
    
    return nil
}
```

#### C. Update Main Ingest Function
```go
func ingestDataset(dataset string, inputPaths []string) {
    // 1. Expand all paths locally
    allFiles, _ := expandPathsLocally(inputPaths, recursive, pattern)
    
    if len(allFiles) == 0 {
        fmt.Println("No files found matching the specified paths/patterns")
        return
    }
    
    fmt.Printf("Found %d files to upload\n", len(allFiles))
    
    // 2. Upload in batches
    uploadExpandedFiles(url, allFiles, batchSize)
}
```

### 2. Server Simplification (`server/api/routers/datasets/datasets.py`)

```python
@router.post("/{dataset}/ingest", response_model=BatchIngestResponse)
async def ingest_files(
    namespace: str,
    project: str,
    dataset: str,
    files: List[UploadFile] = File(...)  # Only accept uploaded files
):
    """
    Ingest uploaded files into dataset.
    All path expansion happens client-side.
    """
    results = []
    
    for file in files:
        # Process each uploaded file
        try:
            # Save file content
            content = await file.read()
            metadata = _save_file_to_data_store(
                namespace, project, dataset, content, file.filename
            )
            
            # Process into vector DB
            result = _process_into_vector_db(
                namespace, project, dataset,
                metadata,
                project_dir, project_config,
                strategy_name, database_name,
                file.filename
            )
            results.append(result)
        except Exception as e:
            results.append({
                "status": "error",
                "filename": file.filename,
                "error": str(e)
            })
    
    # Return batch response
    return BatchIngestResponse(
        total=len(files),
        successful=sum(1 for r in results if r["status"] == "success"),
        failed=sum(1 for r in results if r["status"] == "error"),
        skipped=sum(1 for r in results if r["status"] == "skipped"),
        results=results
    )
```

## Implementation Steps

1. **Update CLI** to expand paths locally
2. **Simplify server** to only handle uploads
3. **Test with Docker** to ensure it works

## What This Fixes

✅ Works with Docker (no server filesystem access)  
✅ Maintains all features (glob patterns, recursive, filtering)  
✅ Actually simpler than current implementation  
✅ More secure (server can't access arbitrary paths)

## What We're NOT Doing (keeping it simple)

❌ No streaming (just regular multipart upload)  
❌ No compression  
❌ No parallel uploads (can add later)  
❌ No progress bars (can add later)