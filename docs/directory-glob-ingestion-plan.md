# Plan for Directory/Glob Pattern Ingestion Support

## Executive Summary
Implement server-side batch processing to enable both the LlamaFarm CLI and UI to ingest entire directories and use glob patterns for batch file processing, eliminating the need to specify each file individually.

## Current Architecture Analysis

### Current Flow
1. CLI (`datasets.go`) expands glob patterns locally using `filepath.Glob()`
2. Uploads files one-by-one via multipart POST to `/datasets/{dataset}/data`
3. Server processes each file individually through `IngestHandler`
4. Test script shows the pain - loops through each file manually (lines 253-260 in test_rag_comprehensive.sh)
5. UI has no bulk upload capability currently

### Key Findings
- RAG core already has `ingest_directory()` method in `ingest_handler.py:423-499`
- CLI currently only supports single file uploads
- No batch upload endpoint exists in the server
- Current approach has significant network overhead for large file sets
- UI cannot leverage bulk processing without server-side support

## Chosen Implementation: Server-Side Batch Processing

### Why Server-Side?
- **Unified Experience**: Both CLI and UI can use the same batch endpoints
- **Performance**: Server can process files in parallel more efficiently
- **Consistency**: Single implementation for batch processing logic
- **UI Compatibility**: Web UI can't access local file system directly for directory walking
- **Future-Proof**: Enables remote directory processing and cloud storage integration

### Architecture Overview
```
                    ┌─────────────────┐
                    │  Single Smart   │
                    │    Endpoint     │
                    │  /datasets/     │
                    │  {name}/ingest  │
                    └────────┬────────┘
                             │
                 ┌───────────┼───────────┐
                 ▼           ▼           ▼
           Auto-detects input type:
         ┌──────────┬──────────┬──────────┐
         │  Files   │Patterns  │Directory  │
         │ Upload   │ *.pdf    │ /docs/    │
         └──────────┴──────────┴──────────┘
                       │
                       ▼
                 IngestHandler
                       │
                       ▼
                 Vector Store

## Implementation Details

### 1. Server API Changes (`server/api/routers/datasets/datasets.py`)

#### Single Smart Endpoint Approach

```python
from typing import List, Optional, Union
import asyncio
from pathlib import Path
import glob
from concurrent.futures import ThreadPoolExecutor
from fastapi import Form

class SmartIngestRequest(BaseModel):
    """Universal request model for all ingestion types"""
    paths: Optional[List[str]] = None  # Can be files, dirs, or patterns
    recursive: bool = False
    pattern: Optional[str] = None  # Additional filter for directories
    batch_size: int = 10
    parallel: bool = True

class IngestItem(BaseModel):
    """Represents a single item to ingest"""
    type: str  # "file", "directory", "pattern", "url"
    value: str  # The actual path/pattern/url
    options: Dict[str, Any] = {}  # Type-specific options

class BatchIngestResponse(BaseModel):
    """Response model for batch operations"""
    total: int
    successful: int
    failed: int
    skipped: int
    results: List[Dict[str, Any]]
    processing_time: float
    detected_types: Dict[str, int]  # Count of each type processed

# Single unified endpoint that handles everything
@router.post("/{dataset}/ingest", response_model=BatchIngestResponse)
async def smart_ingest(
    namespace: str, 
    project: str, 
    dataset: str,
    # Accept either files OR JSON request body
    files: Optional[List[UploadFile]] = None,
    request_body: Optional[str] = Form(None),  # JSON string with paths/patterns
    # Direct parameters for simple cases
    paths: Optional[str] = Form(None),  # Comma-separated paths
    recursive: bool = Form(False),
    pattern: Optional[str] = Form(None),
    parallel: bool = Form(True)
):
    """
    Smart unified endpoint that automatically detects and handles:
    - Direct file uploads (multipart/form-data)
    - File paths (local files on server)
    - Directories (with optional recursion and filtering)
    - Glob patterns (*.pdf, **/*.md, etc.)
    - Mixed inputs (combination of above)
    - URLs (future: for remote file ingestion)
    """
    import time
    import json
    start_time = time.time()
    
    # Parse input based on what was provided
    items_to_process = []
    detected_types = {"files": 0, "directories": 0, "patterns": 0, "urls": 0}
    
    # Case 1: Direct file uploads from UI/CLI
    if files:
        detected_types["files"] = len(files)
        items_to_process.extend([
            IngestItem(type="upload", value=f, options={})
            for f in files
        ])
    
    # Case 2: JSON request with paths/patterns
    elif request_body:
        try:
            data = json.loads(request_body)
            if isinstance(data, dict):
                # Structured request
                request = SmartIngestRequest(**data)
                paths_to_analyze = request.paths or []
            else:
                # Simple array of paths
                paths_to_analyze = data
        except:
            # Fallback to treating as comma-separated paths
            paths_to_analyze = request_body.split(',')
    
    # Case 3: Form data with paths (comma-separated)
    elif paths:
        paths_to_analyze = [p.strip() for p in paths.split(',')]
    else:
        return BatchIngestResponse(
            total=0, successful=0, failed=0, skipped=0,
            results=[], processing_time=0, detected_types={}
        )
    
    # Analyze each path to determine its type
    if 'paths_to_analyze' in locals():
        for path_str in paths_to_analyze:
            item_type, expanded = await detect_and_expand_path(
                path_str, recursive, pattern
            )
            detected_types[item_type] += len(expanded) if isinstance(expanded, list) else 1
            
            if item_type == "pattern":
                # Glob pattern - expand on server
                items_to_process.extend([
                    IngestItem(type="file", value=f, options={"source": "pattern"})
                    for f in expanded
                ])
            elif item_type == "directory":
                # Directory - expand with filters
                items_to_process.extend([
                    IngestItem(type="file", value=f, options={"source": "directory"})
                    for f in expanded
                ])
            elif item_type == "url":
                # URL for remote ingestion (future feature)
                items_to_process.append(
                    IngestItem(type="url", value=path_str, options={})
                )
            else:
                # Regular file
                items_to_process.append(
                    IngestItem(type="file", value=path_str, options={})
                )
    
    # Process all items using the handler
    handler = get_ingest_handler(namespace, project, dataset)
    results = []
    
    if parallel and len(items_to_process) > 1:
        # Parallel processing for multiple items
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for item in items_to_process:
                future = executor.submit(
                    process_item, handler, item
                )
                futures.append(future)
            results = [future.result() for future in futures]
    else:
        # Sequential processing
        for item in items_to_process:
            result = await process_item(handler, item)
            results.append(result)
    
    # Aggregate results
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "error")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    
    return BatchIngestResponse(
        total=len(items_to_process),
        successful=successful,
        failed=failed,
        skipped=skipped,
        results=results,
        processing_time=time.time() - start_time,
        detected_types=detected_types
    )

# Helper function to detect path type and expand
async def detect_and_expand_path(
    path_str: str, 
    recursive: bool = False,
    filter_pattern: Optional[str] = None
) -> Tuple[str, List[str]]:
    """
    Detect the type of path and expand it to actual files.
    Returns: (type, list_of_files)
    """
    # Check if URL
    if path_str.startswith(('http://', 'https://', 'ftp://')):
        return ("url", [path_str])
    
    # Check for glob patterns
    if any(char in path_str for char in ['*', '?', '[', ']']):
        # It's a glob pattern
        matches = glob.glob(path_str, recursive=recursive)
        files = [f for f in matches if Path(f).is_file()]
        return ("pattern", files)
    
    # Check if path exists
    path = Path(path_str)
    if path.exists():
        if path.is_file():
            return ("file", [str(path)])
        elif path.is_dir():
            # Directory - expand based on options
            if recursive:
                pattern = f"**/{filter_pattern or '*'}"
            else:
                pattern = filter_pattern or "*"
            
            matches = list(path.glob(pattern))
            files = [str(f) for f in matches if f.is_file()]
            return ("directory", files)
    
    # Path doesn't exist - might be a pattern without wildcards
    # or a file that will be created
    return ("file", [path_str])

# Unified processor for any item type
async def process_item(handler, item: IngestItem):
    """Process a single item regardless of its type"""
    try:
        if item.type == "upload":
            # Direct upload - file is UploadFile object
            file = item.value
            content = await file.read()
            metadata = {
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(content)
            }
        elif item.type == "file":
            # Local file path
            path = Path(item.value)
            with open(path, 'rb') as f:
                content = f.read()
            metadata = {
                "filename": path.name,
                "filepath": str(path),
                "size": path.stat().st_size if path.exists() else 0,
                "content_type": guess_content_type(path)
            }
        elif item.type == "url":
            # Future: Download from URL
            # content = await download_from_url(item.value)
            # metadata = {...}
            return {"status": "error", "error": "URL ingestion not yet implemented"}
        else:
            return {"status": "error", "error": f"Unknown item type: {item.type}"}
        
        # Add source information from options
        metadata.update(item.options)
        
        # Process through handler
        result = handler.ingest_file(content, metadata)
        return result
        
    except Exception as e:
        logger.error(f"Error processing item {item.value}: {e}")
        return {
            "status": "error",
            "filename": Path(item.value).name if item.type == "file" else item.value,
            "error": str(e)
        }

# Helper functions
async def process_single_file(handler, file: UploadFile):
    """Process a single uploaded file"""
    try:
        content = await file.read()
        metadata = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content)
        }
        result = handler.ingest_file(content, metadata)
        return result
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}")
        return {
            "status": "error",
            "filename": file.filename,
            "error": str(e)
        }

def process_local_file(handler, file_path: str):
    """Process a local file from the server filesystem"""
    try:
        path = Path(file_path)
        with open(file_path, 'rb') as f:
            content = f.read()
        
        metadata = {
            "filename": path.name,
            "filepath": str(path),
            "size": path.stat().st_size,
            "content_type": guess_content_type(path)
        }
        
        result = handler.ingest_file(content, metadata)
        return result
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return {
            "status": "error",
            "filename": Path(file_path).name,
            "error": str(e)
        }
```

### 2. CLI Changes (`cli/cmd/datasets.go`)

The CLI will be simplified to use the single smart endpoint:

```go
// Enhanced datasetsIngestCmd using single smart endpoint
var datasetsIngestCmd = &cobra.Command{
    Use:   "ingest [dataset-name] [paths...]",
    Short: "Upload files, directories, or glob patterns to a dataset",
    Long: `Upload files to a dataset using various methods:
  - Single files: lf datasets ingest my-dataset file.pdf
  - Multiple files: lf datasets ingest my-dataset file1.pdf file2.txt
  - Glob patterns: lf datasets ingest my-dataset "*.pdf" "docs/*.md"
  - Directories: lf datasets ingest my-dataset /path/to/docs/
  - Mixed: lf datasets ingest my-dataset file.pdf /docs/ "*.txt"`,
    Args: cobra.MinimumNArgs(2),
    Run: ingestCommand,
}

// Flags
var (
    recursive bool
    pattern   string
    parallel  bool
    batchSize int
)

func init() {
    datasetsIngestCmd.Flags().BoolVarP(&recursive, "recursive", "r", false, 
        "Recursively process directories")
    datasetsIngestCmd.Flags().StringVarP(&pattern, "pattern", "p", "", 
        "Filter pattern for directory contents (e.g., '*.pdf')")
    datasetsIngestCmd.Flags().BoolVar(&parallel, "parallel", true,
        "Process files in parallel")
    datasetsIngestCmd.Flags().IntVar(&batchSize, "batch-size", 10, 
        "Number of files to upload in each batch")
}

func ingestCommand(cmd *cobra.Command, args []string) {
    datasetName := args[0]
    paths := args[1:]
    
    // Determine the best method based on input
    method := determineIngestMethod(paths)
    
    // Use the single smart endpoint
    url := fmt.Sprintf("/v1/projects/%s/%s/datasets/%s/ingest", 
        namespace, project, datasetName)
    
    switch method {
    case "files":
        // Upload actual files (expand globs locally if needed)
        uploadFiles(url, paths)
        
    case "paths":
        // Send paths to server for processing
        sendPathsToServer(url, paths)
        
    case "mixed":
        // Handle mixed input intelligently
        handleMixedInput(url, paths)
    }
}

func determineIngestMethod(paths []string) string {
    hasLocalFiles := false
    hasPatterns := false
    hasDirectories := false
    
    for _, path := range paths {
        // Check if it's a glob pattern
        if strings.ContainsAny(path, "*?[]") {
            hasPatterns = true
            continue
        }
        
        // Check if it exists locally
        if stat, err := os.Stat(path); err == nil {
            if stat.IsDir() {
                hasDirectories = true
            } else {
                hasLocalFiles = true
            }
        }
    }
    
    // If we have patterns or directories, let server handle them
    if hasPatterns || hasDirectories {
        return "paths"
    }
    
    // If all are local files, upload them directly
    if hasLocalFiles && !hasPatterns && !hasDirectories {
        return "files"
    }
    
    // Mixed case
    return "mixed"
}

func uploadFiles(url string, paths []string) error {
    // Expand any local globs first
    var filesToUpload []string
    for _, p := range paths {
        if strings.ContainsAny(p, "*?[]") {
            // Local glob expansion
            matches, _ := filepath.Glob(p)
            filesToUpload = append(filesToUpload, matches...)
        } else {
            filesToUpload = append(filesToUpload, p)
        }
    }
    
    // Prepare multipart upload
    var buf bytes.Buffer
    writer := multipart.NewWriter(&buf)
    
    // Add files
    for _, file := range filesToUpload {
        part, err := writer.CreateFormFile("files", filepath.Base(file))
        if err != nil {
            continue
        }
        
        content, err := os.ReadFile(file)
        if err != nil {
            fmt.Printf("Warning: Could not read %s: %v\n", file, err)
            continue
        }
        
        part.Write(content)
    }
    
    // Add options
    writer.WriteField("recursive", strconv.FormatBool(recursive))
    writer.WriteField("pattern", pattern)
    writer.WriteField("parallel", strconv.FormatBool(parallel))
    
    writer.Close()
    
    // Send request
    resp, err := http.Post(url, writer.FormDataContentType(), &buf)
    if err != nil {
        return err
    }
    
    // Handle response
    handleResponse(resp)
    return nil
}

func sendPathsToServer(url string, paths []string) error {
    // Send paths/patterns to server for processing
    request := map[string]interface{}{
        "paths":     paths,
        "recursive": recursive,
        "pattern":   pattern,
        "parallel":  parallel,
    }
    
    body, _ := json.Marshal(request)
    
    req, err := http.NewRequest("POST", url, bytes.NewReader(body))
    if err != nil {
        return err
    }
    
    // Use form encoding to pass JSON in request_body field
    form := neturl.Values{}
    form.Set("request_body", string(body))
    form.Set("recursive", strconv.FormatBool(recursive))
    form.Set("pattern", pattern)
    form.Set("parallel", strconv.FormatBool(parallel))
    
    req.Body = io.NopCloser(strings.NewReader(form.Encode()))
    req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
    
    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        return err
    }
    
    // Handle response with progress
    handleResponse(resp)
    return nil
}

func handleMixedInput(url string, paths []string) error {
    // Separate files from patterns/directories
    var localFiles []string
    var serverPaths []string
    
    for _, path := range paths {
        // Patterns and directories go to server
        if strings.ContainsAny(path, "*?[]") {
            serverPaths = append(serverPaths, path)
            continue
        }
        
        // Check if local file or directory
        if stat, err := os.Stat(path); err == nil {
            if stat.IsDir() {
                serverPaths = append(serverPaths, path)
            } else {
                localFiles = append(localFiles, path)
            }
        } else {
            // Doesn't exist locally - let server try
            serverPaths = append(serverPaths, path)
        }
    }
    
    // Create mixed request
    var buf bytes.Buffer
    writer := multipart.NewWriter(&buf)
    
    // Add local files
    for _, file := range localFiles {
        part, _ := writer.CreateFormFile("files", filepath.Base(file))
        content, _ := os.ReadFile(file)
        part.Write(content)
    }
    
    // Add server paths as JSON
    if len(serverPaths) > 0 {
        pathsJSON, _ := json.Marshal(serverPaths)
        writer.WriteField("paths", string(pathsJSON))
    }
    
    // Add options
    writer.WriteField("recursive", strconv.FormatBool(recursive))
    writer.WriteField("pattern", pattern)
    writer.WriteField("parallel", strconv.FormatBool(parallel))
    
    writer.Close()
    
    // Send request
    resp, err := http.Post(url, writer.FormDataContentType(), &buf)
    if err != nil {
        return err
    }
    
    handleResponse(resp)
    return nil
}

func handleResponse(resp *http.Response) {
    defer resp.Body.Close()
    
    var result BatchIngestResponse
    json.NewDecoder(resp.Body).Decode(&result)
    
    // Display results
    fmt.Printf("\n📊 Ingestion Results:\n")
    fmt.Printf("────────────────────────────────────\n")
    fmt.Printf("Total files: %d\n", result.Total)
    fmt.Printf("✅ Successful: %d\n", result.Successful)
    fmt.Printf("⏭️  Skipped: %d\n", result.Skipped)
    fmt.Printf("❌ Failed: %d\n", result.Failed)
    fmt.Printf("⏱️  Time: %.2fs\n", result.ProcessingTime)
    
    if len(result.DetectedTypes) > 0 {
        fmt.Printf("\n📁 Detected Types:\n")
        for typ, count := range result.DetectedTypes {
            if count > 0 {
                fmt.Printf("  - %s: %d\n", typ, count)
            }
        }
    }
    
    // Show any errors
    if result.Failed > 0 {
        fmt.Printf("\n❌ Errors:\n")
        for _, r := range result.Results {
            if r["status"] == "error" {
                fmt.Printf("  - %s: %s\n", r["filename"], r["error"])
            }
        }
    }
}
```

### 3. UI Integration

The web UI can leverage the same single endpoint:

```typescript
// UI TypeScript/JavaScript code - single smart endpoint
class DatasetUploader {
    private endpoint(datasetName: string): string {
        return `/v1/projects/${namespace}/${project}/datasets/${datasetName}/ingest`;
    }
    
    // Upload files directly from browser
    async uploadFiles(datasetName: string, files: File[]) {
        const formData = new FormData();
        
        // Add all files
        files.forEach(file => {
            formData.append('files', file);
        });
        
        // Add options
        formData.append('parallel', 'true');
        
        const response = await fetch(this.endpoint(datasetName), {
            method: 'POST',
            body: formData
        });
        
        return response.json();
    }
    
    // Send patterns/paths for server-side processing
    async ingestByPattern(datasetName: string, patterns: string[], options?: {
        recursive?: boolean;
        filter?: string;
    }) {
        const formData = new FormData();
        
        // Send as JSON in request_body field
        const request = {
            paths: patterns,
            recursive: options?.recursive || false,
            pattern: options?.filter
        };
        
        formData.append('request_body', JSON.stringify(request));
        formData.append('recursive', String(options?.recursive || false));
        formData.append('pattern', options?.filter || '');
        formData.append('parallel', 'true');
        
        const response = await fetch(this.endpoint(datasetName), {
            method: 'POST',
            body: formData
        });
        
        return response.json();
    }
    
    // Mixed upload - files + patterns
    async ingestMixed(datasetName: string, files: File[], patterns: string[]) {
        const formData = new FormData();
        
        // Add files
        files.forEach(file => {
            formData.append('files', file);
        });
        
        // Add patterns as paths
        if (patterns.length > 0) {
            formData.append('paths', patterns.join(','));
        }
        
        formData.append('parallel', 'true');
        
        const response = await fetch(this.endpoint(datasetName), {
            method: 'POST',
            body: formData
        });
        
        return response.json();
    }
}

// React component example
function DatasetIngestionForm({ datasetName }) {
    const [files, setFiles] = useState([]);
    const [pattern, setPattern] = useState('');
    const [recursive, setRecursive] = useState(false);
    
    const handleSubmit = async () => {
        const uploader = new DatasetUploader();
        
        if (files.length > 0 && pattern) {
            // Mixed: files + pattern
            const result = await uploader.ingestMixed(
                datasetName, 
                files, 
                [pattern]
            );
        } else if (files.length > 0) {
            // Just files
            const result = await uploader.uploadFiles(datasetName, files);
        } else if (pattern) {
            // Just pattern
            const result = await uploader.ingestByPattern(
                datasetName, 
                [pattern],
                { recursive }
            );
        }
    };
    
    return (
        <div>
            <FileDropZone onFiles={setFiles} />
            <input 
                placeholder="Or enter pattern: *.pdf, docs/*.md" 
                value={pattern}
                onChange={(e) => setPattern(e.target.value)}
            />
            <label>
                <input 
                    type="checkbox" 
                    checked={recursive}
                    onChange={(e) => setRecursive(e.target.checked)}
                />
                Recursive
            </label>
            <button onClick={handleSubmit}>Ingest</button>
        </div>
    );
}
```

## CLI Command Examples

After implementation, users will be able to use these commands:

### Basic Usage
```bash
# Single file
lf datasets ingest my-dataset document.pdf

# Multiple specific files
lf datasets ingest my-dataset file1.pdf file2.txt file3.md

# All files in a directory (non-recursive)
lf datasets ingest my-dataset /path/to/documents/

# All files in directory tree (recursive)
lf datasets ingest my-dataset /path/to/documents/ --recursive
```

### Glob Patterns
```bash
# All PDFs in current directory
lf datasets ingest my-dataset "*.pdf"

# All markdown files in docs directory
lf datasets ingest my-dataset "docs/*.md"

# All Python files recursively
lf datasets ingest my-dataset "**/*.py"

# Multiple patterns
lf datasets ingest my-dataset "*.pdf" "*.txt" "docs/*.md"

# Pattern with prefix
lf datasets ingest my-dataset "report-*.pdf"

# Complex patterns
lf datasets ingest my-dataset "data/2024-*/*.csv"
```

### Directory with Filters
```bash
# Directory with pattern filter
lf datasets ingest my-dataset /docs/ --pattern "*.pdf"

# Recursive with pattern
lf datasets ingest my-dataset /project/ --recursive --pattern "*.py"

# Multiple directories
lf datasets ingest my-dataset /docs/ /reports/ /archives/ --recursive
```

### Mixed Inputs
```bash
# Mix of files, patterns, and directories
lf datasets ingest my-dataset important.pdf /docs/ "reports/*.csv" --recursive

# Everything in current directory and subdirectories
lf datasets ingest my-dataset . --recursive

# All PDFs and Word docs recursively
lf datasets ingest my-dataset "**/*.pdf" "**/*.docx"
```

### Performance Options
```bash
# Control batch size for large uploads
lf datasets ingest my-dataset /large-dataset/ --batch-size 50

# Disable batch upload (use single file uploads)
lf datasets ingest my-dataset *.pdf --no-batch
```

## Implementation Phases

### Phase 1: Server API Implementation (Week 1)
- [ ] Implement `/batch-ingest` endpoint for multiple file uploads
- [ ] Implement `/ingest-patterns` endpoint for glob pattern processing
- [ ] Implement `/ingest-directory` endpoint for directory traversal
- [ ] Add parallel processing with ThreadPoolExecutor
- [ ] Implement proper error handling and aggregated responses
- [ ] Add request/response models with Pydantic

### Phase 2: CLI Enhancement (Week 1-2)
- [ ] Add intelligent path detection (file vs directory vs pattern)
- [ ] Implement strategy selector for optimal upload method
- [ ] Add `--recursive` flag for directory traversal
- [ ] Add `--pattern` flag for filtering
- [ ] Add `--batch-size` flag for chunking large uploads
- [ ] Implement progress reporting for batch operations
- [ ] Update help text and documentation

### Phase 3: UI Integration (Week 2)
- [ ] Add bulk file upload component
- [ ] Implement drag-and-drop for multiple files
- [ ] Add pattern input field for server-side pattern processing
- [ ] Display batch upload progress
- [ ] Show detailed results per file

### Phase 4: Testing & Optimization (Week 3)
- [ ] Comprehensive unit tests for all endpoints
- [ ] Integration tests with real file scenarios
- [ ] Performance benchmarking
- [ ] Memory optimization for large files
- [ ] Rate limiting implementation

## Testing Strategy

### Server Tests
```python
# Test batch upload endpoint
def test_batch_ingest():
    files = [
        ("files", ("test1.pdf", b"content1", "application/pdf")),
        ("files", ("test2.txt", b"content2", "text/plain"))
    ]
    response = client.post(f"/datasets/{dataset}/batch-ingest", files=files)
    assert response.json()["total"] == 2

# Test pattern endpoint
def test_pattern_ingest():
    request = {"patterns": ["*.pdf", "docs/*.md"]}
    response = client.post(f"/datasets/{dataset}/ingest-patterns", json=request)
    assert response.status_code == 200

# Test directory endpoint
def test_directory_ingest():
    request = {"paths": ["/test/docs"], "recursive": True, "pattern": "*.pdf"}
    response = client.post(f"/datasets/{dataset}/ingest-directory", json=request)
    assert response.json()["successful"] > 0
```

### CLI Tests
```go
// Test pattern detection
func TestDetermineUploadStrategy(t *testing.T) {
    assert.Equal(t, "pattern", determineUploadStrategy([]string{"*.pdf"}))
    assert.Equal(t, "directory", determineUploadStrategy([]string{"/docs/"}))
    assert.Equal(t, "single", determineUploadStrategy([]string{"file.txt"}))
    assert.Equal(t, "batch", determineUploadStrategy([]string{"f1.txt", "f2.pdf"}))
}

// Test batch upload
func TestBatchUpload(t *testing.T) {
    files := []string{"test1.pdf", "test2.txt", "test3.md"}
    err := uploadBatch("test-dataset", files)
    assert.NoError(t, err)
}
```

## Performance Benchmarks

### Expected Performance Improvements
| Scenario | Current (Sequential) | New (Batch/Parallel) | Improvement |
|----------|---------------------|----------------------|-------------|
| 10 files (1MB each) | 10 seconds | 2 seconds | 5x faster |
| 100 files (500KB each) | 100 seconds | 15 seconds | 6.7x faster |
| 1 directory (50 files) | 50 seconds | 5 seconds | 10x faster |
| Mixed (files + patterns) | N/A | 8 seconds | New capability |

### Memory Usage Optimization
- Stream large files (>10MB) instead of loading into memory
- Process in chunks of configurable batch size
- Use async/await for I/O operations
- Implement connection pooling for database operations

## Success Metrics

- **Performance**: 5-10x faster for bulk operations (>10 files)
- **Usability**: Single command for directory ingestion vs multiple commands
- **Reliability**: <1% failure rate with automatic retry
- **Compatibility**: 100% backward compatible
- **UI Feature Parity**: Web UI can perform same bulk operations as CLI

## Migration Guide

### For Existing Scripts
```bash
# Old way (loop through files)
for file in /docs/*.pdf; do
    lf datasets ingest my-dataset "$file"
done

# New way (single command)
lf datasets ingest my-dataset "/docs/*.pdf"
```

### For Test Scripts
```bash
# Old test_rag_comprehensive.sh approach
for pdf in $PDF_FILES; do
    echo "Uploading: $(basename $pdf)"
    ${LF_CMD} datasets ingest "${TEST_DATASET}" "$pdf"
done

# New approach
${LF_CMD} datasets ingest "${TEST_DATASET}" "${SAMPLE_DIR}/fda/*.pdf"
```

## API Documentation

### POST `/datasets/{dataset}/ingest` - Single Smart Endpoint

A unified endpoint that automatically detects and processes different input types.

#### Input Methods

The endpoint accepts multiple input methods and intelligently determines how to process them:

1. **Direct File Upload** (multipart/form-data)
   - `files`: Multiple file uploads
   - `recursive`: Boolean flag for directory recursion
   - `pattern`: Filter pattern for directories
   - `parallel`: Boolean for parallel processing

2. **Path/Pattern Specification** (form-encoded or JSON)
   - `request_body`: JSON string containing paths/patterns
   - `paths`: Comma-separated list of paths/patterns
   - `recursive`: Boolean flag
   - `pattern`: Additional filter pattern

3. **Mixed Mode** (files + paths)
   - Combine file uploads with path specifications

#### Request Examples

**Example 1: Upload Files**
```http
POST /datasets/my-dataset/ingest
Content-Type: multipart/form-data

files: [file1.pdf, file2.txt]
parallel: true
```

**Example 2: Process Patterns**
```http
POST /datasets/my-dataset/ingest
Content-Type: application/x-www-form-urlencoded

request_body: {"paths": ["*.pdf", "docs/**/*.md"], "recursive": true}
parallel: true
```

**Example 3: Process Directory**
```http
POST /datasets/my-dataset/ingest
Content-Type: application/x-www-form-urlencoded

paths: /project/docs
recursive: true
pattern: *.pdf
```

**Example 4: Mixed (Files + Patterns)**
```http
POST /datasets/my-dataset/ingest
Content-Type: multipart/form-data

files: [manual.pdf]
paths: docs/*.md,reports/*.csv
recursive: false
```

#### Response Format

```json
{
    "total": 25,
    "successful": 20,
    "failed": 2,
    "skipped": 3,
    "processing_time": 12.5,
    "detected_types": {
        "files": 10,
        "patterns": 8,
        "directories": 7,
        "urls": 0
    },
    "results": [
        {
            "filename": "document.pdf",
            "status": "success",
            "chunks": 15,
            "embedder": "OllamaEmbedder"
        },
        {
            "filename": "duplicate.txt",
            "status": "skipped",
            "reason": "duplicate"
        },
        {
            "filename": "corrupt.pdf",
            "status": "error",
            "error": "Failed to parse PDF"
        }
    ]
}
```

#### Smart Detection Logic

The endpoint automatically detects input types:

- **URLs**: Strings starting with `http://`, `https://`, `ftp://`
- **Glob Patterns**: Strings containing `*`, `?`, `[`, `]`
- **Directories**: Paths that exist and are directories
- **Files**: Everything else or paths that exist as files

#### Benefits of Single Endpoint

1. **Simplicity**: One endpoint to learn and use
2. **Flexibility**: Handles all input types automatically
3. **Consistency**: Same response format for all operations
4. **Future-Proof**: Easy to add new input types (e.g., S3 URLs)
5. **Backward Compatible**: Can still support old endpoints if needed

## Risk Mitigation

### Server Overload
- **Risk**: Too many files in single request
- **Mitigation**: 
  - Limit batch size to 100 files per request
  - Implement request timeout of 5 minutes
  - Queue large batches for background processing

### Memory Issues
- **Risk**: Large files causing OOM
- **Mitigation**:
  - Stream files >10MB
  - Process in configurable chunks
  - Implement memory monitoring

### Network Failures
- **Risk**: Connection drops during large uploads
- **Mitigation**:
  - Automatic retry with exponential backoff
  - Resume capability for interrupted uploads
  - Progress tracking on both client and server

### Backward Compatibility
- **Risk**: Breaking existing integrations
- **Mitigation**:
  - Keep existing single-file endpoint unchanged
  - Default to single-file mode when batch not available
  - Version API endpoints properly

## Conclusion

This server-side batch processing implementation provides a unified solution for both CLI and UI, enabling efficient bulk file ingestion with support for glob patterns and directory traversal. The approach maintains full backward compatibility while delivering 5-10x performance improvements for bulk operations.