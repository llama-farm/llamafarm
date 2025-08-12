package main

import (
    "bufio"
    "flag"
    "fmt"
    "os"
    "path/filepath"
    "strconv"
    "strings"
)

// Minimal converter from Go coverprofile to LCOV
// Strategy: for each segment, mark every line in the range as hit (1) if count>0 else 0.
// Keeps the max hit value seen per line.

func main() {
    in := flag.String("in", "coverage.out", "input Go coverprofile file")
    out := flag.String("out", "coverage.lcov", "output LCOV file")
    flag.Parse()

    f, err := os.Open(*in)
    if err != nil {
        fmt.Fprintf(os.Stderr, "failed to open input: %v\n", err)
        os.Exit(1)
    }
    defer f.Close()

    // map[filePath]map[line]hits
    fileLines := map[string]map[int]int{}

    scanner := bufio.NewScanner(f)
    first := true
    for scanner.Scan() {
        line := scanner.Text()
        if first {
            first = false
            // skip mode: line
            if strings.HasPrefix(line, "mode:") {
                continue
            }
        }
        if line == "" || strings.HasPrefix(line, "#") {
            continue
        }
        // Format: filename:startLine.startCol,endLine.endCol numStatements count
        // Example: path/file.go:10.2,12.3 1 1
        parts := strings.Fields(line)
        if len(parts) < 3 {
            continue
        }
        fileAndRange := parts[0]
        counts := parts[len(parts)-1]
        // split file and range
        frIdx := strings.LastIndex(fileAndRange, ":")
        if frIdx < 0 {
            continue
        }
        file := fileAndRange[:frIdx]
        rng := fileAndRange[frIdx+1:]
        // range like: 10.2,12.3
        seg := strings.Split(rng, ",")
        if len(seg) != 2 {
            continue
        }
        start := strings.Split(seg[0], ".")
        end := strings.Split(seg[1], ".")
        if len(start) < 1 || len(end) < 1 {
            continue
        }
        startLine, err1 := strconv.Atoi(start[0])
        endLine, err2 := strconv.Atoi(end[0])
        if err1 != nil || err2 != nil {
            continue
        }
        // hit count
        hit, err := strconv.Atoi(counts)
        if err != nil {
            continue
        }
        if _, ok := fileLines[file]; !ok {
            fileLines[file] = map[int]int{}
        }
        for ln := startLine; ln <= endLine; ln++ {
            // store max hits
            if hitVal, ok := fileLines[file][ln]; !ok || hit > hitVal {
                fileLines[file][ln] = hit
            }
        }
    }
    if err := scanner.Err(); err != nil {
        fmt.Fprintf(os.Stderr, "failed reading input: %v\n", err)
        os.Exit(1)
    }

    // write LCOV
    if err := os.MkdirAll(filepath.Dir(*out), 0o755); err != nil && filepath.Dir(*out) != "." {
        fmt.Fprintf(os.Stderr, "failed to create output dir: %v\n", err)
        os.Exit(1)
    }
    w, err := os.Create(*out)
    if err != nil {
        fmt.Fprintf(os.Stderr, "failed to create output: %v\n", err)
        os.Exit(1)
    }
    defer w.Close()

    for file, lines := range fileLines {
        fmt.Fprintf(w, "TN:\n")
        fmt.Fprintf(w, "SF:%s\n", file)
        for ln, hits := range lines {
            // Normalize hits as 0/1; LCOV allows integers
            hitVal := 0
            if hits > 0 {
                hitVal = 1
            }
            fmt.Fprintf(w, "DA:%d,%d\n", ln, hitVal)
        }
        fmt.Fprintf(w, "end_of_record\n")
    }
}
