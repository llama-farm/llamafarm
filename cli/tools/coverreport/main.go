package main

import (
    "bufio"
    "flag"
    "fmt"
    "os"
    "path/filepath"
    "os/exec"
    "sort"
    "strconv"
    "strings"
)

type fileStats struct {
    file              string
    lineCovered       int
    lineTotal         int
    stmtCovered       int
    stmtTotal         int
    funcCovered       int
    funcTotal         int
}

func main() {
    in := flag.String("in", "coverage.out", "input Go coverprofile file")
    format := flag.String("format", "text", "output format: text|markdown")
    trim := flag.String("trim", "llamafarm/", "trim path prefix in file column if present")
    flag.Parse()

    f, err := os.Open(*in)
    if err != nil {
        fmt.Fprintf(os.Stderr, "failed to open input: %v\n", err)
        os.Exit(1)
    }
    defer f.Close()

    // line hits per file
    lineHits := map[string]map[int]int{}
    // statement counts per file
    stmtTotal := map[string]int{}
    stmtCovered := map[string]int{}

    scanner := bufio.NewScanner(f)
    first := true
    for scanner.Scan() {
        line := scanner.Text()
        if first {
            first = false
            if strings.HasPrefix(line, "mode:") {
                continue
            }
        }
        if line == "" || strings.HasPrefix(line, "#") {
            continue
        }
        parts := strings.Fields(line)
        if len(parts) < 3 {
            continue
        }
        fileAndRange := parts[0]
        numStmtStr := parts[1]
        countStr := parts[2]
        frIdx := strings.LastIndex(fileAndRange, ":")
        if frIdx < 0 {
            continue
        }
        file := fileAndRange[:frIdx]
        rng := fileAndRange[frIdx+1:]
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
        nStmt, err3 := strconv.Atoi(numStmtStr)
        hits, err4 := strconv.Atoi(countStr)
        if err3 != nil || err4 != nil {
            continue
        }
        if _, ok := lineHits[file]; !ok {
            lineHits[file] = map[int]int{}
        }
        for ln := startLine; ln <= endLine; ln++ {
            if cur, ok := lineHits[file][ln]; !ok || hits > cur {
                lineHits[file][ln] = hits
            }
        }
        stmtTotal[file] += nStmt
        if hits > 0 {
            stmtCovered[file] += nStmt
        }
    }
    if err := scanner.Err(); err != nil {
        fmt.Fprintf(os.Stderr, "failed reading input: %v\n", err)
        os.Exit(1)
    }

    // Build stats
    files := make([]string, 0, len(lineHits))
    for f := range lineHits {
        files = append(files, f)
    }
    sort.Strings(files)

    // Parse function coverage using `go tool cover -func=coverage.out`
    funcTotals := map[string]int{}
    funcCovered := map[string]int{}
    if path, err := exec.LookPath("go"); err == nil {
        out, err := exec.Command(path, "tool", "cover", "-func="+*in).Output()
        if err == nil {
            scanner := bufio.NewScanner(strings.NewReader(string(out)))
            for scanner.Scan() {
                t := strings.TrimSpace(scanner.Text())
                if t == "" || strings.HasPrefix(t, "total:") { continue }
                // format: path/file.go:FuncName\tXX.X%
                // or: path/file.go:line: FuncName\tXX.X%
                // we'll split on first ':' to get file, then on last '\t'
                parts := strings.SplitN(t, "\t", 2)
                if len(parts) != 2 { continue }
                lhs, pctStr := parts[0], parts[1]
                // file is before the first ':' in lhs
                fp := lhs
                if i := strings.Index(lhs, ":"); i >= 0 {
                    fp = lhs[:i]
                }
                funcTotals[fp]++
                p := strings.TrimSuffix(pctStr, "%")
                p = strings.TrimSpace(p)
                if val, err := strconv.ParseFloat(p, 64); err == nil && val > 0 {
                    funcCovered[fp]++
                }
            }
        }
    }

    stats := make([]fileStats, 0, len(files))
    var totalLines, totalLinesCov, totalStmt, totalStmtCov, totalFunc, totalFuncCov int

    for _, fpath := range files {
        lines := lineHits[fpath]
        // get stable keys
        uniq := make([]int, 0, len(lines))
        for ln := range lines {
            uniq = append(uniq, ln)
        }
        sort.Ints(uniq)
        var ltot, lcov int
        for _, ln := range uniq {
            ltot++
            if lines[ln] > 0 {
                lcov++
            }
        }

        st := fileStats{
            file:        fpath,
            lineCovered: lcov,
            lineTotal:   ltot,
            stmtCovered: stmtCovered[fpath],
            stmtTotal:   stmtTotal[fpath],
            funcCovered: funcCovered[fpath],
            funcTotal:   funcTotals[fpath],
        }
        stats = append(stats, st)
        totalLines += st.lineTotal
        totalLinesCov += st.lineCovered
        totalStmt += st.stmtTotal
        totalStmtCov += st.stmtCovered
        totalFunc += st.funcTotal
        totalFuncCov += st.funcCovered
    }

    // Print
    if *format == "markdown" {
        fmt.Println("| File | Lines (cov/total) | Functions (cov/total) | Statements (cov/total) |")
        fmt.Println("|---|---:|---:|---:|")
        for _, st := range stats {
            display := st.file
            if idx := strings.Index(display, *trim); idx >= 0 {
                display = display[idx+len(*trim):]
            } else {
                display = filepath.ToSlash(display)
            }
            fmt.Printf("| %s | %d/%d (%.1f%%) | %d/%d (%.1f%%) | %d/%d (%.1f%%) |\n",
                display,
                st.lineCovered, st.lineTotal, pct(st.lineCovered, st.lineTotal),
                st.funcCovered, st.funcTotal, pct(st.funcCovered, st.funcTotal),
                st.stmtCovered, st.stmtTotal, pct(st.stmtCovered, st.stmtTotal))
        }
        fmt.Printf("| total | %d/%d (%.1f%%) | %d/%d (%.1f%%) | %d/%d (%.1f%%) |\n",
            totalLinesCov, totalLines, pct(totalLinesCov, totalLines),
            totalFuncCov, totalFunc, pct(totalFuncCov, totalFunc),
            totalStmtCov, totalStmt, pct(totalStmtCov, totalStmt))
    } else {
        // plain text
        fmt.Printf("%-60s %18s %20s %22s\n", "File", "Lines (cov/total)", "Functions (cov/total)", "Statements (cov/total)")
        for _, st := range stats {
            display := st.file
            if idx := strings.Index(display, *trim); idx >= 0 {
                display = display[idx+len(*trim):]
            } else {
                display = filepath.ToSlash(display)
            }
            fmt.Printf("%-60s %7d/%-7d (%5.1f%%) %7d/%-7d (%5.1f%%) %7d/%-7d (%5.1f%%)\n",
                display,
                st.lineCovered, st.lineTotal, pct(st.lineCovered, st.lineTotal),
                st.funcCovered, st.funcTotal, pct(st.funcCovered, st.funcTotal),
                st.stmtCovered, st.stmtTotal, pct(st.stmtCovered, st.stmtTotal))
        }
        fmt.Printf("%-60s %7d/%-7d (%5.1f%%) %7d/%-7d (%5.1f%%) %7d/%-7d (%5.1f%%)\n",
            "total",
            totalLinesCov, totalLines, pct(totalLinesCov, totalLines),
            totalFuncCov, totalFunc, pct(totalFuncCov, totalFunc),
            totalStmtCov, totalStmt, pct(totalStmtCov, totalStmt))
    }
}

func pct(c, t int) float64 {
    if t == 0 {
        return 0
    }
    return (float64(c) * 100.0) / float64(t)
}
