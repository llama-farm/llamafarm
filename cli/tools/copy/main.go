package main

import (
	"io"
	"log"
	"os"
)

// Copies the first file to the second file
func main() {
	if len(os.Args) != 3 {
		log.Fatal("Usage: go run main.go <src> <dst>")
	}

	srcPath := os.Args[1]
	dstPath := os.Args[2]

	src, err := os.Open(srcPath)
	if err != nil {
		log.Fatalf("Failed to open source file: %v", err)
	}
	defer src.Close()

	dst, err := os.Create(dstPath)
	if err != nil {
		log.Fatalf("Failed to create destination file: %v", err)
	}
	defer dst.Close()

	if _, err := io.Copy(dst, src); err != nil {
		log.Fatalf("Failed to copy file: %v", err)
	}
}
