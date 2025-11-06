#!/bin/bash
# Update all model YAML files to add runtime, format, and download_command fields

cd "$(dirname "$0")/text-generation"

# Files to update (skip qwen3 and deepseek which are already updated)
FILES="codellama.yaml granite.yaml llama.yaml mistral.yaml phi.yaml qwen.yaml tinyllama.yaml"

for file in $FILES; do
    echo "Updating $file..."

    # Create a backup
    cp "$file" "$file.bak"

    # Use awk to add fields after specific provider lines
    awk '
    /^      universal:$/ {
        print
        getline
        print
        if ($0 !~ /runtime:/) {
            print "        runtime: universal"
        }
        if ($0 !~ /format:/) {
            print "        format: transformers"
        }
        next
    }
    /^      ollama:$/ {
        print
        getline
        print
        if ($0 !~ /runtime:/) {
            print "        runtime: ollama"
        }
        if ($0 !~ /format:/) {
            print "        format: gguf"
        }
        next
    }
    /^      lemonade:$/ {
        print
        getline
        print
        if ($0 !~ /runtime:/) {
            print "        runtime: lemonade"
        }
        if ($0 !~ /format:/) {
            print "        format: gguf"
        }
        next
    }
    # Convert pull_command to download_command
    /^        pull_command: "ollama pull / {
        sub(/pull_command/, "download_command")
        print
        next
    }
    # Add download_command for universal if "Auto-downloads" note exists but no download_command
    /^        notes: "Auto-downloads from HuggingFace/ {
        print
        getline
        if ($0 !~ /download_command/ && $0 ~ /^      [a-z]/) {
            print "        download_command: \"Auto-downloads from HuggingFace on first use\""
        }
        print
        next
    }
    { print }
    ' "$file.bak" > "$file"

    echo "  ✓ Updated $file"
done

echo ""
echo "✅ Done! Updated $(echo $FILES | wc -w | tr -d ' ') files"
echo "Backups saved as *.bak"
