#!/bin/bash

# Script to reliably start LlamaFarm documentation site

echo "🦙 Starting LlamaFarm Documentation Site..."

# Kill any existing docusaurus processes
pkill -f "docusaurus" 2>/dev/null || true
sleep 2

# Clear any cached data
echo "Clearing cache..."
npm run clear 2>/dev/null || true
rm -rf .docusaurus build node_modules/.cache 2>/dev/null || true

# Build the site
echo "Building site..."
npm run build

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Starting server on http://localhost:3001/"
    echo "Press Ctrl+C to stop the server"
    echo ""
    
    # Serve the built site
    npm run serve -- --port 3001
else
    echo "❌ Build failed. Please check the errors above."
    exit 1
fi