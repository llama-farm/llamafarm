#!/bin/bash

# Docusaurus Quick Start Script
# This script ensures dependencies are installed and starts the development server

set -e

echo "🚀 Starting Docusaurus..."
echo "========================"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    yarn install
else
    echo "✓ Dependencies already installed"
fi

# Start the development server
echo "🌐 Starting development server at http://localhost:3000"
echo "Press Ctrl+C to stop the server"
yarn start