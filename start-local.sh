#!/bin/bash

# Script to start both server and RAG worker locally for development

echo "Starting LlamaFarm services locally..."

# Kill any existing processes
echo "Stopping any existing services..."
pkill -f "nx start server" 2>/dev/null
pkill -f "nx start rag" 2>/dev/null
pkill -f "celery.*LlamaFarm" 2>/dev/null
sleep 2

# Start RAG worker in background
echo "Starting RAG worker..."
nx start rag &
RAG_PID=$!

# Setup cleanup trap before starting foreground process
trap "kill $RAG_PID 2>/dev/null; pkill -f 'celery.*LlamaFarm' 2>/dev/null" EXIT INT TERM

# Give RAG worker time to start
sleep 3

# Start server (foreground)
echo "Starting server..."
nx start server