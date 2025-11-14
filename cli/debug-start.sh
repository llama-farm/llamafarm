#!/bin/bash
# Build with debug symbols and disable optimizations
echo "Building with debug symbols..."
go build -gcflags="all=-N -l" -ldflags="-compressdwarf=false" -o lf main.go

# Start with Delve, waiting for debugger to attach
echo "Starting lf with Delve on port 2345..."
echo "Attach your debugger, then the process will start"
dlv exec ./lf --headless --listen=:2345 --api-version=2 --accept-multiclient --check-go-version=false -- "$@"
