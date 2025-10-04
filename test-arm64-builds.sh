#!/bin/bash
set -euo pipefail

echo "Testing ARM64 Docker builds for LlamaFarm"
echo "=========================================="

# Services to test
SERVICES=("designer" "server" "rag" "runtime")

# Test each service
for SERVICE in "${SERVICES[@]}"; do
    echo ""
    echo "Testing $SERVICE ARM64 build..."
    echo "--------------------------------"
    
    case $SERVICE in
        "designer")
            CONTEXT="./designer"
            DOCKERFILE="./designer/Dockerfile"
            ;;
        "server")
            CONTEXT="./"
            DOCKERFILE="./server/Dockerfile"
            ;;
        "rag")
            CONTEXT="./"
            DOCKERFILE="./rag/Dockerfile"
            ;;
        "runtime")
            CONTEXT="./"
            DOCKERFILE="./runtime/Dockerfile"
            ;;
    esac
    
    # Build for ARM64 platform
    echo "Building $SERVICE for linux/arm64..."
    docker buildx build \
        --platform linux/arm64 \
        --context "$CONTEXT" \
        --file "$DOCKERFILE" \
        --tag "llamafarm-$SERVICE:arm64-test" \
        --build-arg GIT_SHA="$(git rev-parse HEAD)" \
        --load \
        .
    
    echo "✅ $SERVICE ARM64 build successful"
    
    # Inspect the image
    echo "Image details:"
    docker inspect "llamafarm-$SERVICE:arm64-test" | jq -r '.[0].Architecture'
    
    # Clean up
    docker rmi "llamafarm-$SERVICE:arm64-test" || true
done

echo ""
echo "🎉 All ARM64 builds completed successfully!"
echo ""
echo "Note: This test requires:"
echo "- Docker with buildx support"
echo "- ARM64 emulation (QEMU) or native ARM64 machine"
echo "- jq for JSON parsing"
echo ""
echo "GitHub Actions uses ubuntu-24.04-arm runners for efficient ARM64 builds"