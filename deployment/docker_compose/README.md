# LlamaFarm Deployment

This directory contains Docker Compose configurations for running the LlamaFarm application stack.

## Services

- **chromadb-server**: ChromaDB vector database server
- **server**: FastAPI backend server (Python)
- **rag**: RAG service with Celery workers (Python)
- **designer**: React frontend application (TypeScript/Vite)

## ChromaDB Server

The `chromadb-server` service is required to prevent multi-process write conflicts that occur when multiple RAG workers try to write to the same ChromaDB persistent database simultaneously. This was causing "Failed to apply logs to the metadata segment" errors when processing multiple PDFs.

**How it works:**
- ChromaDB runs as a centralized server
- All RAG workers connect as HTTP clients
- Server handles concurrent writes safely
- No more persistent client conflicts

**Configuration:**
- Server runs on port 8001 (mapped from internal 8000)
- Data persists in Docker-managed volume `chromadb_data`
- RAG workers connect via `CHROMADB_HOST` and `CHROMADB_PORT` environment variables
- No manual directory creation required - Docker manages the volume automatically

## Quick Start

### Production

```bash
# Build and run all services
docker-compose up --build

# Run in background
docker-compose up -d --build

# Stop all services
docker-compose down
```

Services will be available at:
- ChromaDB Server: http://localhost:8001
- Backend API: http://localhost:8000
- Frontend: http://localhost:3123

### Development

```bash
# Run development environment with hot reload
docker-compose -f docker-compose.dev.yml up --build

# Run specific service
docker-compose -f docker-compose.dev.yml up server
```

Development services:
- ChromaDB Server: http://localhost:8001
- Backend API: http://localhost:8000 (with auto-reload)
- Frontend: http://localhost:5173 (Vite dev server)

## Environment Variables

You can customize the deployment by creating a `.env` file:

```env
# Frontend environment variables
VITE_APP_API_URL=http://localhost:8000
VITE_APP_ENV=production

# Backend environment variables  
PYTHONUNBUFFERED=1
```

## Scaling

Scale specific services:

```bash
# Scale backend to 3 instances
docker-compose up --scale server=3

# Scale with load balancer (requires additional nginx config)
docker-compose up --scale server=3 --scale designer=2
```

## Monitoring

Check service health:

```bash
# View logs
docker-compose logs -f

# Check service status
docker-compose ps

# Execute commands in running containers
docker-compose exec server bash
docker-compose exec designer sh
```

## Building Individual Services

```bash
# Build specific service
docker-compose build server
docker-compose build rag
docker-compose build designer

# Force rebuild
docker-compose build --no-cache
```

## Verifying ChromaDB Server Fix (Issue #279)

### Check ChromaDB Server Status
```bash
# Check if ChromaDB server is running
curl http://localhost:8001/api/v2/heartbeat

# View ChromaDB server logs
docker-compose logs chromadb-server

# Check collections
curl http://localhost:8001/api/v2/collections
```

### Test Multi-PDF Processing
```bash
# Start the full stack
docker-compose up -d

# Create a test dataset with multiple PDFs
lf datasets create -s pdf_ingest -b main_database test_multi_pdf
lf datasets upload test_multi_pdf ./path/to/pdf1.pdf ./path/to/pdf2.pdf ./path/to/pdf3.pdf

# Process all PDFs (this should now work without conflicts)
lf datasets process test_multi_pdf

# Check logs for HTTP client usage
docker-compose logs rag | grep "ChromaDB HTTP client"
```

### Expected Log Messages
When the fix is working correctly, you should see:
- `Using ChromaDB HTTP client connecting to chromadb-server:8000`
- No "Failed to apply logs to the metadata segment" errors
- All PDFs process successfully

### Troubleshooting

#### ChromaDB Server Issues
If ChromaDB server fails to start or reports as unhealthy:
```bash
# Check ChromaDB server logs
docker-compose logs chromadb-server

# Check service status
docker-compose ps chromadb-server

# Test connectivity manually
curl http://localhost:8001/api/v2/heartbeat
curl http://localhost:8001/api/v2/collections
```

#### Configuration Issues
If you still see persistent client warnings:
1. Check that `host` and `port` are set in your `llamafarm.yaml`
2. Verify `CHROMADB_HOST` and `CHROMADB_PORT` environment variables
3. Ensure ChromaDB server is healthy: `docker-compose ps chromadb-server`

#### Common ChromaDB Server Problems
- **Slow startup**: ChromaDB needs 60-90 seconds to fully initialize
- **Health check failures**: The server may be running but not responding to health checks
- **Port conflicts**: Ensure port 8001 is not in use by other services
- **API version mismatch**: ChromaDB 1.0.20+ uses v2 API endpoints (not v1)

#### CI/CD Integration
For automated testing and deployment:

```bash
# Use the provided wait script to ensure ChromaDB is ready
./scripts/wait-for-chromadb.sh

# Or run the comprehensive CI test
./scripts/ci-chromadb-test.sh
```

The healthcheck now uses a TCP connection test that doesn't require external tools:
```yaml
healthcheck:
  test: ["CMD-SHELL", "timeout 5 bash -c '</dev/tcp/localhost/8000' || exit 1"]
  interval: 10s
  timeout: 10s
  retries: 5
  start_period: 60s
```

#### ChromaDB API Endpoints (v2)
- Heartbeat: `http://localhost:8001/api/v2/heartbeat`
- Version: `http://localhost:8001/api/v2/version`
- Collections: `http://localhost:8001/api/v2/collections`

Note: v1 API endpoints are deprecated in ChromaDB 1.0.20+