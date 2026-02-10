# LlamaFarm Deployment

This directory contains Docker Compose configurations for running the LlamaFarm application stack.

## Services

- **server**: FastAPI backend server (Python)
- **rag**: RAG service with Celery workers (Python)
- **designer**: Designer web UI - React-based visual interface for managing projects, datasets, and configurations (TypeScript/Vite)

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
- Backend API: http://localhost:14345
- **Designer Web UI**: http://localhost:3123 (visual interface for project management)

### Development

```bash
# Run development environment with hot reload
docker-compose -f docker-compose.dev.yml up --build

# Run specific service
docker-compose -f docker-compose.dev.yml up server
```

Development services:
- ChromaDB Server: http://localhost:8001
- Backend API: http://localhost:14345 (with auto-reload)
- **Designer Web UI**: http://localhost:5173 (Vite dev server with hot reload)

## Environment Variables

You can customize the deployment by creating a `.env` file:

```env
# Frontend environment variables
VITE_APP_API_URL=http://localhost:14345
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
