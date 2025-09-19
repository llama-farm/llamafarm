# LlamaFarm Deployment

This directory contains Docker Compose configurations for running the LlamaFarm application stack.

## Services

- **server**: FastAPI backend server (Python)
- **designer**: React frontend application (TypeScript/Vite)
- **runtime**: Python runtime service

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

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

### Development

```bash
# Run development environment with hot reload
docker-compose -f docker-compose.dev.yml up --build

# Run specific service
docker-compose -f docker-compose.dev.yml up server
```

Development services:

- Frontend: http://localhost:5173 (Vite dev server)
- Backend API: http://localhost:8000 (with auto-reload)

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
docker-compose build designer
docker-compose build runtime

# Force rebuild
docker-compose build --no-cache
```
