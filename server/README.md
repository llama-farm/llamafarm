# 🦙 LlamaFarm Server

The FastAPI-based server component of LlamaFarm that provides REST APIs for project management, model inference, and data services.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Commands](#cli-commands)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Development](#development)
- [Docker Support](#docker-support)
- [Monitoring & Logging](#monitoring--logging)
- [Troubleshooting](#troubleshooting)

## ✨ Features

- **🚀 Project Management**: Create, manage, and deploy AI projects with full lifecycle support
- **🤖 Model Integration**: Interface with various AI models and providers (OpenAI, Anthropic, local models)
- **📊 Dataset Management**: Handle and process datasets for AI workflows
- **⚙️ Configuration**: Type-safe configuration management with Pydantic validation
- **🔌 Extensible**: Modular architecture for easy extension
- **📝 Structured Logging**: JSON-formatted logs with correlation IDs
- **🛡️ Error Handling**: Comprehensive error middleware with detailed responses
- **📈 Monitoring**: Built-in health checks and metrics endpoints
- **🔄 Hot Reload**: Development mode with automatic code reloading

## 🏗️ Architecture

```
LlamaFarm Server
├── API Layer (FastAPI)
│   ├── Routers (Projects, Datasets, Inference)
│   ├── Middleware (Error Handling, Logging, Correlation)
│   └── Endpoints (RESTful APIs)
├── Service Layer
│   ├── Data Service
│   ├── Dataset Service
│   └── Project Service
├── Core Layer
│   ├── Settings (Configuration)
│   ├── Logging (Structured Logs)
│   └── Version Management
└── Storage Layer
    └── Local File System
```

## 📦 Installation

### Prerequisites

- Python 3.12+
- UV package manager (`pip install uv`)
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/llamafarm.git
cd llamafarm/server

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync

# Install development dependencies (includes testing tools)
uv sync --dev
```

## 🚀 Quick Start

### Basic Server Launch

```bash
# Start the server with default settings
uv run uvicorn main:app --reload

# Server will be available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Production Server Launch

```bash
# Production mode with specific host and port
uv run uvicorn main:app --host 0.0.0.0 --port 8080 --workers 4

# With environment variables
export LOG_LEVEL=INFO
export LOG_JSON_FORMAT=true
export LF_DATA_DIR=/var/llamafarm/data
uv run uvicorn main:app --host 0.0.0.0 --port 8080
```

## 🖥️ CLI Commands

### Server Management

```bash
# Development server with auto-reload
uv run uvicorn main:app --reload

# Production server
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# With custom settings
uv run uvicorn main:app --env-file .env.production

# Debug mode with verbose logging
LOG_LEVEL=DEBUG uv run uvicorn main:app --reload --log-level debug
```

### Testing Commands

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=. --cov-report=html

# Run specific test file
uv run pytest tests/test_data_service.py

# Run specific test
uv run pytest tests/test_data_service.py::test_specific_function

# Run tests with verbose output
uv run pytest -v

# Run tests with print statements visible
uv run pytest -s

# Run tests in parallel (requires pytest-xdist)
uv run pytest -n auto
```

### System Tests

```bash
# Run integration tests
uv run pytest tests/ -m integration

# Run end-to-end tests
uv run pytest tests/ -m e2e

# Run performance tests
uv run pytest tests/ -m performance

# Run all system tests
uv run pytest tests/ -m "integration or e2e or performance"

# Run tests with specific markers
uv run pytest -v -m "not slow"
```

### Code Quality

```bash
# Format code with ruff
uv run ruff format .

# Lint code
uv run ruff check .

# Auto-fix linting issues
uv run ruff check --fix .

# Type checking (if mypy is configured)
uv run mypy .

# Run pre-commit hooks
uv run pre-commit run --all-files
```

### Database & Migrations (if applicable)

```bash
# Initialize database
uv run python scripts/init_db.py

# Run migrations
uv run alembic upgrade head

# Create new migration
uv run alembic revision --autogenerate -m "Description"

# Rollback migration
uv run alembic downgrade -1
```

## 📚 API Documentation

### Interactive Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### API Endpoints

#### Core Endpoints

```bash
# Health check
curl http://localhost:8000/

# Server info
curl http://localhost:8000/info
```

#### Projects API

```bash
# List all projects
curl http://localhost:8000/v1/projects

# Get specific project
curl http://localhost:8000/v1/projects/{project_id}

# Create new project
curl -X POST http://localhost:8000/v1/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "My Project", "description": "Test project"}'

# Update project
curl -X PUT http://localhost:8000/v1/projects/{project_id} \
  -H "Content-Type: application/json" \
  -d '{"name": "Updated Name"}'

# Delete project
curl -X DELETE http://localhost:8000/v1/projects/{project_id}
```

#### Datasets API

```bash
# List datasets
curl http://localhost:8000/v1/datasets

# Upload dataset
curl -X POST http://localhost:8000/v1/datasets/upload \
  -F "file=@data.csv"

# Get dataset info
curl http://localhost:8000/v1/datasets/{dataset_id}

# Process dataset
curl -X POST http://localhost:8000/v1/datasets/{dataset_id}/process
```

#### Inference API

```bash
# Run inference
curl -X POST http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-3.5-turbo", "prompt": "Hello, world!"}'

# Batch inference
curl -X POST http://localhost:8000/v1/inference/batch \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-3.5-turbo", "prompts": ["Hello", "World"]}'
```

## 🧪 Testing

### Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── test_data_service.py     # Data service tests
├── test_dataset_service.py  # Dataset service tests
├── integration/             # Integration tests
├── e2e/                    # End-to-end tests
└── performance/            # Performance tests
```

### Running Tests

```bash
# Run all tests with coverage
uv run pytest --cov=. --cov-report=term-missing

# Run tests and generate HTML coverage report
uv run pytest --cov=. --cov-report=html
# Open htmlcov/index.html in browser

# Run tests with different verbosity levels
uv run pytest -q    # Quiet
uv run pytest       # Normal
uv run pytest -v    # Verbose
uv run pytest -vv   # Very verbose

# Run failed tests from last run
uv run pytest --lf

# Run tests matching a pattern
uv run pytest -k "dataset"

# Run tests with debugging
uv run pytest --pdb  # Drop into debugger on failure
```

### Writing Tests

```python
# Example test file: tests/test_example.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, World!"}

@pytest.mark.asyncio
async def test_async_operation():
    """Test async operations."""
    # Your async test here
    pass

@pytest.mark.integration
def test_integration():
    """Integration test example."""
    # Integration test code
    pass
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the server directory:

```bash
# Logging
LOG_LEVEL=INFO              # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_JSON_FORMAT=true        # Enable JSON formatted logs

# Data Storage
LF_DATA_DIR=/path/to/data   # Data directory path

# API Keys (if needed)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Server Settings
HOST=0.0.0.0
PORT=8000
WORKERS=4
RELOAD=false

# Database (if applicable)
DATABASE_URL=postgresql://user:pass@localhost/dbname

# Redis (if applicable)
REDIS_URL=redis://localhost:6379/0
```

### Configuration File

The server uses `llamafarm.yaml` for configuration:

```yaml
# llamafarm.yaml
version: "1.0.0"
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4

logging:
  level: "INFO"
  format: "json"

data:
  directory: "./data"
  max_upload_size: 104857600  # 100MB

models:
  default_provider: "openai"
  timeout: 60
  max_retries: 3
```

## 📁 Project Structure

```
server/
├── api/                    # API layer
│   ├── __init__.py
│   ├── main.py            # FastAPI app setup
│   ├── errors.py          # Error definitions
│   ├── middleware/        # Custom middleware
│   │   ├── errors.py      # Error handling
│   │   └── structlog.py   # Structured logging
│   └── routers/           # API routers
│       ├── projects/      # Project endpoints
│       ├── datasets/      # Dataset endpoints
│       └── inference/     # Inference endpoints
├── core/                  # Core functionality
│   ├── __init__.py
│   ├── settings.py        # Configuration settings
│   ├── logging.py         # Logging setup
│   └── version.py         # Version info
├── services/              # Business logic
│   ├── __init__.py
│   ├── data_service.py    # Data operations
│   ├── dataset_service.py # Dataset operations
│   └── project_service.py # Project operations
├── tests/                 # Test suite
│   ├── __init__.py
│   ├── conftest.py        # Test fixtures
│   └── test_*.py          # Test files
├── scripts/               # Utility scripts
│   └── parse_config.py
├── main.py               # Application entry point
├── pyproject.toml        # Project dependencies
├── Dockerfile            # Container definition
├── .env.example          # Environment variables template
└── README.md            # This file
```

## 💻 Development

### Development Workflow

1. **Set up development environment**:
```bash
# Install development dependencies
uv sync --dev

# Set up pre-commit hooks
uv run pre-commit install
```

2. **Make changes and test**:
```bash
# Run server in development mode
uv run uvicorn main:app --reload

# Run tests as you develop
uv run pytest tests/ -v --tb=short

# Check code quality
uv run ruff check .
uv run ruff format .
```

3. **Commit changes**:
```bash
# Pre-commit hooks will run automatically
git add .
git commit -m "feat: Add new feature"
```

### Adding New Endpoints

1. Create a new router in `api/routers/`:
```python
# api/routers/example/__init__.py
from fastapi import APIRouter

router = APIRouter(prefix="/example", tags=["example"])

@router.get("/")
async def list_examples():
    return {"examples": []}

@router.post("/")
async def create_example(data: dict):
    return {"created": data}
```

2. Register the router in `api/main.py`:
```python
from api.routers.example import router as example_router

app.include_router(example_router, prefix=API_PREFIX)
```

### Adding New Services

1. Create service in `services/`:
```python
# services/example_service.py
class ExampleService:
    def __init__(self):
        pass
    
    async def process_data(self, data):
        # Business logic here
        return processed_data
```

2. Use in routers:
```python
from services.example_service import ExampleService

service = ExampleService()

@router.post("/process")
async def process(data: dict):
    result = await service.process_data(data)
    return result
```

## 🐳 Docker Support

### Build and Run with Docker

```bash
# Build Docker image
docker build -t llamafarm-server .

# Run container
docker run -p 8000:8000 -e LOG_LEVEL=INFO llamafarm-server

# Run with volume mount for data persistence
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e LF_DATA_DIR=/app/data \
  llamafarm-server

# Run with docker-compose
docker-compose up -d
```

### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
      - LOG_JSON_FORMAT=true
      - LF_DATA_DIR=/app/data
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

## 📊 Monitoring & Logging

### Structured Logging

The server uses structured logging with correlation IDs:

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "correlation_id": "abc-123-def",
  "event": "request_completed",
  "path": "/v1/projects",
  "method": "GET",
  "status_code": 200,
  "duration_ms": 45
}
```

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/

# Detailed health check (if implemented)
curl http://localhost:8000/health

# Readiness check
curl http://localhost:8000/ready

# Liveness check
curl http://localhost:8000/alive
```

### Metrics (if implemented)

```bash
# Prometheus metrics
curl http://localhost:8000/metrics
```

## 🔧 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
uv run uvicorn main:app --port 8001
```

#### Module Import Errors
```bash
# Ensure you're in the virtual environment
source .venv/bin/activate

# Reinstall dependencies
uv sync --force-reinstall
```

#### Database Connection Issues
```bash
# Check database is running
docker ps | grep postgres

# Test connection
psql $DATABASE_URL -c "SELECT 1"
```

#### Permission Errors
```bash
# Fix data directory permissions
chmod -R 755 ./data
chown -R $(whoami) ./data
```

### Debug Mode

```bash
# Enable debug logging
LOG_LEVEL=DEBUG uv run uvicorn main:app --reload --log-level debug

# Enable Python debugging
PYTHONBREAKPOINT=ipdb.set_trace uv run uvicorn main:app --reload

# Use debugger in code
import pdb; pdb.set_trace()
```

### Performance Profiling

```bash
# Profile with cProfile
python -m cProfile -o profile.stats main.py

# Analyze profile
python -m pstats profile.stats

# Memory profiling (requires memory-profiler)
python -m memory_profiler main.py
```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Structlog Documentation](https://www.structlog.org/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- Pydantic for data validation
- The Python community for amazing tools and libraries