# LlamaFarm Server

The FastAPI-based server component of LlamaFarm that provides REST APIs for project management, model inference, and data services.

## Features

- **Project Management**: Create, manage, and deploy AI projects
- **Model Integration**: Interface with various AI models and providers
- **Dataset Management**: Handle and process datasets for AI workflows
- **Configuration**: Type-safe configuration management with validation
- **Extensible**: Modular architecture for easy extension

## Development

To run the server in development mode:

```bash
cd server
uv sync
uv run uvicorn main:app --reload
```

## API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation.

## Requirements

- Python 3.12+
- FastAPI
- Pydantic
- Additional dependencies listed in `pyproject.toml`
