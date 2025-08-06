"""
HuggingFace model repository integration.

This module provides integration with HuggingFace Hub for model management.
"""

import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import shutil

try:
    from huggingface_hub import (
        HfApi, 
        ModelCard,
        snapshot_download,
        create_repo,
        upload_folder,
        list_models,
        model_info
    )
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

from ...base import BaseModelRepository

logger = logging.getLogger(__name__)


class HuggingFaceRepository(BaseModelRepository):
    """HuggingFace Hub repository implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize HuggingFace repository."""
        super().__init__(config)
        
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError(
                "HuggingFace Hub not installed. Install with: pip install huggingface-hub"
            )
        
        # Get token from config or environment
        self.token = config.get("token") or os.getenv("HUGGINGFACE_TOKEN")
        self.api = HfApi(token=self.token)
        
        # Cache configuration
        self.cache_dir = config.get("cache_dir") or os.getenv("HF_HOME")
        self.default_revision = config.get("default_revision", "main")
    
    def search_models(self, query: str, **filters) -> List[Dict[str, Any]]:
        """Search for models in HuggingFace Hub."""
        try:
            # Build filter string
            filter_str = query
            
            # Add filters
            if "task" in filters:
                filter_str += f" task:{filters['task']}"
            if "library" in filters:
                filter_str += f" library:{filters['library']}"
            if "language" in filters:
                filter_str += f" language:{filters['language']}"
            if "license" in filters:
                filter_str += f" license:{filters['license']}"
            
            # Search models
            models = list(list_models(
                search=filter_str,
                limit=filters.get("limit", 100),
                sort=filters.get("sort", "downloads"),
                direction=filters.get("direction", -1)
            ))
            
            # Convert to dict format
            return [
                {
                    "id": model.modelId,
                    "author": model.author,
                    "downloads": model.downloads,
                    "likes": model.likes,
                    "tags": model.tags,
                    "pipeline_tag": model.pipeline_tag,
                    "library_name": getattr(model, "library_name", None),
                    "created_at": str(model.created_at) if model.created_at else None,
                    "last_modified": str(model.last_modified) if model.last_modified else None
                }
                for model in models
            ]
            
        except Exception as e:
            logger.error(f"Failed to search models: {e}")
            return []
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        try:
            info = model_info(model_id, token=self.token)
            
            return {
                "id": info.modelId,
                "author": info.author,
                "sha": info.sha,
                "downloads": info.downloads,
                "likes": info.likes,
                "tags": info.tags,
                "pipeline_tag": info.pipeline_tag,
                "library_name": getattr(info, "library_name", None),
                "config": info.config if hasattr(info, "config") else {},
                "card_data": info.card_data if hasattr(info, "card_data") else {},
                "siblings": [
                    {
                        "rfilename": s.rfilename,
                        "size": s.size,
                        "blob_id": s.blob_id
                    }
                    for s in (info.siblings or [])
                ],
                "spaces": info.spaces if hasattr(info, "spaces") else [],
                "created_at": str(info.created_at) if info.created_at else None,
                "last_modified": str(info.last_modified) if info.last_modified else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}
    
    def download_model(self, model_id: str, output_path: Path, 
                      revision: Optional[str] = None) -> bool:
        """Download a model from HuggingFace Hub."""
        try:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading model {model_id} to {output_path}")
            
            # Download model snapshot
            downloaded_path = snapshot_download(
                repo_id=model_id,
                revision=revision or self.default_revision,
                cache_dir=self.cache_dir,
                token=self.token,
                local_dir=str(output_path),
                local_dir_use_symlinks=False
            )
            
            logger.info(f"Successfully downloaded model to {downloaded_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return False
    
    def upload_model(self, model_path: Path, model_id: str, 
                    private: bool = False, **metadata) -> bool:
        """Upload a model to HuggingFace Hub."""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model path not found: {model_path}")
            
            logger.info(f"Uploading model from {model_path} to {model_id}")
            
            # Create repository if it doesn't exist
            try:
                create_repo(
                    repo_id=model_id,
                    token=self.token,
                    private=private,
                    repo_type="model",
                    exist_ok=True
                )
            except Exception as e:
                logger.warning(f"Repository might already exist: {e}")
            
            # Create model card if provided
            if "model_card" in metadata:
                card_path = model_path / "README.md"
                with open(card_path, "w") as f:
                    f.write(metadata["model_card"])
            
            # Upload the model
            upload_folder(
                folder_path=str(model_path),
                repo_id=model_id,
                token=self.token,
                commit_message=metadata.get("commit_message", "Upload model"),
                commit_description=metadata.get("commit_description", ""),
                create_pr=metadata.get("create_pr", False)
            )
            
            logger.info(f"Successfully uploaded model to {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload model: {e}")
            return False
    
    def list_user_models(self, username: Optional[str] = None) -> List[Dict[str, Any]]:
        """List models for a specific user."""
        try:
            # If no username provided, try to get current user
            if not username and self.token:
                user_info = self.api.whoami(token=self.token)
                username = user_info["name"]
            
            if not username:
                logger.error("No username provided and couldn't determine current user")
                return []
            
            # List user's models
            models = list(list_models(author=username, token=self.token))
            
            return [
                {
                    "id": model.modelId,
                    "downloads": model.downloads,
                    "likes": model.likes,
                    "tags": model.tags,
                    "pipeline_tag": model.pipeline_tag,
                    "private": model.private,
                    "created_at": str(model.created_at) if model.created_at else None,
                    "last_modified": str(model.last_modified) if model.last_modified else None
                }
                for model in models
            ]
            
        except Exception as e:
            logger.error(f"Failed to list user models: {e}")
            return []
    
    def create_model_card(self, model_info: Dict[str, Any]) -> str:
        """Create a model card for a model."""
        card = ModelCard.from_template(
            card_data=model_info.get("card_data", {}),
            template_str="""
---
{{ card_data }}
---

# {{ model_name | default("Model Name", true) }}

{{ model_description | default("Model description", true) }}

## Model Details

{{ model_details | default("", true) }}

## Training Data

{{ training_data | default("", true) }}

## Training Procedure

{{ training_procedure | default("", true) }}

## Evaluation

{{ evaluation | default("", true) }}

## Usage

```python
{{ usage_example | default("# Usage example", true) }}
```

## Limitations

{{ limitations | default("", true) }}

## Citation

{{ citation | default("", true) }}
"""
        )
        
        return str(card)
    
    def get_model_files(self, model_id: str) -> List[Dict[str, Any]]:
        """List files in a model repository."""
        try:
            files = self.api.list_repo_files(
                repo_id=model_id,
                repo_type="model",
                token=self.token
            )
            
            # Get file info for each file
            file_info = []
            for file in files:
                try:
                    info = self.api.get_file_info(
                        repo_id=model_id,
                        filename=file,
                        repo_type="model",
                        token=self.token
                    )
                    file_info.append({
                        "path": file,
                        "size": info.size,
                        "blob_id": info.blob_id,
                        "lfs": info.lfs if hasattr(info, "lfs") else None
                    })
                except:
                    file_info.append({"path": file})
            
            return file_info
            
        except Exception as e:
            logger.error(f"Failed to list model files: {e}")
            return []
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model from HuggingFace Hub."""
        try:
            self.api.delete_repo(
                repo_id=model_id,
                repo_type="model",
                token=self.token
            )
            logger.info(f"Successfully deleted model: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            return False