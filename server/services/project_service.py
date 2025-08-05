import os
import sys
from pathlib import Path

from pydantic import BaseModel

from core.logging import FastAPIStructLogger
from core.settings import settings

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from config import LlamaFarmConfig, generate_base_config, load_config, save_config  # noqa: E402, I001

logger = FastAPIStructLogger()
class Project(BaseModel):
  namespace: str;
  name: str;
  config: LlamaFarmConfig;

class ProjectService:
  """
  Service for managing projects.
  """

  @classmethod
  def get_namespace_dir(cls, namespace: str):
    if settings.lf_project_dir is None:
      return os.path.join(settings.lf_data_dir, "projects", namespace)
    else:
      return None

  @classmethod
  def get_project_dir(cls, namespace: str, project_id: str):
    if settings.lf_project_dir is None:
      return os.path.join(settings.lf_data_dir, "projects", namespace, project_id)
    else:
      return settings.lf_project_dir

  @classmethod
  def create_project(cls, namespace: str, project_id: str) -> LlamaFarmConfig:
    """
    Create a new project.
    @param project_id: The ID of the project to create. (e.g. MyNamespace/MyProject)
    """
    project_dir = cls.get_project_dir(namespace, project_id)
    os.makedirs(project_dir, exist_ok=True)
    cfg = generate_base_config()
    cfg.update({"name": project_id})
    cls.save_config(namespace, project_id, cfg)
    return cfg

  @classmethod
  def list_projects(cls, namespace: str) -> list[Project]:
    if settings.lf_project_dir is not None:
      logger.info(f"Listing projects in {settings.lf_project_dir}")
      cfg = load_config(directory=settings.lf_project_dir, validate=False)
      return [Project(
        namespace=namespace,
        name=cfg.get("name", "unknown"),
        config=cfg,
      )]

    namespace_dir = cls.get_namespace_dir(namespace)
    logger.info(f"Listing projects in {namespace_dir}")
    projects = []
    for project_name in os.listdir(namespace_dir):
      cfg = load_config(directory=os.path.join(namespace_dir, project_name), validate=False)
      projects.append(Project(
        namespace=namespace,
        name=project_name,
        config=cfg,
      ))
    return projects

  @classmethod
  def get_project(cls, namespace: str, project_id: str) -> Project:
    project_dir = cls.get_project_dir(namespace, project_id)
    cfg = load_config(directory=project_dir, validate=False)
    return Project(
      namespace=namespace,
      name=project_id,
      config=cfg,
    )

  @classmethod
  def load_config(cls, namespace: str, project_id: str) -> LlamaFarmConfig:
    return load_config(cls.get_project_dir(namespace, project_id))

  @classmethod
  def save_config(cls, namespace: str, project_id: str, config: LlamaFarmConfig) -> LlamaFarmConfig:
    logger.debug(
      f"Saving config to {cls.get_project_dir(namespace, project_id)}",
      config=config,
    )
    return save_config(config, cls.get_project_dir(namespace, project_id))
