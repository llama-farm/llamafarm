import os
import sys
from pathlib import Path

from pydantic import BaseModel

from api.errors import NamespaceNotFoundError
from core.logging import FastAPIStructLogger
from core.settings import settings

repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
  sys.path.insert(0, str(repo_root))

from config import load_config, save_config  # noqa: E402
from config.datamodel import LlamaFarmConfig  # noqa: E402
from config.helpers.generator import generate_base_config_from_schema  # noqa: E402

logger = FastAPIStructLogger()

class Project(BaseModel):
  namespace: str
  name: str
  config: LlamaFarmConfig

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
      return os.path.join(
        settings.lf_data_dir,
        "projects",
        namespace,
        project_id,
      )
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

    # Determine schema path from settings; support template selection
    schema_template = getattr(settings, "lf_schema_template", "default")
    schema_dir = getattr(settings, "lf_schema_dir", None)

    if schema_dir is None:
      # default to repo rag/schemas or config/schemas
      candidate_paths = [
        Path(__file__).parent.parent.parent
        / "config"
        / "schemas"
        / f"{schema_template}.yaml",
        Path(__file__).parent.parent.parent / "rag" / "schemas" / "consolidated.yaml",
      ]
    else:
      candidate_paths = [Path(schema_dir) / f"{schema_template}.yaml"]

    schema_path: Path | None = None
    for p in candidate_paths:
      if p.exists():
        schema_path = p
        break

    if schema_path is None:
      raise FileNotFoundError(
        f"No schema file found for template '{schema_template}'. "
        f"Searched: {', '.join(str(p) for p in candidate_paths)}"
      )

    cfg_dict = generate_base_config_from_schema(str(schema_path))
    # Ensure the name matches requested project id
    cfg_dict.update({"name": project_id})

    # Persist
    cfg_model = cls.save_config(namespace, project_id, LlamaFarmConfig(**cfg_dict))
    return cfg_model

  @classmethod
  def list_projects(cls, namespace: str) -> list[Project]:
    if settings.lf_project_dir is not None:
      logger.info(f"Listing projects in {settings.lf_project_dir}")
      cfg = load_config(directory=settings.lf_project_dir, validate=False)
      return [Project(
        namespace=namespace,
        name=cfg.name,
        config=cfg,
      )]

    namespace_dir = cls.get_namespace_dir(namespace)
    logger.info(f"Listing projects in {namespace_dir}")

    dirs: list[str]
    try:
      dirs = os.listdir(namespace_dir)
    except FileNotFoundError as e:
      raise NamespaceNotFoundError(namespace) from e

    projects = []
    for project_name in dirs:
      cfg = load_config(
        directory=os.path.join(namespace_dir, project_name),
        validate=False,
      )
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
  def save_config(
    cls,
    namespace: str,
    project_id: str,
    config: LlamaFarmConfig,
  ) -> LlamaFarmConfig:
    file_path, cfg = save_config(config, cls.get_project_dir(namespace, project_id))
    logger.debug("Saved project config", config=config, file_path=file_path)
    return cfg
