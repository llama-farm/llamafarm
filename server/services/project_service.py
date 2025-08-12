import os
from pathlib import Path

from config import ConfigError, load_config, save_config
from config.datamodel import LlamaFarmConfig
from config.helpers.generator import generate_base_config_from_schema
from pydantic import BaseModel

from api.errors import NamespaceNotFoundError, ProjectConfigError, ProjectNotFoundError
from core.logging import FastAPIStructLogger
from core.settings import settings

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
      project_path = os.path.join(namespace_dir, project_name)

      # Skip non-directories and hidden/system entries (e.g., .DS_Store)
      if not os.path.isdir(project_path) or project_name.startswith("."):
        logger.warning(
          "Skipping non-project entry",
          entry=project_name,
          path=project_path,
        )
        continue

      # Attempt to load project config; skip if invalid/missing
      try:
        cfg = load_config(
          directory=project_path,
          validate=False,
        )
      except ConfigError as e:
        logger.warning(
          "Skipping project without valid config",
          entry=project_name,
          error=str(e),
        )
        continue
      except Exception as e:
        logger.warning(
          "Skipping project due to unexpected error",
          entry=project_name,
          error=str(e),
        )
        continue

      projects.append(Project(
        namespace=namespace,
        name=project_name,
        config=cfg,
      ))
    return projects

  @classmethod
  def get_project(cls, namespace: str, project_id: str) -> Project:
    project_dir = cls.get_project_dir(namespace, project_id)
    # Validate project directory exists (and is a directory)
    if not os.path.isdir(project_dir):
      logger.info(
        "Project directory not found",
        namespace=namespace,
        project_id=project_id,
        path=project_dir,
      )
      raise ProjectNotFoundError(namespace, project_id)

    # Ensure a config file exists inside the directory
    try:
      config_file = None
      try:
        # find_config_file raises ConfigError when directory missing config
        from config.helpers.loader import find_config_file
        config_file = find_config_file(project_dir)
      except Exception:
        config_file = None

      if config_file is None:
        raise ProjectConfigError(
          namespace,
          project_id,
          message="No configuration file found in project directory",
        )

      # Attempt to load config (do not validate here; align with list_projects)
      cfg = load_config(directory=project_dir, validate=False)
    except ProjectConfigError:
      # bubble our structured error
      raise
    except ConfigError as e:
      # Config present but invalid/malformed
      logger.warning(
        "Invalid project config",
        namespace=namespace,
        project_id=project_id,
        error=str(e),
      )
      raise ProjectConfigError(
        namespace,
        project_id,
        message="Invalid project configuration",
      ) from e
    except Exception as e:
      # Unexpected loader errors
      logger.error(
        "Unexpected error loading project config",
        namespace=namespace,
        project_id=project_id,
        error=str(e),
      )
      raise

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

  @classmethod
  def update_project(
    cls,
    namespace: str,
    project_id: str,
    updated_config: LlamaFarmConfig,
  ) -> LlamaFarmConfig:
    """
    Full-replacement update of a project's configuration.
    - Ensures the project exists
    - Validates config via the datamodel when saving
    - Enforces immutable fields (namespace, name alignment)
    - Performs atomic save with backup via loader.save_config
    """
    # Ensure project exists and has a config file
    _ = cls.get_project(namespace, project_id)

    # Enforce immutable name: align to path project_id regardless of payload
    config_dict = updated_config.model_dump(mode="json", exclude_none=True)
    config_dict["name"] = project_id

    # Validate by reconstructing model
    cfg_model = LlamaFarmConfig(**config_dict)

    # Persist (will create a backup and preserve format)
    saved_cfg = cls.save_config(namespace, project_id, cfg_model)
    return saved_cfg
