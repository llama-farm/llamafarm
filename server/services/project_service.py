import os
import sys
from pathlib import Path

from pydantic import BaseModel

from api.errors import (
  NamespaceNotFoundError,
  ProjectConfigError,
  ProjectNotFoundError,
  SchemaNotFoundError,
)
from api.middleware.client_cwd import client_cwd
from core.logging import FastAPIStructLogger
from core.settings import settings

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
from config import (  # noqa: E402
  ConfigError,
  generate_base_config,
  load_config,
  save_config,
)
from config.datamodel import LlamaFarmConfig  # noqa: E402

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
    if not settings.lf_use_data_dir:
      # Prefer client CWD (localhost CLI) when available; fallback to server CWD
      return client_cwd.get() or os.getcwd()
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
  def create_project(
    cls,
    namespace: str,
    project_id: str,
    config_template: str | None = None,
  ) -> LlamaFarmConfig:
    """
    Create a new project.
    @param project_id: The ID of the project to create. (e.g. MyNamespace/MyProject)
    """
    project_dir = cls.get_project_dir(namespace, project_id)
    os.makedirs(project_dir, exist_ok=True)

    # Resolve config template path using shared helper
    schema_path = cls._resolve_template_path(config_template)

    # Generate config directly with correct name
    cfg_dict = generate_base_config(schema_path=str(schema_path), name=project_id)

    # Persist
    cfg_model = cls.save_config(namespace, project_id, LlamaFarmConfig(**cfg_dict))
    return cfg_model

  @classmethod
  def _resolve_template_path(cls, config_template: str | None) -> Path:
    """
    Resolve a config template name to a concrete filesystem path.

    The resolution order is:
    - If settings.lf_template_dir is set: {lf_template_dir}/{template}.yaml
    - Otherwise, look under repo 'config/templates/{template}.yaml'
    - Finally, fall back to 'rag/schemas/consolidated.yaml' as a generic schema
    """
    default_template = settings.lf_config_template
    template = config_template if config_template is not None else default_template
    schema_dir = settings.lf_template_dir

    if schema_dir is None:
      candidate_paths = [
        Path(__file__).parent.parent.parent
        / "config" / "templates" / f"{template}.yaml",
        Path(__file__).parent.parent.parent
        / "rag" / "schemas" / "consolidated.yaml",
      ]
    else:
      candidate_paths = [Path(schema_dir) / f"{template}.yaml"]

    for candidate in candidate_paths:
      if candidate.exists():
        return candidate

    raise SchemaNotFoundError(template, [str(p) for p in candidate_paths])

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
      except OSError as e:
        logger.warning(
          "Skipping project due to filesystem error",
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
      from config.helpers.loader import find_config_file
      config_file = find_config_file(project_dir)
      if not config_file:
        logger.warning(
          "Config file not found in project directory",
          namespace=namespace,
          project_id=project_id,
          path=project_dir,
        )
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
    except OSError as e:
      # Filesystem-related errors
      logger.error(
        "Filesystem error loading project config",
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
