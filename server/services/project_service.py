import os
from config import load_config, LlamaFarmConfig, save_config
from core.config import settings

class ProjectService:
  """
  Service for managing projects.
  """

  def _get_project_dir(self, project_id: str):
    if settings.lf_project_dir is None:
      return os.path.join(settings.lf_data_dir, "projects", project_id)
    else:
      return settings.lf_project_dir

  def load_config(self, project_id: str) -> LlamaFarmConfig:
    return load_config(self._get_project_dir(project_id))

  def save_config(self, project_id: str, config: dict) -> LlamaFarmConfig:
    return save_config(self._get_project_dir(project_id), config)
    