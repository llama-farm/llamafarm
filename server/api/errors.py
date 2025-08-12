class NotFoundError(Exception):
  def __init__(self, message: str | None = None):
    super().__init__(message or "Not found")

class NamespaceNotFoundError(NotFoundError):
  def __init__(self, namespace: str):
    self.namespace = namespace
    super().__init__(f"Namespace {namespace} not found")

class DatasetNotFoundError(NotFoundError):
  def __init__(self, dataset: str):
    self.dataset = dataset
    super().__init__(f"Dataset {dataset} not found")


class ProjectNotFoundError(NotFoundError):
  """Raised when a project doesn't exist."""

  def __init__(self, namespace: str, project_id: str):
    self.namespace = namespace
    self.project_id = project_id
    super().__init__(f"Project {namespace}/{project_id} not found")


class ProjectConfigError(Exception):
  """Raised when project exists but config is invalid or missing."""

  def __init__(self, namespace: str, project_id: str, message: str | None = None):
    self.namespace = namespace
    self.project_id = project_id
    super().__init__(message or f"Invalid or missing config for project {namespace}/{project_id}")