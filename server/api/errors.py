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