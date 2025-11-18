from config.datamodel import Dataset
from pydantic import BaseModel

from services.dataset_service import DatasetWithFileDetails


class ListDatasetsResponse(BaseModel):
    total: int
    datasets: list[Dataset | DatasetWithFileDetails]
