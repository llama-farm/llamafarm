from pydantic import BaseModel


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: str | None = None


