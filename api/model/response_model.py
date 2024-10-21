from pydantic import BaseModel


class ReponseModel(BaseModel):
    message: str | None
    data: None
    status: float