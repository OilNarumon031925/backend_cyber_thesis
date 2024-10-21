from pydantic import BaseModel


class DetectionModel(BaseModel):
    text: str