from pydantic import BaseModel

class ClassificationSchema(BaseModel):
    label: str | None = None
