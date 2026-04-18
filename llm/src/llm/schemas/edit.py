from pydantic import BaseModel

class EditSchema(BaseModel):
    edited: str | None = None
