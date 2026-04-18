from pydantic import BaseModel

class GeneralAnswerSchema(BaseModel):
    answer: str | None = None
