from pydantic import BaseModel
from typing import List

class IngestSchema(BaseModel):
    numbers: List[int]
