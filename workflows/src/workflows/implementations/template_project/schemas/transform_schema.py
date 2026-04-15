from pydantic import BaseModel
from typing import List

class TransformSchema(BaseModel):
    transformed: List[int]
