from pydantic import BaseModel
from typing import Any, Dict

class ExportSchema(BaseModel):
    exported: Dict[str, Any]
