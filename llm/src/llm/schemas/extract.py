from pydantic import BaseModel

class ExtractionSchema(BaseModel):
    # This is dynamic — fields are added at runtime
    pass
