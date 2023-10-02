from pydantic import BaseModel

class CameraUpdateRequest(BaseModel):
    timestamp: int
    density: int
    # TODO: add flux
