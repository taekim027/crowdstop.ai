from pydantic import BaseModel

class Place(BaseModel):
    latitude: float
    longitude: float
    area: float
    polygon: list[list[int]]

class CameraConfig(BaseModel):
    name: str
    latitude: float
    longitude: float
    area: float
    places: list[Place]
