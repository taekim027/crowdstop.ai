from pydantic import BaseModel


class PlaceCreateRequest(BaseModel):
    latitude: float
    longitude: float
    area: float         # approx. area in sqft covered by camera

class CameraCreateRequest(BaseModel):
    latitude: float
    longitude: float
    area: float         # approx. area in sqft covered by camera
    place_ids: list[str]

class CreateResponse(BaseModel):
    uuid: str

class Velocity(BaseModel):
    place_id: str
    movement: int   # Positive indicates towards camera, negative indicates towards the place

class CameraUpdateRequest(BaseModel):
    timestamp: str
    count: int      # people in frame
    velocities: dict[str, float]
