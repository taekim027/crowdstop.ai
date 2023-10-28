import logging
from dateutil import parser
from fastapi import FastAPI

from crowdstop.models.api import CameraCreateRequest, CameraCreateResponse, CameraUpdateRequest
from crowdstop.services.neo4j_client import Neo4jClient

logger = logging.getLogger(__file__)
app = FastAPI()

neo4j_client = Neo4jClient()

@app.post('/camera')
def create_camera(request: CameraCreateRequest) -> CameraCreateResponse:
    logger.info(f'Incoming request to creat camera: {request}')
    camera_id = neo4j_client.create_camera(request.latitude, request.longitude, request.area, request.place_ids)
    return CameraCreateResponse(uuid=camera_id)

@app.put('/camera/{camera_id}')
def update_camera(camera_id: str, request: CameraUpdateRequest) -> None:
    logger.info(f'Incoming request to update camera {camera_id}: {request}')
    neo4j_client.update_camera(
        id=camera_id,
        timestamp=parser.parse(request.timestamp),
        count=request.count,
        velocities=request.velocities
    )
    
@app.delete('/camera/{camera_id}')
def delete_camera(camera_id: str) -> None:
    logger.info(f'Incoming request to update camera {camera_id}: {camera_id}')
    neo4j_client.delete_camera(
        id=camera_id
    )


@app.get("/health")
async def health():
    return {"status": "healthy"}