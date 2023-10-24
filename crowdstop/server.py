import logging

from fastapi import FastAPI

from crowdstop.models.api import CameraUpdateRequest

logger = logging.getLogger(__file__)
app = FastAPI()


@app.put('/camera/{camera_id}')
def update_camera(camera_id: str, request: CameraUpdateRequest) -> None:
    logger.info(f'Incoming request to update camera {camera_id}: {request}')


@app.get("/health")
async def health():
    return {"status": "healthy"}