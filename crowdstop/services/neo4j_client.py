import logging
from crowdstop.models.api import Velocity
from neomodel import config
from datetime import datetime
import uuid

from crowdstop.models.db import Camera, Place

logger = logging.getLogger(__file__)

class Neo4jClient:
    CameraNs = uuid.uuid5(uuid.NAMESPACE_DNS, 'camera.crowdstop.berkeley.edu')
    
    def __init__(self, host_url: str = None) -> None:
        self._host_url = host_url
        config.DATABASE_URL = host_url or 'bolt://neo4j:password@localhost:7687'
    
    def create_camera(self, latitude: float, longitude: float, area: float, place_ids: list[str]) -> str:
        existing = Camera.nodes.filter(latitude=latitude, longitude=longitude)
        if existing:
            logger.info(f'Camera at ({latitude, longitude}) already exists with id {existing[0].uuid}, skipping creation')
            return existing[0].uuid
        
        new_camera = Camera(
            uuid=uuid.uuid5(self.CameraNs, f'{latitude}.{longitude}').hex,
            latitude=latitude,
            longitude=longitude,
            area=area,
        )
        
        for place_id in place_ids:
            place = Place.nodes.get(uuid=place_id)
            assert place is not None, f'Could not find place with id {place_id}'
            new_camera.places.connect(place)
        
        new_camera.save() 
        logger.info(f'Created new camera with id {new_camera.uuid}')
        return new_camera.uuid
    
    def update_camera(self, id: str, timestamp: datetime, count: int, velocities: dict[str, float]):
        logger.info(f'Updating camera {id} with count {count} and velocities {velocities}...')
    
        camera: Camera = Camera.nodes.get(uuid=id)
        camera.people_count = count
        camera.people_velocities = velocities.values()
        camera.last_updated = timestamp
        camera.save()
        
        for place in camera.places.all():
            place: Place
            place.estimated_count -= velocities.get(place.uuid, 0)  # Sign of velocity is positive if coming towards camera
            place.last_updated = max(timestamp, place.last_updated)
            place.save()

    def delete_camera(self, id: str):
        logger.info(f'Deleting camera {id}...')
        camera: Camera = Camera.nodes.get(uuid=id)
        if camera:
            camera.delete()