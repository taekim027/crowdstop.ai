import logging
from crowdstop.models.api import Velocity
from neomodel import config
from datetime import datetime

from crowdstop.models.db import Camera, Place

logger = logging.getLogger(__file__)

class Neo4jClient:
    
    def __init__(self, host_url: str = None) -> None:
        self._host_url = host_url
        config.DATABASE_URL = host_url or 'bolt://neo4j:password@localhost:7687'
    
    def update_camera(self, id: str, timestamp: datetime, count: int, velocities: dict[str, float]):
        logger.info(f'Updating camera {id} with count {count} and velocities {velocities}...')
    
        camera: Camera = Camera.nodes.get(id=id)
        camera.people_count = count
        camera.people_velocities = velocities.values()
        camera.last_updated = timestamp
        camera.save()
        
        for place in camera.places.all():
            place: Place
            place.estimated_count -= velocities.get(place.uuid, 0)  # Sign of velocity is positive if coming towards camera
            place.last_updated = max(timestamp, place.last_updated)
            place.save()
