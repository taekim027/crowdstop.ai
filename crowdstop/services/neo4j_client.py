import logging
from neomodel import config

from crowdstop.models.db import Camera, Place

logger = logging.getLogger(__file__)

class Neo4jClient:
    
    def __init__(self, host_url: str) -> None:
        self._host_url = host_url
        config.DATABASE_URL = host_url # e.g. 'bolt://neo4j:password@localhost:7687'
    
    def update_camera(self, id: str, timestamp: int, count: int, velocities: list[float]):
        logger.info(f'Updating camera {id} with count {count} and velocities {velocities}...')
    
        camera: Camera = Camera.nodes.get(id=id)
        camera.people_count = count
        camera.people_velocities = velocities
        camera.save()
        
        for place, velocity in zip(camera.places.all(), velocities):
            place: Place
            place.estimated_count -= velocity   # Sign of velocity is positive if coming towards camera
            place.save()
