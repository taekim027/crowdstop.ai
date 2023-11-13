import os
from neomodel import (
    StructuredNode, UniqueIdProperty, FloatProperty, StringProperty, ArrayProperty, 
    DateTimeProperty, RelationshipTo, RelationshipFrom, config
)

config.DATABASE_URL = 'bolt://neo4j:password@localhost:7687'

DENSITY_ALERT_THRESHOLD = int(os.getenv('DENSITY_ALERT_THRESHOLD', '10'))


class Camera(StructuredNode):
    # Unchanging properties
    uuid = UniqueIdProperty()
    latitude = FloatProperty(required=True)
    longitude = FloatProperty(required=True)
    area = FloatProperty(required=True)
    
    # Variables, updated via API call
    people_count = FloatProperty(default=0)
    people_velocities = ArrayProperty(base_property=FloatProperty(default=0))
    last_updated = DateTimeProperty(default_now=True)
    
    places = RelationshipTo('Place', 'PLACES')
    
    @property
    def density(self) -> float:
        return self.people_count / self.area
    
class Place(StructuredNode):
    uuid = UniqueIdProperty()
    latitude = FloatProperty(required=True)
    longitude = FloatProperty(required=True)
    area = FloatProperty(required=True)
    
    people_count = FloatProperty(default=0)
    last_updated = DateTimeProperty(default_now=True)
    