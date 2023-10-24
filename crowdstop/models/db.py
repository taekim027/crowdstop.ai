from neomodel import StructuredNode, UniqueIdProperty, FloatProperty, StringProperty, ArrayProperty, RelationshipTo, RelationshipFrom, config

config.DATABASE_URL = 'bolt://neo4j:password@localhost:7687'


class Camera(StructuredNode):
    # Unchanging properties
    uuid = UniqueIdProperty()
    latitude = FloatProperty(required=True)
    longitude = FloatProperty(required=True)
    area = FloatProperty(required=True)
    
    # Variables, updated via API call
    people_count = FloatProperty(default=0)
    people_velocities = ArrayProperty(base_property=FloatProperty(default=0))
    
    places = RelationshipTo('PointOfInterest', 'PLACES')
    
    @property
    def density(self) -> float:
        return self.people_count / self.area
    
class Place(StructuredNode):
    uuid = UniqueIdProperty()
    latitude = FloatProperty(required=True)
    longitude = FloatProperty(required=True)
    area = FloatProperty(required=True)
    
    estimated_count = FloatProperty(default=0)
    
    # cameras
    


# EXAMPLE
# class Book(StructuredNode):
#     title = StringProperty(unique_index=True)
#     author = RelationshipTo('Author', 'AUTHOR')

# class Author(StructuredNode):
#     name = StringProperty(unique_index=True)
#     books = RelationshipFrom('Book', 'AUTHOR')


# harry_potter = Book(title='Harry potter and the..').save()
# rowling =  Author(name='J. K. Rowling').save()
# harry_potter.author.connect(rowling)

# if __name__ == '__main__':
#     # harry_potter = Book(title='Harry potter and the..').save()
#     books = Book.nodes.all()
#     breakpoint()