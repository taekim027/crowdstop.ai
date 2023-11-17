from fastapi.testclient import TestClient
from datetime import datetime, timezone
import random

from crowdstop.models.api import CameraCreateRequest, CreateResponse, CameraUpdateRequest, PlaceCreateRequest

from .conftest import client

def test_docs(client: TestClient):
    response = client.get('/docs')
    assert response.status_code == 200
    for text in ['DOCTYPE html', 'swagger-ui']:
        assert text in response.text

def test_openapi_json(client: TestClient):
    response = client.get('/openapi.json')
    assert response.status_code == 200
    openapi_spec = response.json()
    assert openapi_spec['openapi'].startswith('3.'), \
        f'Expected Open API spec 3+, got {openapi_spec["openapi"]}'
    
def test_health(client: TestClient):
    response = client.get('/health').json()
    assert response.get('status') == 'healthy', 'Expected "status" == "healthy" in response'

def test_update(client: TestClient):
    
    # Create places for the camera to point to
    print('Creating places...')
    place_ids = []
    for i in range(4):
        response = client.post(
            '/place',
            json=PlaceCreateRequest(
                latitude=i,
                longitude=i,
                area=10
            ).model_dump()
        )
        response.raise_for_status()
        place_ids.append(response.json()['uuid'])
    
    # Create the camera
    print('Creating the camera...')
    response = client.post(
        '/camera',
        json=CameraCreateRequest(latitude=10, longitude=10, area=10, place_ids=place_ids).model_dump()
    )
    response.raise_for_status()
    camera_id = CreateResponse(**response.json()).uuid
    
    # Test an example update API call
    print('Sending camera update...')
    velocities = {
        id: random.randint(-10, 10)
        for id in place_ids
    }
    response = client.put(
        f'/camera/{camera_id}', 
        json=CameraUpdateRequest(
            timestamp=str(datetime.now().replace(tzinfo=timezone.utc)), 
            count=10, 
            velocities=velocities
        ).model_dump()
    )
    response.raise_for_status()
    
    # Clean up camera
    response = client.delete(
        url=f'/camera/{camera_id}'
    )
    response.raise_for_status()