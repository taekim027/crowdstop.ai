from fastapi.testclient import TestClient
from datetime import datetime

from crowdstop.models.api import CameraCreateRequest, CameraCreateResponse, CameraUpdateRequest

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
    
    response = client.post(
        '/camera',
        json=CameraCreateRequest(latitude=10, longitude=10, area=10, place_ids=[]).model_dump()
    )
    response.raise_for_status()
    camera_id = CameraCreateResponse(**response.json()).uuid
    
    response = client.put(
        f'/camera/{camera_id}', 
        json=CameraUpdateRequest(timestamp=str(datetime.now()), count=10, velocities=dict()).model_dump()
    )
    response.raise_for_status()
    
    response = client.delete(
        url=f'/camera/{camera_id}'
    )
    response.raise_for_status()